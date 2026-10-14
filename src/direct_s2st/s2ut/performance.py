"""Scoped hooks for the pinned fairseq trainer; never edit third_party sources."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time

from ..train_runtime import (Microbatches, Timings, cached_file, checkpoint_saved,
                             enabled, forward_backward_guard, stopping)


def completed_step(trainer, samples, step, audit, *, max_retries=32):
    """Retry only confirmed pre-optimizer FP16 overflow, never skip a batch."""
    import math
    import torch
    from ..translatotron2.engine import capture_rank_state, restore_rank_state
    optimizer = trainer.optimizer
    fp16 = bool(getattr(getattr(getattr(trainer, 'cfg', None), 'common', None), 'fp16', False))
    scaler = getattr(optimizer, 'scaler', None)
    if fp16 and scaler is None:
        raise ValueError('FP16 retry requires the pinned dynamic loss scaler')
    before = trainer.get_num_updates()
    rng = capture_rank_state(trainer.model) if fp16 else None
    criterion = getattr(trainer, 'criterion', None)
    criterion_rng = capture_rank_state(criterion) if fp16 and isinstance(criterion, torch.nn.Module) else None
    evidence = audit.attempt_state() if audit is not None else None
    controller = getattr(audit, 'monitor', None)
    collector = getattr(controller, 'collector', None)
    entered_optimizer = False

    def attempt(pieces):
        # Also discard evidence from the existing pre-backward OOM retry path.
        if evidence is not None:
            audit.restore_attempt(evidence)
        if collector is not None:
            controller.collector, controller.pending = type(collector)(), None
        return step(pieces)

    original_step = optimizer.step if fp16 else None
    had_local_step = 'step' in vars(optimizer) if fp16 else False
    def guarded_step(*args, **kwargs):
        nonlocal entered_optimizer
        entered_optimizer = True
        return original_step(*args, **kwargs)

    if fp16:
        optimizer.step = guarded_step
    try:
        for retry in range(max_retries+1):
            entered_optimizer = False
            old_scale = float(scaler.loss_scale) if fp16 else None
            if fp16 and (not math.isfinite(old_scale) or old_scale <= 0):
                raise ValueError('invalid FP16 loss scale')
            result = trainer._s2st_tuning.run(trainer.model, optimizer, samples, attempt)
            after = trainer.get_num_updates()
            if after == before+1 and result is not None:
                return result
            new_scale = float(scaler.loss_scale) if fp16 else None
            confirmed = (fp16 and after == before and result is None and not entered_optimizer and
                         math.isfinite(new_scale) and 0 < new_scale < old_scale)
            if not confirmed:
                raise RuntimeError('fairseq did not complete an optimizer update; no confirmed pre-optimizer '
                                   'FP16 overflow, refusing to skip samples or retry an optimizer step')
            if retry == max_retries:
                raise RuntimeError('FP16 overflow retry limit reached; update was not skipped')
            optimizer.zero_grad()
            restore_rank_state(trainer.model, rng)
            if criterion_rng is not None:
                restore_rank_state(criterion, criterion_rng)
            print(f'[fp16-retry] update={before+1} retry={retry+1}/{max_retries} '
                  f'loss_scale={old_scale:g}->{new_scale:g} same_logical_batch=true optimizer_entered=false',
                  file=sys.stderr, flush=True)
    finally:
        if fp16:
            if had_local_step:
                optimizer.step = original_step
            else:
                del optimizer.step


@contextmanager
def runtime_hooks(audit):
    if not enabled():
        if os.environ.get('S2ST_INPUT_MAP'):
            from fairseq.data.audio import speech_to_text_dataset as audio
            from ..drive_staging import local_path
            original = audio.get_features_or_waveform
            def staged_audio(path, *args, **kwargs):
                return original(str(local_path(path)), *args, **kwargs)
            audio.get_features_or_waveform = staged_audio
            try:
                yield
            finally:
                audio.get_features_or_waveform = original
        else:
            yield
        return
    if os.environ.get('S2ST_TRAIN_PRECISION') == 'bf16':
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 training requires a supported CUDA device; no silent precision fallback')
    from fairseq.trainer import Trainer
    from fairseq import checkpoint_utils
    from fairseq.tasks.speech_to_speech import SpeechToSpeechTask
    from fairseq.data.audio import speech_to_text_dataset as audio
    from ..translatotron2.engine import capture_rank_state, restore_rank_state
    from ..preparation import WorkerController, resources
    from ..io import atomic_write_json

    original_step, original_task = Trainer.train_step, SpeechToSpeechTask.train_step
    original_save, original_load = Trainer.save_checkpoint, Trainer.load_checkpoint
    original_publish = checkpoint_utils.save_checkpoint
    original_iterator = Trainer.get_train_iterator
    original_audio = audio.get_features_or_waveform

    def load_audio(path, *args, **kwargs):
        # This benchmark uses WAV source paths. Preserve upstream ZIP handling.
        from ..drive_staging import local_path
        if local_path(path).is_file():
            with cached_file(path) as local:
                return original_audio(str(local or path), *args, **kwargs)
        return original_audio(path, *args, **kwargs)

    def task_step(self, *args, **kwargs):
        if os.environ.get('S2ST_TRAIN_ADAPTIVE_BATCH') == '1':
            with forward_backward_guard():
                return original_task(self, *args, **kwargs)
        return original_task(self, *args, **kwargs)

    def step(self, samples, raise_oom=False):
        if not hasattr(self, '_s2st_tuning'):
            self._s2st_tuning, self._s2st_timing = Microbatches(), Timings()
        if not hasattr(self, '_s2st_timing'):
            self._s2st_timing = Timings()
        if self.data_parallel_world_size != 1:
            raise ValueError('optimized Colab S2UT runtime supports one GPU/process only')
        if not samples or any(not sample for sample in samples):
            raise ValueError('optimized S2UT requires real nonempty logical batches')
        from ..batch_probe import Probe
        if not hasattr(self, '_s2st_probe'):
            self._s2st_probe = Probe()
        self._s2st_probe.begin()
        now = time.monotonic()
        if hasattr(self, '_s2st_last_step'):
            self._s2st_timing.seconds['between_updates_including_io_validation_save'] += now-self._s2st_last_step
        with self._s2st_timing.measure('optimization', 'cuda' if self.cuda else None):
            result = completed_step(self, samples, lambda pieces: original_step(self, pieces, raise_oom=True), audit)
        cap = self._s2st_tuning.fixed
        physical = max(min(int(s['nsentences']), cap) if cap else int(s['nsentences']) for s in samples)
        # The measured interval is the WHOLE logical optimizer update, not
        # just its largest physical piece. Count every accumulated sample so
        # practical-mode speed selection does not reward a larger cap by fiat.
        self._s2st_probe.finish(sum(int(s['nsentences']) for s in samples))
        self._s2st_timing.report(self.get_num_updates(), sum(int(s['ntokens']) for s in samples),
                                 units='target_units', microbatch=physical)
        self._s2st_last_step = time.monotonic()
        if stopping():
            # fairseq validates/saves with its exact iterator position after this step.
            self.cfg.optimization.max_update = self.get_num_updates()
        return result

    def save(self, filename, extra_state):
        extra_state = dict(extra_state)
        if hasattr(self, '_s2st_tuning'):
            extra_state['s2st_performance'] = self._s2st_tuning.state()
            extra_state['s2st_rng'] = capture_rank_state(self.model)
        return original_save(self, filename, extra_state)

    def load(self, *args, **kwargs):
        state = original_load(self, *args, **kwargs)
        if state and 's2st_performance' in state:
            self._s2st_tuning = Microbatches(state['s2st_performance'])
            restore_rank_state(self.model, state['s2st_rng'])
        return state

    def publish(cfg, trainer, epoch_itr, val_loss):
        if cfg.write_checkpoints_asynchronously:
            raise ValueError('upstream asynchronous checkpoint writes cannot be frozen safely')
        # Keep last/best and two local interval checkpoints, rather than filling
        # the Colab disk with hundreds of full model+Adam copies. Drive recovery
        # snapshots are immutable and are not touched by fairseq's local cleanup.
        if getattr(cfg, 'keep_interval_updates', -1) < 0:
            cfg.keep_interval_updates = 2
        if getattr(cfg, 'keep_last_epochs', -1) < 0:
            cfg.keep_last_epochs = 1
        result = original_publish(cfg, trainer, epoch_itr, val_loss)
        checkpoint = Path(cfg.save_dir) / 'checkpoint_last.pt'
        if trainer.should_save_checkpoint_on_current_rank and checkpoint.is_file():
            atomic_write_json(checkpoint.parent.parent / 'gradient-audit-rank-0.json', audit.result(), overwrite=True)
            monitor = getattr(getattr(audit, 'monitor', None), 'monitor', None)
            evaluated = monitor is not None and monitor.update == trainer.get_num_updates() and monitor.due(monitor.update)
            checkpoint_saved(checkpoint.parent.parent, checkpoint, trainer.get_num_updates(),
                             force=evaluated or stopping() or trainer.get_num_updates() >= trainer.cfg.optimization.max_update)
        return result

    def iterator(self, *args, **kwargs):
        if self.cfg.dataset.num_workers == 0 and not os.environ.get('S2ST_BATCH_PROBE'):
            return original_iterator(self, *args, **kwargs)
        if not hasattr(self, '_s2st_workers'):
            cap = min(max(1, self.cfg.dataset.num_workers), os.cpu_count() or 1)
            self._s2st_workers = WorkerController(cap)
            self._s2st_epoch_time = time.monotonic()
            self._s2st_epoch_updates = self.get_num_updates()
        else:
            cpu, ram = resources()
            self._s2st_workers.observe((self.get_num_updates()-self._s2st_epoch_updates) /
                                      max(time.monotonic()-self._s2st_epoch_time, .001), cpu, ram)
            self._s2st_epoch_time, self._s2st_epoch_updates = time.monotonic(), self.get_num_updates()
        self.cfg.dataset.num_workers = self._s2st_workers.workers
        kwargs['disable_iterator_cache'] = True
        print(f'[training-loader] fairseq epoch-boundary workers={self.cfg.dataset.num_workers}',
              file=sys.stderr, flush=True)
        result = original_iterator(self, *args, **kwargs)
        if os.environ.get('S2ST_BATCH_PROBE'):
            # Probe only: run the heaviest legitimate token-budget groups
            # instead of a few randomly shuffled, unusually short examples.
            from ..batch_calibration import stress_fairseq_iterator
            stress_fairseq_iterator(result, int(os.environ['S2ST_BATCH_PROBE_STEPS']),
                                    int(self.cfg.optimization.update_freq[0]))
        return result

    Trainer.train_step, SpeechToSpeechTask.train_step = step, task_step
    Trainer.save_checkpoint, Trainer.load_checkpoint = save, load
    Trainer.get_train_iterator = iterator
    checkpoint_utils.save_checkpoint = publish
    audio.get_features_or_waveform = load_audio
    try:
        yield
    finally:
        Trainer.train_step, SpeechToSpeechTask.train_step = original_step, original_task
        Trainer.save_checkpoint, Trainer.load_checkpoint = original_save, original_load
        Trainer.get_train_iterator = original_iterator
        checkpoint_utils.save_checkpoint = original_publish
        audio.get_features_or_waveform = original_audio
