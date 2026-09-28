"""Scoped hooks for the pinned fairseq trainer; never edit third_party sources."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time

from ..train_runtime import (Microbatches, Timings, cached_file, checkpoint_saved,
                             enabled, forward_backward_guard, stopping)


@contextmanager
def runtime_hooks(audit):
    if not enabled():
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
        if Path(path).is_file():
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
        before = self.get_num_updates()
        if self.data_parallel_world_size != 1:
            raise ValueError('optimized Colab S2UT runtime supports one GPU/process only')
        if not samples or any(not sample for sample in samples):
            raise ValueError('optimized S2UT requires real nonempty logical batches')
        now = time.monotonic()
        if hasattr(self, '_s2st_last_step'):
            self._s2st_timing.seconds['between_updates_including_io_validation_save'] += now-self._s2st_last_step
        with self._s2st_timing.measure('optimization', 'cuda' if self.cuda else None):
            result = self._s2st_tuning.run(self.model, self.optimizer, samples,
                                         lambda pieces: original_step(self, pieces, raise_oom=True))
        if self.get_num_updates() != before + 1:
            raise RuntimeError('fairseq did not complete an optimizer update (e.g. overflow); refusing to skip samples')
        self._s2st_timing.report(self.get_num_updates(), sum(int(s['ntokens']) for s in samples),
                                 units='target_units', microbatch=self._s2st_tuning.size)
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
        result = original_publish(cfg, trainer, epoch_itr, val_loss)
        checkpoint = Path(cfg.save_dir) / 'checkpoint_last.pt'
        if trainer.should_save_checkpoint_on_current_rank and checkpoint.is_file():
            atomic_write_json(checkpoint.parent.parent / 'gradient-audit-rank-0.json', audit.result(), overwrite=True)
            checkpoint_saved(checkpoint.parent.parent, checkpoint, trainer.get_num_updates())
        return result

    def iterator(self, *args, **kwargs):
        if self.cfg.dataset.num_workers == 0:
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
        return original_iterator(self, *args, **kwargs)

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
