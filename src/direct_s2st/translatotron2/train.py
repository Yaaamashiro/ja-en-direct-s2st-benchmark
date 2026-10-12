"""Train the native experimental TT2 core on the prepared train split only."""
import argparse
from ..progress import Progress, operation, track
from dataclasses import asdict
import json
import os
import sys
import time
from pathlib import Path
import torch
from ..io import ExistingOutputError, atomic_write_json
from .data import PreparedDataset, fingerprint
from .engine import load_checkpoint, optimization_step, save_checkpoint, capture_rank_state, restore_rank_state
from .model import ModelConfig, Translatotron2


@operation('translatotron2/train: _main')
def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--max-updates', type=int, default=2)
    parser.add_argument('--save-interval-updates', type=int, default=1)
    parser.add_argument('--model-size', choices=['smoke', 'reference', 'fisher', 'covost2', 'conversational'], default='smoke')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--warmup-updates', type=int)
    parser.add_argument('--update-freq', type=int)
    parser.add_argument('--l2-regularization', type=float)
    parser.add_argument('--reproduction-mode', choices=['smoke', 'paper_exact', 'paper_practical'])
    parser.add_argument('--vocoder-mode', choices=['griffin_lim', 'hifigan'], default='griffin_lim')
    parser.add_argument('--validate-interval', type=int, default=0)
    parser.add_argument('--validation-limit', type=int)
    parser.add_argument('--restore-file', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    from ..recipes import tt2_batch, option, training_metadata, tt2_source_spec
    from ..config import load_config
    repository = Path(__file__).resolve().parents[3]
    fisher = args.model_size in ('fisher', 'reference')
    args.reproduction_mode = args.reproduction_mode or ('paper_exact' if fisher else 'smoke')
    paper = args.reproduction_mode != 'smoke'
    if paper and not fisher:
        parser.error('paper recipe requires model-size fisher (reference is its alias)')
    recipe = load_config(repository / 'configs/translatotron2/train-fisher.yaml')['training']['command'] if fisher else []
    for field, flag, fallback, convert in [('learning_rate', '--learning-rate', 1e-4, float),
        ('warmup_updates', '--warmup-updates', 0, int), ('l2_regularization', '--l2-regularization', 0.0, float)]:
        expected = convert(option(recipe, flag, fallback))
        if getattr(args, field) is None:
            setattr(args, field, expected)
        elif paper and getattr(args, field) != expected:
            parser.error('paper recipe cannot change Fisher optimizer settings: ' + flag)
    world_size, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    args.update_freq = tt2_batch(args.reproduction_mode, args.batch_size, args.update_freq, world_size)
    if args.reproduction_mode == 'paper_exact' and (os.environ.get('S2ST_TRAIN_ADAPTIVE_BATCH') == '1' or
            os.environ.get('S2ST_TRAIN_PRECISION', 'default') not in ('default', 'fp32')):
        parser.error('paper_exact forbids adaptive batch-statistics/mixed-precision changes')
    if args.num_workers < 0 or args.prefetch_factor < 1:
        parser.error('num-workers must be nonnegative and prefetch-factor positive')
    if args.validation_limit is not None and args.validation_limit < 1:
        parser.error('validation-limit must be positive')
    if min(args.max_updates, args.save_interval_updates, args.batch_size, args.update_freq) < 1 or args.learning_rate <= 0 or min(args.warmup_updates, args.validate_interval, args.l2_regularization) < 0:
        parser.error('updates, interval and learning rate must be positive')
    if os.environ.get('CORPUS_ROOT') and args.run_root.resolve().is_relative_to(Path(os.environ['CORPUS_ROOT']).resolve()):
        raise ValueError('run-root must not be inside CORPUS_ROOT')
    checkpoint = args.run_root / 'checkpoints/checkpoint_last.pt'
    if checkpoint.exists() and not (args.restore_file or args.overwrite):
        raise ExistingOutputError(str(checkpoint))
    dataset = PreparedDataset(args.data_root, 'train')
    if paper:
        if (dataset.spec['sample_rate'], dataset.spec['n_mels']) not in ((16000, 80), (24000, 128)):
            raise ValueError('unsupported target frontend for Fisher adaptation')
        dataset.source_spec = tt2_source_spec(repository)
    if world_size > 1 and os.environ.get('S2ST_TRAIN_OPTIMIZE') == '1':
        raise ValueError('optimized Colab runtime supports one GPU/process only')
    if world_size > 1:
        if args.device.startswith('cuda'):
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            args.device = f'cuda:{local_rank}'
        torch.distributed.init_process_group(backend='nccl' if args.device.startswith('cuda') else 'gloo')
    identity = fingerprint(args.data_root)
    identity['training'] = dict(batch_size=args.batch_size, seed=args.seed,
                               learning_rate=args.learning_rate, warmup_updates=args.warmup_updates,
                               world_size=world_size, update_freq=args.update_freq, l2=args.l2_regularization)
    identity['training']['engine_version'] = 'tt2-cpu-mask-rng-vector-prenet-v2'
    if paper:
        identity['training'].update(reproduction_mode=args.reproduction_mode, architecture_preset='fisher',
            vocoder_mode=args.vocoder_mode, source_feature_config=dataset.source_spec,
            target_feature_config=dataset.spec, effective_batch_size=args.batch_size*args.update_freq*world_size)
    from ..train_runtime import Timings, Microbatches, enabled, pin_batch, stopping, checkpoint_saved
    if os.environ.get('S2ST_TRAIN_PRECISION', 'default') != 'default':
        identity['training']['precision'] = os.environ['S2ST_TRAIN_PRECISION']
    if os.environ.get('S2ST_TRAIN_ADAPTIVE_BATCH') == '1':
        identity['training']['adaptive_microbatches'] = True
    torch.manual_seed(args.seed)
    preset = 'fisher' if args.model_size == 'reference' else args.model_size
    config = getattr(ModelConfig, preset)()
    config.input_dim = getattr(dataset, 'source_spec', dataset.spec)['n_mels']
    config.mel_dim = dataset.spec['n_mels']
    completed = 0
    if args.restore_file:
        model, state = load_checkpoint(args.restore_file, device=args.device,
                                        expected_fingerprint=identity, restore_rng=True)
        if state['model_config'] != asdict(config) or state['tokens'] != dataset.tokens:
            raise ValueError('resume configuration/vocabulary mismatch')
        completed = state['updates']
    else:
        model = Translatotron2(config, len(dataset.tokens)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_regularization)
    if args.restore_file:
        optimizer.load_state_dict(state['optimizer'])
    from .monitor import resume_schedule
    requested_warmup = os.environ.get('S2ST_TT2_PRACTICAL_WARMUP')
    effective_warmup, schedule_transition = resume_schedule(
        args.warmup_updates, completed, state if args.restore_file else None,
        int(requested_warmup) if requested_warmup is not None else None)
    if schedule_transition:
        if world_size != 1:
            raise ValueError('practical warmup continuation requires a single process')
        print(f'[training-schedule] mode=paper_practical base_warmup={args.warmup_updates} '
              f'effective_warmup={effective_warmup} adopted_at={schedule_transition["adopted_at"]} '
              'optimizer_state=restored absolute_updates=True', file=sys.stderr, flush=True)
    metadata = None
    if paper and rank == 0:
        command = ['--model-size', 'fisher', '--learning-rate', str(args.learning_rate),
                   '--warmup-updates', str(args.warmup_updates), '--l2-regularization', str(args.l2_regularization),
                   '--batch-size', str(args.batch_size), '--update-freq', str(args.update_freq)]
        metadata = training_metadata(command, 'tt2', args.reproduction_mode, args.vocoder_mode, world_size=world_size)
        metadata['precision'] = os.environ.get('S2ST_TRAIN_PRECISION', 'default')
        if metadata['precision'] == 'default':
            metadata['precision'] = 'fp32'
        import subprocess
        metadata.update(repository_revision=subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(),
            model_config=asdict(config), source_feature_config=dataset.source_spec, target_feature_config=dataset.spec,
            phonemizer=load_config(repository / 'configs/translatotron2/prepare.yaml')['phonemizer'],
            dataset_fingerprint=identity, max_updates=args.max_updates, completed_updates=completed)
        if dataset.spec['sample_rate'] == 24000:
            metadata['deviations'] = [v for v in metadata['deviations'] if 'target retained' not in v]
        if schedule_transition:
            metadata['reproduction_mode'] = 'paper_practical'
            metadata['schedule_transition'] = schedule_transition
            metadata['base_warmup_updates'] = args.warmup_updates
            metadata['warmup_updates'] = effective_warmup
            metadata['effective_warmup_updates'] = effective_warmup
            metadata['deviations'].append('Explicit warmup shortening on resume; original fingerprint records base training, not the effective continuation schedule')
        atomic_write_json(args.run_root / 'research-metadata.json', metadata,
                          resume=True, overwrite=args.overwrite or bool(args.restore_file))
    core = model
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(core,
            device_ids=[torch.cuda.current_device()] if args.device.startswith('cuda') else None,
            broadcast_buffers=False)
    if args.restore_file and state.get('rank_states'):
        restore_rank_state(model, state['rank_states'][rank])
    elif not args.restore_file:
        torch.manual_seed(args.seed+rank)
    if os.environ.get('S2ST_BATCH_PROBE'):
        # Separate process, train-only stress samples, both backward AND Adam
        # state allocation. No probe optimizer/weights enter the real run.
        if args.restore_file or world_size != 1:
            raise ValueError('batch probes require fresh single-process training')
        from ..batch_probe import Probe
        from .batching import collate, learning_rate
        probe = Probe()
        keys = [lambda i: int(dataset.rows[i].get('src_n_frames', dataset.rows[i]['tgt_n_frames'])),
                lambda i: int(dataset.rows[i]['tgt_n_frames']),
                lambda i: len(dataset.labels[dataset.rows[i]['id']].split())]
        extremes = [dataset[max(range(len(dataset)), key=key)] for key in
                    track(keys, 'tt2: load train-only padding extremes')]
        # Cross-example padding can combine max source/target/phone lengths.
        # Use a conservative graph envelope, not a claimed translation pair.
        sample = {key: extremes[index][key] for index, fields in enumerate(
                  [('source', 'source_lengths'), ('target', 'target_lengths'), ('phones', 'phone_lengths')])
                  for key in fields}
        # A completion exception unwinds a suspended track() generator as
        # GeneratorExit. Count optimizer steps explicitly before that signal.
        with Progress('tt2: disposable train-length stress batches',
                      total=probe.limit, counted=True) as progress:
            for step in range(probe.limit):
                progress.current = str(step + 1)
                batch = collate([sample] * args.batch_size)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(args.learning_rate, step + 1, args.warmup_updates)
                probe.begin()
                optimization_step(model, optimizer, batch)
                progress.count = step + 1
                progress.advanced = time.monotonic()
                probe.finish(args.batch_size)
        raise RuntimeError('probe observer did not stop the disposable trainer')
    from .monitor import Monitor, policy_from_env
    requested = policy_from_env()
    monitor = Monitor.restore(requested, state.get('monitor_state') if args.restore_file else None,
                              completed, effective_warmup, args.run_root)
    if monitor and world_size != 1:
        raise ValueError('TT2 monitor currently supports the single-process Colab trainer only')
    if monitor and args.validation_limit is not None:
        raise ValueError('dev-loss stopping requires the full dev split, not a changing subset')
    if monitor:
        args.validate_interval = monitor.policy['interval']
        print('[training-monitor] metric=teacher_forced_dev_loss (not BLEU) '
              f'interval={args.validate_interval} patience={monitor.policy["patience"]} '
              f'warmup_guard={effective_warmup} adopted_at={monitor.origin}', file=sys.stderr, flush=True)
        if args.max_updates <= effective_warmup:
            print('[training-monitor] total_updates <= warmup: early stopping cannot trigger before the requested cap',
                  file=sys.stderr, flush=True)
        if metadata is not None:
            metadata['monitoring'] = dict(policy=monitor.policy, metric='teacher_forced_dev_loss',
                                          warmup_guard=effective_warmup, adopted_at=monitor.origin,
                                          paper_stopping_rule_reproduced=False)
        if monitor.stopped:
            monitor.write_reports(args.run_root)
            print('[training-monitor] restored terminal early-stop checkpoint; no optimizer update', file=sys.stderr, flush=True)
            return
    dev = PreparedDataset(args.data_root, 'dev') if args.validate_interval and rank == 0 else None
    if paper and dev is not None:
        dev.source_spec = dataset.source_spec
    tuning = Microbatches(state.get('performance_state') if args.restore_file else None)
    timing = Timings()
    from .batching import collate, learning_rate, StreamingBatches
    from ..prefetch import ordered_samples
    indices = (((update-1)*args.update_freq+micro)*args.batch_size*world_size + rank*args.batch_size+i
               for update in range(completed+1, args.max_updates+1)
               for micro in range(args.update_freq) for i in range(args.batch_size))
    with ordered_samples(lambda i: dataset[i % len(dataset)], indices,
                         args.num_workers, args.prefetch_factor) as samples:
        for update in track(range(completed + 1, args.max_updates + 1), 'tt2: train updates'):
            from ..stream_inputs import prefetch_dataset
            prefetch_dataset(dataset,
                             (update-1)*args.update_freq*args.batch_size*world_size+rank*args.batch_size,
                             max(128, 2*args.update_freq*args.batch_size*world_size))
            with timing.measure('data_wait_and_collate'):
                streaming = paper and not tuning.active
                batches = (StreamingBatches(samples, dataset, update, args.batch_size, args.update_freq, rank, world_size, timing)
                           if streaming else [collate([next(samples) for _ in range(args.batch_size)])
                                              for _ in range(args.update_freq)])
                if not streaming and enabled() and args.device.startswith('cuda'):
                    batches = [pin_batch(batch) for batch in batches]
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(args.learning_rate, update, effective_warmup)
            with timing.measure('optimization', args.device if enabled() else None):
                losses = (optimization_step(model, optimizer, batches) if streaming else
                          tuning.run(model, optimizer, batches,
                                     lambda pieces: optimization_step(model, optimizer, pieces)))
            if rank == 0:
                atomic_write_json(args.run_root / 'losses' / f'{update:08d}.json',
                                  dict(update=update, **losses), overwrite=args.overwrite or bool(args.restore_file))
            validation = None
            if args.validate_interval and update % args.validate_interval == 0:
                if rank == 0:
                    from .validation import validate
                    validation = validate(core, dev, args.device, args.validation_limit)
                    atomic_write_json(args.run_root / 'validation' / f'{update:08d}.json', validation,
                                      overwrite=args.overwrite or bool(args.restore_file))
                if world_size > 1:
                    torch.distributed.barrier()
            improved = monitor.observe(update, losses, validation, optimizer.param_groups[0]['lr']) if monitor else False
            stop = stopping() or bool(monitor and monitor.stopped)
            if monitor and validation is not None:
                print(f'[training-monitor] update={update} dev_loss={validation["loss"]:.6g} '
                      f'best_update={monitor.best_update} bad_evaluations={monitor.bad_evaluations}/'
                      f'{monitor.policy["patience"]} warmup_guard={update <= monitor.warmup} '
                      f'early_stopped={monitor.stopped}', file=sys.stderr, flush=True)
            if update % args.save_interval_updates == 0 or update == args.max_updates or stop or (monitor and validation is not None):
                local_state = capture_rank_state(model)
                states = [None]*world_size
                if world_size > 1:
                    torch.distributed.all_gather_object(states, local_state)
                else:
                    states = [local_state]
                if rank == 0:
                    with timing.measure('checkpoint_and_local_freeze'):
                        if monitor and improved:
                            monitor.save_best(args.run_root, lambda path: save_checkpoint(
                                path, model, optimizer, update, dataset.tokens, identity, overwrite=True,
                                rank_states=states, performance_state=tuning.state(), schedule_transition=schedule_transition))
                        save_checkpoint(checkpoint, model, optimizer, update, dataset.tokens, identity,
                                        overwrite=args.overwrite or bool(args.restore_file) or update > 1,
                                        rank_states=states, performance_state=tuning.state(),
                                        monitor_state=monitor.state() if monitor else None,
                                        schedule_transition=schedule_transition)
                        if monitor:
                            monitor.write_reports(args.run_root, render=validation is not None or stop or update == args.max_updates)
                        if metadata is not None:
                            atomic_write_json(args.run_root / 'research-metadata.json',
                                              dict(metadata, completed_updates=update), overwrite=True)
                        checkpoint_saved(args.run_root, checkpoint, update,
                                         force=stop or update == args.max_updates)
            if rank == 0:
                print(json.dumps(dict(update=update, **losses)), flush=True)
                if enabled():
                    timing.report(update, batches.source_frames if streaming else sum(int(b['source_lengths'].sum()) for b in batches),
                                  units='source_frames', microbatch=tuning.size if tuning.active else args.batch_size)
            if stop:
                break


def main():
    try:
        _main()
    finally:
        # Cancel queued look-ahead at completion/error, before Python joins the
        # executor at exit. At most two already-started transfers finish.
        from ..stream_inputs import close_current
        close_current()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
