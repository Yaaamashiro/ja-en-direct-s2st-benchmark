"""Train the native experimental TT2 core on the prepared train split only."""
import argparse
from ..progress import operation, track
from dataclasses import asdict
import json
import os
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
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--warmup-updates', type=int, default=0)
    parser.add_argument('--update-freq', type=int, default=1)
    parser.add_argument('--l2-regularization', type=float, default=0.0)
    parser.add_argument('--validate-interval', type=int, default=0)
    parser.add_argument('--validation-limit', type=int)
    parser.add_argument('--restore-file', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
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
    world_size, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
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
    core = model
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(core,
            device_ids=[torch.cuda.current_device()] if args.device.startswith('cuda') else None,
            broadcast_buffers=False)
    if args.restore_file and state.get('rank_states'):
        restore_rank_state(model, state['rank_states'][rank])
    elif not args.restore_file:
        torch.manual_seed(args.seed+rank)
    dev = PreparedDataset(args.data_root, 'dev') if args.validate_interval and rank == 0 else None
    tuning = Microbatches(state.get('performance_state') if args.restore_file else None)
    timing = Timings()
    from .batching import collate, learning_rate
    from ..prefetch import ordered_samples
    indices = (((update-1)*args.update_freq+micro)*args.batch_size*world_size + rank*args.batch_size+i
               for update in range(completed+1, args.max_updates+1)
               for micro in range(args.update_freq) for i in range(args.batch_size))
    with ordered_samples(lambda i: dataset[i % len(dataset)], indices,
                         args.num_workers, args.prefetch_factor) as samples:
        for update in track(range(completed + 1, args.max_updates + 1), 'tt2: train updates'):
            with timing.measure('data_wait_and_collate'):
                batches = [collate([next(samples) for _ in range(args.batch_size)])
                           for _ in range(args.update_freq)]
                if enabled() and args.device.startswith('cuda'):
                    batches = [pin_batch(batch) for batch in batches]
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(args.learning_rate, update, args.warmup_updates)
            with timing.measure('optimization', args.device if enabled() else None):
                losses = tuning.run(model, optimizer, batches,
                                    lambda pieces: optimization_step(model, optimizer, pieces))
            if rank == 0:
                atomic_write_json(args.run_root / 'losses' / f'{update:08d}.json',
                                  dict(update=update, **losses), overwrite=args.overwrite or bool(args.restore_file))
            if args.validate_interval and update % args.validate_interval == 0:
                if rank == 0:
                    from .validation import validate
                    validation = validate(core, dev, args.device, args.validation_limit)
                    atomic_write_json(args.run_root / 'validation' / f'{update:08d}.json', validation,
                                      overwrite=args.overwrite or bool(args.restore_file))
                if world_size > 1:
                    torch.distributed.barrier()
            stop = stopping()
            if update % args.save_interval_updates == 0 or update == args.max_updates or stop:
                local_state = capture_rank_state(model)
                states = [None]*world_size
                if world_size > 1:
                    torch.distributed.all_gather_object(states, local_state)
                else:
                    states = [local_state]
                if rank == 0:
                    with timing.measure('checkpoint_and_local_freeze'):
                        save_checkpoint(checkpoint, model, optimizer, update, dataset.tokens, identity,
                                        overwrite=args.overwrite or bool(args.restore_file) or update > 1,
                                        rank_states=states, performance_state=tuning.state())
                        checkpoint_saved(args.run_root, checkpoint, update)
            if rank == 0:
                print(json.dumps(dict(update=update, **losses)), flush=True)
                if enabled():
                    timing.report(update, sum(int(b['source_lengths'].sum()) for b in batches),
                                  units='source_frames', microbatch=tuning.size if tuning.active else args.batch_size)
            if stop:
                break


def main():
    try:
        _main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
