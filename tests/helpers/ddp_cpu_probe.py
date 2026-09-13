"""Two-process synthetic CPU DDP update/resume probe, never a speech E2E test."""
import argparse
from pathlib import Path
import torch
from direct_s2st.io import atomic_write_json
from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
from direct_s2st.translatotron2.engine import (optimization_step, save_checkpoint,
    load_checkpoint, capture_rank_state, restore_rank_state)


def worker(rank, directory):
    root = Path(directory)
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=(root / 'store').as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(5)
        core = Translatotron2(ModelConfig.smoke(), 6)
        model = torch.nn.parallel.DistributedDataParallel(core, broadcast_buffers=False)
        optimizer = torch.optim.Adam(core.parameters(), lr=1e-4)
        torch.manual_seed(11+rank)
        sample = dict(source=torch.randn(1, 24+rank*4, 80), source_lengths=torch.tensor([24+rank*4]),
            phones=torch.tensor([[3, 4, 2]]), phone_lengths=torch.tensor([3]),
            target=torch.randn(1, 12+rank, 80), target_lengths=torch.tensor([12+rank]))
        first = optimization_step(model, optimizer, [sample, sample])
        rank_states = [None, None]
        torch.distributed.all_gather_object(rank_states, capture_rank_state(model))
        if rank == 0:
            save_checkpoint(root / 'updated.pt', model, optimizer, 1, list('abcdef'), {'fixture': 'ddp'}, rank_states=rank_states)
        torch.distributed.barrier()
        expected = optimization_step(model, optimizer, [sample, sample])
        expected_state = {name: tensor.clone() for name, tensor in core.state_dict().items()}
        restored, state = load_checkpoint(root / 'updated.pt')
        second = torch.nn.parallel.DistributedDataParallel(restored, broadcast_buffers=False)
        opt = torch.optim.Adam(restored.parameters(), lr=1e-4)
        opt.load_state_dict(state['optimizer'])
        restore_rank_state(second, state['rank_states'][rank])
        actual = optimization_step(second, opt, [sample, sample])
        assert actual == expected, (actual, expected)
        for name, tensor in restored.state_dict().items():
            torch.testing.assert_close(tensor, expected_state[name], atol=0, rtol=0)
        atomic_write_json(root / f'rank-{rank}.json', {'status': 'PASS', 'first': first, 'resumed': actual})
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(worker, args=(str(args.directory.resolve()),), nprocs=2, join=True)
