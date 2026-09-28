"""Native PyTorch optimization/checkpoints; no upstream fairseq substitution."""
from dataclasses import asdict
import os
import random
from pathlib import Path
import tempfile
import torch
from ..io import ExistingOutputError
from .model import ModelConfig, Translatotron2


def optimization_step(model, optimizer, batch):
    from contextlib import nullcontext
    from ..train_runtime import forward_backward_guard, precision_context
    batches = batch if isinstance(batch, list) else [batch]
    if not batches:
        raise ValueError('empty optimizer update')
    core = model.module if hasattr(model, 'module') else model
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    counts = torch.tensor([sum(int(b['target_lengths'].sum()) for b in batches),
                           sum(int(b['phone_lengths'].sum()) for b in batches),
                           sum(b['source'].size(0) for b in batches)], device=device, dtype=torch.float64)
    distributed = torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if distributed else 1
    if distributed:
        torch.distributed.all_reduce(counts)
    values = dict(loss=0., mel_loss=0., phone_loss=0., duration_loss=0.)
    guard = forward_backward_guard() if os.environ.get('S2ST_TRAIN_ADAPTIVE_BATCH') == '1' else nullcontext()
    with guard:
        for index, sample in enumerate(batches):
            sample = {key: value.to(device, non_blocking=True) for key, value in sample.items()}
            synchronize = index == len(batches)-1 or not hasattr(model, 'no_sync')
            with (nullcontext() if synchronize else model.no_sync()), precision_context(device):
                output = model(**sample)
                if any(not torch.isfinite(output[name]) for name in values):
                    raise ValueError('nonfinite TT2 objective')
                weights = [float(sample['target_lengths'].sum()) / float(counts[0]),
                           float(sample['phone_lengths'].sum()) / float(counts[1]),
                           sample['source'].size(0) / float(counts[2])]
                weighted = [output[name]*weight for name, weight in zip(
                    ('mel_loss', 'phone_loss', 'duration_loss'), weights)]
                loss = weighted[0] + core.config.phone_weight*weighted[1] + core.config.duration_weight*weighted[2]
                (loss*world_size).backward()
                values['loss'] += float(loss.detach())
                for name, value in zip(('mel_loss', 'phone_loss', 'duration_loss'), weighted):
                    values[name] += float(value.detach())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise ValueError(f'missing/nonfinite gradient: {name}')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite parameter after optimizer update')
    if distributed:
        tensor = torch.tensor(list(values.values()), device=device, dtype=torch.float64)
        torch.distributed.all_reduce(tensor)
        values = dict(zip(values, tensor.tolist()))
    return {**values, 'gradient_norm': float(norm)}


def capture_rank_state(model):
    import numpy as np
    numpy_state = np.random.get_state()
    core = model.module if hasattr(model, 'module') else model
    return dict(rng=torch.get_rng_state(),
                python_rng=random.getstate(),
                numpy_rng=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                buffers={name: buffer.detach().cpu().clone() for name, buffer in core.named_buffers()})


def restore_rank_state(model, state):
    import numpy as np
    core = model.module if hasattr(model, 'module') else model
    with torch.no_grad():
        for name, buffer in core.named_buffers():
            buffer.copy_(state['buffers'][name].to(buffer.device))
    torch.set_rng_state(state['rng'])
    if 'python_rng' in state:
        random.setstate(state['python_rng'])
        numpy_state = state['numpy_rng']
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if state['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def save_checkpoint(path, model, optimizer, updates, tokens, data_fingerprint, *, overwrite=False, rank_states=None, performance_state=None):
    model = model.module if hasattr(model, 'module') else model
    path = Path(path)
    if updates < 1 or not optimizer.state:
        raise ValueError('checkpoint requires real optimizer updates')
    if path.exists() and not overwrite:
        raise ExistingOutputError(f'checkpoint already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format='direct-s2st-tt2-v2', model_config=asdict(model.config),
                   model=model.state_dict(), optimizer=optimizer.state_dict(), updates=updates,
                   tokens=tokens, data_fingerprint=data_fingerprint, rank_states=rank_states, performance_state=performance_state, rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.pt.tmp')
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path, *, device='cpu', expected_fingerprint=None, restore_rng=False):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') not in ('direct-s2st-tt2-v1', 'direct-s2st-tt2-v2') or payload.get('updates', 0) < 1:
        raise ValueError('not an updated native TT2 checkpoint')
    if expected_fingerprint is not None and payload['data_fingerprint'] != expected_fingerprint:
        raise ValueError('checkpoint data fingerprint mismatch')
    if not payload.get('optimizer', {}).get('state'):
        raise ValueError('checkpoint missing optimizer state')
    config = dict(payload['model_config'])
    if payload['format'] == 'direct-s2st-tt2-v1':
        config['architecture_revision'] = 1
    model = Translatotron2(ModelConfig(**config), len(payload['tokens'])).to(device)
    model.load_state_dict(payload['model'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite checkpoint parameters')
    if restore_rng:
        torch.set_rng_state(payload['rng'])
        if payload['cuda_rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return model, payload
