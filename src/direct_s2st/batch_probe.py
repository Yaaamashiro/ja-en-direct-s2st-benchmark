"""Disposable real optimizer trials; never resume or publish probe weights."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

from .io import atomic_write_json
from .progress import operation


class ProbeComplete(BaseException):
    progress_status = 'completed'


class Probe:
    def __init__(self):
        value = os.environ.get('S2ST_BATCH_PROBE')
        self.path = Path(value) if value else None
        self.steps = []
        self.limit = int(os.environ.get('S2ST_BATCH_PROBE_STEPS', '3'))

    def begin(self):
        if not self.path:
            return
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        self.external = max(0, total - free - torch.cuda.memory_reserved())
        self.started = time.monotonic()

    def finish(self, samples):
        if not self.path:
            return
        import torch
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        self.steps.append(dict(seconds=time.monotonic() - self.started, samples=int(samples),
            allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved(),
            used=max(total - free, torch.cuda.max_memory_reserved() + self.external), total=total))
        atomic_write_json(self.path, dict(status='running', steps=self.steps), overwrite=True)
        print(f'[batch-probe] checked={len(self.steps)}/{self.limit} samples={samples} '
              f'vram_used={self.steps[-1]["used"]}/{total}', file=sys.stderr, flush=True)
        if len(self.steps) >= self.limit:
            raise ProbeComplete()


@operation('training: disposable VRAM trial')
def main():
    """Run a supported trainer in a fresh process, with isolated output paths."""
    import torch
    from .train_runtime import RecoverableOOM
    command = sys.argv[1:]
    if len(command) < 3 or command[0] != '-m' or not os.environ.get('S2ST_BATCH_PROBE'):
        raise ValueError('probe requires -m trainer and a private result path')
    if not torch.cuda.is_available():
        raise ValueError('VRAM calibration requires CUDA')
    if any(v in command for v in ('--restore-file', '--resume')):
        raise ValueError('probe must start from fresh weights')
    if os.environ.get('S2ST_TRAIN_STAGING'):
        raise ValueError('probe must not publish training checkpoints')
    path = Path(os.environ['S2ST_BATCH_PROBE'])
    sys.argv = command[1:]
    try:
        runpy.run_module(command[1], run_name='__main__')
        raise RuntimeError('trainer exited before completing probe measurements')
    except ProbeComplete:
        result = json.loads(path.read_text(encoding='utf-8'))
        result['status'] = 'ok'
    except (torch.cuda.OutOfMemoryError, RecoverableOOM):
        result = dict(status='oom')
    atomic_write_json(path, result, overwrite=True)


if __name__ == '__main__':
    main()
