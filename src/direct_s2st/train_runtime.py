"""Opt-in training performance controls; no changes to the logical sampler."""
from collections import OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

from .progress import operation


def enabled():
    return os.environ.get('S2ST_TRAIN_OPTIMIZE') == '1'


def stopping():
    deadline = os.environ.get('S2ST_TRAIN_DEADLINE')
    return deadline is not None and time.monotonic() >= float(deadline)


def gpu_telemetry():
    """Best-effort hardware sample; null means unavailable, never idle/zero."""
    import subprocess
    import torch
    if not torch.cuda.is_available():
        return None
    try:
        device = torch.cuda.current_device()
        selector = getattr(torch.cuda.get_device_properties(device), 'uuid', None)
        if not selector:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            selector = visible.split(',')[device] if visible else str(device)
        output = subprocess.check_output(['nvidia-smi', '-i', str(selector),
            '--query-gpu=utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'],
            text=True, timeout=1, stderr=subprocess.DEVNULL)
        compute, memory = map(float, output.strip().split(','))
        return dict(sm_util_pct_sample=compute, memory_util_pct_sample=memory,
                    source='nvidia-smi hardware sample, not an update average')
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class Timings:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.started = self.reported = time.monotonic()
        self.units = 0

    @contextmanager
    def measure(self, name, device=None):
        # CUDA events measure queued device work without synchronizing every op.
        import torch
        cuda = device is not None and torch.device(device).type == 'cuda'
        start = time.monotonic()
        if cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        try:
            yield
        finally:
            if cuda:
                end.record()
                end.synchronize()
                self.seconds[name + '_cuda'] += begin.elapsed_time(end) / 1000
            self.seconds[name] += time.monotonic() - start

    def report(self, update, amount=0, **extra):
        self.units += amount
        now = time.monotonic()
        if now - self.reported < 10 and update != 1:
            return
        import torch
        from .preparation import resources
        cpu, ram = resources()
        gpu = {}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu = dict(vram_free=free, vram_total=total,
                       vram_peak=torch.cuda.max_memory_allocated())
        value = dict(update=update, units_per_s=self.units / max(now-self.started, .001),
                     seconds=dict(self.seconds), cpu_pct=cpu, free_ram_ratio=ram,
                     gpu_telemetry=gpu_telemetry(), **gpu, **extra)
        if 'stream_data_wait_and_collate' in self.seconds:
            value['optimization_wall_excluding_stream_loading'] = max(0.,
                self.seconds['optimization'] - self.seconds['stream_data_wait_and_collate'])
        print('[training-performance] ' + json.dumps(value), file=sys.stderr, flush=True)
        self.reported = now


class LocalCache:
    """Process-private bounded LRU. Leases prevent eviction during readers.

    Fairseq loader processes each get a share of the configured total budget.
    Nothing is written to the source tree; oversized files bypass the cache.
    """
    def __init__(self, root, budget, reserve=1024**3, total_budget=None):
        root = Path(root).resolve()
        corpus = os.environ.get('CORPUS_ROOT')
        if corpus and root.is_relative_to(Path(corpus).resolve()):
            raise ValueError('training cache must be outside CORPUS_ROOT')
        root.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=f'reader-{os.getpid()}-', dir=root))
        self.budget, self.reserve = budget, reserve
        self.shared_root, self.total_budget = root, total_budget
        self.entries, self.used = OrderedDict(), 0
        self.lock = threading.RLock()

    def reserve_bytes(self, amount):
        # A session-wide ledger also bounds caches left by exited loader workers.
        # Failed processes may leak reservations, never exceed the capacity.
        if self.total_budget is None:
            return True
        from filelock import FileLock
        from .io import atomic_write_json
        ledger = self.shared_root / 'budget.json'
        with FileLock(str(self.shared_root / 'budget.lock')):
            used = json.loads(ledger.read_text())['used'] if ledger.exists() else 0
            if used + amount > self.total_budget:
                return False
            atomic_write_json(ledger, dict(used=max(0, used + amount)), overwrite=True)
        return True

    @contextmanager
    def acquire(self, source, offset=None, length=None):
        source = Path(source)
        before = source.stat()
        size = before.st_size if length is None else length
        offset = 0 if offset is None else offset
        if offset < 0 or size < 0 or offset + size > before.st_size:
            raise ValueError('invalid cached file range')
        key = (str(source.resolve()), before.st_size, before.st_mtime_ns, offset, size)
        path = None
        with self.lock:
            if key not in self.entries and size <= self.budget:
                for old_key in list(self.entries):
                    if self.used + size <= self.budget and shutil.disk_usage(self.root).free >= size + self.reserve:
                        break
                    old = self.entries[old_key]
                    if old[2] == 0:
                        old[0].unlink()
                        self.used -= old[1]
                        self.reserve_bytes(-old[1])
                        del self.entries[old_key]
                if (self.used + size <= self.budget and shutil.disk_usage(self.root).free >= size + self.reserve
                        and self.reserve_bytes(size)):
                    fd, name = tempfile.mkstemp(dir=self.root, suffix=source.suffix if length is None else '.cache')
                    path = Path(name)
                    try:
                        expected = hashlib.sha256()
                        with os.fdopen(fd, 'wb') as target, source.open('rb') as stream:
                            stream.seek(offset)
                            remaining = size
                            while remaining:
                                block = stream.read(min(1024**2, remaining))
                                if not block:
                                    raise OSError('source truncated during cache copy')
                                target.write(block)
                                expected.update(block)
                                remaining -= len(block)
                        after = source.stat()
                        from .hashing import sha256_file
                        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                                or sha256_file(path) != expected.hexdigest()):
                            raise OSError('training cache copy verification failed')
                        self.entries[key] = [path, size, 0]
                        self.used += size
                    except BaseException:
                        path.unlink(missing_ok=True)
                        self.reserve_bytes(-size)
                        raise
            if key in self.entries:
                self.entries.move_to_end(key)
                self.entries[key][2] += 1
                path = self.entries[key][0]
        try:
            # None means caller uses the original path/range without copying.
            yield path
        finally:
            if path is not None:
                with self.lock:
                    self.entries[key][2] -= 1


_cache = None
_cache_pid = None
_cache_lock = threading.Lock()


@contextmanager
def cached_file(path, offset=None, length=None):
    from .drive_staging import local_path
    staged = local_path(path)
    if staged != Path(path) and length is None:
        yield staged
        return
    if os.environ.get('S2ST_TRAIN_STRICT_LOCAL') == '1' and str(path).replace('\\', '/').startswith('/content/drive/'):
        raise RuntimeError(f'Training input was not staged locally: {path}')
    global _cache, _cache_pid
    root = os.environ.get('S2ST_TRAIN_CACHE')
    if not enabled() or not root:
        yield None
        return
    with _cache_lock:
        if _cache is None or _cache_pid != os.getpid():
            readers = max(1, int(os.environ.get('S2ST_TRAIN_READERS', '1')))
            total = int(float(os.environ.get('S2ST_TRAIN_CACHE_GB', '8')) * 1024**3)
            _cache = LocalCache(root, total // readers, total_budget=total)
            _cache_pid = os.getpid()
    with _cache.acquire(path, offset, length) as local:
        yield local


def pin_batch(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.pin_memory() if value.device.type == 'cpu' and torch.cuda.is_available() else value
    if isinstance(value, dict):
        return {k: pin_batch(v) for k, v in value.items()}
    return value


def precision_context(device):
    import torch
    precision = os.environ.get('S2ST_TRAIN_PRECISION', 'default')
    if precision in ('default', 'fp32'):
        return nullcontext()
    if precision != 'bf16' or torch.device(device).type != 'cuda' or not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 training requires a supported CUDA device; no silent precision fallback')
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


class RecoverableOOM(BaseException):
    """Only raised before optimizer.step; bypass fairseq's sample-skipping handler."""


@contextmanager
def forward_backward_guard():
    import torch
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        raise RecoverableOOM('forward/backward allocation failed') from None


def slice_batch(batch, start, end, count):
    """Slice already-collated samples without dropping or reordering examples."""
    import torch
    if isinstance(batch, torch.Tensor):
        return batch[start:end] if batch.ndim and batch.size(0) == count else batch
    if not isinstance(batch, dict):
        return batch
    result = {k: slice_batch(v, start, end, count) for k, v in batch.items()}
    if 'nsentences' in result:
        result['nsentences'] = end-start
    if 'ntokens' in result and 'target_lengths' in result:
        result['ntokens'] = int(result['target_lengths'].sum())
    return result


class Microbatches:
    """Opt-in split tuning; logical sample order and update count stay fixed.

    Batch statistics/dropout can change: this is NOT bitwise-equivalent training.
    The optimizer phase is deliberately outside the retry boundary.
    """
    def __init__(self, state=None):
        import torch
        self.active = os.environ.get('S2ST_TRAIN_ADAPTIVE_BATCH') == '1'
        self.fixed = int(os.environ.get('S2ST_TRAIN_FIXED_MICROBATCH', '0'))
        if self.fixed < 0 or (self.fixed and self.active):
            raise ValueError('fixed microbatch is nonnegative and cannot combine with online adaptation')
        if state and state.get('fixed', 0) != self.fixed:
            raise ValueError('saved fixed microbatch differs from the runtime recipe; use a new run, not a silent batch change')
        self.hardware = (torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else ('cpu', 0)
        if state and tuple(state.get('hardware', ())) != self.hardware:
            state = None
        self.size = self.fixed or int((state or {}).get('microbatch', 1))
        self.trials = int((state or {}).get('trials', 0))
        self.previous = None

    def state(self):
        return dict(microbatch=self.size, trials=self.trials, hardware=self.hardware, fixed=self.fixed)

    def run(self, model, optimizer, batches, step):
        import torch
        from .translatotron2.engine import capture_rank_state, restore_rank_state
        if self.fixed:
            if torch.distributed.is_initialized():
                raise ValueError('fixed microbatches require a single training process')
            chunks = []
            for batch in batches:
                count = int(batch['nsentences']) if 'nsentences' in batch else batch['source'].size(0)
                chunks.extend(slice_batch(batch, start, min(start+self.fixed, count), count)
                              for start in range(0, count, self.fixed))
            return step(chunks)  # Fixed from startup: no OOM retries/sample skips.
        if not self.active:
            return step(batches)
        if torch.distributed.is_initialized():
            raise ValueError('adaptive microbatches currently require a single training process')
        maximum = max(int(b['nsentences']) if 'nsentences' in b else b['source'].size(0) for b in batches)
        original = capture_rank_state(model)
        while True:
            chunks = []
            for batch in batches:
                count = int(batch['nsentences']) if 'nsentences' in batch else batch['source'].size(0)
                chunks.extend(slice_batch(batch, start, min(start+self.size, count), count)
                              for start in range(0, count, self.size))
            started = time.monotonic()
            failed = False
            try:
                result = step(chunks)
            except RecoverableOOM:
                failed = True
            if not failed:
                break
            optimizer.zero_grad()
            restore_rank_state(model, original)
            torch.cuda.empty_cache()
            if self.size == 1:
                raise RuntimeError('OOM at one sample; update was not skipped, resume from durable checkpoint')
            self.size = max(1, self.size // 2)
            self.trials, self.previous = 9, None
            print(f'[adaptive-training] OOM retry microbatch={self.size}', file=sys.stderr, flush=True)
        elapsed = time.monotonic()-started
        self.trials += 1
        # Calibrate briefly, then freeze. Runtime OOM can still lower the cap.
        if torch.cuda.is_available() and self.trials <= 8:
            free, total = torch.cuda.mem_get_info()
            if self.previous and self.size > self.previous[0] and elapsed > self.previous[1]*1.15:
                self.size = self.previous[0]
                self.trials = 9
            elif free / total > .3:
                self.previous = self.size, elapsed
                self.size = min(maximum, self.size*2)
        return result


_last_frozen = None
_last_frozen_time = None


@operation('training: freeze recovery checkpoint')
def checkpoint_saved(work, checkpoint, updates, *, force=False):
    """Called while the writer is stopped. Publish only immutable local copies."""
    staging = os.environ.get('S2ST_TRAIN_STAGING')
    if not staging:
        return
    global _last_frozen, _last_frozen_time
    if _last_frozen == (staging, updates):
        return
    interval = float(os.environ.get('S2ST_BACKUP_MIN_SECONDS', '0'))
    if interval < 0 or not __import__('math').isfinite(interval):
        raise ValueError('backup interval must be finite and nonnegative')
    if (not force and _last_frozen is not None and _last_frozen[0] == staging
            and _last_frozen_time is not None and time.monotonic() - _last_frozen_time < interval):
        print(f'[recovery] local_update={updates} upload_deferred=true min_seconds={interval}',
              file=sys.stderr, flush=True)
        return
    from .io import atomic_write_json
    work, checkpoint, root = Path(work).resolve(), Path(checkpoint).resolve(), Path(staging).resolve()
    relative = checkpoint.relative_to(work)
    # Bounded queue: at most two local snapshots, including one being uploaded.
    while len(list(root.glob('*/ready.json'))) >= 2:
        time.sleep(.25)
    directory = Path(tempfile.mkdtemp(prefix=f'{updates:012d}-', dir=root))
    target = directory / 'files' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, target)
    for name in ('config.json', 'gradient-audit-rank-0.json', 'batch-calibration.json',
                 'research-metadata.json', 'session-research-metadata.json', 's2ut-recipe-lock.json'):
        if (work / name).is_file():
            shutil.copyfile(work / name, directory / 'files' / name)
    atomic_write_json(directory / 'ready.json', dict(updates=updates))
    _last_frozen = (staging, updates)
    _last_frozen_time = time.monotonic()
