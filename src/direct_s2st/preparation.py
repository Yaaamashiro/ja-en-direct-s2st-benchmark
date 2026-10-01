"""Bounded preparation workers and append-only, atomic recovery chunks.

Only the coordinator publishes checkpoints. A killed process loses at most the
unpublished chunk, not every preceding sample. No SQLite database lives on Drive.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time
import uuid

from .io import atomic_write_json
from .journal import digest
from .progress import track


def file_stamp(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


class Checkpoints:
    def __init__(self, root, identity, *, resume=False, chunk_size=64, overwrite=False):
        self.identity = identity
        # Keep paths below Windows' legacy limit; full identity is checked in chunks.
        self.root = Path(root) / digest(identity)[:24]
        corpus = os.environ.get('CORPUS_ROOT')
        if corpus and self.root.resolve().is_relative_to(Path(corpus).resolve()):
            raise ValueError('preparation checkpoints must be outside CORPUS_ROOT')
        active = self.root / 'active.json'
        if overwrite:
            # Retain old generations; never mix explicitly regenerated results.
            atomic_write_json(active, dict(identity=identity, generation=uuid.uuid4().hex[:12]),
                              overwrite=True)
        if active.is_file():
            pointer = json.loads(active.read_text(encoding='utf-8'))
            generation = pointer['generation']
            if (pointer['identity'] != identity or len(generation) != 12
                    or any(c not in '0123456789abcdef' for c in generation)):
                raise ValueError(f'corrupt checkpoint generation: {active}')
            self.root = self.root / generation
        self.rows, self.pending = {}, {}
        self.chunk_size = chunk_size
        self.flushed = time.monotonic()
        self.reused = 0
        if resume:
            for path in track(sorted(self.root.glob('chunk-*.json')), 'preparation: load checkpoints'):
                document = json.loads(path.read_text(encoding='utf-8'))
                values = document['rows']
                if document['identity'] != identity or document['sha256'] != digest(values):
                    raise ValueError(f'corrupt preparation checkpoint: {path}')
                for key, value in values.items():
                    if key in self.rows and self.rows[key] != value:
                        raise ValueError(f'conflicting preparation checkpoint: {key}')
                    self.rows[key] = value
        print(f'[checkpoint] path={self.root} loaded={len(self.rows)} resume={resume}',
              file=sys.stderr, flush=True)

    def get(self, key):
        value = self.rows.get(key)
        if value is not None:
            self.reused += 1
        return value

    def record(self, key, value):
        if key in self.rows and self.rows[key] != value:
            raise ValueError('checkpoint key produced different results; use a new output root')
        if key not in self.rows:
            self.rows[key] = value
            self.pending[key] = value
        if len(self.pending) >= self.chunk_size or time.monotonic() - self.flushed >= 10:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        values = dict(self.pending)
        atomic_write_json(self.root / f'chunk-{uuid.uuid4().hex}.json',
                          dict(identity=self.identity, rows=values, sha256=digest(values)))
        self.pending.clear()
        self.flushed = time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        self.flush()
        print(f'[checkpoint] {self.root.name[:12]} saved={len(self.rows)} reused={self.reused}',
              file=sys.stderr, flush=True)


class WorkerController:
    def __init__(self, maximum):
        self.maximum = max(1, maximum)
        self.workers = min(2, self.maximum)
        self.previous_rate = None

    def observe(self, rate, cpu=None, free_ram=None):
        pressure = (free_ram is not None and free_ram < .15)
        slower = self.previous_rate is not None and rate < self.previous_rate * .85
        if pressure or slower:
            self.workers = max(1, self.workers - 1)
        elif (cpu is None or cpu < 90) and (free_ram is None or free_ram > .25):
            self.workers = min(self.maximum, self.workers + 1)
        self.previous_rate = rate
        return self.workers


def resources():
    try:
        import psutil
        memory = psutil.virtual_memory()
        return psutil.cpu_percent(), memory.available / memory.total
    except ImportError:
        return None, None


def worker_limit():
    cap = int(os.environ.get('S2ST_PREP_WORKERS', '8'))
    if not 1 <= cap <= 32:
        raise ValueError('S2ST_PREP_WORKERS must be between 1 and 32')
    return min(cap, os.cpu_count() or 1)


def adaptive_map(function, items, *, maximum=None):
    """Ordered results, bounded in-flight work, throughput/RAM/CPU feedback.

Threads overlap remote I/O, eSpeak subprocesses and native feature extraction.
They do not promise parallel speedup for pure-Python CPU loops.
"""
    controller = WorkerController(worker_limit() if maximum is None else maximum)
    source, pending = iter(items), deque()
    started, completed, exhausted = time.monotonic(), 0, False
    adaptive = os.environ.get('S2ST_PREP_ADAPTIVE', '1') != '0'
    if not adaptive:
        controller.workers = controller.maximum
    try:
        from threadpoolctl import threadpool_limits
        native_threads = threadpool_limits(limits=1)
    except ImportError:
        native_threads = nullcontext()
    with native_threads, ThreadPoolExecutor(max_workers=controller.maximum) as pool:
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < controller.workers:
                    try:
                        item = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append(pool.submit(function, item))
                if not pending:
                    break
                yield pending.popleft().result()
                completed += 1
                elapsed = time.monotonic() - started
                if elapsed >= 10:
                    cpu, ram = resources()
                    rate = completed / elapsed
                    if adaptive:
                        controller.observe(rate, cpu, ram)
                    print(f'[adaptive] workers={controller.workers}/{controller.maximum} '
                          f'items_per_s={rate:.3f} cpu_pct={cpu} free_ram_ratio={ram}',
                          file=sys.stderr, flush=True)
                    started, completed = time.monotonic(), 0
        finally:
            for future in pending:
                future.cancel()


def checkpoint_map(function, items, cache, key, phase, *, total=None):
    """Parallelize metadata keys too; only the coordinator publishes results.

    Workers consult a frozen index, never mutate the live checkpoint dictionary.
    This avoids serial Drive stat/resolve calls before each job is submitted.
    """
    saved = dict(cache.rows)
    def work(item):
        identity = key(item)
        prior = saved.get(identity)
        label = item.get('pair_id', '-') if isinstance(item, dict) else getattr(item, 'pair_id', '-')
        if isinstance(item, tuple) and item and isinstance(item[0], str):
            label = item[0]
        return dict(pair_id=label, identity=identity,
                    reused=prior is not None,
                    result=prior if prior is not None else function(item))

    reported = time.monotonic()
    computed = 0
    for completed in track(adaptive_map(work, items), phase, total=total):
        cache.reused += completed['reused']
        computed += completed['identity'] not in cache.rows
        cache.record(completed['identity'], completed['result'])
        if time.monotonic() - reported >= 10:
            print(f'[checkpoint-progress] {phase} reused={cache.reused} computed={computed} '
                  f'persisted={len(cache.rows)-len(cache.pending)}', file=sys.stderr, flush=True)
            reported = time.monotonic()
        yield completed['result']
