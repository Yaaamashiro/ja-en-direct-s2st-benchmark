"""Bounded, ordered CPU sample loading; sampling/RNG stay in the trainer."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import sys
import time
import threading


@contextmanager
def ordered_samples(load, indices, workers=0, prefetch=2):
    if workers < 0 or prefetch < 1:
        raise ValueError('workers must be nonnegative and prefetch positive')
    if not workers:
        yield (load(index) for index in indices)
        return
    executor = ThreadPoolExecutor(max_workers=workers)
    adaptive = os.environ.get('S2ST_TRAIN_OPTIMIZE') == '1'
    from .preparation import WorkerController, resources
    controller = WorkerController(workers)
    if not adaptive:
        controller.workers = workers
    reported, completed = time.monotonic(), 0
    condition = threading.Condition()
    active = 0
    def limited_load(index):
        nonlocal active
        with condition:
            while active >= controller.workers:
                condition.wait()
            active += 1
        try:
            return load(index)
        finally:
            with condition:
                active -= 1
                condition.notify_all()
    pending = deque()
    indices = iter(indices)
    def enqueue():
        try:
            index = next(indices)
        except StopIteration:
            return False
        pending.append(executor.submit(limited_load, index))
        return True
    def consume():
        nonlocal reported, completed
        while pending:
            value = pending.popleft().result()
            completed += 1
            elapsed = time.monotonic()-reported
            if adaptive and elapsed >= 10:
                cpu, ram = resources()
                controller.observe(completed/elapsed, cpu, ram)
                with condition:
                    condition.notify_all()
                print(f'[training-loader] workers={controller.workers}/{workers} '
                      f'prefetch={controller.workers*prefetch} samples_per_s={completed/elapsed:.3f} '
                      f'cpu_pct={cpu} free_ram_ratio={ram}', file=sys.stderr, flush=True)
                reported, completed = time.monotonic(), 0
            while len(pending) < controller.workers * prefetch and enqueue():
                pass
            yield value
    try:
        for _ in range(controller.workers * prefetch):
            if not enqueue():
                break
        yield consume()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
