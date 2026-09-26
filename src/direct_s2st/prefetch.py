"""Bounded, ordered CPU sample loading; sampling/RNG stay in the trainer."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def ordered_samples(load, indices, workers=0, prefetch=2):
    if workers < 0 or prefetch < 1:
        raise ValueError('workers must be nonnegative and prefetch positive')
    if not workers:
        yield (load(index) for index in indices)
        return
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    indices = iter(indices)
    def enqueue():
        try:
            index = next(indices)
        except StopIteration:
            return False
        pending.append(executor.submit(load, index))
        return True
    def consume():
        while pending:
            value = pending.popleft().result()
            enqueue()
            yield value
    try:
        for _ in range(workers * prefetch):
            if not enqueue():
                break
        yield consume()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
