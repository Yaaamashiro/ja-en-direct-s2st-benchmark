"""Bounded CPU ZIP work; scheduling settings never enter artifact identity."""
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import threading
import time

from .journal import digest
from .progress import Progress


def validate_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError('audio pack workers must be an integer between 1 and 4')


@contextmanager
def archive_lock(archive):
    """One publisher per ZIP on this VM, including independent Python processes.

    Locks live on the VM, not Drive. OS release also covers abrupt process exit.
    A single active CPU VM must own a persistent pack directory.
    """
    root = Path(tempfile.gettempdir()) / 's2st-audio-pack-locks'
    root.mkdir(exist_ok=True)
    with (root / (digest(str(archive.absolute())) + '.lock')).open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.write(b'\0')
            stream.flush()
            stream.seek(0)
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError(f'audio pack publisher already active on this VM: {archive}') from error
        try:
            yield
        finally:
            stream.seek(0)
            release()


class PackProgress(Progress):
    def __init__(self, split, total, workers, audio):
        super().__init__('stage WAV ZIPs: ' + split, total=total, counted=True)
        self.workers, self.audio = workers, audio
        self.built = self.reused = self.failed = self.bytes = self.wavs = 0
        self.seconds = dict(zip_build=0., originals=0., drive_copy=0., readback=0., reuse_verify=0.)

    def record(self, result):
        _, built, size, wavs, seconds = result
        self.built += int(built)
        self.reused += int(not built)
        self.bytes += size
        self.wavs += wavs
        for key, value in seconds.items():
            self.seconds[key] += value
        self.count += 1
        self.advanced = time.monotonic()

    def emit(self, status):
        elapsed = max(time.monotonic() - self.started, 1e-9)
        self.activity = ('[audio-packs] '
            f'workers={self.workers} completed={self.count}/{self.total} '
            f'zip={self.count}/{self.total} audio={self.wavs}/{self.audio} '
            f'built={self.built} reused={self.reused} failed={self.failed} '
            f'packs_per_sec={self.count/elapsed:.3f} MiB_per_sec={self.bytes/2**20/elapsed:.3f} '
            + ' '.join(f'{key}_seconds={value:.3f}' for key, value in self.seconds.items()))
        super().emit(status)


class DiskBudget:
    def __init__(self, local, capacity):
        self.local, self.capacity = local, capacity
        self.lock = threading.Lock()
        self.reserved = 0
        self.peak = 0

    @contextmanager
    def reserve(self, amount):
        with self.lock:
            # Include reservations of every active worker, plus the normal VM reserve.
            self.capacity(self.local, self.reserved + amount)
            self.reserved += amount
            self.peak = max(self.peak, self.reserved)
        try:
            yield
        finally:
            with self.lock:
                self.reserved -= amount


def completed_packs(jobs, function, workers, stop, progress):
    """At most workers futures, no preloaded executor queue. Join on all exits."""
    if workers == 1:
        for job in jobs:
            try:
                result = function(job)
            except BaseException:
                progress.failed += 1
                raise
            progress.record(result)
            yield job, result
        return
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='audio-pack')
    pending = {}
    jobs = iter(jobs)
    try:
        while True:
            while len(pending) < workers:
                try:
                    job = next(jobs)
                except StopIteration:
                    break
                pending[pool.submit(function, job)] = job
            if not pending:
                break
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Inspect failures before replenishing the bounded queue.
            results, error = [], None
            for future in ready:
                job = pending.pop(future)
                try:
                    result = future.result()
                except BaseException as failure:
                    progress.failed += 1
                    error = error or failure
                else:
                    progress.record(result)
                    results.append((job, result))
            if error is not None:
                raise error
            yield from results
    finally:
        stop.set()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        # Include workers that committed while another worker was failing.
        for future in pending:
            try:
                result = future.result()
            except CancelledError:
                pass
            except BaseException:
                progress.failed += 1
            else:
                progress.record(result)


def check_stop(stop):
    if stop.is_set():
        raise CancelledError('audio pack work cancelled after interruption/failure')
