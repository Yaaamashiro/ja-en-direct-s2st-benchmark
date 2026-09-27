"""Flushed, line-oriented progress on stderr; stdout remains machine-readable.

Heartbeats report liveness, not proof of forward progress. Counts advance only
after an item has returned to the iterator (including checked/reused items).
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import sys
import threading
import time


_active = ContextVar('progress', default=None)
_output_lock = threading.Lock()


class Progress:
    def __init__(self, phase, *, total=None, interval=10.0, counted=False):
        if interval <= 0:
            raise ValueError('progress interval must be positive')
        self.phase, self.total, self.interval = phase, total, interval
        self.count = 0
        self.counted = counted
        self.current = '-'
        self.activity = '-'
        self.stop = threading.Event()

    def emit(self, status):
        now = time.monotonic()
        total = '?' if self.total is None else self.total
        counts = (f'checked={self.count}/{total} no_advance={now-self.advanced:.1f}s '
                  if self.counted else '')
        if self.counted and self.total is not None:
            counts += f'remaining={max(0, self.total-self.count)} '
        message = (f'[progress] {self.phase} status={status} '
                   f'{counts}elapsed={now-self.started:.1f}s '
                   f'current={self.current} activity={self.activity}')
        with _output_lock:
            print(message.replace('\r', ' ').replace('\n', ' '), file=sys.stderr, flush=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.emit('running')

    def __enter__(self):
        self.started = self.advanced = time.monotonic()
        self.token = _active.set(self)
        self.emit('started')
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name='s2st-progress')
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join()
        _active.reset(self.token)
        status = 'completed' if kind is None else 'interrupted' if issubclass(kind, (KeyboardInterrupt, GeneratorExit)) else 'failed'
        self.emit(status)
        return False


def operation(phase):
    """Monitor an entire operation, including blocking model loads/subprocesses."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with Progress(phase):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def track(items, phase, *, total=None, interval=10.0):
    """Stream without pre-reading/counting a manifest or changing item order."""
    if total is None:
        try:
            total = len(items)
        except TypeError:
            pass
    with Progress(phase, total=total, interval=interval, counted=True) as progress:
        for item in items:
            displayed = item[0] if isinstance(item, tuple) and item and isinstance(item[0], dict) else item
            if isinstance(displayed, dict):
                label = displayed.get('pair_id', displayed.get('id', displayed.get('stage', '-')))
            else:
                label = getattr(displayed, 'pair_id', displayed)
            progress.current = str(label)[:160]
            yield item
            progress.count += 1
            progress.advanced = time.monotonic()


@contextmanager
def activity(label):
    """Annotate blocking I/O without creating a thread or log per audio file."""
    progress = _active.get()
    previous = progress.activity if progress else None
    if progress:
        progress.activity = str(label)
    try:
        yield progress
    finally:
        if progress:
            progress.activity = previous
