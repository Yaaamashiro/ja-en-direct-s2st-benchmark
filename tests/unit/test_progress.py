import threading

import pytest

from direct_s2st.progress import Progress, activity, operation, track, _active


def test_track_streams_and_counts_after_processing(capsys):
    consumed = []

    def source():
        for i in range(3):
            consumed.append(i)
            yield {'pair_id': f'pair-{i}'}

    iterator = track(source(), 'stream')
    assert consumed == []
    assert next(iterator)['pair_id'] == 'pair-0'
    assert consumed == [0]
    assert _active.get().count == 0
    next(iterator)
    assert _active.get().count == 1
    list(iterator)
    assert _active.get() is None
    logs = capsys.readouterr()
    assert logs.out == ''
    assert 'checked=3/?' in logs.err
    assert 'status=completed' in logs.err


def test_heartbeat_during_blocking_item_without_fake_progress(capsys, monkeypatch):
    heartbeat = threading.Event()
    original = Progress.emit

    def emit(self, status):
        original(self, status)
        if status == 'running':
            heartbeat.set()

    monkeypatch.setattr(Progress, 'emit', emit)
    with Progress('slow', total=2, counted=True, interval=0.01) as progress:
        with activity('reading WAV'):
            assert heartbeat.wait(2), 'no heartbeat during blocked work'
            assert progress.count == 0
    assert not progress.thread.is_alive()
    logs = capsys.readouterr().err
    assert 'status=running' in logs
    assert 'checked=0/2' in logs
    assert 'activity=reading WAV' in logs
    assert 'no_advance=' in logs


@pytest.mark.parametrize('error,status', [(ValueError, 'failed'), (KeyboardInterrupt, 'interrupted')])
def test_operation_failure_cleans_up_and_propagates(error, status, capsys):
    states = []

    @operation('failure')
    def fail():
        states.append(_active.get())
        raise error('test')

    with pytest.raises(error):
        fail()
    assert _active.get() is None
    assert not states[0].thread.is_alive()
    logs = capsys.readouterr().err
    assert f'status={status}' in logs
    assert 'status=completed' not in logs


def test_nested_scope_and_closed_iterator_do_not_leak(capsys):
    with Progress('parent') as parent:
        iterator = track([1, 2], 'child')
        next(iterator)
        child = _active.get()
        iterator.close()
        assert not child.thread.is_alive()
        assert _active.get() is parent
    assert _active.get() is None
    assert 'child status=interrupted' in capsys.readouterr().err


def test_known_total_and_empty_input(capsys):
    assert list(track([1, 2], 'known')) == [1, 2]
    assert list(track([], 'empty')) == []
    logs = capsys.readouterr().err
    assert 'checked=2/2' in logs
    assert 'checked=0/0' in logs


def test_fast_loop_does_not_log_per_item(capsys):
    assert sum(track(range(10000), 'fast', interval=3600)) == sum(range(10000))
    assert len(capsys.readouterr().err.splitlines()) == 2
