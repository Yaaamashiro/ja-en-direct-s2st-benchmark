import json
from pathlib import Path

import pytest
import torch

from direct_s2st.train_runtime import (LocalCache, Microbatches, checkpoint_saved,
    forward_backward_guard, precision_context, slice_batch)


def test_cache_ranges_leases_eviction_and_global_budget(tmp_path):
    source = tmp_path / 'original.wav'
    source.write_bytes(b'abcdefgh')
    cache = LocalCache(tmp_path/'cache', 4, reserve=0, total_budget=4)
    other = LocalCache(tmp_path/'cache', 4, reserve=0, total_budget=4)
    with cache.acquire(source, 0, 4) as first:
        assert first.read_bytes() == b'abcd'
        with cache.acquire(source, 4, 4) as bypass:
            assert bypass is None  # leased data cannot be evicted
        with other.acquire(source, 4, 4) as bypass:
            assert bypass is None  # other process caches share the total cap
    with cache.acquire(source, 4, 4) as second:
        assert second.read_bytes() == b'efgh'
        assert not first.exists()
    with cache.acquire(source) as oversized:
        assert oversized is None
    assert source.read_bytes() == b'abcdefgh'
    assert json.loads((tmp_path/'cache/budget.json').read_text())['used'] == 4


def test_cache_source_changes_and_corpus_guard(tmp_path, monkeypatch):
    source = tmp_path/'data'
    source.write_bytes(b'a')
    cache = LocalCache(tmp_path/'cache', 10, reserve=0)
    with cache.acquire(source) as local:
        assert local.read_bytes() == b'a'
    source.write_bytes(b'ab')
    with cache.acquire(source) as local:
        assert local.read_bytes() == b'ab'
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='CORPUS_ROOT'):
        LocalCache(tmp_path/'unsafe', 10)


def test_slice_recomputes_auxiliary_counts_without_reordering():
    batch = dict(id=torch.arange(3), nsentences=3, ntokens=9,
                 target_lengths=torch.tensor([2, 3, 4]),
                 multitask={'text': dict(target_lengths=torch.tensor([1, 2, 3]), ntokens=6)})
    result = slice_batch(batch, 1, 3, 3)
    assert result['id'].tolist() == [1, 2]
    assert result['nsentences'] == 2 and result['ntokens'] == 7
    assert result['multitask']['text']['ntokens'] == 5
    assert batch['ntokens'] == 9


def test_microbatch_retry_restores_rng_buffers_and_preserves_samples(monkeypatch):
    monkeypatch.setenv('S2ST_TRAIN_ADAPTIVE_BATCH', '1')
    model = torch.nn.Linear(1, 1)
    model.register_buffer('counter', torch.zeros(1))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    tuning = Microbatches()
    tuning.size = 4
    seen, draws = [], []
    def step(chunks):
        draws.append(torch.rand(1).item())
        seen.append([x for c in chunks for x in c['id'].tolist()])
        model.counter.add_(1)
        with forward_backward_guard():
            if len(chunks) == 1:
                raise torch.cuda.OutOfMemoryError('synthetic pre-update OOM')
        assert model.counter.item() == 1
        return 'done'
    batch = dict(id=torch.arange(4), nsentences=4)
    assert tuning.run(model, optimizer, [batch], step) == 'done'
    assert seen == [list(range(4)), list(range(4))]
    assert draws[0] == draws[1]
    assert tuning.size == 2 and tuning.trials > 8
    restored = Microbatches(tuning.state())
    assert restored.size == 2
    assert Microbatches(dict(tuning.state(), hardware=('different', 1))).size == 1


def test_optimizer_oom_is_never_retried(monkeypatch):
    monkeypatch.setenv('S2ST_TRAIN_ADAPTIVE_BATCH', '1')
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    calls = []
    def step(chunks):
        calls.append(chunks)
        raise torch.cuda.OutOfMemoryError('synthetic optimizer OOM')
    with pytest.raises(torch.cuda.OutOfMemoryError):
        Microbatches().run(model, optimizer, [dict(nsentences=2)], step)
    assert len(calls) == 1


def test_single_sample_oom_is_fatal(monkeypatch):
    monkeypatch.setenv('S2ST_TRAIN_ADAPTIVE_BATCH', '1')
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    def step(chunks):
        with forward_backward_guard():
            raise torch.cuda.OutOfMemoryError('synthetic')
    with pytest.raises(RuntimeError, match='not skipped'):
        Microbatches().run(model, optimizer, [dict(nsentences=1)], step)


def test_bf16_rejects_cpu(monkeypatch):
    monkeypatch.setenv('S2ST_TRAIN_PRECISION', 'bf16')
    with pytest.raises(ValueError, match='no silent'):
        precision_context('cpu')


def test_frozen_checkpoint_is_immutable_and_deduplicated(tmp_path, monkeypatch):
    staging, work = tmp_path/'staging', tmp_path/'work'
    staging.mkdir()
    work.mkdir()
    checkpoint = work/'checkpoint.pt'
    checkpoint.write_text('one')
    monkeypatch.setenv('S2ST_TRAIN_STAGING', str(staging))
    checkpoint_saved(work, checkpoint, 1)
    checkpoint.write_text('two')
    checkpoint_saved(work, checkpoint, 1)
    ready = list(staging.glob('*/ready.json'))
    assert len(ready) == 1
    assert (ready[0].parent/'files/checkpoint.pt').read_text() == 'one'


def test_continuous_session_single_process_and_verified_snapshots(tmp_path):
    from direct_s2st.session_runtime import run_continuous
    from direct_s2st.colab import publish, latest
    calls = []
    class Process:
        returncode = 0
        def __init__(self, args, env):
            calls.append(args)
            for update in (1, 2):
                directory = Path(env['S2ST_TRAIN_STAGING']) / str(update)
                (directory/'files').mkdir(parents=True)
                (directory/'files/checkpoint.pt').write_text(str(update))
                (directory/'ready.json').write_text(json.dumps(dict(updates=update)))
        def poll(self):
            return self.returncode
    config = dict(kind='tt2', checkpoint='checkpoint.pt', command=['fixture', '{updates}'],
                  resume_args=[], runtime=dict(cache_gb=0, readers=1, precision='default', adaptive_batch=False))
    result = run_continuous(config, work=tmp_path/'work', backup=tmp_path/'drive',
        identity={'fixture': True}, completed=0, total=2, seconds=10,
        inspect_checkpoint=lambda p,k: int(p.read_text()), publish=publish, popen=Process)
    assert result == dict(status='COMPLETE', durable_updates=2)
    assert calls == [['fixture', '2']]
    snapshot, state = latest(tmp_path/'drive', {'fixture': True})
    assert state['updates'] == 2
    assert (snapshot/'files/checkpoint.pt').read_text() == '2'
    assert not list(tmp_path.glob('.s2st-training-*'))


def test_failed_upload_retains_previous_verified_checkpoint(tmp_path):
    from direct_s2st.session_runtime import run_continuous
    from direct_s2st.colab import publish, latest
    work, backup = tmp_path/'work', tmp_path/'drive'
    work.mkdir()
    (work/'checkpoint.pt').write_text('1')
    identity = {'fixture': True}
    publish(work, backup, identity, 1)
    class Process:
        returncode = 0
        def __init__(self, args, env):
            assert args[-1] == '--resume'
            root = Path(env['S2ST_TRAIN_STAGING'])/'2'
            (root/'files').mkdir(parents=True)
            (root/'files/checkpoint.pt').write_text('2')
            (root/'ready.json').write_text('{"updates": 2}')
        def poll(self):
            return self.returncode
    def failed(*args):
        raise OSError('Drive unavailable')
    config = dict(kind='tt2', checkpoint='checkpoint.pt', command=['fixture'],
        resume_args=['--resume'], runtime=dict(cache_gb=0, readers=1, precision='default', adaptive_batch=False))
    with pytest.raises(OSError, match='Drive unavailable'):
        run_continuous(config, work=work, backup=backup, identity=identity,
            completed=1, total=2, seconds=10, inspect_checkpoint=lambda p,k: int(p.read_text()),
            publish=failed, popen=Process)
    assert latest(backup, identity)[1]['updates'] == 1


def test_optimized_config_is_explicit_and_gan_split_rejected(tmp_path):
    from direct_s2st.colab import make_config
    root = tmp_path/'data/translatotron2/fairseq'
    root.mkdir(parents=True)
    (root/'fixture').write_text('prepared')
    args = (tmp_path, tmp_path/'data', tmp_path/'environment.json')
    assert 'runtime' not in make_config(*args)
    cfg = make_config(*args, optimize=True, num_workers=8)
    assert cfg['runtime'] == dict(cache_gb=8, readers=1, adaptive_batch=False, precision='default')
    with pytest.raises(ValueError, match='two-optimizer'):
        make_config(*args, kind='unit', optimize=True, adaptive_batch=True)
    with pytest.raises(ValueError, match='requires optimize'):
        make_config(*args, adaptive_batch=True)


def test_rank_state_restores_python_numpy_and_torch_rng():
    import random
    import numpy as np
    from direct_s2st.translatotron2.engine import capture_rank_state, restore_rank_state
    model = torch.nn.Linear(1, 1)
    state = capture_rank_state(model)
    expected = random.random(), np.random.random(), torch.rand(1).item()
    restore_rank_state(model, state)
    assert (random.random(), np.random.random(), torch.rand(1).item()) == expected
