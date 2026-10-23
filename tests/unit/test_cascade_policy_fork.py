from pathlib import Path

import pytest

from direct_s2st.benchmark_storage import read
from direct_s2st.drive_staging import raw_sha
from direct_s2st.io import atomic_write_json
from direct_s2st.journal import digest
from direct_s2st.cascade import benchmark_worker as worker
from test_cascade_benchmark import fixture, models, session
from test_cascade_tts_resume import files


def legacy(tmp_path, monkeypatch):
    persistent, data, local, _, plan = fixture(tmp_path, monkeypatch)
    # Model the existing legacy plan without its newly added generation policy.
    plan['pipeline']['tts'].pop('generation')
    path = Path(plan['output'])/'plan.json'
    atomic_write_json(path, plan, overwrite=True)
    atomic_write_json(persistent/'benchmarks/cascade/latest.json', dict(plan=str(path), sha256=raw_sha(path)), overwrite=True)
    models(monkeypatch)
    worker.text_phase(plan, 'asr')
    worker.text_phase(plan, 'mt')
    worker.synthesize(plan)
    return persistent, data, local, plan


def test_sampled_policy_fork_keeps_legacy_text_audio_and_runtime_byte_for_byte(tmp_path, monkeypatch):
    persistent, data, local, old = legacy(tmp_path, monkeypatch)
    old_root = Path(old['output'])
    before = files(old_root)
    previous_pointer = (persistent/'benchmarks/cascade/latest.json').read_bytes()
    plan = session.prepare_cascade(persistent, data, local, 'b'*40)
    assert plan['identity'] != old['identity'] and plan['output'] != old['output']
    assert plan['pipeline']['tts']['generation']['policy'] == 'official_sampling_eos_v1'
    assert files(old_root) == before
    assert (Path(plan['output'])/'previous-latest.json').read_bytes() == previous_pointer
    assert not (Path(plan['output'])/'audio-index.json').exists()
    body = dict(plan=plan['identity'], environment_sha256='b'*64)
    atomic_write_json(Path(plan['output'])/'runtime.json', dict(body=body, sha256=digest(body)))
    loads, calls = models(monkeypatch)
    worker.verify_text(plan)
    result = worker.synthesize(plan)
    assert loads == ['tts'] and len(calls) == 4  # New TTS + separate preflight; no ASR/MT.
    assert all(row['run_id'] == plan['run']['name'] for row in result)
    assert all(str(Path(plan['local'])/'audio') in row['output_audio'] for row in result)
    assert files(old_root) == before
    assert session.prepare_cascade(persistent, data, local, 'b'*40) == plan


@pytest.mark.parametrize('field', ['text/asr.json', 'text/mt.json', 'runtime.json'])
def test_source_pointer_change_is_fatal_without_regeneration(tmp_path, monkeypatch, field):
    persistent, data, local, old = legacy(tmp_path, monkeypatch)
    plan = session.prepare_cascade(persistent, data, local, 'b'*40)
    (Path(old['output'])/field).write_text('{}')
    monkeypatch.setattr(worker, 'loader', lambda *args: pytest.fail('must not regenerate'))
    with pytest.raises(ValueError, match='text-source pointer/runtime changed'):
        worker.verify_text(plan)


def test_legacy_run_cannot_silently_use_official_sampling_in_place(tmp_path, monkeypatch):
    persistent, _, _, old = legacy(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='separate policy evaluation'):
        session.execute_cascade(persistent, 'run', lambda *args: pytest.fail('must fail before execution'), 'a'*40)


@pytest.mark.parametrize('matches', [True, False])
def test_policy_fork_requires_the_original_dependency_environment(tmp_path, monkeypatch, matches):
    persistent, data, local, old = legacy(tmp_path, monkeypatch)
    plan = session.prepare_cascade(persistent, data, local, 'b'*40)
    prefix = tmp_path/'runtime'
    atomic_write_json(prefix/'environment.json', dict(fixture='rebuilt environment'))
    checksum = raw_sha(prefix/'environment.json')
    original_sha = session.raw_sha
    monkeypatch.setattr(session, 'raw_sha', lambda path: 'b'*64 if matches and Path(path) == prefix/'environment.json' else original_sha(path))
    calls = []
    def ensure(repo, *args, **kwargs):
        calls.append((repo, kwargs['expected']))
        return prefix/'venv/bin/python'
    monkeypatch.setattr('environment_recovery.ensure_environment', ensure)
    before = files(Path(old['output']))
    build = tmp_path/'reviewed-original-build'
    if matches:
        assert session.cascade_environment(persistent, plan, lambda *args: None, prefix=prefix, build=build) == prefix/'venv/bin/python'
    else:
        assert checksum != 'b'*64
        with pytest.raises(ValueError, match='runtime differs'):
            session.cascade_environment(persistent, plan, lambda *args: None, prefix=prefix, build=build)
        assert not (Path(plan['output'])/'runtime.json').exists()
    assert calls == [(build, 'b'*64)]
    assert files(Path(old['output'])) == before
