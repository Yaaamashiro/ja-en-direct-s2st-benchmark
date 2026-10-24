from pathlib import Path

import pytest

from direct_s2st.benchmark_storage import read
from direct_s2st.drive_staging import raw_sha
from direct_s2st.io import atomic_write_json
from test_interactive_comparison import session, worker, fake_phases, ROOT


def setup(tmp_path, monkeypatch):
    persistent, local = tmp_path/'d', tmp_path/'l'
    runtime = tmp_path/'runtime'
    atomic_write_json(runtime/'environment.json', {'packages':'fixed'})
    monkeypatch.setattr(session, 'RUNTIME', runtime)
    monkeypatch.setenv('PYTHONPATH', '')
    monkeypatch.setenv('S2ST_CONFIG_ROOT', '')
    monkeypatch.setenv('S2ST_INTERACTIVE_MEASUREMENT_SESSION', '')
    environments = []
    def environment(*args, **kwargs):
        environments.append((args, kwargs))
        return '/runtime/python'
    monkeypatch.setattr('environment_recovery.ensure_environment', environment)
    monkeypatch.setattr('inference_session.saved_run', lambda *args: pytest.fail('no unselected model restore'))
    calls, sources, _ = fake_phases(monkeypatch)
    invoked = []
    def invoke(*args):
        invoked.append(args)
        if '-m' in args:
            plan = read(Path(args[args.index('--plan')+1]))
            phase = args[-1]
            if phase == 'report':
                worker.report(plan)
            else:
                worker.worker(plan, phase)
    return persistent, local, invoke, environments, calls, invoked


def test_normal_launcher_no_training_or_corpus_staging_and_cached_complete_needs_no_gpu(tmp_path, monkeypatch):
    persistent, local, invoke, environments, calls, invoked = setup(tmp_path, monkeypatch)
    args = (tmp_path/'repo', persistent, local, '日本語です。', 's2t_tts', '', 'a'*40, invoke)
    plan = session.launch(*args)
    assert calls == list(worker.PHASES)
    assert [a[-1] for a in invoked if '-m' in a] == [*worker.PHASES, 'report']
    assert len(environments) == 1
    assert read(persistent/'interactive/latest.json')['report_sha256'] == raw_sha(Path(plan['output'])/'comparison.json')
    # Entire local VM/cache gone; complete report/audio can be listened to without setup/download/GPU.
    local.rename(tmp_path/'lost-vm')
    assert session.launch(*args) == plan
    assert len(environments) == 1 and calls == list(worker.PHASES)
    assert not (persistent/'benchmarks').exists() and not (persistent/'data').exists()


def test_interruption_resumes_stages_and_remeasurement_creates_separate_results(tmp_path, monkeypatch):
    persistent, local, invoke, environments, calls, invoked = setup(tmp_path, monkeypatch)
    def interrupted(*args):
        if args[-1] == 'candidate-model':
            raise KeyboardInterrupt()
        invoke(*args)
    common = (tmp_path/'repo', persistent, local, '日本語です。', 's2t_tts', '', 'a'*40)
    with pytest.raises(KeyboardInterrupt):
        session.launch(*common, interrupted, remeasure=True)
    partial = session.saved_plan(persistent)
    assert partial['remeasurement']
    old_source = worker.read_stage(partial, 'source')
    resumed = session.launch(*common, invoke, remeasure=False)
    assert resumed['identity'] == partial['identity'] and calls.count('source') == 1
    assert worker.read_stage(resumed, 'source') == old_source
    before = {str(p): p.read_bytes() for p in Path(resumed['output']).rglob('*') if p.is_file()}
    measured = session.launch(*common, invoke, remeasure=True)
    assert measured['identity'] != resumed['identity']
    assert calls.count('source') == 2 and all(Path(p).read_bytes() == body for p, body in before.items())


def test_only_bound_frontend_metadata_and_verified_best_are_used(tmp_path, monkeypatch):
    root, work = tmp_path/'prepared', tmp_path/'work'
    for name in ('config.yaml', 'config_multitask.yaml', 'data-lock.json', 'dict.txt'):
        atomic_write_json(root/name, {'unit_configuration':{}})
    checkpoint = work/'checkpoints/checkpoint_best.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'bound best checkpoint')
    run = dict(name='s2ut-original', updates=100, work=str(work),
               config=dict(command=['python', '-m', 'fairseq_cli.train', str(root)]),
               identity=dict(inputs={str(p):raw_sha(p) for p in root.iterdir()}),
               files={'checkpoints/checkpoint_best.pt':raw_sha(checkpoint)})
    def select(path, choice):
        assert path == work and choice == 'best'
        return checkpoint
    monkeypatch.setattr('direct_s2st.translatotron2.monitor.select_checkpoint', select)
    monkeypatch.setattr('direct_s2st.vocoders.reference.validate_reference_units', lambda value: None)
    selected, metadata = session.restored_metadata(run, 's2ut')
    assert selected['checkpoint'] == str(checkpoint) and metadata[str(checkpoint)] == raw_sha(checkpoint)
    assert all(not key.endswith('.tsv') for key in metadata)
    (root/'dict.txt').write_text('changed')
    with pytest.raises(ValueError, match='metadata changed'):
        session.restored_metadata(run, 's2ut')
