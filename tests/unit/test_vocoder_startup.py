"""Default calibrated-cell startup must not be confused with overwrite/resume."""
import json
from pathlib import Path
import subprocess

import pytest

from direct_s2st.io import ExistingOutputError
from direct_s2st.vocoders.verification import check_output, preflight_command


def preflight(work, directory):
    def runner(command, **kwargs):
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_text('{}')
    command = ['python', '-m', 'direct_s2st.vocoders.train', '--output-root', str(work)]
    args = preflight_command(command, directory, runner=runner)
    return Path(args[args.index('--initial-output-receipt')+1]) if '--initial-output-receipt' in args else None


@pytest.mark.parametrize('names', [[], ['batch-calibration.json'],
    ['batch-calibration.json', 'session-research-metadata.json']])
def test_private_preflight_permits_only_exact_launcher_metadata(tmp_path, names):
    work = tmp_path/'run'
    work.mkdir()
    for name in names:
        (work/name).write_text('{"selected":16}')
    before = {p.name: p.read_bytes() for p in work.iterdir()}
    if names:
        with pytest.raises(ExistingOutputError):
            check_output(work)
    receipt = preflight(work, tmp_path/'private')
    check_output(work, initial_receipt=receipt)
    assert before == {p.name: p.read_bytes() for p in work.iterdir()}
    assert not any(p.suffix == '.pt' for p in work.iterdir())


def test_output_absent_and_empty_directory_need_no_overwrite(tmp_path):
    work = tmp_path/'new-run'
    check_output(work)
    receipt = preflight(work, tmp_path/'private')
    check_output(work, initial_receipt=receipt)
    work.mkdir()
    check_output(work, initial_receipt=receipt)


@pytest.mark.parametrize('entry', ['generator.pt', 'config.json', 'training-monitor.json',
                                  'notes.json', 'checkpoints', 'generator.pt.tmp'])
def test_unknown_files_models_reports_and_directories_are_never_ignored(tmp_path, entry):
    work = tmp_path/'run'
    work.mkdir()
    target = work/entry
    target.mkdir() if entry == 'checkpoints' else target.write_bytes(b'retain')
    with pytest.raises((ExistingOutputError, ValueError)):
        receipt = preflight(work, tmp_path/'private')
        check_output(work, initial_receipt=receipt)
    assert target.exists()
    if target.is_file():
        assert target.read_bytes() == b'retain'


@pytest.mark.parametrize('change', ['metadata', 'added-model', 'added-report', 'other-directory', 'corrupt-receipt'])
def test_files_cannot_change_between_preflight_and_fresh_trainer(tmp_path, change):
    work = tmp_path/'run'
    work.mkdir()
    metadata = work/'batch-calibration.json'
    metadata.write_text('{"selected":16}')
    receipt = preflight(work, tmp_path/'private')
    if change == 'metadata':
        metadata.write_text('{"selected":32}')
    elif change == 'added-model':
        (work/'generator.pt').write_text('retain optimizer')
    elif change == 'added-report':
        (work/'training-monitor.json').write_text('retain history')
    elif change == 'other-directory':
        other = tmp_path/'other'
        other.mkdir()
        (other/metadata.name).write_bytes(metadata.read_bytes())
        work = other
    else:
        document = json.loads(receipt.read_text())
        document['sha256'] = '0'*64
        receipt.write_text(json.dumps(document))
    before = {p.name: p.read_bytes() for p in work.iterdir()}
    with pytest.raises(ExistingOutputError):
        check_output(work, initial_receipt=receipt)
    assert before == {p.name: p.read_bytes() for p in work.iterdir()}


def test_explicit_resume_retains_model_and_does_not_get_fresh_start_permission(tmp_path):
    work = tmp_path/'run'
    work.mkdir()
    (work/'generator.pt').write_bytes(b'optimizer checkpoint')
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_text('{}')
    command = ['python', '-m', 'direct_s2st.vocoders.train', '--output-root', str(work), '--resume']
    args = preflight_command(command, tmp_path/'private', runner=runner)
    assert '--initial-output-receipt' not in args and '--resume' in calls[0]
    check_output(work, resume=True)
    with pytest.raises(ExistingOutputError):
        check_output(work)
    assert (work/'generator.pt').read_bytes() == b'optimizer checkpoint'


@pytest.mark.parametrize('kind', ['unit', 'mel'])
def test_default_session_passes_bootstrap_after_calibration_before_spawn(tmp_path, monkeypatch, kind):
    from direct_s2st import session_runtime, batch_calibration, colab
    from direct_s2st.io import atomic_write_json
    from direct_s2st.recipes import option
    work, backup, data = tmp_path/'work', tmp_path/'backup', tmp_path/'data'
    data.mkdir()
    lock = data/'dataset-lock.json'
    lock.write_text('{}')
    command = ['{python}', '-m', 'direct_s2st.vocoders.train', '--kind', kind,
        '--output-root', '{run_root}', '--batch-size', '1', '--max-updates', '{updates}']
    config = dict(kind='vocoder', checkpoint='generator.pt', command=command,
        identity_files=[str(lock)], resume_args=['--resume'],
        runtime=dict(cache_gb=0, readers=1, precision='default', adaptive_batch=False),
        batch_calibration=dict(max_batch=1, steps=1, reserve_ratio=.1, trial_seconds=1, objective='capacity'))
    monkeypatch.setattr(batch_calibration, 'hardware', lambda: {'fixture':True})
    calibrations = []
    def calibrate(cfg, inputs, record_path, parent, **kwargs):
        calibrations.append(kwargs['inspect_only'])
        record_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(record_path, dict(selected=1, verified_hardware=[{'fixture':True}]), resume=True)
        return cfg
    monkeypatch.setattr(batch_calibration, 'calibrate', calibrate)
    def preflight_runner(command, **kwargs):
        assert '--verify-only' in command
        Path(command[-1]).write_text('{}')
    monkeypatch.setattr('direct_s2st.vocoders.verification.subprocess.run', preflight_runner)
    frozen_metadata = []
    class Process:
        returncode = 0
        def __init__(self, args, env):
            receipt = Path(option(args, '--initial-output-receipt'))
            assert receipt.is_file() and '--resume' not in args and '--overwrite' not in args
            check_output(work, initial_receipt=receipt)
            frozen_metadata.append((work/'batch-calibration.json').read_bytes())
            # No GPU/model is run here. Publish a small typed optimizer receipt
            # to exercise the real outer staging/upload/result orchestration.
            import torch
            torch.save(dict(format='direct-s2st-vocoder-v1', updates=1, optimizer_g={'fixture':1},
                            optimizer_d={'fixture':1}), work/'generator.pt')
            from direct_s2st.recovery import freeze_files
            directory = Path(env['S2ST_TRAIN_STAGING'])/'1'
            freeze_files(work, work/'generator.pt', directory/'files')
            atomic_write_json(directory/'ready.json', dict(updates=1))
        def poll(self):
            return self.returncode
    original = session_runtime.run_continuous
    monkeypatch.setattr(session_runtime, 'run_continuous', lambda *a, **kw: original(*a, **kw, popen=Process))
    result = colab.run_session(config, work=work, backup=backup, total=1, auto_resume=True)
    assert result == dict(status='COMPLETE', durable_updates=1)
    assert len(calibrations) == 1 and frozen_metadata
    assert (work/'batch-calibration.json').read_bytes() == frozen_metadata[0]
    # A repeat finds the durable checkpoint, not a fresh model or repeated probe.
    assert colab.run_session(config, work=work, backup=backup, total=1, auto_resume=True) == result
    assert calibrations == [False, True]
