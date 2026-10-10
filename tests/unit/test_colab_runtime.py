import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_setup_returns_to_python310_without_overlay():
    path = ROOT/'scripts/colab/setup.py'
    spec = importlib.util.spec_from_file_location('colab_setup', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PYTHON_VERSION == '3.10.18'
    source = path.read_text()
    assert 'compat313' not in source and 'check313' not in source
    assert 'pip==24.0' in source and 'colab310.txt' in source
    assert '--inference' in source
    req = (ROOT/'requirements/colab310.txt').read_text()
    assert '-r preparation.txt' in req and 'torch==2.7.1' in req
    assert 'hydra-core==1.0.7' in req and 'omegaconf==2.0.6' in req


def test_notebook_preserves_four_system_workflow():
    notebook = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'colab313' not in source and '3.13.7' not in source
    assert 'ensure_runtime' in source and 'RUN_S2T_TTS' in source
    assert 'RUN_CASCADE' in source and '--restore-only' in source
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))


def test_notebooks_separate_data_models_and_training_guards(monkeypatch):
    import os
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    settings = {}
    for name in ('training', 'smoke'):
        notebook = json.loads((ROOT/f'notebooks/colab_{name}.ipynb').read_text(encoding='utf-8'))
        codes = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        for source in codes:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in ('builder', 'prepare_comparison')
                    for target in node.targets
                ):
                    ast.parse(ast.literal_eval(node.value))
        cfg = {}
        exec(codes[0], cfg)
        settings[name] = cfg
        assert cfg['CONFIRM_FULL_DATA'] is True
        assert cfg['PREP_MAX_WORKERS'] == cfg['HUBERT_MAX_BATCH'] == 32
        calls = []
        preparation = next(s for s in codes if 'limit_args =' in s)
        if name == 'training':
            import pytest
            with pytest.raises(AssertionError, match='CONFIRM_FULL_DATA'):
                exec(preparation, cfg | {'CONFIRM_FULL_DATA': False,
                     'cli': lambda *args: calls.append(args), 'DATA': 'fixture'})
            assert not calls
        runtime_calls = []
        exec(preparation, cfg | {'CONFIRM_FULL_DATA': True, 'cli': lambda *args: calls.append(args),
             'run': lambda *args: calls.append(args), 'PYTHON': 'python',
             'DATA': 'fixture', 'ensure_runtime': lambda **kw: runtime_calls.append(kw)})
        assert runtime_calls == [{'require_gpu': False}]
        assert all(call[0] != 's2ut' for call in calls)
        importing = calls[0]
        assert ('--limit' in importing) == (name == 'smoke')
        whole = '\n'.join(codes)
        assert "MODE / EXPERIMENT" in whole
        assert "MODE + '-' + EXPERIMENT" in whole
        assert 'stderr=subprocess.STDOUT' in whole and 'session.log' in whole
    assert settings['smoke']['DATA_LIMIT'] == 5
    assert settings['smoke']['TOTAL_UPDATES'] == 2
    assert settings['smoke']['MODEL_SIZE'] == 'smoke'
    assert settings['training']['DATA_LIMIT'] is None
    assert settings['training']['MODEL_SIZE'] == 'fisher'
    assert settings['training']['REPRODUCTION_MODE'] == 'paper_exact'
    assert settings['training']['TT2_VOCODER_MODE'] == 'griffin_lim'
    assert settings['training']['S2UT_VOCODER_MODE'] == 'trained'
    assert settings['training']['S2UT_TOTAL_UPDATES'] == 400000
    assert settings['training']['TRAIN_CALIBRATION_OBJECTIVE'] == 'throughput'
    assert not settings['training']['RUN_TRAINING']
    assert not settings['training']['CONFIRM_TRAINING']


def test_notebooks_split_gpu_stage_and_restore_before_training(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    for name in ('training', 'smoke'):
        nb = json.loads((ROOT/f'notebooks/colab_{name}.ipynb').read_text(encoding='utf-8'))
        codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        cfg = {}
        exec(codes[0], cfg)  # Defining bootstrap functions must not mount/install.
        calls = []
        common = tmp_path / 'common'
        common.mkdir(exist_ok=True)
        (common / 'dataset-lock.json').write_text('{}')
        stage = next(s for s in codes if s.startswith('#@title 4B.'))
        exec(stage, cfg | {'CONFIRM_FULL_DATA': True, 'DATA': tmp_path,
             'run': lambda *args: calls.append(args), 'PYTHON': 'python',
             'ensure_runtime': lambda **kw: calls.append(('runtime', kw)),
             'cli': lambda *args: calls.append(args)})
        assert calls[0] == ('runtime', {'require_gpu': True})
        assert len(calls) == 2
        assert calls[1][:5] == ('python', '-m', 'direct_s2st.preparation_workflow', '--stage', '4b')
        if name == 'training':
            calls.clear()
            cpu = next(s for s in codes if s.startswith('#@title 4A.5.'))
            exec(cpu, cfg | {'CONFIRM_FULL_DATA': True,
                 'run': lambda *args: calls.append(args), 'PYTHON': 'python',
                 'ensure_runtime': lambda **kw: calls.append(('runtime', kw))})
            assert calls[0] == ('runtime', {'require_gpu': False})
            assert calls[1][:5] == ('python', '-m', 'direct_s2st.preparation_workflow', '--stage', 'audio-packs')
            assert len(calls) == 2
        train = next(s for s in codes if s.startswith('#@title 5.'))
        assert train.index('ensure_runtime(require_gpu=True)') < train.index('CONFIG =')
        exec(train, cfg | {'RUN_TRAINING': False,
             'ensure_runtime': lambda **kw: (_ for _ in ()).throw(AssertionError('unexpected setup'))})


def _runtime_module():
    spec = importlib.util.spec_from_file_location('notebook_runtime', ROOT/'scripts/colab/runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_runtime(prefix):
    for name in ('venv/bin/python', 'fairseq/setup.py', 'espeak/bin/espeak-ng',
                 'espeak/share/espeak-ng-data/en_dict'):
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    (prefix/'environment.json').write_text(json.dumps(dict(repository='pinned', inference_dependencies=True)))


def test_runtime_restores_after_vm_loss_and_reuses_for_gpu(tmp_path, monkeypatch):
    module = _runtime_module()
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: 'pinned\n')
    # Keep runtime environment changes local to this test.
    monkeypatch.setattr(module.os, 'environ', dict(module.os.environ))
    repo, persistent, prefix = tmp_path/'repo', tmp_path/'drive', tmp_path/'vm'
    calls = []
    def run(*args):
        calls.append(args)
        if any(str(arg).endswith('setup.py') for arg in args):
            _fake_runtime(prefix)
    module.ensure_runtime(repo, persistent, run, prefix=prefix)
    assert '--allow-cpu' in calls[0]
    calls.clear()
    module.ensure_runtime(repo, persistent, run, prefix=prefix, require_gpu=True)
    assert not any(any(str(a).endswith('setup.py') for a in c) for c in calls)
    assert 'assert False or torch.cuda.is_available()' in calls[0][2]
    # A new VM has no local files: setup is automatically invoked again.
    prefix = tmp_path/'new-vm'
    calls.clear()
    module.ensure_runtime(repo, persistent, run, prefix=prefix, require_gpu=True)
    assert '--inference' in calls[0] and '--allow-cpu' not in calls[0]
    assert module.os.environ['EXPERIMENT_DATA_ROOT'] == str(persistent/'data')


def test_incomplete_or_wrong_revision_runtime_is_not_reused(tmp_path, monkeypatch):
    import pytest
    module = _runtime_module()
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: 'different')
    _fake_runtime(tmp_path/'vm')
    assert not module.ready(tmp_path/'repo', tmp_path/'vm')
    with pytest.raises(RuntimeError, match='complete environment'):
        module.ensure_runtime(tmp_path/'repo', tmp_path/'drive', lambda *a: None, prefix=tmp_path/'vm')
