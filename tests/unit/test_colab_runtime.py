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
    assert 'colab310.txt' in source and 'RUN_S2T_TTS' in source
    assert 'RUN_CASCADE' in source and '--restore-only' in source
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))


def test_notebooks_separate_data_models_and_training_guards():
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
        calls = []
        preparation = next(s for s in codes if 'limit_args =' in s)
        if name == 'training':
            import pytest
            with pytest.raises(AssertionError, match='CONFIRM_FULL_DATA'):
                exec(preparation, cfg | {'cli': lambda *args: calls.append(args), 'DATA': 'fixture'})
            assert not calls
        exec(preparation, cfg | {'CONFIRM_FULL_DATA': True, 'cli': lambda *args: calls.append(args), 'DATA': 'fixture'})
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
    assert settings['training']['MODEL_SIZE'] == 'reference'
    assert not settings['training']['RUN_TRAINING']
    assert not settings['training']['CONFIRM_TRAINING']
