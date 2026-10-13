"""Exercise pip's real include parser, not a mocked environment setup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
spec = importlib.util.spec_from_file_location('restore_setup', ROOT/'scripts/colab/setup.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def parse_with_pip(path, *, cwd):
    # uv development venvs need not install pip. Use Python's bundled wheel
    # without modifying the development environment or accessing the network.
    env = setup.bootstrap_environment()
    selected = os.environ.get('S2ST_TEST_PIP_WHEEL')
    if selected:
        assert Path(selected).is_file()
        env['PYTHONPATH'] = selected+os.pathsep+env.get('PYTHONPATH', '')
    elif importlib.util.find_spec('pip') is None:
        bundled = Path(sys.base_prefix)/('Lib' if os.name == 'nt' else f'lib/python{sys.version_info.major}.{sys.version_info.minor}')/'ensurepip/_bundled'
        wheels = sorted(bundled.glob('pip-*.whl'))
        if not wheels:
            pytest.skip('real pip parser unavailable')
        env['PYTHONPATH'] = str(wheels[-1])+os.pathsep+env.get('PYTHONPATH', '')
    code = ('import json,sys\nfrom pip._internal.network.session import PipSession\n'
            'from pip._internal.req.req_file import parse_requirements\n'
            'rows=list(parse_requirements(sys.argv[1],session=PipSession(),constraint=True))\n'
            'print(json.dumps([r.requirement for r in rows]))')
    return subprocess.run([sys.executable, '-c', code, str(path)], cwd=cwd, env=env,
                          text=True, capture_output=True)


def test_real_pip_resolves_saved_constraints_from_original_checkout(tmp_path):
    repo, runtime = tmp_path/'saved checkout with spaces', tmp_path/'runtime'
    requirements = repo/'requirements'
    requirements.mkdir(parents=True)
    runtime.mkdir()
    base = requirements/'colab310.txt'
    base.write_text('# pinned base\n-r preparation.txt\ntorch==2.7.1\n', encoding='utf-8')
    (requirements/'preparation.txt').write_text('-r nested/audio.txt\nnumpy==1.26.4\n', encoding='utf-8')
    (requirements/'nested').mkdir()
    (requirements/'nested/audio.txt').write_text('soundfile==0.13.1\n', encoding='utf-8')
    before = {p: p.read_bytes() for p in requirements.rglob('*.txt')}
    constraint = runtime/'restore-constraints.txt'
    # Reproduce the user's exact failure using the old file-generation logic.
    constraint.write_text(base.read_text()+'\npydantic==2.14.0\n', encoding='utf-8')
    failed = parse_with_pip(constraint, cwd=runtime)
    assert failed.returncode != 0
    assert 'preparation.txt' in failed.stderr
    assert str(runtime) in failed.stderr.replace(chr(92)*2, chr(92))
    # A restart regenerates only the derived runtime file, not saved inputs.
    written = setup.write_restore_constraints(repo, runtime,
        '# editable package\n-e /content/benchmark\npydantic==2.14.0\ntorch==2.7.1+cu126\n')
    assert written == constraint
    assert constraint.read_text().splitlines()[0] == '-c '+json.dumps(base.resolve().as_posix())
    result = parse_with_pip(constraint, cwd=runtime)
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)) == {
        'torch==2.7.1', 'torch==2.7.1+cu126', 'numpy==1.26.4', 'soundfile==0.13.1', 'pydantic==2.14.0'}
    assert before == {p: p.read_bytes() for p in before}
    first = constraint.read_bytes()
    setup.write_restore_constraints(repo, runtime, 'torch==2.7.1+cu126\npydantic==2.14.0\n')
    assert constraint.read_bytes() == first
    assert not (runtime/'preparation.txt').exists()


def test_missing_repository_requirements_never_uses_a_runtime_copy(tmp_path):
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    (runtime/'preparation.txt').write_text('untrusted-package==1\n')
    with pytest.raises(FileNotFoundError):
        setup.write_restore_constraints(tmp_path/'missing repo', runtime, 'pydantic==2.14.0\n')
    assert not (runtime/'restore-constraints.txt').exists()
