import ast
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('order', ['before_fairseq', 'after_fairseq', 'later_explicit_debug'])
def test_debug_is_suppressed_but_info_warnings_errors_progress_and_json_survive(order):
    # Run in a fresh process, not pytest's globally configured capture handlers.
    # Execute the actual offending basicConfig call from pinned fairseq without
    # importing its Python-3.10-only model registry or requiring GPU/dependencies.
    source = ROOT/'third_party/fairseq/fairseq/models/wav2vec/wav2vec2_classification.py'
    code = '''
import ast, json, logging, warnings
from pathlib import Path
from direct_s2st.inference_logging import configure_logging
from direct_s2st.progress import track
source = Path(SOURCE)
tree = ast.parse(source.read_text(encoding='utf-8'))
node = next(n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'basicConfig')
def fairseq_import_configuration():
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), {'logging':logging})
if ORDER == 'after_fairseq':
    fairseq_import_configuration()
configure_logging()
fairseq_import_configuration()
if ORDER == 'later_explicit_debug':
    logging.getLogger().setLevel(logging.DEBUG)
configure_logging()
for name in ('numba.core.byteflow', 'numba.core.ssa', 'filelock', 'fairseq'):
    logger = logging.getLogger(name)
    # Explicit child DEBUG bypasses the parent logger's level; handler filtering
    # must still keep this record out while allowing more important messages.
    logger.setLevel(logging.DEBUG)
    logger.debug('hidden-debug-' + name)
    logger.info('kept-info-' + name)
    logger.warning('kept-warning-' + name)
    logger.error('kept-error-' + name)
try:
    raise ValueError('kept-exception')
except ValueError:
    logging.getLogger('fairseq').exception('kept-traceback')
warnings.warn('kept-python-warning')
list(track([{'pair_id':'example'}], 'logging-test'))
print(json.dumps({'status':'success'}))
'''
    code = 'SOURCE = '+repr(str(source))+'\nORDER = '+repr(order)+'\n'+code
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                            text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'status':'success'}
    assert 'hidden-debug' not in result.stderr and 'DEBUG:' not in result.stderr
    for name in ('numba.core.byteflow', 'numba.core.ssa', 'filelock', 'fairseq'):
        for level in ('info', 'warning', 'error'):
            assert f'kept-{level}-{name}' in result.stderr
    for text in ('Traceback', 'ValueError: kept-exception', 'kept-python-warning',
                 '[progress]', 'status=completed', 'checked=1/1'):
        assert text in result.stderr


def test_logging_setup_is_idempotent_and_retains_existing_handlers_and_stricter_levels(monkeypatch):
    from direct_s2st.inference_logging import configure_logging
    root = logging.Logger('isolated', level=logging.DEBUG)
    info, warning = logging.StreamHandler(), logging.StreamHandler()
    info.setLevel(logging.DEBUG)
    warning.setLevel(logging.WARNING)
    root.addHandler(info)
    root.addHandler(warning)
    streams = [h.stream for h in root.handlers]
    monkeypatch.setattr(logging, 'getLogger', lambda: root)
    monkeypatch.setattr(logging, 'basicConfig', lambda **kw: None)
    configure_logging()
    configure_logging()
    assert root.level == logging.INFO
    assert root.handlers == [info, warning]
    assert [h.stream for h in root.handlers] == streams
    assert [h.level for h in root.handlers] == [logging.INFO, logging.WARNING]
    root.setLevel(logging.ERROR)
    configure_logging()
    assert root.level == logging.ERROR  # Do not weaken a caller's stricter policy.


def test_real_numba_compilation_keeps_results_without_debug_dump():
    pytest.importorskip('numba')
    code = '''
import json, logging
logging.basicConfig(level=logging.DEBUG)
from direct_s2st.inference_logging import configure_logging
configure_logging()
import numba, numpy as np
@numba.njit
def squares(values):
    total = 0.0
    for x in values:
        total += x*x
    return total
result = squares(np.arange(8, dtype=np.float64))
print(json.dumps({'result':result, 'compiled':bool(squares.nopython_signatures)}))
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT/'src')), text=True, encoding='utf-8', timeout=45)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(result=140.0, compiled=True)
    assert 'DEBUG:' not in result.stderr and 'Prune PHIs' not in result.stderr


@pytest.mark.parametrize('relative', [
    'scripts/colab/inference_session.py', 'src/direct_s2st/preview.py',
    'src/direct_s2st/translatotron2/infer.py', 'src/direct_s2st/s2ut/fairseq_infer.py',
    'src/direct_s2st/s2ut/preview_generate.py'])
def test_preview_parent_and_every_inference_child_configure_logging_at_entry(relative):
    tree = ast.parse((ROOT/relative).read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert isinstance(main.body[0], ast.ImportFrom)
    assert main.body[0].module.endswith('inference_logging')
    assert isinstance(main.body[1], ast.Expr)
    assert main.body[1].value.func.id == 'configure_logging'
