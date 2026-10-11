"""Real -m launcher regressions, with explicit CPU-only CUDA/trainer doubles."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module', [
    'direct_s2st.translatotron2.train',
    'direct_s2st.s2ut.fairseq_train',
    'direct_s2st.vocoders.train',
])
@pytest.mark.parametrize('outcome', ['ok', 'oom', 'recoverable_oom', 'error', 'early_exit'])
def test_real_module_launcher_protocol(tmp_path, module, outcome):
    # No GPU or actual trainer is impersonated as real execution: these
    # doubles test Python module identity and launcher exit/result semantics.
    (tmp_path / 'torch.py').write_text('''
class cuda:
    class OutOfMemoryError(RuntimeError): pass
    @staticmethod
    def is_available(): return True
    @staticmethod
    def synchronize(): pass
    @staticmethod
    def reset_peak_memory_stats(): pass
    @staticmethod
    def mem_get_info(): return 60, 100
    @staticmethod
    def memory_reserved(): return 20
    @staticmethod
    def max_memory_allocated(): return 50
    @staticmethod
    def max_memory_reserved(): return 60
''', encoding='utf-8')
    (tmp_path / 'sitecustomize.py').write_text('''
import runpy
original = runpy.run_module
def run_module(name, **kwargs):
    if name in ('direct_s2st.translatotron2.train',
                'direct_s2st.s2ut.fairseq_train', 'direct_s2st.vocoders.train'):
        name = 'probe_fixture'
    return original(name, **kwargs)
runpy.run_module = run_module
''', encoding='utf-8')
    (tmp_path / 'probe_fixture.py').write_text('''
import os
import torch
from direct_s2st.batch_probe import Probe
from direct_s2st.train_runtime import RecoverableOOM
outcome = os.environ['PROBE_FIXTURE_OUTCOME']
if outcome == 'oom': raise torch.cuda.OutOfMemoryError('fixture allocation')
if outcome == 'recoverable_oom': raise RecoverableOOM('fixture backward')
if outcome == 'error': raise ValueError('fixture dictionary error')
probe = Probe()
for _ in range(probe.limit - (outcome == 'early_exit')):
    probe.begin()
    probe.finish(8)
if outcome == 'ok':
    raise AssertionError('normal execution must stop before checkpoint publication')
raise SystemExit(0)
''', encoding='utf-8')
    path = tmp_path / 'measurement.json'
    env = dict(os.environ, S2ST_BATCH_PROBE=str(path), S2ST_BATCH_PROBE_STEPS='3',
               PROBE_FIXTURE_OUTCOME=outcome,
               PYTHONPATH=os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[2] / 'src')]))
    env.pop('S2ST_TRAIN_STAGING', None)
    result = subprocess.run([sys.executable, '-m', 'direct_s2st.batch_probe',
                             '-m', module, '--fixture'], env=env, text=True,
                            capture_output=True, timeout=30)
    if outcome in ('ok', 'oom', 'recoverable_oom'):
        assert result.returncode == 0, result.stderr
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert saved['status'] == ('ok' if outcome == 'ok' else 'oom')
        if outcome == 'ok':
            assert len(saved['steps']) == 3
            assert all(step['used'] == 80 for step in saved['steps'])
            assert 'ProbeComplete' not in result.stderr
    else:
        assert result.returncode != 0
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8'))['status'] != 'ok'
        assert ('fixture dictionary error' if outcome == 'error' else
                'trainer exited before completing probe measurements') in result.stderr
