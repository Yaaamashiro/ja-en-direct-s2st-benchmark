"""Real two-process Gloo mechanics using explicit synthetic input tensors."""
import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_two_process_cpu_checkpoint_resume(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip('CPU Gloo backend unavailable')
    root = Path(__file__).resolve().parents[2]
    import os
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(root / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, str(root / 'tests/helpers/ddp_cpu_probe.py'), str(tmp_path / 'ddp')],
                            capture_output=True, text=True, timeout=90, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        assert json.loads((tmp_path / f'ddp/rank-{rank}.json').read_text())['status'] == 'PASS'
