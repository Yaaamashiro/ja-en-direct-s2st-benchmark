"""Explicit opt-in GPU test; requires an existing validated S2UT interactive plan.

Creates a separate temporary comparison; never edits the supplied run/results.
Actual Qwen/Whisper/NLLB/S2UT/vocoder downloads and CUDA inference cost time/CU.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest


@pytest.mark.gpu
@pytest.mark.skipif(not os.environ.get('S2ST_INTERACTIVE_GPU_PLAN'), reason='explicit real-model plan opt-in required')
def test_real_sequential_source_only_s2ut_comparison(tmp_path):
    from direct_s2st.interactive_comparison import PURPOSE, PHASES, read_stage, report
    from direct_s2st.io import atomic_write_json
    from direct_s2st.journal import digest
    original = json.loads(Path(os.environ['S2ST_INTERACTIVE_GPU_PLAN']).read_text(encoding='utf-8'))
    assert original['purpose'] == PURPOSE and original['model'] == 's2ut'
    plan = dict(original, local=str(tmp_path/'local'), output=str(tmp_path/'output'),
                identity=digest(dict(original=original['identity'], test=uuid.uuid4().hex)))
    path = tmp_path/'plan.json'
    atomic_write_json(path, plan)
    env = dict(os.environ, S2ST_INTERACTIVE_MEASUREMENT_SESSION=uuid.uuid4().hex)
    for phase in PHASES:
        subprocess.run([sys.executable, '-m', 'direct_s2st.interactive_comparison', '--plan', str(path), '--phase', phase],
                       env=env, check=True)
        assert read_stage(plan, phase)['status'] == 'success'
    result = report(plan)
    assert set(result['audio']) == {'source', 'cascade', 'candidate'}
    assert result['performance_same_session']
    assert result['human_reference'] is False
    for branch in ('cascade', 'candidate'):
        assert result['branches'][branch]['performance']['torch_allocated_peak_bytes'] > 0
        assert 0 <= result['evaluation']['branches'][branch]['sentence_bleu'] <= 100
