"""Opt-in real L4/SDK/weights test. No mocks and no test references.

Run ONLY in the pinned Colab environment with S2ST_RUN_QWEN_GPU_TEST=1.
This downloads the pinned model if absent; excluded from the CPU test suite.
"""
import os

import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('S2ST_RUN_QWEN_GPU_TEST') != '1', reason='explicit real-model GPU opt-in required')
def test_real_l4_bf16_model_tokenizer_and_synthesis(tmp_path):
    from direct_s2st.cascade.tts import QwenTTS
    from direct_s2st.cascade.benchmark_worker import checked_wav
    from direct_s2st.config import load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[2]/'configs/cascade/default.yaml')['tts']
    adapter = QwenTTS(model_id=cfg['model'], revision=cfg['revision'],
                      speaker=cfg['speaker'], language=cfg['language'])
    assert 'L4' in adapter.runtime['gpu'] and adapter.runtime['bf16_supported']
    path = tmp_path/'preflight.wav'
    adapter('This is a synthesis test.', path)
    assert checked_wav(path) > 0
