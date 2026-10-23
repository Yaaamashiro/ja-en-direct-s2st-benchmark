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
    from direct_s2st.cascade.qwen_generation import verify_success
    verify_success(adapter.generation, 'This is a synthesis test.', adapter.last_generation)
    assert adapter.last_generation['generated_tokens'] <= 1501


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('S2ST_RUN_QWEN_AB_TEST') != '1', reason='separate bounded A/B GPU opt-in required')
def test_bounded_real_greedy_vs_official_sampling(tmp_path, monkeypatch):
    """Diagnostic only: never feeds benchmark journals or chooses a lucky output.

    Both modes use the same seed/text/budget. Missing EOS is a recorded failure,
    not accepted truncated audio. Results do not imply a universal root cause.
    """
    from pathlib import Path
    import json
    from direct_s2st.cascade import qwen_generation as generation
    from direct_s2st.cascade.tts import QwenTTS
    from direct_s2st.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[2]/'configs/cascade/default.yaml')['tts']
    adapter = QwenTTS(model_id=cfg['model'], revision=cfg['revision'], speaker=cfg['speaker'],
                      language=cfg['language'], generation=dict(max_audio_seconds=20, max_wall_seconds=180))
    base = generation.generation_kwargs
    outcomes = []
    texts = ['This is a synthesis test.', 'The victim"s name is Sebastian Jones.',
             'Please put the blue book on the table.']
    for mode in ('greedy_diagnostic_only', 'official_sampling'):
        def options(policy):
            return dict(base(policy), do_sample=mode == 'official_sampling', subtalker_dosample=mode == 'official_sampling')
        monkeypatch.setattr(generation, 'generation_kwargs', options)
        for index, text in enumerate(texts):
            try:
                adapter(text, tmp_path/f'{mode}-{index}.wav')
            except generation.GenerationFailure:
                pass
            outcomes.append(dict(mode=mode, text=text, generation=dict(adapter.last_generation)))
    (tmp_path/'ab-results.json').write_text(json.dumps(outcomes, indent=2), encoding='utf-8')
    assert len(outcomes) == 6
