from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from direct_s2st.cascade import tts
from direct_s2st.io import atomic_write_json
from direct_s2st.journal import digest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
import cascade_loader_overlay as repair

REVISION = 'f00cf133b78d3c2c35857faba3b1be9b98c4f971'


def snapshot(tmp_path, monkeypatch):
    import huggingface_hub
    root = tmp_path/'hub/models--Qwen--TTS/snapshots'/REVISION
    for name in tts.REQUIRED_FILES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"fixture":true}' if name.endswith('.json') else b'fixture bytes')
    # The exact feature extractor contract in the published pinned revision.
    atomic_write_json(root/'speech_tokenizer/preprocessor_config.json', dict(
        chunk_length_s=None, feature_extractor_type='EncodecFeatureExtractor', feature_size=1,
        overlap=None, padding_side='right', padding_value=0.0,
        return_attention_mask=True, sampling_rate=24000), overwrite=True)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(root)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    monkeypatch.setattr(huggingface_hub.constants, 'HF_HUB_CACHE', str(tmp_path/'hub'))
    return root, calls


def test_complete_pinned_local_bundle_prevents_sdk_nested_main_resolution(tmp_path, monkeypatch):
    import torch
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'NVIDIA L4')
    root, downloads = snapshot(tmp_path, monkeypatch)
    from transformers.utils.hub import cached_file
    from transformers import AutoFeatureExtractor
    sdk_calls = []
    def load(path, **kwargs):
        sdk_calls.append((path, kwargs))
        assert Path(path) == root and kwargs['local_files_only'] is True
        assert 'revision' not in kwargs  # Do not rely on the SDK forwarding it.
        # Reproduce qwen-tts 0.1.1's nested cached_file(revision=None) call.
        config = cached_file(path, 'speech_tokenizer/config.json', revision=None)
        extractor = AutoFeatureExtractor.from_pretrained(str(Path(config).parent), local_files_only=True)
        assert extractor.sampling_rate == 24000
        return SimpleNamespace(generate_custom_voice=lambda **kwargs: ([], 24000))
    package = ModuleType('qwen_tts')
    package.Qwen3TTSModel = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, 'qwen_tts', package)
    adapter = tts.QwenTTS(model_id='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', revision=REVISION,
                          speaker='Ono_Anna', language='English')
    assert adapter.speaker == 'Ono_Anna' and len(sdk_calls) == 1
    assert downloads == [dict(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', revision=REVISION)]
    assert sdk_calls[0][1]['attn_implementation'] == 'sdpa'
    assert sdk_calls[0][1]['dtype'] == torch.bfloat16
    assert adapter.runtime['gpu'] == 'NVIDIA L4' and adapter.runtime['bf16_supported']


@pytest.mark.parametrize('name', ['speech_tokenizer/preprocessor_config.json', 'preprocessor_config.json',
                                 'speech_tokenizer/model.safetensors'])
def test_missing_bundle_member_fails_before_gpu_model_load(tmp_path, monkeypatch, name):
    root, _ = snapshot(tmp_path, monkeypatch)
    (root/name).unlink()
    monkeypatch.setattr(tts, 'check_runtime', lambda device: {})
    with pytest.raises(ValueError, match='incomplete pinned Qwen snapshot'):
        tts.QwenTTS(model_id='Qwen/model', revision=REVISION, speaker='x', language='English')


def test_invalid_revision_cache_location_and_json_are_rejected(tmp_path, monkeypatch):
    root, downloads = snapshot(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='immutable revision'):
        tts.pinned_snapshot('Qwen/model', 'main')
    assert not downloads
    atomic_write_json(root/'config.json', {}, overwrite=True)
    with pytest.raises(ValueError, match='invalid Qwen JSON'):
        tts.pinned_snapshot('Qwen/model', REVISION)
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='outside CORPUS_ROOT'):
        tts.pinned_snapshot('Qwen/model', REVISION)


def test_sdk_cannot_receive_another_revision_snapshot(tmp_path, monkeypatch):
    root, _ = snapshot(tmp_path, monkeypatch)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda **kwargs: str(root.parent/('0'*40)))
    with pytest.raises(ValueError, match='snapshot path'):
        tts.pinned_snapshot('Qwen/model', REVISION)


@pytest.mark.parametrize('kind', ['nan', 'inf', 'silent', 'stereo'])
def test_raw_qwen_waveform_is_checked_before_pcm_conversion(tmp_path, kind):
    import numpy as np
    values = dict(nan=np.array([.1, float('nan')]), inf=np.array([.1, float('inf')]),
                  silent=np.zeros(10), stereo=np.ones((10, 2))*.1)
    adapter = object.__new__(tts.QwenTTS)
    adapter.speaker, adapter.language = 'Ono_Anna', 'English'
    adapter.model = SimpleNamespace(generate_custom_voice=lambda **kwargs: ([values[kind]], 24000))
    path = tmp_path/'result.wav'
    with pytest.raises(ValueError, match='invalid/non-finite'):
        adapter('This is a synthesis test.', path)
    assert not path.exists()


def test_valid_qwen_generation_keeps_original_decoding_and_pcm_bytes(tmp_path):
    import numpy as np
    import soundfile as sf
    waveform = np.sin(np.arange(1000)/10)*.1
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return [waveform], 24000
    adapter = object.__new__(tts.QwenTTS)
    adapter.speaker, adapter.language = 'Ono_Anna', 'English'
    adapter.model = SimpleNamespace(generate_custom_voice=generate)
    path, original = tmp_path/'new.wav', tmp_path/'original.wav'
    adapter('saved MT output', path)
    sf.write(original, waveform, 24000, subtype='PCM_16')
    assert path.read_bytes() == original.read_bytes()
    assert calls == [dict(text='saved MT output', language='English', speaker='Ono_Anna',
                          do_sample=False, subtalker_dosample=False)]


def overlay_plan(tmp_path):
    path = tmp_path/'drive/plan.json'
    plan = dict(identity='a'*64, execution_revision=repair.BASE_REVISION,
                output=str(path.parent), local=str(tmp_path/'local'))
    atomic_write_json(path, plan)
    return path, plan


def git_stub(root, *args):
    if args == ('rev-parse','HEAD'):
        return 'b'*40
    if args[:1] == ('status',):
        return ''
    if args[:1] == ('merge-base',):
        return repair.BASE_REVISION
    if args[:1] == ('diff',):
        return 'src/direct_s2st/cascade/tts.py\nscripts/colab/cascade_loader_overlay.py'
    raise AssertionError(args)


def test_reviewed_overlay_preserves_plan_and_can_resume_audio_created_after_adoption(tmp_path, monkeypatch):
    path, plan = overlay_plan(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(repair, 'git', git_stub)
    with pytest.raises(ValueError, match='not adopted'):
        repair.approve(tmp_path, path, plan, 'b'*40)
    receipt = repair.approve(tmp_path, path, plan, 'b'*40, adopt=True)
    assert path.read_bytes() == before
    wav = Path(plan['local'])/'audio/generated-after-repair.wav'
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b'later generated WAV')
    assert repair.approve(tmp_path, path, plan, 'b'*40, adopt=True) == receipt
    assert receipt['body']['base_revision'] == repair.BASE_REVISION


def test_old_tts_audio_and_unreviewed_changes_refuse_automatic_reuse(tmp_path, monkeypatch):
    path, plan = overlay_plan(tmp_path)
    monkeypatch.setattr(repair, 'git', git_stub)
    body = dict(identity=plan['identity'], samples=[dict(status='success')])
    atomic_write_json(Path(plan['output'])/'audio-index.json', dict(body=body, sha256=digest(body)))
    with pytest.raises(ValueError, match='already generated audio'):
        repair.approve(tmp_path, path, plan, 'b'*40, adopt=True)
    def changed(root, *args):
        return 'src/direct_s2st/cascade/asr.py' if args[:1] == ('diff',) else git_stub(root,*args)
    monkeypatch.setattr(repair, 'git', changed)
    with pytest.raises(ValueError, match='unreviewed'):
        repair.approve(tmp_path, path, plan, 'b'*40, adopt=True)


def test_uncommitted_archive_refuses_automatic_loader_adoption(tmp_path, monkeypatch):
    path, plan = overlay_plan(tmp_path)
    monkeypatch.setattr(repair, 'git', git_stub)
    archive = Path(plan['output'])/'audio/unfinished.zip'
    archive.parent.mkdir()
    archive.write_bytes(b'uncommitted data')
    with pytest.raises(ValueError, match='uncommitted old TTS archives'):
        repair.approve(tmp_path, path, plan, 'b'*40, adopt=True)
