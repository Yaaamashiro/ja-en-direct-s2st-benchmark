from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from direct_s2st.cascade import qwen_generation as generation
from direct_s2st.cascade.tts import QwenTTS


class FakeTalker:
    def __init__(self, codes=None, *, error=None):
        self.codes = codes if codes is not None else [8, 9, generation.CODEC_EOS]
        self.error = error
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        ids = torch.tensor([[1]])
        for code in self.codes:
            if self.error:
                raise self.error
            torch.rand(1)
            ids = torch.cat((ids, torch.tensor([[code]])), dim=1)
            for criterion in kwargs.get('stopping_criteria', []):
                criterion(ids, None)
        return SimpleNamespace(sequences=ids)


class FakeSDK:
    def __init__(self, codes=None, waveform=None, error=None):
        self.model = SimpleNamespace(config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=2150)),
                                     talker=FakeTalker(codes, error=error))
        self.waveform = waveform
        self.decode_count = 0
        self.calls = []

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        # qwen-tts 0.1.1 explicitly rebuilds talker kwargs and drops extras,
        # including stopping_criteria passed through the public SDK method.
        forwarded = {k: v for k, v in kwargs.items() if k in generation.generation_kwargs(generation.generation_policy())}
        forwarded['eos_token_id'] = generation.CODEC_EOS
        self.model.talker.generate(**forwarded)
        self.decode_count += 1
        wave = self.waveform if self.waveform is not None else np.sin(np.arange(1000)/10)*.1
        return [wave], 24000


def adapter(sdk=None, policy=None):
    value = object.__new__(QwenTTS)
    value.speaker, value.language = 'Ono_Anna', 'English'
    value.model = sdk or FakeSDK()
    value.generation = generation.generation_policy(policy)
    value._generation_lock = threading.Lock()
    value.last_generation = None
    return value


def test_eos_verified_before_decode_and_official_flags_are_used(tmp_path):
    value = adapter()
    value('saved MT', tmp_path/'sample.wav')
    assert value.model.decode_count == 1
    assert value.last_generation['eos_observed'] is True
    assert value.last_generation['generated_tokens'] == 3
    assert value.last_generation['stop_reason'] == 'eos'
    assert value.last_generation['status'] == 'success'
    assert value.model.calls[0] == dict(text='saved MT', language='English', speaker='Ono_Anna',
                                       non_streaming_mode=True, **generation.generation_kwargs(value.generation))
    assert 'generate' not in vars(value.model.model.talker)
    generation.verify_success(value.generation, 'saved MT', value.last_generation)


@pytest.mark.parametrize('field,value', [('policy', {}), ('eos_observed', False), ('seed', -1),
                                         ('stop_reason', 'token_limit_without_eos'), ('generated_tokens', 1502)])
def test_resume_rejects_missing_or_mismatched_generation_evidence(tmp_path, field, value):
    model = adapter()
    model('saved MT', tmp_path/'sample.wav')
    record = dict(model.last_generation, **{field: value})
    with pytest.raises(generation.GenerationContractError, match='verified EOS/policy evidence'):
        generation.verify_success(model.generation, 'saved MT', record)


def test_non_eos_limit_rejects_before_codec_decode_and_keeps_existing_wav(tmp_path):
    value = adapter(FakeSDK(codes=[8]*4), dict(max_audio_seconds=.16))  # cap=3 (2 frames + EOS).
    path = tmp_path/'existing.wav'
    path.write_bytes(b'original artifact')
    with pytest.raises(generation.GenerationFailure, match='limit 3 reached without audio EOS'):
        value('saved MT', path)
    assert value.model.decode_count == 0 and path.read_bytes() == b'original artifact'
    assert value.last_generation['stop_reason'] == 'token_limit_without_eos'
    assert value.last_generation['status'] == 'failed'
    assert 'generate' not in vars(value.model.model.talker)


def test_eos_on_final_allowed_step_is_not_truncation(tmp_path):
    value = adapter(policy=dict(max_audio_seconds=.16))
    value('saved MT', tmp_path/'last-step.wav')
    assert value.last_generation['stop_reason'] == 'eos'
    assert value.last_generation['generated_tokens'] == value.last_generation['max_new_tokens'] == 3


def test_return_without_eos_rejects_before_decode(tmp_path):
    value = adapter(FakeSDK(codes=[8]))
    with pytest.raises(generation.GenerationFailure, match='without verified audio EOS'):
        value('saved MT', tmp_path/'missing-eos.wav')
    assert value.model.decode_count == 0 and not list(tmp_path.glob('*.wav'))


def test_cooperative_deadline_without_eos(tmp_path, monkeypatch):
    watch_class = generation.GenerationWatch
    times = iter([0., 601.])
    monkeypatch.setattr(generation, 'GenerationWatch',
                        lambda *args: watch_class(*args, clock=lambda: next(times)))
    value = adapter()
    with pytest.raises(generation.GenerationFailure, match='time limit'):
        value('saved MT', tmp_path/'deadline.wav')
    assert value.model.decode_count == 0
    assert value.last_generation['stop_reason'] == 'time_limit_without_eos'


@pytest.mark.parametrize('error', [RuntimeError('CUDA out of memory'), KeyboardInterrupt()])
def test_failure_restores_instance_method_rng_and_lock(tmp_path, error):
    value = adapter(FakeSDK(error=error))
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()
    with pytest.raises(type(error)):
        value('saved MT', tmp_path/'failed.wav')
    assert torch.equal(state, torch.get_rng_state())
    assert 'generate' not in vars(value.model.model.talker)
    assert not value._generation_lock.locked() and value.model.decode_count == 0


def test_seed_is_text_based_not_call_order_or_output_location(tmp_path):
    value = adapter()
    value('same MT', tmp_path/'first.wav')
    seed = value.last_generation['seed']
    value('different MT', tmp_path/'another.wav')
    assert value.last_generation['seed'] != seed
    value('same MT', tmp_path/'resumed.wav')
    assert value.last_generation['seed'] == seed
    assert (tmp_path/'first.wav').read_bytes() == (tmp_path/'resumed.wav').read_bytes()


def test_wrong_audio_eos_or_decoding_flags_are_fatal_contract_errors():
    for changed in (dict(eos_token_id=151673), dict(do_sample=False)):
        sdk = FakeSDK()
        watch = generation.GenerationWatch(generation.generation_policy(), {}, 'sample')
        options = dict(generation.generation_kwargs(watch.policy), eos_token_id=2150)
        options.update(changed)
        with watch.attach(sdk), pytest.raises(generation.GenerationContractError):
            sdk.model.talker.generate(**options)


def test_declared_duration_cap_uses_actual_12_point_5_hz_not_model_name():
    assert generation.generation_kwargs(generation.generation_policy())['max_new_tokens'] == 1501


@pytest.mark.parametrize('values', [dict(seed=True), dict(seed=-1), dict(max_audio_seconds=121),
                                  dict(max_wall_seconds=float('inf')), dict(max_wall_seconds=0),
                                  dict(policy='legacy'), dict(do_sample=False)])
def test_invalid_or_unreviewed_policy_is_rejected(values):
    with pytest.raises(ValueError):
        generation.generation_policy(values)


def test_actual_transformers_generation_eos_and_main_counter():
    from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList, LogitsProcessor

    class Tokens(LogitsProcessor):
        def __call__(self, ids, scores):
            scores.fill_(-float('inf'))
            scores[:, 2150 if ids.shape[1] >= 3 else 8] = 0
            return scores

    tiny = GPT2LMHeadModel(GPT2Config(vocab_size=2151, n_embd=8, n_layer=1, n_head=1,
                                    n_positions=16, eos_token_id=2150, pad_token_id=0)).eval()
    class Talker:
        def generate(self, **kwargs):
            options = {k: v for k, v in kwargs.items() if not k.startswith('subtalker_')}
            return tiny.generate(torch.tensor([[1]]), return_dict_in_generate=True,
                                 logits_processor=LogitsProcessorList([Tokens()]), **options)
    sdk = SimpleNamespace(model=SimpleNamespace(talker=Talker()))
    diagnostic = {}
    policy = generation.generation_policy(dict(max_audio_seconds=.8))
    watch = generation.GenerationWatch(policy, diagnostic, 'real-transformers')
    with watch.attach(sdk):
        result = sdk.model.talker.generate(**generation.generation_kwargs(policy), eos_token_id=2150)
    assert result.sequences.tolist() == [[1, 8, 8, 2150]]
    assert diagnostic['eos_observed'] and diagnostic['generated_tokens'] == 3


def test_fixed_sdk_source_rebuilds_kwargs_instead_of_forwarding_stopping_criteria():
    """Inspect the exact wheel when present; no model weights/imports required."""
    import ast
    import os
    import zipfile
    wheel = Path(os.environ.get('S2ST_QWEN_SDK_WHEEL',
                 str(Path(__file__).resolve().parents[3]/'loss-review/qwen-loader-diagnosis/qwen_tts-0.1.1-py3-none-any.whl')))
    if not wheel.is_file():
        pytest.skip('exact qwen-tts 0.1.1 wheel not available for source contract test')
    with zipfile.ZipFile(wheel) as archive:
        tree = ast.parse(archive.read('qwen_tts/core/models/modeling_qwen3_tts.py').decode())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3TTSForConditionalGeneration')
    generate = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
    calls = [n for n in ast.walk(generate) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self.talker.generate']
    assert len(calls) == 1
    assert [ast.unparse(k.value) for k in calls[0].keywords if k.arg is None] == ['talker_kwargs']
    mapping = next(n.value for n in ast.walk(generate) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'talker_kwargs' for t in n.targets))
    assert isinstance(mapping, ast.Dict) and not any(k.value == 'stopping_criteria' for k in mapping.keys)
