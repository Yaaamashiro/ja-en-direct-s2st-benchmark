"""Audited qwen-tts 0.1.1 generation contract; no model imports at launch time.

The SDK discards extra kwargs before talker.generate. Observe the actual main
talker, not the 15-code sub-talker, and reject missing EOS BEFORE codec decode.
"""
from contextlib import contextmanager
import hashlib
import math
import random
import time

from ..progress import Progress

POLICY = 'official_sampling_eos_v1'
OFFICIAL = dict(do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
                repetition_penalty=1.05, subtalker_dosample=True,
                subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)
CODEC_EOS = 2150
CODEC_RATE = 12.5  # Fixed tokenizer: 24000 samples/sec / 1920 samples/frame.


class GenerationFailure(ValueError):
    """A sample failed; never publish its truncated waveform as success."""


class GenerationContractError(RuntimeError):
    fatal_tts = True


def generation_policy(value=None):
    defaults = dict(policy=POLICY, seed=17, max_audio_seconds=120.0, max_wall_seconds=600.0)
    value = defaults if value is None else dict(value)
    if value.keys() - defaults.keys():
        raise ValueError('unreviewed Qwen generation policy fields')
    result = dict(defaults, **value)
    if result['policy'] != POLICY:
        raise ValueError('unsupported Qwen generation policy; do not reuse legacy greedy WAVs')
    if type(result['seed']) is not int or not 0 <= result['seed'] < 2**32:
        raise ValueError('Qwen seed must be an integer in [0, 2**32)')
    for key, maximum in [('max_audio_seconds', 120), ('max_wall_seconds', 3600)]:
        item = result[key]
        if isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item) or not 0 < item <= maximum:
            raise ValueError('invalid Qwen '+key)
    return result


def generation_kwargs(policy):
    # Preserve the previous accepted waveform ceiling, plus one EOS step.
    cap = math.floor(policy['max_audio_seconds']*CODEC_RATE)+1
    return dict(OFFICIAL, max_new_tokens=cap)


def text_seed(text, policy):
    key = str(policy['seed'])+'\0'+text
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:4], 'big')


def verify_success(policy, text, record):
    policy = generation_policy(policy)
    if (not isinstance(record, dict) or record.get('policy') != policy
            or record.get('seed') != text_seed(text, policy) or record.get('status') != 'success'
            or record.get('stop_reason') != 'eos' or record.get('eos_observed') is not True
            or record.get('eos_token_id') != CODEC_EOS
            or record.get('max_new_tokens') != generation_kwargs(policy)['max_new_tokens']
            or type(record.get('generated_tokens')) is not int
            or not 1 <= record['generated_tokens'] <= record['max_new_tokens']):
        raise GenerationContractError('Qwen successful WAV lacks matching verified EOS/policy evidence')


@contextmanager
def isolated_rng(seed):
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class GenerationWatch:
    def __init__(self, policy, diagnostic, sample, *, clock=time.monotonic):
        self.policy, self.diagnostic, self.clock = policy, diagnostic, clock
        self.cap = generation_kwargs(policy)['max_new_tokens']
        self.progress = Progress('Qwen TTS: main audio tokens', total=self.cap, counted=True)
        self.sample = sample
        self.started = None
        self.initial_length = None
        self.calls = 0
        self.false = None

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise GenerationContractError('Qwen EOS monitoring requires exactly one sequence')
        length = input_ids.shape[1]
        if self.initial_length is None:
            self.initial_length = length-1
        steps = length-self.initial_length
        if steps <= self.diagnostic.get('generated_tokens', 0):
            raise GenerationContractError('Qwen main token counter did not advance')
        eos = input_ids[0, -1].item() == CODEC_EOS
        elapsed = self.clock()-self.started
        self.diagnostic.update(generated_tokens=steps, eos_observed=eos, generation_seconds=elapsed)
        self.progress.count = steps
        self.progress.advanced = time.monotonic()
        self.progress.current = self.sample
        self.progress.activity = f'main_codec eos={eos} tokens_per_s={steps/max(elapsed, .001):.2f}'
        if not eos and steps >= self.cap:
            self.diagnostic['stop_reason'] = 'token_limit_without_eos'
            raise GenerationFailure(f'Qwen token limit {self.cap} reached without audio EOS; no WAV decoded')
        if not eos and elapsed >= self.policy['max_wall_seconds']:
            self.diagnostic['stop_reason'] = 'time_limit_without_eos'
            raise GenerationFailure('Qwen cooperative time limit reached without audio EOS; no WAV decoded')
        if self.false is None:
            self.false = torch.zeros(1, dtype=torch.bool, device=input_ids.device)
        return self.false  # Normal EOS stopping remains Hugging Face's own rule.

    @contextmanager
    def attach(self, sdk):
        """Instance-only, reversible interception at the SDK's actual boundary."""
        from transformers import StoppingCriteriaList
        talker = sdk.model.talker
        original = talker.generate
        had_override = 'generate' in vars(talker)
        override = vars(talker).get('generate')

        def generate(*args, **kwargs):
            self.calls += 1
            if self.calls != 1:
                raise GenerationContractError('unexpected repeated Qwen main-talker generation')
            if kwargs.get('eos_token_id') != CODEC_EOS:
                raise GenerationContractError('Qwen runtime audio EOS differs from fixed codec EOS 2150')
            expected = generation_kwargs(self.policy)
            if any(kwargs.get(key) != value for key, value in expected.items()):
                raise GenerationContractError('Qwen runtime decoding arguments differ from declared policy')
            existing = list(kwargs.get('stopping_criteria') or [])
            kwargs['stopping_criteria'] = StoppingCriteriaList(existing+[self])
            self.started = self.clock()
            self.diagnostic.update(stage='main_token_generation', generated_tokens=0,
                                   eos_token_id=CODEC_EOS, eos_observed=False,
                                   max_new_tokens=self.cap, stop_reason=None,
                                   effective_generation_kwargs=dict(expected))
            with self.progress:
                self.progress.current = self.sample
                self.progress.activity = 'waiting_for_first_main_codec_token'
                result = original(*args, **kwargs)
                if (self.progress.count == 0 or not self.diagnostic.get('eos_observed')
                        or result.sequences.shape[0] != 1
                        or result.sequences[0, -1].item() != CODEC_EOS):
                    self.diagnostic['stop_reason'] = 'missing_eos'
                    raise GenerationFailure('Qwen returned without verified audio EOS; no WAV decoded')
                self.diagnostic.update(stop_reason='eos', stage='waveform_decode')
            return result

        talker.generate = generate
        try:
            yield
            if self.calls != 1:
                raise GenerationContractError('Qwen SDK bypassed the monitored main talker')
        finally:
            if had_override:
                talker.generate = override
            else:
                delattr(talker, 'generate')
