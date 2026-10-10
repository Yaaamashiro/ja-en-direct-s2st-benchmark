"""Paper recipe contracts. Exact batch does not mean identical paper data/results."""
import json
from pathlib import Path

from .config import load_config

MODES = ('smoke', 'paper_exact', 'paper_practical')
TT2_VOCODERS = ('griffin_lim', 'hifigan')
S2UT_VOCODERS = ('paper', 'trained')
FAIRSEQ_REVISION = '3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99'


def option(command, name, default=None):
    return command[command.index(name) + 1] if name in command else default


def replace_option(command, name, value):
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend([name, str(value)])


def tt2_batch(mode, batch_size, update_freq=None, world_size=1):
    if mode not in MODES or min(batch_size, world_size) < 1:
        raise ValueError('invalid recipe/batch/world size')
    if mode == 'paper_exact':
        denominator = batch_size * world_size
        if 1024 % denominator:
            raise ValueError('paper_exact batch_size * world_size must divide 1024')
        required = 1024 // denominator
        if update_freq is not None and update_freq != required:
            raise ValueError('paper_exact requires effective global batch 1024')
        return required
    result = 1 if update_freq is None else update_freq
    if result < 1:
        raise ValueError('update_freq must be positive')
    return result


def tt2_source_spec(repository):
    from .translatotron2.prepare_fairseq import _mel_settings
    values = load_config(Path(repository) / 'configs/translatotron2/prepare-paper.yaml')['source_mel']
    return dict(_mel_settings(values), log_transform='natural_log_clamp_eps', normalization='utterance_cmvn')


def prepared_metadata(root):
    """Read actual immutable prepared metadata, never infer shape from a preset name."""
    root = Path(root)
    result = {'dataset_fingerprint': json.loads((root / 'data-lock.json').read_text(encoding='utf-8'))}
    for name, field in [('mel-spec.json', 'target_feature_config'),
                        ('source-mel-spec.json', 'source_feature_config')]:
        if (root / name).is_file():
            result[field] = json.loads((root / name).read_text(encoding='utf-8'))
    if 'unit_configuration' in result['dataset_fingerprint']:
        if result['dataset_fingerprint'].get('target') != 'reduced_units':
            raise ValueError('paper S2UT requires prepared reduced units')
        import yaml
        if (root / 'config.yaml').is_file():
            result['source_feature_config'] = dict(sample_rate=16000, implementation='pinned fairseq frontend',
                prepared_config=yaml.safe_load((root / 'config.yaml').read_text(encoding='utf-8')))
        result['target_feature_config'] = dict(type='reduced_units',
                                              **result['dataset_fingerprint']['unit_configuration'])
        if (root / 'config_multitask.yaml').is_file():
            result['auxiliary_tasks'] = yaml.safe_load((root / 'config_multitask.yaml').read_text(encoding='utf-8'))
    return result


def training_metadata(command, system, mode, vocoder_mode, *, world_size=1):
    value = lambda name, default=None: option(command, name, default)
    tt2 = system == 'tt2'
    size = int(value('--batch-size', '1')) if tt2 else None
    frequency = int(value('--update-freq', '1'))
    return dict(system='translatotron2' if tt2 else system, reproduction_mode=mode,
        architecture_preset=value('--model-size') if tt2 else 'fisher',
        model_architecture='native_translatotron2' if tt2 else value('--arch'),
        optimizer='Adam', learning_rate=float(value('--learning-rate' if tt2 else '--lr')),
        optimizer_betas=[0.9, 0.999] if tt2 else [0.9, 0.98],
        precision='fp32' if tt2 else ('bf16' if '--bf16' in command else 'fp16' if '--fp16' in command else 'fp32'),
        lr_scheduler='warmup_inverse_sqrt' if tt2 else value('--lr-scheduler'),
        warmup_updates=int(value('--warmup-updates', '0')),
        l2=float(value('--l2-regularization' if tt2 else '--weight-decay', '0')),
        gradient_clipping=1.0 if tt2 else float(value('--clip-norm', '0')),
        physical_batch_size=size, max_tokens=None if tt2 else int(value('--max-tokens')),
        update_freq=frequency, world_size=world_size,
        effective_batch_size=size * frequency * world_size if tt2 else None,
        batch_unit='utterances' if tt2 else 'variable-size token-budget batches',
        batch_policy=('official Fisher token-budget batches; no additional splitting'
                      if system == 's2ut' and mode == 'paper_exact' else 'explicit configured batches'),
        physical_batch_cap=None,
        vocoder_mode=vocoder_mode, vocoder_checkpoint_sha256=None,
        engine_version='tt2-cpu-mask-rng-vector-prenet-v2' if tt2 else 'pinned-fairseq-unigram-v2',
        ctc_tokenizer=None if tt2 else 'sentencepiece-unigram-1000-v1',
        vocoder_training_condition=(None if tt2 else 'target_corpus_train_only' if vocoder_mode == 'trained'
                                    else 'released_LJSpeech_checkpoint_not_Fisher_train'),
        training_length_source='researcher_TOTAL_UPDATES',
        recommended_total_updates=None if tt2 else 400000,
        paper_total_updates_status='unknown/not specified in Appendix A' if tt2 else '400000 in pinned fairseq recipe',
        deviations=(['Japanese→English synthetic corpus', 'eSpeak-NG replaces proprietary Google G2P',
                     'voice preservation disabled', '16-kHz source adapted from Fisher 8-kHz',
                     '16-kHz/80-bin target retained by user instead of paper 24-kHz/128-bin',
                     ('Griffin-Lim iterations/phase initialization are implementation choices' if vocoder_mode == 'griffin_lim' else 'HiFi-GAN replaces paper BLEU Griffin-Lim'),
                     'Whisper ASR-BLEU replaces Google ASR evaluator'] if tt2 else
                    ['Japanese→English synthetic corpus', 'Japanese/English character CE; train-only Unigram-1000 CTC',
                     'official single-GPU accumulation simulates paper four-GPU training',
                     ('independent Code HiFi-GAN training on target train corpus; frontend/crop/batch differ from original Fisher vocoder'
                      if vocoder_mode == 'trained' else 'released LJSpeech vocoder differs from paper Fisher-trained vocoder')]),
        numerical_equivalence=('not established; TT2 BatchNorm statistics remain microbatch-local' if tt2 else 'not established for the adapted corpus'),
        implementation='independent PyTorch, not Google code' if tt2 else 'official pinned fairseq',
        fairseq_revision=FAIRSEQ_REVISION)


def validate_s2ut_runtime(mode, *, fixed_microbatch=0, adaptive_batch=False, precision='default'):
    """Exact means preserving per-GPU token batches, not only effective volume."""
    if mode == 'paper_exact' and (fixed_microbatch != 0 or adaptive_batch or precision != 'default'):
        raise ValueError('S2UT paper_exact forbids additional batch splitting/adaptation or precision changes; use paper_practical in a new run')


def validate_s2ut_paper(command, *, exact=True):
    expected = {'--arch': 's2ut_transformer_fisher', '--criterion': 'speech_to_unit',
        '--label-smoothing': '0.2', '--optimizer': 'adam', '--adam-betas': '(0.9,0.98)',
        '--lr': '0.0005', '--lr-scheduler': 'inverse_sqrt', '--warmup-init-lr': '1e-7',
        '--warmup-updates': '10000', '--clip-norm': '10.0', '--max-target-positions': '3000',
        '--task': 'speech_to_speech', '--target-code-size': '100',
        '--dropout': '0.1', '--attention-dropout': '0.1', '--relu-dropout': '0.1'}
    if exact:
        expected.update({'--max-tokens': '20000', '--update-freq': '4'})
        if any(key in command for key in ('--batch-size', '--max-sentences')):
            raise ValueError('S2UT paper_exact uses official token-budget batches, not an utterance batch cap')
    required = ('--target-is-code', '--share-decoder-input-output-embed')
    if any(command.count(key) != 1 or option(command, key) != val for key, val in expected.items()) or any(key not in command for key in required) or (exact and ('--fp16' not in command or '--bf16' in command or '--memory-efficient-fp16' in command)):
        raise ValueError('S2UT paper_exact requires the pinned Fisher architecture/optimizer/batch/FP16 recipe')
