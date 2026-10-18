"""An explicit new S2UT run, not an in-place schedule change to old runs."""
import json
import os

LEGACY = 'legacy_short'
PAPER = 'paper_fisher_v1'
RECIPES = (LEGACY, PAPER)
MAX_UPDATES = 400000
MILESTONES = (1000, 5000, 10000, 25000, 50000, 100000, 200000, 400000)


def report_policy(interval=1000, train_interval=100):
    return dict(version=1, mode='report_only', interval=interval, train_interval=train_interval,
                max_update=MAX_UPDATES)


def bleu_policy():
    return dict(version=1, aux_interval=1000, aux_samples=500, asr_samples=100,
                asr_updates=list(MILESTONES), seed=1, beam=10, max_len_a=1, max_len_b=200,
                max_aux_tokens=256, max_seconds=1800,
                asr_model='openai/whisper-large-v3-turbo',
                asr_revision='60be3615a4d667e1258e8ad29130467587c489aa')


def recipe_from_config(config):
    from ..recipes import option
    return option(config.get('command', []), '--s2ut-recipe', LEGACY)


def activate(config):
    """Run-scoped options are bound to the immutable config, not notebook leftovers."""
    if config.get('kind') != 's2ut' or recipe_from_config(config) != PAPER:
        os.environ.pop('S2ST_S2UT_PAPER_MONITOR', None)
        os.environ.pop('S2ST_S2UT_BLEU_POLICY', None)
        return
    settings = config['s2ut_paper']
    validate_report_policy(settings['monitor'])
    validate_bleu_policy(settings['bleu'])
    os.environ.pop('S2ST_S2UT_MONITOR_POLICY', None)
    os.environ['S2ST_S2UT_PAPER_MONITOR'] = json.dumps(settings['monitor'])
    os.environ['S2ST_S2UT_BLEU_POLICY'] = json.dumps(settings['bleu'])


def validate_report_policy(value):
    if (not isinstance(value,dict) or set(value) != set(report_policy())
            or type(value['version']) is not int or value['version'] != 1 or value['mode'] != 'report_only'
            or value['max_update'] != MAX_UPDATES
            or any(type(value[k]) is not int or value[k] < 1 for k in ('interval','train_interval'))
            or value['interval'] % value['train_interval']):
        raise ValueError('invalid S2UT paper report-only policy')
    return value


def validate_bleu_policy(value):
    if (not isinstance(value,dict) or set(value) != set(bleu_policy())
            or type(value['version']) is not int or value['version'] != 1
            or any(type(value[k]) is not int or value[k] < 1 for k in
                   ('aux_interval','aux_samples','asr_samples','beam','max_len_b','max_aux_tokens','max_seconds'))
            or not 1 <= value['asr_samples'] <= value['aux_samples'] <= 500
            or value['beam'] != 10 or type(value['max_len_a']) is not int or value['max_len_a'] != 1
            or type(value['seed']) is not int or value['seed'] != 1
            or not isinstance(value['asr_updates'],list) or not value['asr_updates']
            or any(type(u) is not int for u in value['asr_updates'])
            or value['asr_updates'] != sorted(set(value['asr_updates']))
            or any(type(u) is not int or not 1 <= u <= MAX_UPDATES or u % value['aux_interval']
                   for u in value['asr_updates'])
            or value['asr_model'] != bleu_policy()['asr_model']
            or value['asr_revision'] != bleu_policy()['asr_revision']):
        raise ValueError('invalid S2UT online BLEU policy')
    return value


def preflight_dependencies():
    """Fail before expensive input staging, never install into a saved run silently."""
    import importlib
    import importlib.metadata
    names = ('torch', 'torchaudio', 'librosa', 'numpy', 'soundfile', 'sacrebleu', 'accelerate',
             'transformers', 'sentencepiece', 'editdistance')
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError as error:
            raise RuntimeError(f'S2UT paper/BLEU requires {name}; rebuild via normal cell 5') from error
    try:
        # Lazy top-level imports alone do not detect broken audio pipelines or
        # an incompatible fairseq registry before spending hours training.
        getattr(importlib.import_module('transformers'), 'pipeline')
        importlib.import_module('fairseq.models.text_to_speech.codehifigan')
        importlib.import_module('fairseq.sequence_generator')
    except Exception as error:
        raise RuntimeError('S2UT BLEU native fairseq/transformers audio imports failed; '
                           'rebuild the pinned Python 3.10 runtime via normal cell 5') from error
    return {name: importlib.metadata.version(name) for name in names}
