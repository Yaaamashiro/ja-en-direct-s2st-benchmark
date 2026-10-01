import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from direct_s2st.hashing import sha256_file
from direct_s2st.preparation import Checkpoints, checkpoint_map
from direct_s2st.translatotron2 import phonemize as phonemes
from direct_s2st.translatotron2.prepare_fairseq import validate_phonemes, prepare_fairseq
from direct_s2st.vocoders import verification
from test_translatotron2_prepare import _common, _mel_fixture


@pytest.fixture
def common(tmp_path, monkeypatch):
    monkeypatch.delenv('CORPUS_ROOT', raising=False)
    monkeypatch.setenv('S2ST_PREP_WORKERS', '1')
    root = tmp_path / 'common'
    _common(root)
    for split in ('train', 'dev', 'test'):
        path = root / f'{split}.jsonl'
        row = json.loads(path.read_text())
        row['en_sha256'] = sha256_file(Path(row['en_audio']))
        path.write_text(json.dumps(row) + '\n')
    return root


def mock_espeak(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='eSpeak NG 1.52.0' if '--version' in args else 'a', stderr='')
    monkeypatch.setattr(phonemes.subprocess, 'run', run)
    return calls


def test_lexical_words_are_not_split_and_cli_options_are_terminated(monkeypatch):
    calls = mock_espeak(monkeypatch)
    engine = phonemes.EspeakNgPhonemizer(expected_version='1.52.0')
    engine("Don't prefecture’s dogs' 3.14 1,000 U.S.A. well-known 12:30 -5 1/2!")
    assert [args[-1] for args in calls[1:]] == [
        "Don't", "prefecture's", "dogs'", '3.14', '1,000', 'U.S.A.', 'well-known', '12:30', '-5', '1/2']
    assert all(args[-2] == '--' for args in calls[1:])


def test_inventory_queries_fixed_engine_definitions(monkeypatch):
    calls = mock_espeak(monkeypatch)
    engine = phonemes.EspeakNgPhonemizer(expected_version='1.52.0')
    assert engine.fixed_vocabulary() == ('a', '|')
    definitions = [args[-1] for args in calls if args[-1].startswith('[[')]
    assert len(definitions) == len(phonemes._ESPEAK_EN_US_PHONEMES)
    assert '[[aI@]]' in definitions and '[[aU@]]' in definitions


def test_unknown_phone_fails_before_publishing_bad_label(common, tmp_path):
    output = tmp_path / 'phones'
    with pytest.raises(ValueError, match="pair-train.*unknown phonemes"):
        phonemes.phonemize_manifests(common, output, phonemizer=lambda _: 'UNEXPECTED',
            engine='fixture', version='1', fixed_vocabulary=['a'], resume=True)
    assert not (output / 'train.tsv').exists()
    assert not list((output / '.checkpoints').rglob('chunk-*.json'))


def test_legacy_complete_labels_keep_their_algorithm(common, tmp_path, monkeypatch, capsys):
    mock_espeak(monkeypatch)
    output = tmp_path / 'phones'
    kwargs = dict(engine='espeak-ng', version='1.52.0', resume=True)
    old = phonemes.EspeakNgPhonemizer(expected_version='1.52.0', text_processing_version='legacy-v1')
    phonemes.phonemize_manifests(common, output, phonemizer=old, **kwargs)
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    new = phonemes.EspeakNgPhonemizer(expected_version='1.52.0')
    phonemes.phonemize_manifests(common, output, phonemizer=new, **kwargs)
    assert new.text_processing_version == 'legacy-v1'
    assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before
    assert '--regenerate-phonemes' in capsys.readouterr().err


def test_partial_old_generation_requires_explicit_migration(common, tmp_path, monkeypatch):
    mock_espeak(monkeypatch)
    output = tmp_path / 'phones'
    output.mkdir()
    (output / 'inventory.txt').write_text('OLD\n')
    with pytest.raises(ValueError, match='regenerate-phonemes'):
        phonemes.phonemize_manifests(common, output,
            phonemizer=phonemes.EspeakNgPhonemizer(expected_version='1.52.0'),
            engine='espeak-ng', version='1.52.0', resume=True)
    assert (output / 'inventory.txt').read_text() == 'OLD\n'


@pytest.mark.parametrize('bad_sequence', ['   ', 'UNKNOWN'])
def test_saved_label_preflight_never_opens_audio(common, tmp_path, monkeypatch, bad_sequence):
    output = tmp_path / 'phones'
    phonemes.phonemize_manifests(common, output, phonemizer=lambda _: 'a',
        engine='fixture', version='1', fixed_vocabulary=['a'])
    (output / 'dev.tsv').write_text('pair-dev\t' + bad_sequence + '\n')
    def fail(*args, **kwargs):
        pytest.fail('audio resolved before invalid labels were rejected')
    from direct_s2st.translatotron2 import prepare_fairseq as module
    monkeypatch.setattr(module, 'read_common_manifest', fail)
    with pytest.raises(ValueError):
        prepare_fairseq(common, output, tmp_path / 'prepared', mel_config={'n_mels': 80})
    assert not (tmp_path / 'prepared').exists()


def test_label_checksum_detects_corruption(common, tmp_path, monkeypatch):
    mock_espeak(monkeypatch)
    output = tmp_path / 'phones'
    phonemes.phonemize_manifests(common, output,
        phonemizer=phonemes.EspeakNgPhonemizer(expected_version='1.52.0'),
        engine='espeak-ng', version='1.52.0')
    assert validate_phonemes(common, output)['wav_reads'] == 0
    (output / 'train.tsv').write_text('pair-train\ta a\n')
    with pytest.raises(ValueError, match='labels differ'):
        validate_phonemes(common, output)


def test_vocoder_verification_reuses_hash_and_rechecks_changes(common, monkeypatch):
    original = verification.sha256_file
    calls = []
    monkeypatch.setattr(verification, 'sha256_file', lambda path: calls.append(Path(path)) or original(path))
    verified = verification.verify_inputs(common, 'mel')
    assert list(verified['audio']) == ['pair-train']
    assert [p.name for p in calls if p.suffix == '.wav'] == ['train-en.wav']
    calls.clear()
    assert verification.verify_inputs(common, 'mel')['audio'] == verified['audio']
    assert not [p for p in calls if p.suffix == '.wav']
    monkeypatch.setenv('S2ST_PREP_RECHECK', '1')
    verification.verify_inputs(common, 'mel')
    assert [p.name for p in calls if p.suffix == '.wav'] == ['train-en.wav']
    monkeypatch.setenv('S2ST_PREP_RECHECK', '0')
    path = common / 'train-en.wav'
    path.write_bytes(path.read_bytes() + b'\0\0')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verification.verify_inputs(common, 'mel')


def test_vocoder_checks_actual_manifest_hash_field(common):
    path = common / 'train.jsonl'
    row = json.loads(path.read_text())
    row['en_audio_sha256'] = row['en_sha256']
    row['en_sha256'] = '0' * 64
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verification.verify_inputs(common, 'mel')
    del row['en_sha256']
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='expected en_sha256'):
        verification.verify_inputs(common, 'mel')


def test_unit_verification_is_train_only_and_receipt_is_authenticated(common, tmp_path):
    units = tmp_path / 'units'
    (units / 'train/original').mkdir(parents=True)
    (units / 'train/original/pair-train.units').write_text('1 2 99\n')
    (units / 'unit-lock.json').write_text('{"hubert_layer":6,"kmeans_clusters":100}')
    verified = verification.verify_inputs(common, 'unit', units)
    assert verified['sequences'] == {'pair-train': [1, 2, 99]}
    receipt = tmp_path / 'receipt.json'
    verification.save_receipt(receipt, verified)
    assert verification.load_receipt(receipt, common, 'unit', units) == verified
    document = json.loads(receipt.read_text())
    document['value']['sequences']['pair-train'] = [88]
    receipt.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='receipt mismatch'):
        verification.load_receipt(receipt, common, 'unit', units)


def test_receipt_rejects_changed_common(common, tmp_path):
    receipt = tmp_path / 'receipt.json'
    verification.save_receipt(receipt, verification.verify_inputs(common, 'mel'))
    with (common / 'train.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='receipt mismatch'):
        verification.load_receipt(receipt, common, 'mel')


def test_interrupted_vocoder_verification_resumes_completed_rows(common, monkeypatch):
    path = common / 'train.jsonl'
    first = json.loads(path.read_text())
    second = dict(first, pair_id='pair-second')
    path.write_text(json.dumps(first) + '\n' + json.dumps(second) + '\n')
    original = verification.sha256_file
    calls = []
    def interrupted(path):
        if Path(path).suffix == '.wav':
            calls.append(path)
            if len(calls) == 2:
                raise KeyboardInterrupt()
        return original(path)
    monkeypatch.setattr(verification, 'sha256_file', interrupted)
    with pytest.raises(KeyboardInterrupt):
        verification.verify_inputs(common, 'mel')
    calls.clear()
    monkeypatch.setattr(verification, 'sha256_file', lambda p: calls.append(p) or original(p))
    assert len(verification.verify_inputs(common, 'mel')['audio']) == 2
    assert len([p for p in calls if Path(p).suffix == '.wav']) == 1


def test_preflight_command_passes_private_receipt_only_to_vocoder(tmp_path):
    command = ['python', '-m', 'direct_s2st.vocoders.train', '--kind', 'mel']
    calls = []
    def runner(args, **kwargs):
        calls.append(args)
        Path(args[-1]).write_text('verified fixture')
    verified = verification.preflight_command(command, tmp_path, runner=runner)
    assert calls[0][-3:] == ['--verify-only', '--verification-result', str(tmp_path / 'vocoder-inputs.json')]
    assert verified[-2:] == ['--verified-inputs', str(tmp_path / 'vocoder-inputs.json')]
    assert verification.preflight_command(['fixture'], tmp_path, runner=runner) == ['fixture']
    assert len(calls) == 1


def test_checkpoint_metadata_keys_run_in_workers(tmp_path, monkeypatch):
    monkeypatch.setenv('S2ST_PREP_WORKERS', '2')
    monkeypatch.setenv('S2ST_PREP_ADAPTIVE', '0')
    owner = threading.get_ident()
    seen = []
    def key(n):
        seen.append(threading.get_ident())
        return str(n)
    with Checkpoints(tmp_path, {'fixture': 1}) as cache:
        assert list(checkpoint_map(lambda n: n, range(5), cache, key, 'fixture')) == list(range(5))
    assert len(seen) == 5 and owner not in seen


def test_hubert_caller_can_supply_full_32_batch_and_reuse(common, tmp_path, monkeypatch):
    from direct_s2st.s2ut.extract_units import _recoverable_units
    monkeypatch.setenv('S2ST_HUBERT_BATCH_MAX', '32')
    batches = []
    extractor = SimpleNamespace(extract_many=lambda paths: batches.append(len(paths)) or [[1]] * len(paths))
    rows = [dict(pair_id=str(i), en_audio=str(common / 'train-en.wav')) for i in range(40)]
    for resume in (False, True):
        with Checkpoints(tmp_path / 'units-cache', {'fixture': 1}, resume=resume) as cache:
            assert len(list(_recoverable_units(rows, extractor, cache, 100))) == 40
    assert batches == [32, 8]
    monkeypatch.setenv('S2ST_HUBERT_BATCH_MAX', '33')
    with pytest.raises(ValueError, match='between 1 and 32'):
        list(_recoverable_units(rows, extractor, cache, 100))


def test_explicit_relabeling_retains_old_dictionary_and_mel(common, tmp_path):
    from test_preparation_recovery import module
    recovery = module('resume_preparation')
    phones = tmp_path / 'translatotron2/phonemes'
    prepared = tmp_path / 'translatotron2/fairseq'
    kwargs = dict(engine='espeak-ng', version='1.52.0', fixed_vocabulary=['a'], resume=True)
    phonemes.phonemize_manifests(common, phones, phonemizer=lambda _: 'a', **kwargs)
    prepare_fairseq(common, phones, prepared, mel_config={'n_mels': 80}, feature_extractor=_mel_fixture, resume=True)
    old_zip = (prepared / 'logmelspec80.zip').read_bytes()
    old_dictionary = (prepared / 'target_phoneme/dict.txt').read_bytes()
    calls = []
    def run(*command):
        calls.append(command)
        phonemes.phonemize_manifests(common, phones, phonemizer=lambda _: 'b',
            engine='espeak-ng', version='1.52.0', fixed_vocabulary=['b'], resume=True)
    backup = recovery.regenerate_phonemes(tmp_path, ['fixture'], run)
    assert (backup / 'phonemes/train.tsv').read_text() == 'pair-train\ta\n'
    assert (backup / 'target_phoneme/dict.txt').read_bytes() == old_dictionary
    assert (backup / 'data-lock.json').is_file()
    assert (prepared / 'logmelspec80.zip').read_bytes() == old_zip
    assert recovery.regenerate_phonemes(tmp_path, ['fixture'], run) == backup
    assert len(calls) == 1
    def unused(*args, **kwargs):
        pytest.fail('retained Mel features should be reused')
    prepare_fairseq(common, phones, prepared, mel_config={'n_mels': 80}, feature_extractor=unused, resume=True)
    assert (prepared / 'logmelspec80.zip').read_bytes() == old_zip
    assert 'b 1\n' in (prepared / 'target_phoneme/dict.txt').read_text(encoding='utf-8')


def test_explicit_relabeling_can_resume_after_interruption(common, tmp_path):
    from test_preparation_recovery import module
    recovery = module('resume_preparation')
    phones = tmp_path / 'translatotron2/phonemes'
    phonemes.phonemize_manifests(common, phones, phonemizer=lambda _: 'a',
        engine='espeak-ng', version='1.52.0', fixed_vocabulary=['a'], resume=True)
    def interrupted(*command):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        recovery.regenerate_phonemes(tmp_path, ['fixture'], interrupted)
    old = next((tmp_path / 'translatotron2/phoneme-migrations').glob('*/phonemes/train.tsv'))
    before = old.read_bytes()
    def complete(*command):
        phonemes.phonemize_manifests(common, phones, phonemizer=lambda _: 'b',
            engine='espeak-ng', version='1.52.0', fixed_vocabulary=['b'], resume=True)
    backup = recovery.regenerate_phonemes(tmp_path, ['fixture'], complete)
    assert (backup / 'phonemes/train.tsv').read_bytes() == before
    assert (phones / 'train.tsv').read_text() == 'pair-train\tb\n'
