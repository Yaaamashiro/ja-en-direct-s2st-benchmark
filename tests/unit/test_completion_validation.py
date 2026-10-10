import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from direct_s2st.cli import main
from direct_s2st.config import RootPaths
from direct_s2st.manifests.paths import resolve_audio_path
from direct_s2st.manifests.reader import read_common_manifest
from direct_s2st.s2ut.multitask import prepare_labels, read_tsv, tokenize, validate_prepared
from direct_s2st.s2ut.fairseq_infer import parse_generated
from direct_s2st.s2ut.prepare_fairseq import prepare_fairseq
from direct_s2st.s2ut.extract_units import extract_units
from direct_s2st.translatotron2.status import require_implemented
from direct_s2st.vocoders.inference import MEL_KEYS, validate_config, write_waveform
from test_s2ut_units import _common


@pytest.mark.parametrize("old", ["/content/drive/MyDrive/dataset/production/audio/16k/en/a.wav",
                                  "D:\\old\\production\\audio\\16k\\en\\a.wav"])
def test_rebase_foreign_absolute_paths(tmp_path, old, monkeypatch):
    # The old path is intentionally absent. Do not probe the host's real D:
    # drive (it may be mounted, unreadable, or contain unrelated user files).
    is_file = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda path, **kw: False if path == Path(old) else is_file(path, **kw))
    audio = tmp_path / "production/audio/16k/en/a.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"path fixture")
    assert resolve_audio_path(old, tmp_path) == audio.resolve()


def test_rebase_rejects_ambiguous_and_basename_matches(tmp_path):
    for suffix in ("audio/16k/en/a.wav", "production/audio/16k/en/a.wav"):
        audio = tmp_path / suffix
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b"path fixture")
    with pytest.raises(ValueError, match="2 candidates"):
        resolve_audio_path("/old/audio/16k/en/a.wav", tmp_path)
    with pytest.raises(ValueError, match="0 candidates"):
        resolve_audio_path("/old/a.wav", tmp_path)


def test_relative_paths_do_not_fallback_or_escape(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_audio_path("missing.wav", tmp_path)
    with pytest.raises(ValueError, match="traversal"):
        resolve_audio_path("../a.wav", tmp_path)


def test_corpus_cannot_be_an_output_root(tmp_path):
    with pytest.raises(ValueError, match="CORPUS_ROOT"):
        RootPaths(tmp_path, tmp_path / "derived", tmp_path / "runs", tmp_path / "cache").validate_output_roots()


def test_existing_absolute_file_is_preserved(tmp_path):
    audio = tmp_path / "existing.wav"
    audio.write_bytes(b"path fixture")
    assert resolve_audio_path(str(audio.resolve()), tmp_path / "other") == audio.resolve()


def test_portable_common_manifest_uses_new_corpus_root_without_rewriting_input(tmp_path):
    root = tmp_path / "new-corpus"
    root.mkdir()
    (root / "ja.wav").write_bytes(b"path fixture")
    (root / "en.wav").write_bytes(b"path fixture")
    path = tmp_path / "common.jsonl"
    path.write_text(json.dumps({"ja_audio": "/old/ja.wav", "en_audio": "/old/en.wav",
                                "ja_audio_corpus_relative": "ja.wav", "en_audio_corpus_relative": "en.wav"}))
    before = path.read_bytes()
    row = next(read_common_manifest(path, corpus_root=root))
    assert row["ja_audio"] == str((root / "ja.wav").resolve())
    assert path.read_bytes() == before


def test_tokenization_is_character_level_and_does_not_silently_normalize():
    assert tokenize(" 日本語  A!\n") == ["日", "本", "語", "<space>", "A", "!"]
    assert tokenize("é") != tokenize("e\u0301")
    assert tokenize("Aa") == ["A", "a"]
    for bad in ("", "  ", "x\x00", "x\u3000y"):
        with pytest.raises(ValueError):
            tokenize(bad)


def prepared(tmp_path):
    common, units, target = (tmp_path / name for name in ("common", "units", "prepared"))
    _common(common)
    extract_units(common, units, extractor=lambda _: [1, 1, 2, 3], split=None,
                  clusters=100, hubert_model="fixture", hubert_revision="a" * 40,
                  hubert_layer=6, kmeans_sha256="b" * 64)
    prepare_fairseq(common, units, target)
    return target


def test_all_multitask_ids_align_and_reference_weights_are_explicit(tmp_path):
    target = prepared(tmp_path)
    assert validate_prepared(target)["splits"] == {"train": 1, "dev": 1, "test": 1}
    cfg = yaml.safe_load((target / "config_multitask.yaml").read_text())
    assert cfg["source_letter"]["encoder_layer"] == 6
    assert cfg["target_letter"]["encoder_layer"] == 8
    assert cfg["decoder_target_ctc"]["decoder_layer"] == 3
    assert [cfg[name]["loss_weight"] for name in ("source_letter", "target_letter", "decoder_target_ctc")] == [8.0, 8.0, 1.6]


def test_literal_tsv_quotes_never_swallow_rows_or_change_labels(tmp_path):
    path = tmp_path/'labels.tsv'
    path.write_text('id\ttgt_text\nfirst\t" 日\nsecond\t本 " 語\nthird\t" "\n', encoding='utf-8')
    rows = read_tsv(path, ('id', 'tgt_text'))
    assert list(rows) == ['first', 'second', 'third']
    assert rows['first']['tgt_text'] == '" 日'
    assert rows['second']['tgt_text'] == '本 " 語'
    assert rows['third']['tgt_text'] == '" "'


def test_real_prepare_preserves_quoted_transcripts_and_reuses_outputs(tmp_path):
    common, units, target = tmp_path/'common', tmp_path/'units', tmp_path/'prepared'
    _common(common)
    for split in ('train', 'dev', 'test'):
        row = json.loads((common/f'{split}.jsonl').read_text(encoding='utf-8'))
        row.update(ja_text='"日本語', ja_tts_text='"日本語',
                   en_text='"English', en_tts_text='"English')
        second = dict(row, pair_id=row['pair_id']+'-second', ja_text='日本語"', ja_tts_text='日本語"',
                      en_text='English"', en_tts_text='English"')
        (common/f'{split}.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(second)+'\n', encoding='utf-8')
    extract_units(common, units, extractor=lambda _: [1, 2, 3], split=None, clusters=100,
                  hubert_model='fixture', hubert_revision='a'*40, hubert_layer=6, kmeans_sha256='b'*64)
    prepare_fairseq(common, units, target)
    before = {p: p.read_bytes() for folder in (target, units) for p in folder.rglob('*') if p.is_file()}
    assert validate_prepared(target)['splits'] == dict(train=2, dev=2, test=2)
    prepare_fairseq(common, units, target, resume=True)
    assert all(p.read_bytes() == contents for p, contents in before.items())


def test_alignment_failure_in_test_is_reported_before_any_audio_probe(tmp_path, monkeypatch):
    target = prepared(tmp_path)
    (target/'source_letter/test.tsv').write_text('id\ttgt_text\nwrong-id\t日\n', encoding='utf-8')
    original = Path.is_file
    def is_file(path):
        if path.suffix == '.wav':
            pytest.fail('must detect alignment before any audio check')
        return original(path)
    monkeypatch.setattr(Path, 'is_file', is_file)
    with pytest.raises(ValueError, match=r'main=1 auxiliary=1 missing=1.*extra=1'):
        validate_prepared(target)


@pytest.mark.parametrize("task", ["source_letter", "target_letter", "decoder_target_ctc"])
def test_missing_auxiliary_sample_rejected_for_each_task(tmp_path, task):
    target = prepared(tmp_path)
    (target / task / "dev.tsv").write_text("id\ttgt_text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="alignment"):
        validate_prepared(target)


@pytest.mark.parametrize("tokens", ["", "UNKNOWN"])
def test_empty_and_out_of_dictionary_labels_rejected(tmp_path, tokens):
    target = prepared(tmp_path)
    (target / "target_letter/test.tsv").write_text(f"id\ttgt_text\npair-test\t{tokens}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dictionary coverage"):
        validate_prepared(target)


def test_zero_weight_auxiliary_cannot_be_silently_disabled(tmp_path):
    target = prepared(tmp_path)
    path = target / "config_multitask.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["source_letter"]["loss_weight"] = 0
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="loss_weight"):
        validate_prepared(target)


def test_heldout_characters_registered_without_training_ctc_on_heldout(tmp_path):
    rows = {split: [{"pair_id": split, "ja_text": "日", "en_text": "a"}] for split in ("train", "dev", "test")}
    rows["test"][0]["ja_text"] = "未"
    rows['dev'][0]['ja_text'] = '汎'
    rows['test'][0]['en_text'] = 'a Ω'
    target = tmp_path/'out'
    lock = prepare_labels(rows, target)
    assert (target/'source_letter/dict.txt').read_text(encoding='utf-8').splitlines() == ['日 1', '未 1', '汎 1']
    assert 'Ω 1' in (target/'target_letter/dict.txt').read_text(encoding='utf-8')
    assert lock['heldout_only_characters']['source_letter'] == ['未', '汎']
    assert lock['unknown_token_counts']['decoder_target_ctc']['test'] == 1
    assert '<unk>' in (target/'decoder_target_ctc/test.tsv').read_text(encoding='utf-8')
    assert lock['ctc_tokenizer']['vocabulary_source'] == 'train_only'
    assert lock['model_training_split'] == 'train'
    assert prepare_labels(rows, target, resume=True) == lock


def test_tts_content_difference_blocks_preparation(tmp_path):
    rows = {split: [{"pair_id": split, "ja_text": "日", "en_text": "a", "en_tts_text": "b"}] for split in ("train", "dev", "test")}
    with pytest.raises(ValueError, match="differs"):
        prepare_labels(rows, tmp_path / "out")


def test_fairseq_generation_alignment_and_incomplete_outputs(tmp_path):
    path = tmp_path / "generate.txt"
    path.write_text("D-1\t-0.5\t2 3\nD-0\t-0.5\t1 4\n")
    assert parse_generated(path, ["a", "b"]) == {"b": [2, 3], "a": [1, 4]}
    path.write_text("D-0\t-0.5\t1 4\n")
    with pytest.raises(ValueError, match="incomplete"):
        parse_generated(path, ["a", "b"])


def test_vocoder_rejects_missing_duration_predictor_and_wrong_sample_rate():
    config = {"num_embeddings": 100, "sampling_rate": 16000}
    with pytest.raises(ValueError, match="duration predictor"):
        validate_config("unit", config, sample_rate=16000)
    with pytest.raises(ValueError, match="sampling_rate"):
        validate_config("unit", config, sample_rate=22050)


def test_mel_compatibility_requires_every_parameter():
    mel = {**dict.fromkeys(MEL_KEYS, 1), "sample_rate": 16000, "n_mels": 80, "hop_length": 256}
    config = {"sampling_rate": 16000, "mel": dict(mel), "upsample_rates": [8, 8, 2, 2]}
    validate_config("mel", config, sample_rate=16000, mel=mel)
    for key in MEL_KEYS:
        changed = {**mel, key: 2}
        with pytest.raises(ValueError, match="mel specification"):
            validate_config("mel", config, sample_rate=16000, mel=changed)


def test_waveform_serialization_readback_and_silence_rejection(tmp_path):
    # Tests file serialization only, not neural vocoder inference or E2E success.
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)
    assert write_waveform(tmp_path / "tone.wav", tone, 16000) == 0.1
    for bad in (np.zeros(100), np.array([]), np.array([float("nan")])):
        with pytest.raises(ValueError, match="waveform"):
            write_waveform(tmp_path / "invalid.wav", bad, 16000)


def test_translatotron2_rejects_duration_free_substitute(tmp_path):
    from direct_s2st.training import train_system
    with pytest.raises(NotImplementedError, match='duration-free substitute'):
        train_system(tmp_path, tmp_path / 'run', tmp_path / 'data',
                     run_id='tt2-smoke', config={'architecture': 's2spect2_conformer'}, profile='smoke')
