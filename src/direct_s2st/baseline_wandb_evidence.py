"""Read evidence from published snapshots without loading a model."""
import hashlib
import json
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

from .hashing import sha256_file

MY_DRIVE = '/content/drive/MyDrive/'
FINGERPRINT = re.compile('[0-9a-f]{64}')
EXPECTED_SIGNATURE = 'nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1'
NORMALIZATION = 'benchmark.normalize_english (lowercase/NFKC/punctuation except apostrophes)'


def publication_key(path):
    return _relative_key(str(path).removeprefix(MY_DRIVE))


def _relative_key(key):
    if '\\' in key or any(part in ('', '.', '..') for part in key.split('/')):
        raise ValueError('expected the original nonempty MyDrive-relative publication key')
    return key


def baseline_wandb_run_id(key, fingerprint):
    key = _relative_key(key)
    if not FINGERPRINT.fullmatch(fingerprint):
        raise ValueError('expected the saved lowercase SHA256 training fingerprint')
    payload = json.dumps(['baseline', key, fingerprint], ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:24]


class PublishedFiles:
    def __init__(self, marker, receipt):
        self.directory = Path(marker).parent
        self.receipt = receipt
        self.archive = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.archive is not None:
            self.archive.close()

    def read(self, name):
        expected = self.receipt['files'].get(name)
        if not expected:
            raise FileNotFoundError(name)
        if self.receipt.get('storage') == 'zip-v1':
            if self.archive is None:
                self.archive = ZipFile(self.directory/'snapshot.zip')
            payload = self.archive.read(name)
        else:
            payload = (self.directory/'files'/name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'published file hash mismatch: {name}')
        return payload


def registered_fingerprint(registration, receipt):
    """Use the registration verifier's proof and tie it to this exact snapshot."""
    path = Path(registration)
    published = json.loads(path.read_bytes())
    if published.get('storage') == 'zip-v1':
        if sha256_file(path.parent/'snapshot.zip') != published['zip_sha256']:
            raise ValueError('registration archive hash mismatch')
    with PublishedFiles(path, published) as files:
        record = json.loads(files.read('baseline.json'))
    raw = {key: value for key, value in record.items() if key != 'record_hash'}
    encoded = json.dumps(raw, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    expected = hashlib.sha256(encoded).hexdigest()
    fingerprint = record.get('baseline_fingerprint', '')
    work = receipt['identity']['work'].rstrip('/')
    checkpoint = record.get('checkpoint', '').removeprefix(work + '/')
    if (record.get('format') != 'native-fairseq-s2ut-comparison-v1'
            or record.get('method') != 'baseline' or expected != record.get('record_hash')
            or not FINGERPRINT.fullmatch(fingerprint)
            or record.get('metadata', {}).get('training_fingerprint') != fingerprint
            or checkpoint not in ('checkpoints/checkpoint_best.pt', 'checkpoints/checkpoint_last.pt')
            or record.get('checkpoint') != work + '/' + checkpoint
            or not 0 < record['checkpoint_updates'] <= receipt['updates']):
        raise ValueError('invalid or unrelated Baseline registration')
    for name, value in (
        (checkpoint, record.get('checkpoint_hash')),
        ('research-metadata.json', record['provenance'].get('metadata_hash')),
        ('s2ut-recipe-lock.json', record['provenance'].get('recipe_lock_hash')),
    ):
        if not value or receipt['files'].get(name) != value:
            raise ValueError(f'Baseline registration does not bind snapshot file: {name}')
    return fingerprint


def snapshot_fingerprint(receipt):
    evidence = receipt.get('s2ut_lineage')
    if evidence is None:
        return None
    fingerprint = evidence.get('training_fingerprint', '')
    if (not FINGERPRINT.fullmatch(fingerprint)
            or evidence.get('updates') != receipt['updates']
            or any(not FINGERPRINT.fullmatch(evidence.get(name, ''))
                   or evidence[name] != receipt['files'].get(name) for name in (
                       'checkpoints/checkpoint_last.pt', 'research-metadata.json',
                       's2ut-recipe-lock.json'))):
        raise ValueError('invalid published S2UT checkpoint lineage')
    return fingerprint


def _decoder_failure(mode, error):
    # Exact messages from the pinned decoder and PR #8's generation-only policy.
    if not isinstance(error, str):
        return False
    if error.startswith(('ValueError: ', 'GenerationFailure: ')):
        error = error.split(': ', 1)[1]
    if mode == 'aux':
        return error in ('nonfinite auxiliary translation logits',
                         'auxiliary decoding reached max_aux_tokens without EOS')
    return (error in ('Unit decoding did not terminate with EOS', 'unit sequence must not be empty',
                      *(f'invalid literal for int() with base 10: {t!r}' for t in ('<unk>', '<pad>', '<s>', '</s>')))
            or error.startswith('unit IDs are out of range [0, 100); received '))


def verify_bleu(item, mode, files):
    """Return comparable, legacy_unclassified, or unmatched for one metric."""
    scoring = item.get(mode+'_scoring') or {}
    count, failures = item.get(mode+'_samples'), item.get(mode+'_failures')
    identity = item.get('identity') or {}
    policy = identity.get('policy') or {}
    ids = identity.get('ids') or []
    references = identity.get('references') or {}
    expected_count = 500 if mode == 'aux' else 100
    if (type(failures) is not int or failures < 0 or type(count) is not int
            or count <= 0 or failures > count):
        return 'legacy_unclassified'
    if (count != expected_count or scoring.get('signature') != EXPECTED_SIGNATURE
            or scoring.get('normalization') != NORMALIZATION
            or any(scoring.get(key) != item.get(mode+'_'+key) for key in ('bleu', 'samples', 'failures'))
            or item.get('status') != ('completed_with_failures'
                                      if item.get('aux_failures') or item.get('asr_failures') else 'completed')
            or identity.get('update') != item['update']
            or not FINGERPRINT.fullmatch(identity.get('weights', ''))
            or len(ids) != 500 or len(set(ids)) != 500 or set(references) != set(ids)
            or not all(isinstance(ref, str) for ref in references.values())
            or ids != sorted(ids, key=lambda key: hashlib.sha256(f'1:{key}'.encode()).hexdigest())
            or identity.get('subset') != 'fixed_hash_sample_of_dev'
            or identity.get('generation') != 'source_audio_only'
            or any(policy.get(key) != value for key, value in dict(
                seed=1, aux_samples=500, asr_samples=100, max_aux_tokens=256, beam=10,
                max_len_a=1, max_len_b=200, asr_model='openai/whisper-large-v3-turbo',
                asr_revision='60be3615a4d667e1258e8ad29130467587c489aa').items())
            or (mode == 'aux' and identity.get('auxiliary_decoding') != 'greedy encoder-attached target-letter decoder')
            or (mode == 'asr' and (identity.get('unit_vocoder') != 'official LJSpeech (not Fisher-trained)'
                                  or identity.get('asr') != 'benchmark Whisper (not original-paper wav2vec2)'))):
        return 'legacy_unclassified' if failures else 'unmatched'
    if failures == 0:
        return 'comparable'
    if files is None:
        return 'legacy_unclassified'
    prefix = f'training-bleu/{item["update"]:09d}/{mode}'
    try:
        lock = json.loads(files.read(prefix+'.lock.json'))
        rows = [json.loads(line) for line in files.read(prefix+'.jsonl').splitlines()]
    except (OSError, ValueError, KeyError, BadZipFile):
        return 'legacy_unclassified'
    if (lock != identity or len(rows) != count
            or not all(isinstance(row, dict) and isinstance(row.get('pair_id'), str) for row in rows)
            or {row['pair_id'] for row in rows} != set(ids[:count])):
        return 'legacy_unclassified'
    failed = [row for row in rows if row.get('status') == 'failed']
    if (len(failed) != failures or any(row.get('status') not in ('failed', 'success') for row in rows)
            or any(row.get('hypothesis') != '' or not _decoder_failure(mode, row.get('error')) for row in failed)
            or any(not isinstance(row.get('hypothesis'), str) or row.get('error') is not None
                   for row in rows if row.get('status') == 'success')):
        return 'legacy_unclassified'
    return 'comparable'
