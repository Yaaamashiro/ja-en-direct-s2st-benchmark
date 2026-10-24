"""Official, fixed-revision evaluation data. No datasets scripts or audio decoder."""
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import unicodedata

from .corpus_diagnostic import checksum
from ..journal import digest
from ..progress import operation, track

FLEURS = 'google/fleurs'
FLEURS_REVISION = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'
FLORES = 'openlanguagedata/flores_plus'
FLORES_REVISION = 'e707e62e762a9e2faca3933aa315f4665cc3cd8f'
MATCHING = 'unique_nfkc_unicode_whitespace_exact_v1'


def sha_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def normalized(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('dataset has empty/missing canonical text')
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def join(fleurs, japanese, english, limit=300):
    """Match text only; FLORES English is joined by (split,id), never row order."""
    if type(limit) is not int or limit < 0:
        raise ValueError('FLORES_FLEURS_LIMIT must be 0 (all) or positive')
    jp, en = defaultdict(list), defaultdict(list)
    for row in japanese:
        if row['split'] not in ('dev', 'devtest'):
            raise ValueError('FLORES training/other split is forbidden')
        jp[normalized(row['text'])].append(row)
    for row in english:
        if row['split'] not in ('dev', 'devtest'):
            raise ValueError('FLORES training/other split is forbidden')
        en[(row['split'], str(row['id']))].append(row)
    recordings = defaultdict(list)
    for row in fleurs:
        if row['split'] not in ('validation', 'test'):
            raise ValueError('FLEURS training/other split is forbidden')
        recordings[normalized(row['raw_transcription'])].append(row)
    counts = dict(matched=0, unmatched=0, ambiguous=0,
                  duplicate_recordings=sum(len(group)-1 for group in recordings.values()),
                  input_recordings=sum(map(len, recordings.values())))
    selected, rejected = [], []
    for key, group in recordings.items():
        candidates = jp.get(key, [])
        english_rows = en.get((candidates[0]['split'], str(candidates[0]['id'])), []) if len(candidates)==1 else []
        reason = 'unmatched' if not candidates or (len(candidates)==1 and not english_rows) else 'ambiguous'
        if len(candidates)!=1 or len(english_rows)!=1:
            counts[reason] += 1
            rejected.append(dict(text_sha256=sha_text(key), reason=reason, recordings=len(group)))
            continue
        ja, reference = candidates[0], english_rows[0]
        natural = min(group, key=lambda r:sha_text(json.dumps(
            [r['split'],str(r['id']),r['path'],r['raw_transcription']], ensure_ascii=False, separators=(',',':'))))
        identifier = 'flores:'+ja['split']+':'+str(ja['id'])
        # Use the safely normalized common text in ALL THREE branches.
        selected.append(dict(sample_id=identifier, ja_text=key, en_reference=reference['text'],
            ja_text_sha256=sha_text(key), en_text_sha256=sha_text(reference['text']),
            flores_split=ja['split'], flores_id=str(ja['id']), fleurs_split=natural['split'],
            fleurs_id=str(natural['id']), gender=natural.get('gender'), fleurs_path=natural['path'],
            raw_transcription=natural['raw_transcription'], locator=natural['locator']))
    counts['matched'] = len(selected)
    if len(selected)<300:
        raise ValueError('Fewer than 300 unique exact FLORES/FLEURS sentences; no fuzzy fallback. '+json.dumps(counts))
    selected.sort(key=lambda r:sha_text(r['sample_id']))
    selected = selected[:limit] if limit else selected
    return selected, dict(matching_algorithm=MATCHING,
                         recording_selection='min_sha256_utf8_compact_json_array(split,id,path,raw_transcription)',
                         sample_selection='sha256_utf8_canonical_sample_id',counts=counts, selected=len(selected),
                         rejected=sorted(rejected, key=lambda r:r['text_sha256']))


class OfficialData:
    """Range-read only text columns first; selected audio rows are fetched later.

    FLEURS main contains official parquet-data, so no converted/mirror revision
    or arbitrary remote Python script is needed. All larger source data stays
    in local Hub/range caches; only selected validated WAVs go to diagnostics.
    """
    def __init__(self, cache):
        self.cache = Path(cache)
        self.files, self.audio_tables = {}, {}

    def parquet(self, split):
        if split not in ('validation','test'):
            raise ValueError('only FLEURS validation/test may be loaded')
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
        path = f'parquet-data/ja_jp/{split}-00000-of-00001.parquet'
        remote = f'datasets/{FLEURS}@{FLEURS_REVISION}/{path}'
        stream = HfFileSystem().open(remote, 'rb', block_size=4*1024**2)
        return pq.ParquetFile(stream), stream, path

    @operation('8C: official gated FLORES+ texts and FLEURS metadata only')
    def load(self, limit=300):
        from huggingface_hub import hf_hub_download
        langs = {}
        for lang in ('jpn_Jpan','eng_Latn'):
            langs[lang] = []
            for split in ('dev','devtest'):
                filename = f'{split}/{lang}.jsonl'
                try:
                    path = Path(hf_hub_download(repo_id=FLORES, repo_type='dataset', revision=FLORES_REVISION,
                                               filename=filename, cache_dir=str(self.cache/'datasets')))
                except Exception as error:
                    raise RuntimeError('FLORES+ could not be loaded from the official Hugging Face repository. '
                        'Accept the dataset terms and authenticate with Hugging Face, then rerun Cell 8C. '
                        'No third-party mirror fallback. '+type(error).__name__) from error
                self.files[filename] = dict(sha256=checksum(path), bytes=path.stat().st_size)
                for line in path.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        row = json.loads(line)
                        langs[lang].append(dict(split=split, id=str(row['id']), text=row['text']))
        recordings = []
        for split in ('validation','test'):
            parquet, stream, path = self.parquet(split)
            try:
                required = {'id','path','raw_transcription','gender','audio'}
                if not required <= set(parquet.schema_arrow.names):
                    raise RuntimeError('official FLEURS schema changed at pinned revision: '+str(parquet.schema_arrow))
                texts = []
                for rg in track(range(parquet.num_row_groups),'8C: FLEURS text columns '+split):
                    group = parquet.read_row_group(rg, columns=['id','path','raw_transcription','gender']).to_pylist()
                    for offset, row in enumerate(group):
                        row.update(split=split,locator=dict(split=split,row_group=rg,offset=offset))
                        recordings.append(row)
                    texts.extend(group)
                self.files[path] = dict(metadata_sha256=digest(texts), rows=parquet.metadata.num_rows,
                                        row_groups=parquet.num_row_groups, selected_audio_only=True)
            finally:
                stream.close()
        selected, audit = join(recordings, langs['jpn_Jpan'], langs['eng_Latn'], limit)
        audit.update(datasets=[dict(repository=FLEURS, revision=FLEURS_REVISION, config='ja_jp', splits=['validation','test']),
                               dict(repository=FLORES, revision=FLORES_REVISION, configs=['jpn_Jpan','eng_Latn'],splits=['dev','devtest'])],
                     download_metadata=self.files, selected_samples=selected, limit=limit)
        return selected, audit

    def audio(self, sample, destination):
        """At most one row group's audio in RAM; never persist the full shard."""
        import numpy as np
        import soundfile as sf
        loc = sample['locator']
        key = (loc['split'],loc['row_group'])
        if key not in self.audio_tables:
            self.audio_tables.clear()
            parquet, stream, _ = self.parquet(loc['split'])
            try:
                self.audio_tables[key] = parquet.read_row_group(loc['row_group'],
                    columns=['audio','id','path','raw_transcription']).to_pylist()
            finally:
                stream.close()
        row = self.audio_tables[key][loc['offset']]
        if (str(row['id'])!=sample['fleurs_id'] or row['path']!=sample['fleurs_path']
                or row['raw_transcription']!=sample['raw_transcription']):
            raise RuntimeError('selected FLEURS audio row no longer matches verified metadata')
        encoded = row['audio']['bytes']
        if not encoded:
            raise RuntimeError('official selected FLEURS row has no embedded audio bytes')
        audio, rate = sf.read(io.BytesIO(encoded), dtype='float32', always_2d=False)
        if audio.ndim!=1 or not np.isfinite(audio).all() or not np.any(audio) or len(audio)==0:
            raise ValueError('invalid/silent/non-mono natural FLEURS audio')
        info=sf.info(io.BytesIO(encoded))
        if info.format=='WAV':
            # Preserve the official natural WAV bytes; do not requantize them.
            Path(destination).write_bytes(encoded)
        else:
            # Lossless float32 materialization for an unexpected supported codec.
            sf.write(destination,audio,rate,subtype='FLOAT')
        return dict(original_audio_sha256=hashlib.sha256(encoded).hexdigest(),
                    original_format=info.format,original_subtype=info.subtype)
