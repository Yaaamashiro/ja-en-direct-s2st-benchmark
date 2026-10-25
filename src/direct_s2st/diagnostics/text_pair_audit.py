"""CPU-only, human-labelled audit of accepted experiment text pairs.

No audio paths are resolved. No inference/scoring packages are imported.
The queue and sampling audit are INTERNAL: do not inspect them before unblinding.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import threading

from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_text
from ..progress import operation, track

AUDIT_VERSION = 'blinded-accepted-text-pair-v1'
AUDIT_SEED = 20261025
POPULATIONS = ('train', 'test')
CORPORA = ('kftt', 'jesc')
LABELS = ('correct', 'acceptable', 'partial_mismatch', 'wrong_alignment', 'uncertain')
FLAGS = ('boundary_mismatch', 'missing_content', 'extra_content', 'named_entity_mismatch',
         'number_or_date_mismatch', 'negation_or_polarity_mismatch', 'malformed_japanese',
         'malformed_english', 'fragment', 'subtitle_artifact', 'domain_or_context_hard',
         'unnatural_translationese', 'other')
CONFIDENCES = ('high', 'medium', 'low')
TEXT_FIELDS = ('ja_text', 'en_text', 'ja_text_raw', 'en_text_raw', 'ja_tts_text', 'en_tts_text')
SAMPLING = 'sha256(version\\0seed\\0population\\0corpus\\0pair_id)_ascending'
BLIND_ORDER = 'sha256(version\\0blind\\0seed\\0pair_id)_ascending'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def priority(*parts):
    return hashlib.sha256('\0'.join(map(str, parts)).encode()).hexdigest()


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f'manifest row {number} must be an object')
                yield value


def corpus_name(row):
    value = row.get('corpus')
    return value.strip().lower() if isinstance(value, str) and value.strip().lower() in CORPORA else None


def text_hash(row):
    return digest({key: row.get(key) for key in TEXT_FIELDS})


class ProvenanceError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit  # Never shown with corpus/sample mapping by the annotation UI.


@operation('8D: read accepted experiment texts and validate provenance (no audio)')
def collect(experiment, *, corpus_root=None, source_manifest=None):
    experiment = Path(experiment).resolve()
    common = experiment/'data/common'
    lock_path = common/'dataset-lock.json'
    paths = [lock_path, *(common/(p+'.jsonl') for p in POPULATIONS)]
    inputs = {str(path): checksum(path) for path in paths}
    lock = read(lock_path)
    if lock.get('schema_version') != 1 or not isinstance(lock.get('source_manifest'), str) or not re.fullmatch(
            '[0-9a-f]{64}', lock.get('source_manifest_sha256', '')):
        raise ValueError('8D requires the saved accepted-manifest dataset lock, not raw corpus input')
    audit = dict(excluded=[], manifest_rows={}, eligible={}, metadata_source='explicit common-manifest corpus field',
                 accepted_manifest=dict(path=lock['source_manifest'], sha256=lock['source_manifest_sha256']))
    candidates, missing, seen = [], [], set()
    for population in POPULATIONS:
        count = 0
        for row in track(rows(common/(population+'.jsonl')), '8D: read experiment text rows', key=lambda _: '-'):
            count += 1
            key = row.get('pair_id')
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError('8D missing/duplicate pair_id; cannot prove a unique experiment population')
            seen.add(key)
            if row.get('split') != population:
                raise ValueError('8D manifest split differs from its experiment population')
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('ja_text', 'en_text')):
                audit['excluded'].append(dict(pair_id=key, population=population, reason='missing_canonical_text'))
                continue
            if any(row.get(k) is not None and not isinstance(row[k], str) for k in TEXT_FIELDS):
                raise ValueError('8D optional raw/TTS text must be a string when present')
            # The common manifest is the actual model/reference text. If TTS text
            # differs, retain it in a blind text-only disclosure and the export.
            item = {k: row.get(k) for k in TEXT_FIELDS}
            item.update(pair_id=key, population=population,
                        original_split=row.get('original_split', row['split']), corpus=corpus_name(row),
                        source_id=row.get('source_id'))
            if item['corpus']:
                candidates.append(item)
            else:
                missing.append(item)
        audit['manifest_rows'][population] = count
        expected = lock.get('splits', {}).get(population, {}).get('pairs')
        if type(expected) is not int or expected != count:
            raise ValueError('8D experiment manifest count differs from its saved dataset lock')
    if missing:
        source = Path(source_manifest or lock['source_manifest'])
        if source.is_file():
            if corpus_root is None or not source.resolve().is_relative_to(Path(corpus_root).resolve()):
                raise ValueError('8D fallback must be the locked accepted manifest inside CORPUS_ROOT')
            if checksum(source) != lock['source_manifest_sha256']:
                raise ValueError('8D accepted-manifest SHA mismatch; no prefix-based corpus inference')
            inputs[str(source.resolve())] = checksum(source)
            required = {r['pair_id']: r for r in missing}
            joined, duplicates = {}, set()
            for row in track(rows(source), '8D: strict saved-key metadata join', key=lambda _: '-'):
                key = row.get('pair_id')
                if key not in required:
                    continue
                if key in joined:
                    duplicates.add(key)
                joined[key] = row
            for item in missing:
                raw = joined.get(item['pair_id'], {})
                valid = (item['pair_id'] not in duplicates and corpus_name(raw)
                         and raw.get('split') == item['population']
                         and all(isinstance(raw.get(k), str) and raw[k].strip() == item[k] for k in ('ja_text', 'en_text'))
                         and (not item['source_id'] or raw.get('source_id') == item['source_id']))
                if valid:
                    item['corpus'] = corpus_name(raw)
                    candidates.append(item)
                else:
                    audit['excluded'].append(dict(pair_id=item['pair_id'], population=item['population'],
                        reason='missing_or_nonunique_or_text_mismatched_accepted_provenance'))
            audit['metadata_source'] += '; SHA-verified accepted-manifest pair_id + split + text join'
        else:
            audit['excluded'].extend(dict(pair_id=r['pair_id'], population=r['population'],
                reason='missing_explicit_corpus_and_locked_accepted_manifest_unavailable') for r in missing)
    for population in POPULATIONS:
        audit['eligible'][population] = {corpus: sum(r['population'] == population and r['corpus'] == corpus
                                                   for r in candidates) for corpus in CORPORA}
        if not sum(audit['eligible'][population].values()):
            raise ProvenanceError('8D cannot prove corpus provenance for a required population; audit stopped', audit)
    # Prepared model data, when present, must belong to the same accepted dataset.
    for system in ('s2ut', 'translatotron2'):
        prepared = experiment/'data'/system/'fairseq/data-lock.json'
        if prepared.is_file():
            if read(prepared).get('common_dataset_lock_sha256') != inputs[str(lock_path)]:
                raise ValueError('8D prepared data belongs to a different common dataset lock')
            inputs[str(prepared)] = checksum(prepared)
    for path, sha in inputs.items():
        if checksum(path) != sha:
            raise ValueError('8D input changed while reading; restart without modifying the experiment')
    return candidates, inputs, audit


def select(candidates, *, per_stratum=100, seed=AUDIT_SEED):
    if type(per_stratum) is not int or per_stratum < 1 or type(seed) is not int:
        raise ValueError('8D requires a positive integer per-stratum limit and an integer seed')
    selected, strata = [], []
    for population in POPULATIONS:
        for corpus in CORPORA:
            pool = [r for r in candidates if r['population'] == population and r['corpus'] == corpus]
            pool.sort(key=lambda r: (priority(AUDIT_VERSION, seed, population, corpus, r['pair_id']), r['pair_id']))
            chosen = pool[:per_stratum]
            selected.extend(chosen)
            strata.append(dict(population=population, corpus=corpus, requested=per_stratum,
                               available=len(pool), selected=len(chosen)))
    selected.sort(key=lambda r: (priority(AUDIT_VERSION, 'blind', seed, r['pair_id']), r['pair_id']))
    return selected, strata


def owned_root(experiment, corpus_root=None):
    root = Path(experiment).absolute()/'diagnostics/text-pair-audit'
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('8D output path may not traverse symbolic links')
    if corpus_root and root.resolve().is_relative_to(Path(corpus_root).resolve()):
        raise ValueError('8D never writes inside CORPUS_ROOT')
    return root


@operation('8D: deterministic blinded queue / resume')
def prepare(experiment, *, corpus_root=None, source_manifest=None, per_stratum=100, seed=AUDIT_SEED):
    if type(per_stratum) is not int or per_stratum < 1 or type(seed) is not int:
        raise ValueError('8D requires a positive integer per-stratum limit and an integer seed')
    base = owned_root(experiment, corpus_root)
    try:
        candidates, inputs, audit = collect(experiment, corpus_root=corpus_root, source_manifest=source_manifest)
    except ProvenanceError as error:
        # A diagnostic-only failure receipt; never discard unproven rows silently.
        failure = dict(audit_version=AUDIT_VERSION, experiment_root=str(Path(experiment).resolve()),
                       error=str(error), provenance_audit=error.audit)
        atomic_write_json(base/'provenance-failures'/(digest(failure)+'.json'), failure, resume=True)
        raise
    selected, strata = select(candidates, per_stratum=per_stratum, seed=seed)
    identity = dict(audit_version=AUDIT_VERSION, experiment_root=str(Path(experiment).resolve()),
        inputs_sha256=inputs, sampling_seed=seed, per_stratum_limit=per_stratum,
        sampling_algorithm=SAMPLING, blind_order_algorithm=BLIND_ORDER,
        selected_pairs=[dict(pair_id=r['pair_id'], population=r['population'], corpus=r['corpus'],
                             original_split=r['original_split'], text_sha256=text_hash(r)) for r in selected])
    root = base/digest(identity)
    if root.exists() and not (root/'identity.json').is_file():
        raise ValueError('8D found an unowned existing output directory; preserved without overwrite')
    atomic_write_json(root/'identity.json', identity, resume=True)
    audit.update(strata=strata, total_selected=len(selected), random_representative_only=True)
    atomic_write_json(root/'sampling-audit.json', audit, resume=True)
    queue = [dict(r, audit_id=priority(digest(identity), r['pair_id'])[:16]) for r in selected]
    if len({r['audit_id'] for r in queue}) != len(queue):
        raise ValueError('8D blind audit_id collision')
    atomic_write_jsonl(root/'audit_queue.jsonl', queue, resume=True)
    if not (root/'annotations.jsonl').exists():
        atomic_write_jsonl(root/'annotations.jsonl', [])
    store = AuditStore(root)
    store.publish()
    return store


def wilson(successes, total):
    if type(total) is not int or type(successes) is not int or not 0 <= successes <= total:
        raise ValueError('invalid Wilson counts')
    if not total:
        return dict(estimate=None, lower=None, upper=None)
    z = 1.959963984540054
    p, denominator = successes/total, 1 + z*z/total
    centre = (p + z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return dict(estimate=p, lower=max(0., centre-half), upper=min(1., centre+half))


def aggregate(queue, annotations):
    total, done = len(queue), len(annotations)
    result = dict(complete=done == total, total=total, annotated=done, remaining=total-done)
    if not result['complete']:
        return result
    groups = []
    for population in POPULATIONS:
        for corpus in CORPORA:
            labels = Counter(annotations[r['audit_id']]['label'] for r in queue
                             if r['population'] == population and r['corpus'] == corpus)
            n, uncertain = sum(labels.values()), labels['uncertain']
            decided = n-uncertain
            counts = dict(N=n, decided=decided, **{key: labels[key] for key in LABELS})
            numerators = dict(strict_correct_rate=labels['correct'], acceptable_rate=labels['acceptable'],
                usable_rate=labels['correct']+labels['acceptable'], partial_rate=labels['partial_mismatch'],
                wrong_alignment_rate=labels['wrong_alignment'],
                serious_error_rate=labels['partial_mismatch']+labels['wrong_alignment'])
            rates = {key: wilson(value, decided) for key, value in numerators.items()}
            rates['uncertain_rate'] = wilson(uncertain, n)
            groups.append(dict(population=population, corpus=corpus, **counts, rates=rates))
    comparisons = {}
    for population in POPULATIONS:
        values = {r['corpus']: r['rates']['serious_error_rate']['estimate'] for r in groups if r['population'] == population}
        comparisons[population] = None if any(v is None for v in values.values()) else 100*(values['jesc']-values['kftt'])
    flag_counts = {corpus: dict(Counter(flag for r in queue if r['corpus'] == corpus
                                      for flag in annotations[r['audit_id']]['flags'])) for corpus in CORPORA}
    return dict(result, strata=groups, jesc_minus_kftt_serious_error_pp=comparisons, flags_by_corpus=flag_counts)


def percent(value):
    return 'N/A' if value is None else f'{100*value:.2f}%'


def report(summary):
    if not summary['complete']:
        return f"Annotated: {summary['annotated']} / {summary['total']}\nRemaining: {summary['remaining']}\n"
    lines = ['# 8D: KFTT/JESC Text-Pair Quality Audit', '',
             'Human labels only. Uncertain is excluded from decided-rate denominators.', '',
             '| Population | Corpus | N | Correct | Acceptable | Partial mismatch | Wrong alignment | Uncertain |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for group in summary['strata']:
        lines.append('| '+' | '.join(str(group[k]) for k in ('population', 'corpus', 'N', *LABELS))+' |')
    lines += ['', '| Population | Corpus | Rate | Estimate | Wilson 95% lower | Wilson 95% upper |',
              '|---|---|---|---:|---:|---:|']
    for group in summary['strata']:
        for name, value in group['rates'].items():
            lines.append('| '+' | '.join([group['population'], group['corpus'], name,
                                         *(percent(value[k]) for k in ('estimate', 'lower', 'upper'))])+' |')
    lines += ['', 'JESC serious error rate − KFTT serious error rate (percentage points):']
    for key, value in summary['jesc_minus_kftt_serious_error_pp'].items():
        lines.append(f'- {key}: '+('N/A (no decided cases)' if value is None else f'{value:+.2f} pp'))
    for corpus, flags in summary['flags_by_corpus'].items():
        lines += ['', 'Most common '+corpus.upper()+' flags (count / audited pairs; flags may overlap):']
        n = sum(r['N'] for r in summary['strata'] if r['corpus'] == corpus)
        for name in ('boundary_mismatch', 'missing_content', 'extra_content'):
            lines.append(f'- {name}: {flags.get(name, 0)}/{n}')
        for name, count in sorted(flags.items(), key=lambda p: (-p[1], p[0])):
            if name not in ('boundary_mismatch', 'missing_content', 'extra_content'):
                lines.append(f'- {name}: {count}/{n}')
    lines += ['', 'This diagnostic measures sampled text-pair quality only.',
              'It does not by itself prove the cause of Direct S2ST performance.',
              'The four strata are sampled separately; no unweighted pooled corpus-quality claim is made.', '']
    if 'coverage' in summary:
        lines += ['## Sampling and provenance coverage', '',
                  '| Population | Corpus | Requested | Eligible available | Selected |',
                  '|---|---|---:|---:|---:|']
        for group in summary['coverage']['strata']:
            lines.append('| '+' | '.join(str(group[k]) for k in ('population', 'corpus', 'requested', 'available', 'selected'))+' |')
        lines += ['', 'Excluded rows: '+str(summary['coverage']['excluded_rows']),
                  'Exclusion reasons: '+json.dumps(summary['coverage']['exclusion_reasons']),
                  'Rates describe eligible sampled pairs; excluded/unproven rows limit generalization.',
                  'Corpus provenance: '+summary['coverage']['metadata_source'], '', 'Input manifests / provenance SHA256:']
        lines += ['- '+path+' — '+sha for path, sha in summary['inputs_sha256'].items()]
        lines += ['']
    return '\n'.join(lines)


def csv_text(values, fields):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(values)
    return output.getvalue()


class AuditStore:
    def __init__(self, root):
        self.root = Path(root)
        self.identity = read(self.root/'identity.json')
        if self.root.name != digest(self.identity):
            raise ValueError('8D identity hash mismatch; annotations were not reset')
        self.queue = list(rows(self.root/'audit_queue.jsonl'))
        self.by_id = {r['audit_id']: r for r in self.queue}
        expected = self.identity['selected_pairs']
        if len(self.queue) != len(expected) or len(self.by_id) != len(expected):
            raise ValueError('8D queue count / audit IDs differ from the saved identity')
        for item, proof in zip(self.queue, expected):
            if (any(item[k] != proof[k] for k in ('pair_id', 'population', 'corpus', 'original_split'))
                    or text_hash(item) != proof['text_sha256']
                    or item['audit_id'] != priority(self.root.name, item['pair_id'])[:16]):
                raise ValueError('8D queue text / blind mapping differs from the saved identity')
        self._lock = threading.RLock()
        self.annotations = self._read_annotations()
        self._identity_sha = checksum(self.root/'identity.json')
        self._queue_sha = checksum(self.root/'audit_queue.jsonl')
        self._check_saved_progress()

    def _check_saved_progress(self):
        path = self.root/'summary.json'
        if path.is_file() and read(path).get('annotated', 0) > len(self.annotations):
            raise ValueError('8D annotations are missing compared with saved progress; no answers were reset')

    def _read_annotations(self):
        result = {}
        for row in rows(self.root/'annotations.jsonl'):
            if (set(row) != {'audit_id', 'label', 'flags', 'confidence', 'note', 'timestamp'}
                    or row['audit_id'] not in self.by_id or row['audit_id'] in result):
                raise ValueError('8D unknown/duplicate annotation or non-blind annotation fields')
            self.validate(row['label'], row['flags'], row['confidence'], row['note'])
            datetime.fromisoformat(row['timestamp'])
            result[row['audit_id']] = row
        return result

    @staticmethod
    def validate(label, flags, confidence, note):
        if label not in LABELS or confidence not in CONFIDENCES:
            raise ValueError('Select one human label and a valid confidence')
        if not isinstance(flags, (list, tuple)) or any(f not in FLAGS for f in flags) or len(set(flags)) != len(flags):
            raise ValueError('Invalid / duplicate secondary flags')
        if not isinstance(note, str):
            raise ValueError('Note must be text')

    @contextmanager
    def writing(self):
        # OS advisory locks release on process death: no stale lock deletion is
        # needed after a runtime reset. Use only one annotating runtime per audit.
        path = self.root/'.annotation-write.lock'
        with self._lock:
            handle = path.open('a+b')
            try:
                if os.name == 'nt':
                    import msvcrt
                    handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                handle.close()
                raise RuntimeError('8D save lock unavailable. Close other audit UIs and retry; '
                                   'annotations are preserved. The mount must support advisory locks.') from error
            try:
                yield
            finally:
                handle.close()

    def save(self, audit_id, label, *, flags=(), confidence='high', note='', allow_update=False):
        if audit_id not in self.by_id:
            raise ValueError('Unknown blind audit_id')
        self.validate(label, flags, confidence, note)
        with self.writing():
            if checksum(self.root/'identity.json') != self._identity_sha or checksum(self.root/'audit_queue.jsonl') != self._queue_sha:
                raise ValueError('8D identity/queue changed during annotation; rerun the cell without modifying saved files')
            current = self._read_annotations()
            if audit_id in current and not allow_update:
                raise ValueError('Existing annotation preserved; use Previous to explicitly edit it')
            if audit_id in current and current[audit_id] != self.annotations.get(audit_id):
                raise ValueError('8D this answer changed in another UI; rerun the cell before editing')
            current[audit_id] = dict(audit_id=audit_id, label=label, flags=list(flags),
                confidence='low' if label == 'uncertain' else confidence, note=note,
                timestamp=datetime.now(timezone.utc).isoformat())
            # Commit every individual answer before moving the UI forward.
            atomic_write_jsonl(self.root/'annotations.jsonl',
                              [current[r['audit_id']] for r in self.queue if r['audit_id'] in current], overwrite=True)
            self.annotations = current
            self._publish()

    def first_pending(self):
        return next((i for i, r in enumerate(self.queue) if r['audit_id'] not in self.annotations), len(self.queue))

    def blind_sample(self, position):
        row = self.queue[position]
        return dict(position=position+1, total=len(self.queue), ja_text=row['ja_text'], en_text=row['en_text'],
                    provenance={k: row[k] for k in TEXT_FIELDS if k not in ('ja_text', 'en_text') and row.get(k) is not None})

    def publish(self):
        with self.writing():
            self.annotations = self._read_annotations()
            self._check_saved_progress()
            return self._publish()

    def _publish(self):
        summary = aggregate(self.queue, self.annotations)
        if summary['complete']:
            sampling = read(self.root/'sampling-audit.json')
            summary.update(inputs_sha256=self.identity['inputs_sha256'],
                coverage=dict(strata=sampling['strata'], excluded_rows=len(sampling['excluded']),
                              exclusion_reasons=dict(Counter(r['reason'] for r in sampling['excluded'])),
                              metadata_source=sampling['metadata_source']))
        atomic_write_json(self.root/'summary.json', summary, overwrite=True)
        atomic_write_text(self.root/'report.md', report(summary), overwrite=True)
        if summary['complete']:
            exported = []
            for item in self.queue:
                row = dict(item, **self.annotations[item['audit_id']])
                row['flags'] = ';'.join(row['flags'])
                exported.append(row)
            fields = ('audit_id', 'corpus', 'population', 'original_split', 'pair_id', *TEXT_FIELDS,
                      'label', 'flags', 'confidence', 'note', 'timestamp')
            atomic_write_text(self.root/'per_sample.csv', csv_text(exported, fields), overwrite=True)
            flat = []
            for group in summary['strata']:
                row = {k: v for k, v in group.items() if k != 'rates'}
                for name, values in group['rates'].items():
                    row.update({name+'_'+key: value for key, value in values.items()})
                flat.append(row)
            atomic_write_text(self.root/'per_stratum.csv', csv_text(flat, list(flat[0])), overwrite=True)
        return summary
