"""Separate, resumable gold-JA vs saved-ASR-JA NLLB diagnostic. No audio IO."""
import csv
import io
import json
from pathlib import Path
import re
import sys
import time

from .corpus_diagnostic import (aggregate, checksum, diagnose, index, joined_samples, mapping,
                                read, rows, text)
from ..evaluation.bleu import corpus_bleu, normalize_english, sentence_bleu
from ..io import atomic_write_json, atomic_write_text, atomic_write_bytes
from ..journal import digest
from ..progress import operation, track, activity

VERSION = 1
PURPOSE = 'gold_ja_vs_saved_asr_ja_nllb_diagnostic_not_formal_benchmark'
CONDITIONS = ('gold', 'asr')


def validate_model(config):
    config = mapping(config, 'saved MT model settings')
    if (config.get('model') != 'facebook/nllb-200-distilled-600M'
            or not re.fullmatch('[0-9a-f]{40}', str(config.get('revision', '')))
            or config.get('source_language') != 'jpn_Jpan' or config.get('target_language') != 'eng_Latn'):
        raise RuntimeError('8B requires saved SHA-pinned NLLB-600M jpn_Jpan -> eng_Latn settings; no guessed defaults')
    if set(config) != {'model', 'revision', 'source_language', 'target_language'}:
        raise RuntimeError('unsupported extra saved MT settings; review them before reproducing generation')
    return config


@operation('8B: verify saved Cascade scores and text-only inputs')
def saved_context(manifest, evaluation_root, limit=0):
    if type(limit) is not int or limit < 0:
        raise ValueError('8B limit must be 0 (all test samples) or a positive integer')
    root = Path(evaluation_root)
    baseline = diagnose(manifest, root)  # Includes official All BLEU round trip and exact ID/reference checks.
    benchmark = (root/'plan.json').is_file()
    predictions_path = root/('predictions.jsonl' if benchmark else 'predictions/predictions.jsonl')
    scores = root/('results' if benchmark else 'metrics')
    samples = joined_samples(index(rows(manifest), 'common/test'), index(rows(predictions_path), 'saved Cascade'),
                             index(rows(scores/'per_sample.jsonl'), 'saved evaluation'), baseline['run_id'])
    if benchmark:
        plan = read(root/'plan.json')
        body = {key:value for key,value in plan.items() if key not in ('identity','local','output','run')}
        if digest(body) != plan['identity'] or read(scores/'metrics.json').get('pipeline') != plan.get('pipeline'):
            raise RuntimeError('8B saved Cascade model settings are not bound to the original plan/metrics')
        settings = plan['pipeline']
    else:
        settings = read(predictions_path.with_suffix('.lock.json')).get('models')
    config = validate_model(mapping(settings, 'saved Cascade pipeline').get('mt'))
    if limit:
        samples = sorted(samples, key=lambda row: digest(row['pair_id']))[:limit]
    samples = sorted(samples, key=lambda row: row['pair_id'])
    evidence = dict(baseline['inputs_sha256'])
    for path in ((root/'plan.json',) if benchmark else (predictions_path.with_suffix('.lock.json'),)):
        evidence[str(path)] = checksum(path)
    return dict(baseline=baseline, samples=samples, model=config, inputs_sha256=evidence,
                selected_ids=[row['pair_id'] for row in samples], full_test_samples=baseline['groups']['All']['samples'])


def safe_output(persistent, evaluation_root, corpus=None):
    root = Path(persistent)/'diagnostics/cascade-gold-mt'
    protected = [Path(evaluation_root), Path(persistent)/'data'] + ([Path(corpus)] if corpus else [])
    if any(root.resolve().is_relative_to(path.resolve()) for path in protected):
        raise RuntimeError('8B output overlaps official results, prepared data or CORPUS_ROOT')
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise RuntimeError('8B diagnostic output must not contain symlinks')
    return root


class TextChunks:
    """Immutable small JSON chunks; at most seven new branch results between saves."""
    def __init__(self, root, identity, expected):
        self.root, self.identity, self.expected = Path(root), digest(identity), dict(expected)
        self.values, self.pending = {}, []
        if self.root.exists():
            for path in track(sorted(self.root.glob('*.json')), '8B: verify/reuse saved text chunks'):
                document = read(path)
                body = mapping(document.get('body'), '8B text chunk')
                if document.get('sha256') != digest(body) or body.get('identity') != self.identity:
                    raise RuntimeError('8B text checkpoint checksum/identity mismatch: '+str(path))
                if not isinstance(body.get('rows'), list):
                    raise RuntimeError('8B text checkpoint has no rows')
                for row in body['rows']:
                    self._accept(row)

    def _accept(self, row):
        row = mapping(row, '8B saved translation')
        if not isinstance(row.get('pair_id'), str) or row.get('condition') not in CONDITIONS:
            raise RuntimeError('8B malformed saved pair/condition')
        key = (row.get('pair_id'), row.get('condition'))
        if key not in self.expected or key in self.values:
            raise RuntimeError('8B duplicate or unexpected saved translation: '+str(key))
        if row.get('input_sha256') != self.expected[key]:
            raise RuntimeError('8B saved translation input differs from the selected Japanese text')
        if row.get('status') not in ('success', 'failed') or not isinstance(row.get('translation'), str):
            raise RuntimeError('8B malformed saved translation')
        if row['status'] == 'failed' and row['translation']:
            raise RuntimeError('8B failed translation must have an empty hypothesis')
        if row['status'] == 'success' and not row['translation'].strip():
            raise RuntimeError('8B successful saved translation is empty')
        self.values[key] = row

    def record(self, row):
        self._accept(row)
        self.pending.append(row)
        if len(self.pending) >= 8:
            self.flush()

    def flush(self):
        if self.pending:
            body = dict(identity=self.identity, rows=self.pending)
            sha = digest(body)
            atomic_write_json(self.root/(sha+'.json'), dict(body=body, sha256=sha), resume=True)
            print(f'[8b-checkpoint] durable={len(self.values)} branch_results', flush=True, file=sys.stderr)
            self.pending = []


def fatal(error):
    return isinstance(error, RuntimeError) and any(word in str(error).lower() for word in
        ('cuda', 'out of memory', 'device-side assert', 'cublas', 'cudnn'))


class NllbProbe:
    """Same tokenizer/model/generate arguments as cascade.mt.NllbTranslator."""
    @operation('8B: load SHA-pinned NLLB only on GPU')
    def __init__(self, config):
        from .mt import NllbTranslator
        self.adapter = NllbTranslator(model_id=config['model'], revision=config['revision'],
            source_language=config['source_language'], target_language=config['target_language'], device='cuda')
        self.torch, self.tokenizer, self.model = self.adapter.torch, self.adapter.tokenizer, self.adapter.model
        if self.adapter.device.type != 'cuda':
            raise RuntimeError('8B requires a GPU; CPU fallback is not permitted')
        if self.model.training or self.model.generation_config.do_sample:
            raise RuntimeError('8B requires the original eval-mode deterministic NLLB settings; review unexpected training/sampling state')
        self.forced_bos = self.tokenizer.convert_tokens_to_ids(config['target_language'])
        source_id = self.tokenizer.convert_tokens_to_ids(config['source_language'])
        if (any(type(token) is not int or not 0 <= token < self.model.config.vocab_size for token in (source_id,self.forced_bos))
                or self.tokenizer.src_lang != config['source_language'] or self.forced_bos == self.tokenizer.unk_token_id
                or source_id == self.tokenizer.unk_token_id or self.forced_bos == source_id):
            raise RuntimeError('8B NLLB language token validation failed')
        self.audit = dict(tokenizer_class=type(self.tokenizer).__name__, source_language_token_id=source_id,
            forced_bos_token_id=self.forced_bos, tokenizer_init=self.tokenizer.init_kwargs,
            generation_config=self.model.generation_config.to_dict(),
            generate_kwargs='only forced_bos_token_id; no beam/repetition/length/precision overrides',
            dtype=str(next(self.model.parameters()).dtype), device=str(self.adapter.device))
        self.audit = json.loads(json.dumps(self.audit, default=str))

    def __call__(self, source):
        encoded = self.tokenizer(source, return_tensors='pt').to(self.adapter.device)
        source_ids = encoded['input_ids'][0].tolist()
        if self.audit['source_language_token_id'] not in source_ids:
            raise RuntimeError('8B tokenized source lost the Japanese language token')
        if len(source_ids) > self.model.config.max_position_embeddings:
            raise ValueError('source exceeds saved NLLB position limit; no silent truncation')
        with self.torch.inference_mode():
            tokens = self.model.generate(**encoded, forced_bos_token_id=self.forced_bos)
        ids = tokens[0].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return dict(translation=str(self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]).strip(),
                    source_token_count=len(source_ids), generated_token_count=len(ids),
                    eos_observed=any(token in eos for token in ids[1:]),
                    generation_limit_reached=len(ids) >= (1+self.model.generation_config.max_new_tokens
                        if self.model.generation_config.max_new_tokens is not None else self.model.generation_config.max_length),
                    source_prefix_ids=source_ids[:4], output_prefix_ids=ids[:4])


def groups(context, values):
    result = {}
    for name, corpus in (('All', None), ('KFTT', 'kftt'), ('JESC', 'jesc')):
        samples = [r for r in context['samples'] if corpus is None or r['corpus'] == corpus]
        if not samples:
            result[name] = None
            continue
        references = [normalize_english(row['en_text']) for row in samples]
        metric = dict(samples=len(samples), saved_mt_bleu=corpus_bleu(
            [normalize_english(row['mt_en_text']) for row in samples], references))
        for condition in CONDITIONS:
            translated = [values[(row['pair_id'], condition)] for row in samples]
            metric[condition+'_mt_bleu'] = corpus_bleu([normalize_english(r['translation']) for r in translated], references)
            metric[condition+'_failures'] = sum(r['status'] != 'success' for r in translated)
            metric[condition+'_generation_limit_reached'] = sum(r.get('generation_limit_reached') is True for r in translated)
        metric['gold_minus_asr_bleu'] = metric['gold_mt_bleu']-metric['asr_mt_bleu']
        metric['asr_rerun_matches_saved'] = sum(normalize_english(values[(r['pair_id'], 'asr')]['translation'])
                                               == normalize_english(r['mt_en_text']) for r in samples)
        result[name] = metric
    return result


@operation('8B: controlled gold-JA / saved-ASR-JA MT diagnostic')
def run(context, persistent, environment, *, factory=NllbProbe, corpus=None, limit=0, label='default'):
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}', label):
        raise ValueError('8B run label must contain 1..80 safe letters/digits/_/-')
    identity = dict(version=VERSION, purpose=PURPOSE, model=context['model'], environment=environment,
        selected_ids=context['selected_ids'], inputs=context['inputs_sha256'], label=label,
        implementation=checksum(Path(__file__)), adapter=checksum(Path(__file__).with_name('mt.py')),
        limit=limit, failure_policy='all_selected_samples_failed_as_empty', generation='unchanged_saved_mt_adapter_defaults')
    base = safe_output(persistent, context['baseline']['evaluation_root'], corpus)
    output = base/digest(identity)[:24]
    if output.is_symlink() or (output/'chunks').is_symlink() or not output.resolve().is_relative_to(base.resolve()):
        raise RuntimeError('8B output/chunks must stay inside the separate diagnostic directory')
    atomic_write_json(output/'identity.json', identity, resume=True)
    expected = {(r['pair_id'], condition):digest(r['ja_text'] if condition == 'gold' else r['asr_ja_text'])
                for r in context['samples'] for condition in CONDITIONS}
    journal = TextChunks(output/'chunks', identity, expected)
    audit_path = output/'model-audit.json'
    audit_sha = checksum(audit_path) if audit_path.is_file() else None
    if journal.values and (audit_sha is None or any(row.get('model_audit_sha256') != audit_sha for row in journal.values.values())):
        raise RuntimeError('8B saved generation audit is missing or differs from the saved translation chunks')
    model = None
    try:
        for sample in track(context['samples'], '8B: translate/reuse gold JA and ASR JA', key=lambda r:r['pair_id']):
            for condition in CONDITIONS:
                key = sample['pair_id'], condition
                if key in journal.values:
                    continue
                source = sample['ja_text'] if condition == 'gold' else sample['asr_ja_text']
                row = dict(pair_id=sample['pair_id'], condition=condition, status='failed', translation='', error=None,
                           input_sha256=digest(source))
                started = time.perf_counter()
                if not source.strip():
                    row['error'] = 'saved Japanese ASR text is missing/empty; no new ASR was run'
                else:
                    if model is None:
                        model = factory(context['model'])  # Loading/network/dependency failures are fatal, not 1196 failed rows.
                        atomic_write_json(audit_path, model.audit, resume=True)
                        audit_sha = checksum(audit_path)
                    started = time.perf_counter()
                    try:
                        with activity(condition+' NLLB: '+sample['pair_id']):
                            generated = model(source)
                        hypothesis = text(generated.get('translation'), '8B NLLB output')
                        if not hypothesis:
                            raise ValueError('empty NLLB output')
                        row.update(generated, status='success')
                    except Exception as error:
                        if fatal(error) or not isinstance(error, ValueError):
                            raise
                        row['error'] = f'{type(error).__name__}: {error}'
                row['seconds'] = time.perf_counter()-started
                row['model_audit_sha256'] = audit_sha
                journal.record(row)
    finally:
        journal.flush()
    for path, expected_sha in context['inputs_sha256'].items():
        if checksum(path) != expected_sha:
            raise RuntimeError('8B saved source metadata changed during diagnosis: '+path)
    metrics = groups(context, journal.values)
    per_sample = []
    for sample in context['samples']:
        row = {k:sample[k] for k in ('pair_id','corpus','ja_text','asr_ja_text','en_text')}
        row['saved_mt_en_text'] = sample['mt_en_text']
        for condition in CONDITIONS:
            value = journal.values[(sample['pair_id'], condition)]
            for key, item in value.items():
                if key not in ('pair_id', 'condition'):
                    row[condition+'_'+key] = item
            row[condition+'_sentence_bleu'] = sentence_bleu(normalize_english(value['translation']), normalize_english(sample['en_text']))
        per_sample.append(row)
    display = ['=== 8B: gold Japanese vs saved ASR Japanese -> same NLLB ===',
               'Full test samples: '+str(context['full_test_samples'])+'; selected: '+str(len(context['samples'])),
               'All metrics include failed hypotheses as empty. Corpus BLEU, not mean sentence BLEU.',
               'No new Whisper/TTS/audio inference. Official saved results are unchanged.', '']
    for name, metric in metrics.items():
        display.append(name+': '+('not present' if metric is None else
            f"n={metric['samples']} saved_MT={metric['saved_mt_bleu']:.6f} "
            f"gold_JA_MT={metric['gold_mt_bleu']:.6f} ASR_JA_MT_rerun={metric['asr_mt_bleu']:.6f} "
            f"gap={metric['gold_minus_asr_bleu']:+.6f} rerun_matches_saved={metric['asr_rerun_matches_saved']}/{metric['samples']} "
            f"failures(gold/asr)={metric['gold_failures']}/{metric['asr_failures']} "
            f"length_limit(gold/asr)={metric['gold_generation_limit_reached']}/{metric['asr_generation_limit_reached']}"))
    display += ['', 'A gold/ASR gap does not alone establish a sole cause. Low gold BLEU does not prove reference corruption.',
                'If ASR reruns differ from saved MT, inspect the recorded environment/generation audit before attribution.',
                'Separate diagnostic results: '+str(output)]
    report = dict(purpose=PURPOSE, identity_sha256=digest(identity), output_root=str(output), groups=metrics,
                  full_test_samples=context['full_test_samples'], selected_samples=len(context['samples']),
                  all_score_matches_saved=True, display='\n'.join(display), per_sample=per_sample)
    atomic_write_json(output/'metrics.json', report, resume=True)
    atomic_write_text(output/'report.md', report['display']+'\n', resume=True)
    stream = io.StringIO(newline='')
    first = ['pair_id','corpus','ja_text','asr_ja_text','en_text','saved_mt_en_text']
    fields = first+sorted({key for row in per_sample for key in row}-set(first))
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(per_sample)
    atomic_write_bytes(output/'per_sample.csv', stream.getvalue().encode('utf-8'), resume=True)
    return report
