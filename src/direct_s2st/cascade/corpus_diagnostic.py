"""Read-only corpus diagnostics over completed Cell 8 Cascade text artifacts.

Never reads WAVs/checkpoints, restores archives, runs inference, or saves output.
"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import unicodedata

from ..evaluation.bleu import corpus_bleu, normalize_english
from ..journal import digest
from ..progress import operation, track

NORMALIZATION = 'nfkc_lower_english_v1'
ALL_FAILED_EMPTY = 'all_samples_failed_as_empty'
SUCCESS_ONLY = 'successful_evaluations_only'
SACREBLEU_VERSION = '2.5.1'  # Existing requirements/evaluation-core.txt; not a new dependency.


def read(path):
    try:
        with Path(path).open(encoding='utf-8-sig') as stream:
            return mapping(json.load(stream), str(path))
    except (OSError, ValueError) as error:
        raise RuntimeError(f'cannot read saved diagnostic metadata {path}: {error}; '
                           'no restore, regeneration, or result writes attempted') from error


def mapping(value, label):
    if not isinstance(value, dict):
        raise RuntimeError(label+': expected a saved JSON/YAML object')
    return value


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024**2):
            result.update(chunk)
    return result.hexdigest()


def rows(path):
    """Literal metadata path: no audio-path resolution/staging or filesystem scan."""
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f'{path}:{number}: expected a JSON object')
                yield value


def index(collection, label):
    values = {}
    for row in collection:
        key = row.get('pair_id')
        if not isinstance(key, str) or not key:
            raise RuntimeError(label+': missing/invalid pair_id')
        if key in values:
            raise RuntimeError(label+': duplicate pair_id: '+key)
        values[key] = row
    if not values:
        raise RuntimeError(label+': no rows')
    return values


def same_ids(expected, actual, label):
    if set(expected) != set(actual):
        missing, extra = sorted(set(expected)-set(actual)), sorted(set(actual)-set(expected))
        raise RuntimeError(f'{label}: incomplete pair_id join; missing={missing[:8]} extra={extra[:8]}. '
                           'A missing row is not a recorded failed hypothesis; no samples were dropped or fabricated.')


def normalize_japanese_cer(text):
    """Explicit diagnostic-only convention approved by the user; retain punctuation."""
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if not c.isspace())


def edit_distance(reference, hypothesis):
    """Unicode-codepoint Levenshtein, linear memory, no new dependency."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis)+1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    return previous[-1]


def text(value, label):
    if not isinstance(value, str):
        raise RuntimeError(label+': expected saved text string')
    return value


def optional_stage_text(row, field):
    value = row.get(field)
    if value is None:
        if row['status'] != 'failed':
            raise RuntimeError(row['pair_id']+': successful Cascade output has no '+field)
        return ''
    return text(value, row['pair_id']+'/'+field)


@operation('Cascade corpus diagnostic: join saved metadata without audio or inference')
def joined_samples(manifest, predictions, evaluated, run_id):
    same_ids(manifest, predictions, 'Cascade predictions')
    same_ids(manifest, evaluated, 'Cell 8 evaluation')
    corpuses = {key: text(row.get('corpus'), key+'/corpus').lower() for key, row in manifest.items()}
    if any(not value for value in corpuses.values()):
        raise RuntimeError('common test has an empty corpus field; no inference from pair_id is allowed')
    if 'kftt' not in corpuses.values():
        raise RuntimeError('common/test.jsonl contains no corpus == kftt samples')
    result = []
    for key in track(sorted(manifest), 'Cascade diagnostic: validate/join pair_id'):
        row, predicted, score = manifest[key], predictions[key], evaluated[key]
        if row.get('split') != 'test' or predicted.get('split') != 'test':
            raise RuntimeError(key+': diagnostic requires saved test results, not dev/preview')
        if any(value.get('system_id') != 'cascade' or value.get('run_id') != run_id for value in (predicted, score)):
            raise RuntimeError(key+': mixed Cascade run/system identity')
        if predicted.get('status') not in ('success', 'failed') or score.get('evaluation_status') not in ('success', 'failed'):
            raise RuntimeError(key+': unrecognized generation/evaluation status')
        if score.get('generation_status') != predicted['status']:
            raise RuntimeError(key+': generation status differs from saved evaluation')
        success = score['evaluation_status'] == 'success'
        if success and predicted['status'] != 'success':
            raise RuntimeError(key+': a failed generation cannot have successful ASR evaluation')
        reference = text(row.get('en_text'), key+'/en_text')
        ja_reference = normalize_japanese_cer(text(row.get('ja_text'), key+'/ja_text'))
        if not ja_reference:
            raise RuntimeError(key+': CER reference is empty after NFKC/whitespace normalization')
        if predicted.get('reference_text') != reference or score.get('reference_raw') != reference:
            raise RuntimeError(key+': saved English reference differs from the pair_id-matched manifest')
        reference_normalized = normalize_english(reference)
        if score.get('reference_normalized') != reference_normalized:
            raise RuntimeError(key+': saved reference normalization differs from Cell 8 implementation')
        hypothesis = text(score.get('hypothesis_raw'), key+'/hypothesis_raw') if success else ''
        normalized = normalize_english(hypothesis)
        if success and score.get('hypothesis_normalized') != normalized:
            raise RuntimeError(key+': saved hypothesis normalization differs from Cell 8 implementation')
        if not success and (score.get('hypothesis_raw') not in (None, '') or score.get('hypothesis_normalized') not in (None, '')):
            raise RuntimeError(key+': failed evaluation retains unexpected nonempty hypothesis')
        asr_text = optional_stage_text(predicted, 'asr_ja_text')
        mt_text = optional_stage_text(predicted, 'mt_en_text')
        ja_hypothesis = normalize_japanese_cer(asr_text)
        sentence = score.get('sentence_bleu')
        if sentence is not None and (isinstance(sentence, bool) or not isinstance(sentence, (int, float))
                                     or not math.isfinite(sentence) or not 0 <= sentence <= 100):
            raise RuntimeError(key+': invalid saved sentence BLEU')
        result.append(dict(pair_id=key, corpus=corpuses[key], ja_text=row['ja_text'], asr_ja_text=asr_text,
                           en_text=reference, mt_en_text=mt_text, generated_en_asr_text=hypothesis,
                           sentence_bleu=sentence, generation_status=predicted['status'],
                           evaluation_status=score['evaluation_status'], status='success' if success else 'failed',
                           cer_edits=edit_distance(ja_reference, ja_hypothesis), cer_characters=len(ja_reference),
                           ja_exact_match=ja_reference == ja_hypothesis,
                           source_asr_missing=predicted.get('asr_ja_text') is None,
                           mt_missing=predicted.get('mt_en_text') is None,
                           reference_normalized=reference_normalized, hypothesis_normalized=normalized))
    return result


def aggregate(samples, policy):
    if not samples:
        return None
    final = samples if policy == ALL_FAILED_EMPTY else [r for r in samples if r['evaluation_status'] == 'success']
    edits, characters = sum(r['cer_edits'] for r in samples), sum(r['cer_characters'] for r in samples)
    return dict(samples=len(samples), ja_asr_cer=edits/characters, ja_asr_cer_percent=100*edits/characters,
                cer_edits=edits, cer_reference_characters=characters,
                ja_asr_exact_match_rate=sum(r['ja_exact_match'] for r in samples)/len(samples),
                source_asr_missing_samples=sum(r['source_asr_missing'] for r in samples),
                mt_missing_samples=sum(r['mt_missing'] for r in samples),
                mt_bleu=corpus_bleu([normalize_english(r['mt_en_text']) for r in samples],
                                    [r['reference_normalized'] for r in samples]),
                final_asr_bleu=corpus_bleu([r['hypothesis_normalized'] for r in final],
                                         [r['reference_normalized'] for r in final]),
                final_bleu_samples=len(final), successful_samples=sum(r['evaluation_status'] == 'success' for r in samples),
                failed_samples=sum(r['evaluation_status'] != 'success' for r in samples),
                generation_failures=sum(r['generation_status'] == 'failed' for r in samples),
                asr_failures=sum(r['generation_status'] == 'success' and r['evaluation_status'] != 'success' for r in samples))


def display(report):
    lines = ['=== Cascade corpus diagnostic ===', '', 'Saved evaluation: '+report['evaluation_root'],
             'Final BLEU denominator: '+report['failure_policy'],
             'JA CER: NFKC + remove whitespace, KEEP punctuation; sum edits / sum reference characters.',
             'All ASR-BLEU matches the saved Cell 8 metric. No inference or result files were written.', '']
    for group in ('All', 'KFTT', 'JESC'):
        metric = report['groups'][group]
        if metric is None:
            lines += [group+': not present', '']
            continue
        def number(value):
            return 'not available (no eligible hypotheses)' if value is None else f'{value:.6f}'
        lines += [group+':', f"  samples        : {metric['samples']}",
                  '  JA ASR CER %   : '+number(metric['ja_asr_cer_percent']),
                  '  MT BLEU        : '+number(metric['mt_bleu']),
                  '  Final ASR-BLEU : '+number(metric['final_asr_bleu']),
                  f"  successful/failed : {metric['successful_samples']}/{metric['failed_samples']}",
                  f"  Final BLEU samples: {metric['final_bleu_samples']}", '']
    lines += ['MT BLEU uses all stage outputs, including available MT text when only TTS/final ASR failed.',
              'A KFTT/JESC difference alone does NOT prove corpus quality caused the difference.']
    return '\n'.join(lines)


def locate(persistent, target, *, comparison_config=None, runs_root=None, evaluation_root=None):
    if evaluation_root:
        return Path(evaluation_root)
    persistent = Path(persistent)
    if target == 'cascade':
        pointer = read(persistent/'benchmarks/cascade/latest.json')
        path = Path(pointer['plan'])
        if checksum(path) != pointer['sha256']:
            raise RuntimeError('saved Cascade latest/plan checksum mismatch; no repair/inference attempted')
        return path.parent
    if target == 'all':
        if comparison_config is None or runs_root is None:
            raise RuntimeError('four-system mode needs the current Cell 8 COMPARE_CONFIG and RUNS values, or an explicit evaluation root')
        import yaml
        with (Path(comparison_config)/'cascade-eval.yaml').open(encoding='utf-8') as stream:
            config = mapping(yaml.safe_load(stream), 'current Cascade evaluation config')
        run_id = config.get('run_id')
        if not isinstance(run_id, str) or not re.fullmatch('[A-Za-z0-9_-]+', run_id):
            raise RuntimeError('current Cascade evaluation config has an invalid run_id')
        return Path(runs_root)/run_id
    raise RuntimeError('current Cell 8 target is not Cascade/all. Select an explicit saved Cascade evaluation root; do not rerun models.')


@operation('Cascade KFTT-only diagnostic: read saved Cell 8 results only')
def diagnose(manifest_path, evaluation_root):
    try:
        version = importlib.metadata.version('sacrebleu')
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError('select an existing CPU Python with sacrebleu=='+SACREBLEU_VERSION+
                           '; this diagnostic does not install libraries or rebuild the runtime') from error
    if version != SACREBLEU_VERSION:
        raise RuntimeError('read-only diagnostic requires existing sacrebleu=='+SACREBLEU_VERSION+'; no packages were installed')
    manifest_path, root = Path(manifest_path), Path(evaluation_root)
    plan_path = root/'plan.json'
    benchmark = plan_path.is_file()
    if benchmark:
        plan = read(plan_path)
        if plan.get('system_id') != 'cascade' or plan.get('split') != 'test' or plan.get('version') != 1:
            raise RuntimeError('selected plan is not a Cascade full-test evaluation')
        if checksum(manifest_path) != plan['common_sha256']:
            raise RuntimeError('current test manifest differs from saved Cascade plan')
        if checksum(manifest_path.parent/'dataset-lock.json') != plan['dataset_lock_sha256']:
            raise RuntimeError('current common dataset lock differs from saved Cascade plan')
        scores_root, predictions_path = root/'results', root/'predictions.jsonl'
    else:
        plan = None
        scores_root, predictions_path = root/'metrics', root/'predictions/predictions.jsonl'
        prediction_lock = read(predictions_path.with_suffix('.lock.json'))
        if prediction_lock.get('common') != checksum(manifest_path):
            raise RuntimeError('current test manifest differs from saved Cascade prediction journal')
    metrics = read(scores_root/'metrics.json')
    if metrics.get('system_id') != 'cascade' or not isinstance(metrics.get('run_id'), str):
        raise RuntimeError('saved metrics are not a Cascade run')
    run_id = metrics['run_id']
    identity = metrics.get('evaluation_identity')
    if not isinstance(identity, dict) or not identity:
        raise RuntimeError('saved evaluation identity is missing')
    normalization = identity.get('normalization', NORMALIZATION)
    if normalization != NORMALIZATION:
        raise RuntimeError('unsupported saved normalization; do not reinterpret the formal evaluation')
    if benchmark:
        policy = metrics.get('bleu_denominator')
        if policy != ALL_FAILED_EMPTY or plan.get('bleu_denominator') != policy or identity.get('bleu_denominator') != policy:
            raise RuntimeError('Cascade benchmark failure policy differs from saved Cell 8 conditions')
        if run_id != plan['run']['name'] or identity.get('plan') != plan['identity'] or identity.get('predictions') != checksum(predictions_path):
            raise RuntimeError('Cascade metrics/plan/predictions identity mismatch')
        receipt = read(root/'progress'/('asr-'+digest(identity)[:24]+'.json'))
        body = mapping(receipt.get('body'), 'saved Cell 8 score receipt body')
        files = mapping(body.get('files'), 'saved Cell 8 score receipt files')
        sample_file = mapping(files.get('per_sample.jsonl'), 'saved Cell 8 score receipt per_sample.jsonl')
        if (receipt.get('sha256') != digest(body) or body.get('identity') != identity
                or sample_file.get('sha256') != checksum(scores_root/'per_sample.jsonl')):
            raise RuntimeError('saved Cell 8 score receipt/checksum mismatch; no archive restore attempted')
    else:
        if metrics.get('bleu_denominator') not in (None, SUCCESS_ONLY):
            raise RuntimeError('unrecognized four-system failure policy')
        policy = SUCCESS_ONLY  # evaluate_predictions default used by the unchanged generic Cell 8 CLI.
        lock = read(scores_root/'per_sample.lock.json')
        if lock.get('predictions') != checksum(predictions_path) or lock.get('evaluation') != identity:
            raise RuntimeError('saved Cascade evaluation is stale relative to its predictions')
    manifest = index(rows(manifest_path), 'common/test.jsonl')
    if plan:
        if (not isinstance(plan.get('ids'), list)
                or any(not isinstance(key, str) or not key for key in plan['ids'])
                or len(plan['ids']) != len(set(plan['ids']))):
            raise RuntimeError('duplicate/invalid Cascade plan IDs')
        same_ids(manifest, plan['ids'], 'Cascade plan')
    predictions = index(rows(predictions_path), 'Cascade predictions')
    evaluated = index(rows(scores_root/'per_sample.jsonl'), 'Cell 8 per_sample.jsonl')
    samples = joined_samples(manifest, predictions, evaluated, run_id)
    groups = dict(All=aggregate(samples, policy), KFTT=aggregate([r for r in samples if r['corpus'] == 'kftt'], policy),
                  JESC=aggregate([r for r in samples if r['corpus'] == 'jesc'], policy))
    all_metrics = groups['All']
    if (any(type(metrics.get(field)) is not int for field in ('samples', 'generation_failures'))
            or metrics['samples'] != len(samples) or metrics['generation_failures'] != all_metrics['generation_failures']):
        raise RuntimeError('saved Cell 8 counts differ from joined metadata')
    rate = metrics.get('asr_success_rate')
    if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate)
            or not math.isclose(rate, all_metrics['successful_samples']/len(samples), rel_tol=0, abs_tol=1e-12)):
        raise RuntimeError('saved Cell 8 ASR success counts differ')
    if 'asr_failures' in metrics and metrics['asr_failures'] != all_metrics['asr_failures']:
        raise RuntimeError('saved Cell 8 ASR failure counts differ')
    formal, recomputed = metrics.get('bleu'), all_metrics['final_asr_bleu']
    if (formal is None) != (recomputed is None) or (formal is not None and (
            isinstance(formal, bool) or not isinstance(formal, (int, float)) or not math.isfinite(formal)
            or not math.isclose(formal, recomputed, rel_tol=0, abs_tol=1e-8))):
        raise RuntimeError(f'All ASR-BLEU differs from saved Cell 8: saved={formal}, recomputed={recomputed}. '
                           'No KFTT/JESC score was accepted; check result selection/scoring environment.')
    report = dict(purpose='read_only_cascade_by_manifest_corpus_diagnostic_not_a_new_benchmark',
                  evaluation_root=str(root), run_id=run_id, failure_policy=policy,
                  scoring=dict(shared_implementation='direct_s2st.evaluation.bleu', normalization=NORMALIZATION,
                               sacrebleu_version=SACREBLEU_VERSION, sentence_bleu_averaged=False,
                               ja_cer='micro Levenshtein after NFKC + remove whitespace; punctuation retained'),
                  saved_all_asr_bleu=formal, all_score_matches_saved=True, groups=groups,
                  kftt_samples=[r for r in samples if r['corpus'] == 'kftt'],
                  inputs_sha256={str(path):checksum(path) for path in
                                 (manifest_path, predictions_path, scores_root/'per_sample.jsonl', scores_root/'metrics.json')})
    report['display'] = display(report)
    return report
