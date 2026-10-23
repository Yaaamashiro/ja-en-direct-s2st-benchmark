"""GPU phases of full-test S2UT evaluation; audio is durably saved in ZIPs."""
import argparse
import csv
import io
from pathlib import Path
import subprocess
import sys

from ..benchmark_storage import guarded, publish_tree, read, restore_tree
from ..drive_staging import raw_sha
from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_text, read_jsonl
from ..journal import digest
from ..progress import operation, track


def ordered(rows, ids):
    by_id = {r['pair_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError('benchmark requires exactly all test IDs, without duplicates')
    return [by_id[key] for key in ids]


@operation('benchmark: generate/reuse full-test Unit shards')
def units(plan):
    local, output = Path(plan['local']), Path(plan['output'])
    subprocess.run([sys.executable, '-m', 'direct_s2st.s2ut.fairseq_infer',
                    '--data-root', plan['data_root'], '--run-root', plan['run']['work'],
                    '--common-root', plan['common_root'], '--predictions', str(local/'units/predictions.jsonl'),
                    '--split', 'test', '--beam', str(plan['beam']), '--device', 'cuda',
                    '--checkpoint-selection', 'best', '--resume',
                    '--generation-backup', str(output/'generation')], check=True)


def save_index(output, identity, samples):
    body = dict(version=1, identity=identity, samples=samples)
    atomic_write_json(output/'audio-index.json', dict(body=body, sha256=digest(body)), overwrite=True)


@operation('benchmark: reconstruct and archive all generated test WAVs')
def audio(plan):
    local, output = Path(plan['local']), guarded(plan['output'])
    records = ordered(list(read_jsonl(local/'units/predictions.jsonl')), plan['ids'])
    from ..vocoders.reference import ensure_reference, CHECKPOINT_SHA256, CONFIG_SHA256
    from ..vocoders.inference import vocode
    checkpoint, config = local/'official-unit/g_00500000', local/'official-unit/config.json'
    # Fetch/load the vocoder only if a part actually needs synthesis.
    samples, predictions = [], []
    for start in track(range(0, len(records), 32), 'benchmark: generate/reuse/persist WAV groups'):
        batch = records[start:start+32]
        label = f'part-{start//32:05d}'
        directory = local/'audio'/label
        identity = dict(plan=plan['identity'], inputs=digest(batch),
                        vocoder=CHECKPOINT_SHA256, config=CONFIG_SHA256)
        if not directory.exists() and (output/'audio'/(label+'.json')).is_file():
            restore_tree(output/'audio', label, directory, identity)
        guarded(directory)
        input_path = directory/'units.jsonl'
        atomic_write_jsonl(input_path, batch, resume=True)
        predicted = directory/'predictions.jsonl'
        previous = list(read_jsonl(predicted)) if predicted.is_file() else []
        ready = (len(previous) == len(batch) and
                 {r['pair_id'] for r in previous} == {r['pair_id'] for r in batch} and
                 all(r.get('status') == 'success' and Path(r['output_audio']).is_file()
                     and raw_sha(r['output_audio']) == r.get('output_sha256') for r in previous))
        if not ready:
            ensure_reference(checkpoint, config)
            vocode('unit', input_path, directory, checkpoint, config,
                   sample_rate=read(config)['sampling_rate'], device='cuda',
                   duration_prediction=True, resume=True, fatal_cuda=True)
        expected_lock = dict(backend='fairseq_unit_hifigan', checkpoint_sha256=CHECKPOINT_SHA256,
                             config_sha256=CONFIG_SHA256, input_sha256=raw_sha(input_path),
                             sample_rate=16000, duration_prediction=True, mel=None)
        if read(directory/'predictions.lock.json') != expected_lock:
            raise ValueError('saved vocoder journal does not match official model and Unit inputs')
        values = ordered(list(read_jsonl(predicted)), [r['pair_id'] for r in batch])
        # Per-part atomic ZIP+receipt precedes its listing in the global index.
        receipt = publish_tree(directory, output/'audio', label, identity)
        for row in values:
            member = Path(row['output_audio']).relative_to(directory).as_posix() if row['status'] == 'success' else None
            samples.append(dict(pair_id=row['pair_id'], status=row['status'], error=row['error'],
                                part=label, part_identity=identity, member=member,
                                archive=receipt['body']['archive'], archive_sha256=receipt['body']['sha256'],
                                output_sha256=row.get('output_sha256'), reference_text=row['reference_text']))
        predictions.extend(values)
        save_index(output, plan['identity'], samples)
    atomic_write_jsonl(local/'predictions.jsonl', predictions, overwrite=True)
    atomic_write_jsonl(output/'predictions.jsonl', predictions, overwrite=True)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['pair_id','status','archive','member','output_sha256','reference_text','error'])
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in samples)
    atomic_write_text(output/'audio-index.csv', stream.getvalue(), overwrite=True)
    print('Generated test WAVs saved:', output/'audio', flush=True)


@operation('benchmark: restore generated WAVs for full-test ASR')
def restore_audio(plan):
    local, output = Path(plan['local']), Path(plan['output'])
    document = read(output/'audio-index.json')
    body = document['body']
    if document['sha256'] != digest(body) or body['identity'] != plan['identity']:
        raise ValueError('benchmark audio index identity/checksum mismatch')
    ordered(body['samples'], plan['ids'])
    for label in track(sorted({r['part'] for r in body['samples']}), 'benchmark: verify saved WAV groups'):
        group = [r for r in body['samples'] if r['part'] == label]
        restore_tree(output/'audio', label, local/'audio'/label, group[0]['part_identity'])
    predictions = ordered(list(read_jsonl(output/'predictions.jsonl')), plan['ids'])
    common = {r['pair_id']: r for r in read_jsonl(Path(plan['common_root'])/'test.jsonl')}
    from ..drive_staging import local_path
    for row, item in zip(predictions, body['samples']):
        if row['pair_id'] != item['pair_id'] or row['reference_text'] != common[row['pair_id']]['en_text']:
            raise ValueError('saved test references/IDs changed')
        if row['status'] == 'success':
            expected = str(local/'audio'/item['part']/item['member'])
            if row['output_audio'] != expected or raw_sha(expected) != item['output_sha256']:
                raise ValueError('saved test waveform path/checksum mismatch')
        for language, field in [('ja', 'source_audio'), ('en', 'reference_audio')]:
            if raw_sha(local_path(row[field])) != common[row['pair_id']][language+'_sha256']:
                raise ValueError('source/reference test audio checksum mismatch')
    atomic_write_jsonl(local/'predictions.jsonl', predictions, overwrite=True)
    return predictions


@operation('cell 8: full-test S2UT ASR-BLEU, with durable recovery')
def evaluate(plan):
    local, output = Path(plan['local']), Path(plan['output'])
    predictions = restore_audio(plan)
    identity = dict(plan=plan['identity'], predictions=raw_sha(local/'predictions.jsonl'),
                    asr=plan['asr'], bleu_denominator=plan['bleu_denominator'],
                    normalization='nfkc_lower_english_v1', optional_metrics=False)
    label = 'asr-'+digest(identity)[:24]
    scores = local/label
    if not scores.exists() and (output/'progress'/(label+'.json')).is_file():
        restore_tree(output/'progress', label, scores, identity)
    guarded(scores)
    from ..evaluation.run import evaluate_predictions
    from ..evaluation.transcribe import EvaluationASR
    from ..progress import Progress
    # Loading ASR is explicit, pinned, and never runs in the listening action.
    with Progress('benchmark: load pinned English evaluation ASR'):
        asr = EvaluationASR(**plan['asr'], device=0)
    def persist(count, total):
        if count % 32 == 0 or count == total:
            publish_tree(scores, output/'progress', label, identity)
    metrics = evaluate_predictions(local/'predictions.jsonl', scores, transcribe=asr,
                                   resume=True, evaluation_identity=identity,
                                   include_failed_in_bleu=True, on_record=persist)
    metrics.update(split='test', selected_checkpoint_sha256=plan['checkpoint_sha256'],
                   selected_best_update=plan['best_update'], training_updates=plan['run']['updates'],
                   evaluation_scope='S2UT only, not four-system acceptance', audio_index=str(output/'audio-index.json'))
    results = guarded(output/'results')
    atomic_write_json(results/'metrics.json', metrics, overwrite=True)
    atomic_write_jsonl(results/'per_sample.jsonl', read_jsonl(scores/'per_sample.jsonl'), overwrite=True)
    stream = io.StringIO(newline='')
    fields = ['pair_id','generation_status','evaluation_status','reference_raw','hypothesis_raw','sentence_bleu','evaluation_error']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in fields} for row in read_jsonl(scores/'per_sample.jsonl'))
    atomic_write_text(results/'per_sample.csv', stream.getvalue(), overwrite=True)
    report = ('# S2UT full-test ASR-BLEU\n\n'
              f"Run: `{plan['run']['name']}`; dev-best update: {plan['best_update']}; "
              f"training reached: {plan['run']['updates']}.\n\n"
              f"Test samples: {len(predictions)}; ASR-BLEU: {metrics['bleu']:.4f}; "
              f"generation failures: {metrics['generation_failures']}; ASR failures: {metrics['asr_failures']}.\n\n"
              'All test references are retained in the BLEU denominator; failed hypotheses are empty. '
              'Official LJSpeech Unit HiFi-GAN, beam 10, pinned English Whisper. '
              'No TT2, retraining, model selection on test, BLASER or speaker-similarity evaluation. '
              'This is not a four-system comparison/acceptance report.\n\n'
              'Generated WAV ZIPs: `../audio/`; sample/ZIP index: `../audio-index.json`. '
              'Use normal cell 9 to listen without a GPU.\n')
    atomic_write_text(results/'report.md', report, overwrite=True)
    publish_tree(scores, output/'progress', label, identity)


def main():
    from ..inference_logging import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--action', required=True, choices=('units', 'audio', 'evaluate'))
    args = parser.parse_args()
    plan = read(args.plan)
    if plan.get('split') != 'test' or plan.get('version') != 1:
        raise ValueError('benchmark worker requires a full-test plan')
    {'units': units, 'audio': audio, 'evaluate': evaluate}[args.action](plan)


if __name__ == '__main__':
    main()
