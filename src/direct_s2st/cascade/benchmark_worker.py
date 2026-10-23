"""Cascade-only test inference, with independent ASR/MT/TTS recovery units."""
import argparse
import csv
import io
import os
from pathlib import Path
import re
import time

from ..benchmark_storage import guarded, publish_tree, read, restore_tree
from ..drive_staging import local_path, raw_sha
from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_text, read_jsonl
from ..journal import Journal, digest
from ..manifests.reader import read_common_manifest
from ..progress import operation, track
from ..s2ut.benchmark_worker import ordered, save_index
from .pipeline import _fatal_accelerator_error


def runtime_identity(plan):
    receipt = read(Path(plan['output'])/'runtime.json')
    body = receipt['body']
    if (receipt['sha256'] != digest(body) or body['plan'] != plan['identity']
            or not re.fullmatch('[0-9a-f]{64}', body['environment_sha256'])
            or os.environ.get('S2ST_BENCHMARK_ENVIRONMENT_SHA256') != body['environment_sha256']):
        raise ValueError('cascade worker requires the verified evaluation dependency environment')
    return body['environment_sha256']


def common_rows(plan):
    root = Path(plan['common_root'])
    if (raw_sha(root/'test.jsonl') != plan['common_sha256']
            or raw_sha(root/'dataset-lock.json') != plan['dataset_lock_sha256']):
        raise ValueError('cascade test metadata changed')
    rows = ordered(list(read_common_manifest(root/'test.jsonl')), plan['ids'])
    if any(r['split'] != 'test' for r in rows):
        raise ValueError('cascade cannot evaluate another split')
    for row in track(rows, 'cascade: verify only test WAV hashes'):
        for language in ('ja', 'en'):
            if raw_sha(local_path(row[language+'_audio'])) != row[language+'_sha256']:
                raise ValueError('cascade source/reference WAV checksum mismatch')
    return rows


def phase_identity(plan, phase, inputs):
    return dict(plan=plan['identity'], runtime=runtime_identity(plan), phase=phase, inputs=digest(inputs))


def loader(plan, phase):
    cfg = plan['pipeline'][phase]
    if phase == 'asr':
        from .asr import WhisperASR
        return WhisperASR(model_id=cfg['model'], revision=cfg['revision'], device=0)
    if phase == 'mt':
        from .mt import NllbTranslator
        return NllbTranslator(model_id=cfg['model'], revision=cfg['revision'],
                              source_language=cfg['source_language'], target_language=cfg['target_language'])
    from .tts import QwenTTS
    return QwenTTS(model_id=cfg['model'], revision=cfg['revision'],
                   speaker=cfg['speaker'], language=cfg['language'])


def read_phase(plan, phase):
    pointer = read(Path(plan['output'])/'text'/(phase+'.json'))
    if pointer['sha256'] != digest(pointer['body']) or pointer['body']['plan'] != plan['identity']:
        raise ValueError('cascade stage pointer differs from evaluation')
    body = pointer['body']
    if body['identity']['runtime'] != runtime_identity(plan):
        raise ValueError('cascade stage environment changed')
    directory = Path(plan['local'])/'text'/body['label']
    if not directory.exists():
        restore_tree(Path(plan['output'])/'progress', body['label'], directory, body['identity'])
    path = directory/'predictions.jsonl'
    if raw_sha(path) != body['predictions_sha256']:
        raise ValueError('cascade completed text journal changed; rerun normal cell 7')
    return ordered(list(read_jsonl(path)), plan['ids'])


@operation('cascade: resumable Japanese ASR / English translation')
def text_phase(plan, phase):
    if phase not in ('asr', 'mt'):
        raise ValueError('invalid cascade text stage')
    inputs = common_rows(plan) if phase == 'asr' else read_phase(plan, 'asr')
    identity = phase_identity(plan, phase, inputs)
    label = phase+'-'+digest(identity)[:24]
    local, output = Path(plan['local']), Path(plan['output'])
    directory = local/'text'/label
    if not directory.exists() and (output/'progress'/(label+'.json')).is_file():
        restore_tree(output/'progress', label, directory, identity)
    journal = Journal(directory/'predictions.jsonl', identity, resume=True)
    field, timing = ('asr_ja_text', 'asr_seconds') if phase == 'asr' else ('mt_en_text', 'mt_seconds')
    model = None
    results = []
    for index, row in enumerate(track(inputs, f'cascade: {phase} compute/reuse')):
        saved = journal.rows.get(row['pair_id'])
        if saved and saved['status'] == 'success':
            if not isinstance(saved.get(field), str) or not saved[field].strip():
                raise ValueError('saved cascade text is empty/corrupt')
            result = saved
        else:
            result = dict(row, status='failed', error=None, **{field:None, timing:0.0})
            if phase == 'mt' and row['status'] != 'success':
                result['error'] = 'upstream Japanese ASR failed: '+str(row['error'])
            else:
                if model is None:
                    model = loader(plan, phase)
                started = time.perf_counter()
                try:
                    argument = local_path(row['ja_audio']) if phase == 'asr' else row['asr_ja_text']
                    text = model(argument)
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError('empty cascade '+phase+' text')
                    result.update({field:text, 'status':'success'})
                except Exception as error:
                    if _fatal_accelerator_error(error):
                        raise
                    result['error'] = f'{type(error).__name__}: {error}'
                result[timing] = time.perf_counter()-started
            journal.record(result)
        results.append(result)
        if (index+1) % 32 == 0:
            publish_tree(directory, output/'progress', label, identity)
    publish_tree(directory, output/'progress', label, identity)
    body = dict(plan=plan['identity'], label=label, identity=identity,
                predictions_sha256=raw_sha(journal.path))
    atomic_write_json(output/'text'/(phase+'.json'), dict(body=body, sha256=digest(body)), overwrite=True)
    return results


def valid_saved(row):
    if row.get('status') != 'success':
        return False
    path = Path(row['output_audio'])
    if not path.is_file() or raw_sha(path) != row['output_sha256']:
        raise ValueError('saved cascade WAV changed; retain files and review before retrying')
    return True


@operation('cascade: English TTS and durable WAV ZIPs')
def synthesize(plan):
    inputs = read_phase(plan, 'mt')
    local, output = Path(plan['local']), Path(plan['output'])
    model, samples, predictions = None, [], []
    for start in track(range(0, len(inputs), 32), 'cascade: synthesize/reuse/persist WAV groups'):
        batch = inputs[start:start+32]
        identity = phase_identity(plan, 'tts', batch)
        label = f'part-{start//32:05d}-'+digest(identity)[:8]
        directory = local/'audio'/label
        if not directory.exists() and (output/'audio'/(label+'.json')).is_file():
            restore_tree(output/'audio', label, directory, identity)
        journal = Journal(directory/'predictions.jsonl', identity, resume=True)
        values = []
        for row in track(batch, 'cascade: TTS generate/reuse'):
            saved = journal.rows.get(row['pair_id'])
            if saved and valid_saved(saved):
                result = saved
            else:
                result = dict(pair_id=row['pair_id'], system_id='cascade', run_id=plan['run']['name'],
                              split='test', source_audio=row['ja_audio'], reference_audio=row['en_audio'],
                              reference_text=row['en_text'], asr_ja_text=row.get('asr_ja_text'),
                              mt_en_text=row.get('mt_en_text'), asr_seconds=row.get('asr_seconds', 0),
                              mt_seconds=row.get('mt_seconds', 0), tts_seconds=0.0, output_audio=None,
                              output_duration=None, real_time_factor=None, status='failed', error=None)
                if row['status'] != 'success':
                    result['error'] = 'upstream cascade text failed: '+str(row['error'])
                else:
                    if model is None:
                        model = loader(plan, 'tts')
                    path = directory/'audio'/(row['pair_id']+'.wav')
                    started = time.perf_counter()
                    try:
                        model(row['mt_en_text'], path)
                        import numpy as np
                        import soundfile as sf
                        waveform, rate = sf.read(path, always_2d=True)
                        duration = len(waveform)/rate
                        if (waveform.shape[1] != 1 or not 0 < duration <= 120
                                or not np.isfinite(waveform).all() or not np.any(waveform)):
                            raise ValueError('invalid/empty/silent or over-120s cascade TTS WAV')
                        result.update(output_audio=str(path), output_sha256=raw_sha(path),
                                      output_duration=duration, status='success')
                    except Exception as error:
                        if _fatal_accelerator_error(error):
                            raise
                        result['error'] = f'{type(error).__name__}: {error}'
                    result['tts_seconds'] = time.perf_counter()-started
                result['inference_seconds'] = sum(result[key] for key in ('asr_seconds','mt_seconds','tts_seconds'))
                if result['output_duration']:
                    result['real_time_factor'] = result['inference_seconds']/result['output_duration']
                result['timing_scope'] = 'per-sample ASR+MT+TTS wall time; excludes model loading and backup'
                journal.record(result)
            values.append(result)
        receipt = publish_tree(directory, output/'audio', label, identity)
        for row in values:
            member = Path(row['output_audio']).relative_to(directory).as_posix() if row['status'] == 'success' else None
            samples.append(dict(pair_id=row['pair_id'], status=row['status'], error=row['error'], part=label,
                                part_identity=identity, member=member, archive=receipt['body']['archive'],
                                archive_sha256=receipt['body']['sha256'], output_sha256=row.get('output_sha256'),
                                reference_text=row['reference_text']))
        predictions.extend(values)
        save_index(output, plan['identity'], samples)
    atomic_write_jsonl(output/'predictions.jsonl', predictions, overwrite=True)
    atomic_write_jsonl(local/'predictions.jsonl', predictions, overwrite=True)
    stream = io.StringIO(newline='')
    fields = ['pair_id','status','archive','member','output_sha256','reference_text','error']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in fields} for row in samples)
    atomic_write_text(output/'audio-index.csv', stream.getvalue(), overwrite=True)
    return predictions


def main():
    from ..inference_logging import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--action', required=True, choices=('asr', 'mt', 'tts', 'evaluate'))
    args = parser.parse_args()
    plan = read(args.plan)
    if plan.get('system_id') != 'cascade' or plan.get('split') != 'test' or plan.get('version') != 1:
        raise ValueError('cascade worker requires its own full-test plan')
    runtime_identity(plan)
    if args.action in ('asr', 'mt'):
        text_phase(plan, args.action)
    elif args.action == 'tts':
        synthesize(plan)
    else:
        from ..s2ut.benchmark_worker import evaluate
        evaluate(plan)


if __name__ == '__main__':
    main()
