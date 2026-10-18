"""Small dev listening preview, deliberately separate from benchmark results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .drive_staging import copy_verified, raw_sha
from .io import atomic_write_json, read_jsonl
from .progress import operation, track
from .recipes import option


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@operation('preview: run one model without optimizer or fitting')
def worker(plan, kind):
    import os
    from .preview_selection import sample_ids
    root = Path(plan['local'])
    ids_path = root/'sample-ids.json'
    if plan['purpose'] != 'dev_listening_only_not_benchmark' or plan['split'] != 'dev' or sample_ids(ids_path) != plan['ids']:
        raise ValueError('not a bounded dev preview plan')
    os.environ['S2ST_INPUT_MAP'] = str(root/'input-map.json')
    run = plan['runs'][kind]
    if kind in ('tt2', 's2ut'):
        module = 'translatotron2.infer' if kind == 'tt2' else 's2ut.fairseq_infer'
        command = [sys.executable, '-m', 'direct_s2st.'+module,
                   '--data-root', run['data_root'], '--run-root', run['work'],
                   '--common-root', plan['common_root'], '--predictions', str(root/kind/'predictions.jsonl'),
                   '--split', 'dev', '--device', 'cuda', '--sample-ids', str(ids_path), '--resume']
        subprocess.run(command, check=True)
        return
    from .vocoders.inference import vocode
    from .vocoders.monitor import select_generator
    from .journal import Journal
    checkpoint = select_generator(Path(run['work']))
    config_path = Path(option(run['config']['command'], '--config'))
    config = read(config_path)
    spec = config.get('mel') if kind == 'mel' else None
    rate = config['sampling_rate']
    predicted = root/('tt2' if kind == 'mel' else 's2ut')/'predictions.jsonl'
    if kind == 'mel' and spec != read(Path(plan['runs']['tt2']['data_root'])/'mel-spec.json'):
        # Matching required keys are checked by vocode as well. Complete spec
        # may include harmless metadata; compare the actual frontend contract.
        from .vocoders.inference import MEL_KEYS
        target = read(Path(plan['runs']['tt2']['data_root'])/'mel-spec.json')
        if any(spec.get(k) != target.get(k) for k in MEL_KEYS):
            raise ValueError('trained Mel vocoder and TT2 target frontends differ')
    vocode(kind, predicted, root/(kind+'-translated'), checkpoint, config_path,
           sample_rate=rate, device='cuda', mel=spec, duration_prediction=True, resume=True)
    references = root/(kind+'-reference-input.jsonl')
    identity = dict(purpose='reference_reconstruction_not_translation', ids=plan['ids'],
                    config=raw_sha(config_path), common=plan['common_sha256'])
    if kind == 'unit':
        from .s2ut.multitask import read_tsv
        from .s2ut.reduce_units import validate_units
        table = Path(plan['runs']['s2ut']['data_root'])/'dev.tsv'
        identity['prepared_units_sha256'] = raw_sha(table)
        units = read_tsv(table, ('id', 'src_audio', 'src_n_frames', 'tgt_audio', 'tgt_n_frames'))
    with_reference = Journal(references, identity, resume=True)
    for row in track(plan['rows'], f'preview: {kind} reference reconstruction inputs'):
        pair_id = row['pair_id']
        previous = with_reference.rows.get(pair_id)
        if previous and previous['status'] == 'success':
            if kind == 'unit' or (Path(previous['mel_path']).is_file() and raw_sha(previous['mel_path']) == previous['mel_sha256']):
                continue
        record = dict(pair_id=pair_id, split='dev', run_id=run['name'],
                      system_id='translatotron2' if kind == 'mel' else 's2ut',
                      reference_text=row['en_text'], inference_seconds=0,
                      reference_reconstruction=True, duration_mode='predicted' if kind == 'unit' else 'reference_mel')
        try:
            if kind == 'unit':
                record['units'] = validate_units([int(v) for v in units[pair_id]['tgt_audio'].split()], clusters=100)
            else:
                import os
                import tempfile
                from .translatotron2.prepare_fairseq import _extract_logmel_official
                feature = root/'reference-mels'/(pair_id+'.npy')
                feature.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(dir=feature.parent, suffix='.npy')
                os.close(fd)
                try:
                    _extract_logmel_official(Path(row['en_local']), Path(temporary), spec)
                    import numpy as np
                    values = np.load(temporary, allow_pickle=False)
                    if values.ndim != 2 or values.shape[1] != spec['n_mels'] or not values.size or not np.isfinite(values).all():
                        raise ValueError('invalid reference Mel')
                    os.replace(temporary, feature)
                finally:
                    Path(temporary).unlink(missing_ok=True)
                record.update(mel_path=str(feature), mel_sha256=raw_sha(feature))
            record.update(status='success', error=None)
        except Exception as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
        with_reference.record(record)
    vocode(kind, references, root/(kind+'-reference'), checkpoint, config_path,
           sample_rate=rate, device='cuda', mel=spec, duration_prediction=True, resume=True)


@operation('preview: publish small verified listening results to Drive')
def publish(plan):
    root, output = Path(plan['local']), Path(plan['output'])
    entries = {r['pair_id']: dict(pair_id=r['pair_id'], ja_text=r['ja_text'], en_text=r['en_text'], audio={})
               for r in plan['rows']}
    def audio(pair_id, label, source, expected):
        target = output/'audio'/pair_id/(label+'.wav')
        if not target.is_file():
            copy_verified(Path(source), target, expected)
        if raw_sha(target) != expected:
            raise ValueError('published preview changed; original outputs preserved')
        entries[pair_id]['audio'][label] = dict(status='success', path=str(target), sha256=expected)
    for row in track(plan['rows'], 'preview: publish selected originals'):
        for language in ('ja', 'en'):
            audio(row['pair_id'], 'original_'+language, row[language+'_local'], row[language+'_sha256'])
    for label in ('mel-reference', 'unit-reference', 'mel-translated', 'unit-translated'):
        records = list(read_jsonl(root/label/'predictions.jsonl'))
        if {r['pair_id'] for r in records} != set(entries) or len(records) != len(entries):
            raise ValueError('preview outputs do not match selected IDs')
        for row in track(records, 'preview: publish '+label):
            pair_id = row['pair_id']
            if row['status'] == 'success':
                audio(pair_id, label, row['output_audio'], row['output_sha256'])
            else:
                entries[pair_id]['audio'][label] = dict(status='failed', error=row.get('error'))
    result = dict(purpose=plan['purpose'], split='dev', execution_revision=plan['execution_revision'],
                  runs={k: dict(name=v['name'], durable_updates=v['updates'], snapshot=v['snapshot'])
                        for k,v in plan['runs'].items()},
                  notes=['Not a test benchmark. No translation-quality score is computed.',
                         'mel-reference: reference English audio -> saved Mel frontend -> trained Mel HiFi-GAN.',
                         'unit-reference: reference reduced units -> trained Unit HiFi-GAN with PREDICTED durations.',
                         'mel-translated: Japanese -> TT2 -> trained Mel HiFi-GAN (not the formal Griffin-Lim mode).',
                         'unit-translated: Japanese -> S2UT -> trained Unit HiFi-GAN.'],
                  samples=[entries[i] for i in plan['ids']])
    atomic_write_json(output/'listening.json', result, overwrite=True)
    if read(output/'listening.json') != result:
        raise OSError('preview report readback mismatch')
    print('Preview saved:', output/'listening.json', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--kind', choices=('tt2', 's2ut', 'unit', 'mel'), required=True)
    args = parser.parse_args()
    worker(read(args.plan), args.kind)


if __name__ == '__main__':
    main()
