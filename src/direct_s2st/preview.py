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


def vocoder_settings(mel='hifigan', unit='trained'):
    """Explicit preview identity; old plans keep their trained-only semantics."""
    from .recipes import TT2_VOCODERS, S2UT_VOCODERS
    if mel not in TT2_VOCODERS or unit not in S2UT_VOCODERS:
        raise ValueError('invalid preview vocoder mode')
    values = dict(mel=dict(mode=mel), unit=dict(mode=unit))
    if mel == 'griffin_lim':
        values['mel'].update(iterations=32, seed=1)
    if unit == 'paper':
        from .vocoders.reference import BASE, CHECKPOINT_SHA256, CONFIG_SHA256
        values['unit'].update(source=BASE, checkpoint_sha256=CHECKPOINT_SHA256,
                              config_sha256=CONFIG_SHA256, training_data='LJSpeech')
    return values


def settings(plan):
    values = plan.get('vocoders', vocoder_settings())
    if values != vocoder_settings(values['mel']['mode'], values['unit']['mode']):
        raise ValueError('preview vocoder identity changed; rerun cell 6')
    return values


def listening_labels(plan):
    values = settings(plan)
    mel = 'Griffin–Lim' if values['mel']['mode'] == 'griffin_lim' else '自前Mel HiFi-GAN'
    unit = '公式Unit HiFi-GAN（LJSpeech）' if values['unit']['mode'] == 'paper' else '自前Unit HiFi-GAN'
    return dict(original_ja='日本語原音', original_en='英語正解',
                **{'mel-reference': f'正解Mel → {mel}',
                   'unit-reference': f'正解Unit → {unit}（予測duration）',
                   'mel-translated': f'TT2翻訳 → {mel}',
                   'unit-translated': f'S2UT翻訳 → {unit}'})


@operation('preview: run one model without optimizer or fitting')
def worker(plan, kind):
    import os
    from .preview_selection import sample_ids
    root = Path(plan['local'])
    ids_path = root/'sample-ids.json'
    if plan['purpose'] != 'dev_listening_only_not_benchmark' or plan['split'] != 'dev' or sample_ids(ids_path) != plan['ids']:
        raise ValueError('not a bounded dev preview plan')
    os.environ['S2ST_INPUT_MAP'] = str(root/'input-map.json')
    if kind in ('tt2', 's2ut'):
        run = plan['runs'][kind]
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
    selected = settings(plan)[kind]
    mode = selected['mode']
    owner = 'tt2' if kind == 'mel' else 's2ut'
    run = plan['runs'].get(kind, plan['runs'][owner])
    spec_path = Path(plan['runs'][owner]['data_root'])/'mel-spec.json' if kind == 'mel' else None
    if mode == 'paper':
        from .vocoders.reference import ensure_reference, validate_reference_units
        lock = read(Path(plan['runs']['s2ut']['data_root'])/'data-lock.json')
        validate_reference_units(lock['unit_configuration'])
        checkpoint, config_path = root/'official-unit/g_00500000', root/'official-unit/config.json'
        ensure_reference(checkpoint, config_path)
        config = read(config_path)
        spec, rate = None, config['sampling_rate']
    elif mode == 'griffin_lim':
        from .vocoders import griffin_lim
        spec = read(spec_path)
        griffin_lim.validate_spec(spec)
        config_path, rate = spec_path, spec['sample_rate']
    else:
        checkpoint = select_generator(Path(run['work']))
        config_path = Path(option(run['config']['command'], '--config'))
        config = read(config_path)
        spec = config.get('mel') if kind == 'mel' else None
        rate = config['sampling_rate']
    predicted = root/('tt2' if kind == 'mel' else 's2ut')/'predictions.jsonl'
    if kind == 'mel' and spec != read(spec_path):
        # Matching required keys are checked by vocode as well. Complete spec
        # may include harmless metadata; compare the actual frontend contract.
        from .vocoders.inference import MEL_KEYS
        target = read(spec_path)
        if any(spec.get(k) != target.get(k) for k in MEL_KEYS):
            raise ValueError('trained Mel vocoder and TT2 target frontends differ')
    def reconstruct(inputs, output):
        if mode == 'griffin_lim':
            return griffin_lim.vocode(inputs, output, spec_path, iterations=selected['iterations'],
                                      seed=selected['seed'], resume=True)
        return vocode(kind, inputs, output, checkpoint, config_path,
                      sample_rate=rate, device='cuda', mel=spec, duration_prediction=True, resume=True)
    reconstruct(predicted, root/(kind+'-translated'))
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
    reconstruct(references, root/(kind+'-reference'))


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
                  vocoders=settings(plan), labels=listening_labels(plan),
                  runs={k: dict(name=v['name'], durable_updates=v['updates'], snapshot=v['snapshot'])
                        for k,v in plan['runs'].items()},
                  notes=['Not a test benchmark. No translation-quality score is computed.',
                         'mel-reference: reference English audio -> saved Mel frontend -> selected Mel backend.',
                         'unit-reference: reference reduced units -> selected Unit backend with PREDICTED durations.',
                         'mel-translated: Japanese -> TT2 -> selected Mel backend.',
                         'unit-translated: Japanese -> S2UT -> selected Unit backend.',
                         'Official Unit weights are LJSpeech-trained, not Fisher/target-corpus-trained.'],
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
