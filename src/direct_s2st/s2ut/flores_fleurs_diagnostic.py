"""8E: frozen 8C natural inputs, existing dev-best S2UT, separate results.

Nothing in the corpus, training metadata or 8C namespace is written here.
The English reference is attached only AFTER the fairseq subprocess exits.
"""
import csv
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from ..benchmark_storage import guarded, publish_tree, read, restore_tree
from ..cascade.flores_fleurs_data import sha_text
from ..cascade.flores_fleurs_diagnostic import PURPOSE as CASCADE_PURPOSE, inside, audio_audit
from ..drive_staging import copy_verified, raw_sha
from ..evaluation.bleu import normalize_english, sentence_bleu
from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_text, read_jsonl
from ..journal import digest
from ..progress import operation, track

VERSION = 1
PURPOSE = 'external_flores_fleurs_s2ut_natural_not_formal_benchmark'
NORMALIZATION = 'nfkc_lower_english_v1'
ASR = dict(model_id='openai/whisper-large-v3-turbo',
           revision='60be3615a4d667e1258e8ad29130467587c489aa')
SAMPLE_FIELDS = ('sample_id', 'flores_id', 'flores_split', 'fleurs_id', 'fleurs_split',
                 'ja_text', 'en_reference', 'ja_text_sha256', 'en_text_sha256')


def indexed(rows, field='sample_id'):
    values = {}
    for row in rows:
        key = row.get(field)
        if not isinstance(key, str) or not key or key in values:
            raise ValueError('missing/duplicate sample ID: '+str(key))
        values[key] = row
    if not values:
        raise ValueError('empty sample set')
    return values


def choose_cascade(persistent, requested=''):
    base = Path(persistent)/'diagnostics/flores-fleurs-cascade'
    if requested:
        root = Path(requested) if Path(requested).is_absolute() else base/requested
        inside(base, root)
        if root.parent != base or not (root/'metrics.json').is_file():
            raise ValueError('8E requires a completed saved 8C identity (not a benchmark/8B folder)')
        return root
    # Prefer an existing complete 300-item experiment. Never load/select new data.
    candidates = []
    for path in track(sorted(base.glob('*/metrics.json')), '8E: inspect saved 8C evaluations'):
        doc = read(path)
        if doc.get('purpose') != CASCADE_PURPOSE:
            continue
        n = doc.get('metrics', {}).get('samples', 0)
        if n:
            candidates.append((n == 300, path.stat().st_mtime_ns, path.parent.name, path.parent))
    if not candidates:
        raise ValueError('保存済み8C結果がありません。同じOUTPUT_BASE・MODE・EXPERIMENTを確認してください')
    return max(candidates)[-1]


@operation('8E: verify frozen 8C IDs, references and natural WAV evidence')
def source_context(root, *, verify_audio=True):
    root = Path(root).absolute()
    inside(root, root)
    plan = read(root/'identity.json')
    identity = digest(plan)
    if plan.get('purpose') != CASCADE_PURPOSE or root.name != identity[:24]:
        raise ValueError('8C identity/path mismatch')
    audit = read(root/'dataset-audit.json')
    metrics = read(root/'metrics.json')
    if (audit != plan['dataset'] or metrics.get('identity_sha256') != identity
            or metrics.get('purpose') != CASCADE_PURPOSE):
        raise ValueError('8C saved dataset/metrics identity mismatch')
    selected = indexed(plan['selected'])
    audited = indexed(audit['selected_samples'])
    rows = indexed(list(read_jsonl(root/'per_sample.jsonl')))
    embedded = indexed(metrics['per_sample'])
    if not (selected.keys() == audited.keys() == rows.keys() == embedded.keys()
            and metrics['metrics']['samples'] == len(selected) == audit['selected']):
        raise ValueError('missing/extra 8C samples; exact sample set required')
    asr = plan['context']['evaluation_asr']
    asr = dict(model_id=asr.get('model_id', asr.get('model')), revision=asr['revision'])
    if asr != ASR:
        raise ValueError('8C evaluation ASR differs from existing S2UT benchmark; no silent substitution')
    # This also binds the actual Whisper generation config/dtype, not just its Hub ID.
    asr_audit = read(root/'execution-audit/evaluation_asr.json')
    if asr_audit.get('adapter') != 'EvaluationASR English transcribe':
        raise ValueError('unsupported 8C evaluation ASR adapter')
    code = Path(__file__).resolve().parents[1]
    # 8C recorded Linux source-byte hashes. Local Windows verification may have
    # checkout CRLF; LF normalization here is solely for source-code comparison.
    for relative in ('cascade/asr.py','cascade/whisper.py','evaluation/transcribe.py','evaluation/bleu.py'):
        value = (code/relative).read_bytes()
        candidates = {hashlib.sha256(v).hexdigest() for v in (value,value.replace(b'\r\n',b'\n'))}
        if plan.get('implementation',{}).get(relative) not in candidates:
            raise ValueError('8C ASR/normalization implementation differs: '+relative)
    datasets = audit.get('datasets',[])
    if ({d.get('repository') for d in datasets} != {'google/fleurs','openlanguagedata/flores_plus'}
            or len(datasets)!=2 or any(not re.fullmatch('[0-9a-f]{40}',str(d.get('revision',''))) for d in datasets)):
        raise ValueError('8C source dataset revisions must be immutable and complete')
    samples = []
    files = {str(root/name): raw_sha(root/name) for name in
             ('identity.json', 'dataset-audit.json', 'metrics.json', 'per_sample.jsonl',
              'execution-audit/evaluation_asr.json')}
    for key in track(sorted(selected), '8E: validate identical natural sample set'):
        item, row = selected[key], rows[key]
        if item != audited[key] or row != embedded[key]:
            raise ValueError('8C per-sample report differs from saved identity/metrics')
        if any(row.get(k) != item.get(k) for k in SAMPLE_FIELDS):
            raise ValueError('8C sample/reference text mismatch: '+key)
        if not all(isinstance(item.get(k), str) and item[k].strip() for k in SAMPLE_FIELDS):
            raise ValueError('empty/invalid saved 8C sample fields')
        if (sha_text(item['ja_text']) != item['ja_text_sha256']
                or sha_text(item['en_reference']) != item['en_text_sha256']):
            raise ValueError('8C source/reference text SHA mismatch')
        if (item['sample_id'] != 'flores:'+item['flores_split']+':'+item['flores_id']
                or item['flores_split'] not in ('dev', 'devtest')
                or item['fleurs_split'] not in ('validation', 'test')):
            raise ValueError('invalid canonical FLORES/FLEURS identifier')
        # The stage receipt binds the recording locator, dataset revisions and WAV bytes.
        pointer = root/'stage-results'/(digest([key, 'natural', 'natural_audio'])+'.json')
        receipt = read(pointer)
        body = receipt['body']
        expected = dict(run=identity, sample_id=key, condition='natural', stage='natural_audio',
                        inputs=dict(locator=item['locator'], dataset=plan['dataset']['datasets'],
                                    text_sha=item['ja_text_sha256']))
        if (receipt['sha256'] != digest(body) or body.get('request') != expected
                or body.get('status') != 'success' or len(body.get('audio', [])) != 1):
            raise ValueError('missing/invalid saved 8C natural audio receipt')
        wav = body['audio'][0]
        path = inside(root, Path(wav['path']))
        if (row.get('natural_input_audio_status') != 'success'
                or row.get('natural_input_audio_sha256') != wav['sha256']
                or row.get('natural_audio_path') != str(path)
                or row.get('natural_input_audio_path') != str(path)
                or wav['sample_rate'] != 16000):
            raise ValueError('8C natural WAV metadata mismatch; no resampling/selection fallback')
        if verify_audio:
            actual = audio_audit(path)
            if any(actual[k] != wav[k] for k in actual):
                raise ValueError('8C natural audio SHA/format mismatch: '+key)
        files[str(pointer)] = raw_sha(pointer)
        samples.append({**{k: item[k] for k in SAMPLE_FIELDS},
                        'pair_id': 'ff_'+sha_text(key)[:32],
                        'source_audio': str(path), 'source_sha256': wav['sha256'],
                        'source_sample_rate': 16000, 'source_duration': wav['duration']})
    if len({r['pair_id'] for r in samples}) != len(samples):
        raise ValueError('external numeric mapping collision')
    baseline = metrics['metrics']['natural_final_ASR_BLEU']
    if not isinstance(baseline, (int, float)) or not math.isfinite(baseline):
        raise ValueError('8C saved Natural ASR-BLEU missing/nonfinite')
    return dict(root=str(root), identity=identity, samples=samples, files=files,
                dataset_revisions=audit['datasets'], asr=asr, asr_audit=asr_audit,
                environment=plan['environment'], cascade_natural_bleu=baseline,
                normalization=NORMALIZATION, selection='exact_saved_8c_natural_set_no_reselection')


def verify_source(ctx, *, audio=False):
    for path, sha in track(ctx['files'].items(), '8E: verify unchanged 8C artifacts'):
        if raw_sha(path) != sha:
            raise ValueError('8C source artifact changed; original 8E identity retained: '+path)
    if audio:
        for row in ctx['samples']:
            if raw_sha(row['source_audio']) != row['source_sha256']:
                raise ValueError('8C natural audio SHA mismatch')


def implementation_files():
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[1]
    implementations = [Path(__file__), package/'s2ut/external_generate.py',package/'s2ut/benchmark_worker.py',
        package/'s2ut/fairseq_infer.py',package/'s2ut/staged_generate.py',package/'benchmark_storage.py',
        package/'evaluation/run.py',package/'evaluation/transcribe.py',package/'evaluation/bleu.py',
        package/'cascade/whisper.py',package/'cascade/asr.py',package/'vocoders/inference.py',
        package/'vocoders/reference.py',repo/'scripts/colab/s2ut_flores_fleurs_diagnostic.py',
        repo/'scripts/colab/flores_fleurs_runtime.py',repo/'requirements/diagnostic-flores.txt',
        package/'s2ut/flores_fleurs_resume.py']
    return {str(p):raw_sha(p) for p in implementations}


def prepare_plan(persistent, local, ctx, run, metadata, checkpoint, best_update, revision, *, resume=False):
    from ..vocoders.reference import BASE, CHECKPOINT_SHA256, CONFIG_SHA256
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('8E requires immutable code revision')
    if type(best_update) is not int or not 1 <= best_update <= run['updates']:
        raise ValueError('invalid saved dev-best update')
    files = implementation_files()
    body = dict(version=VERSION, purpose=PURPOSE, source=ctx, run=run, metadata=metadata,
                workspace_base=str(Path(local).absolute()), persistent_base=str(Path(persistent).absolute()),
                execution_files=files,
                checkpoint=str(checkpoint), checkpoint_sha256=raw_sha(checkpoint), best_update=best_update,
                execution_revision=revision, beam=10, asr=ctx['asr'], normalization=NORMALIZATION,
                ids=[r['pair_id'] for r in ctx['samples']],
                vocoder=dict(url=BASE, checkpoint='g_00500000', checkpoint_sha256=CHECKPOINT_SHA256,
                             config_sha256=CONFIG_SHA256, duration_prediction=True),
                bleu_denominator='all_samples_failed_as_empty')
    token = digest(body)[:24]
    base = Path(persistent).absolute()/'diagnostics/flores-fleurs-s2ut'
    output = base/token
    # Only dedicated local inputs, never prepared data/common or the training run.
    owned = Path(local).absolute()/token
    protected = [Path(persistent).absolute(), Path(ctx['root']).absolute()]
    if any(owned.resolve().is_relative_to(p.resolve()) for p in protected):
        raise ValueError('8E local workspace must be outside persistent experiments')
    guarded(output)
    guarded(owned)
    if resume:
        from .flores_fleurs_resume import adopt
        previous = adopt(body, base, files)
        if previous is not None:
            verify_plan(previous)
            return previous
    plan = dict(body, identity=digest(body), output=str(output), local=str(owned))
    atomic_write_json(output/'plan.json', plan, resume=True)
    return plan


def verify_plan(plan):
    body = {k:v for k,v in plan.items() if k not in ('identity', 'local', 'output')}
    output = Path(plan['output'])
    if (digest(body) != plan['identity'] or plan['purpose'] != PURPOSE or plan['version'] != VERSION
            or output.name != plan['identity'][:24] or output.parent.name != 'flores-fleurs-s2ut'
            or output.parent.parent.name != 'diagnostics'):
        raise ValueError('8E plan identity/owned path mismatch')
    if (Path(plan['local']) != Path(plan['workspace_base'])/plan['identity'][:24]
            or output != Path(plan['persistent_base'])/'diagnostics/flores-fleurs-s2ut'/plan['identity'][:24]
            or Path(plan['local']).resolve().is_relative_to(Path(plan['persistent_base']).resolve())):
        raise ValueError('8E owned local/output paths changed')
    inside(output, output)
    verify_source(plan['source'])
    for name, sha in plan['metadata'].items():
        if raw_sha(name) != sha:
            raise ValueError('checkpoint-bound inference metadata changed: '+name)
    current_files = implementation_files()
    if plan['execution_files'] != current_files:
        from .flores_fleurs_resume import verify_execution
        verify_execution(plan, current_files)
    if raw_sha(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('S2UT dev-best checkpoint SHA mismatch')


def copy_asset(source, target, sha):
    target = Path(target)
    guarded(target.parent)
    if target.exists():
        if not target.is_file() or target.is_symlink() or raw_sha(target) != sha:
            raise ValueError('different local external prepared asset retained: '+str(target))
        return
    copy_verified(source, target, sha)


@operation('8E: stage identical natural WAVs and reference-free external fairseq data')
def prepare_external(plan):
    import yaml
    verify_plan(plan)
    local = Path(plan['local'])
    directory = guarded(local/'prepared')
    original = Path(plan['run']['data_root'])
    metadata = plan['metadata']
    # Inference config must be byte-identical. Unbound assets/absolute feature
    # paths are rejected rather than silently normalizing features differently.
    config = read_yaml(original/'config.yaml')
    if config.get('target_speaker_embed') or config.get('prepend_tgt_lang_tag_as_bos'):
        raise ValueError('8E does not approximate speaker-conditioned/language-tagged S2UT tasks')
    for p in (original/'config.yaml', original/'dict.txt'):
        if str(p) not in metadata:
            raise ValueError('unbound saved S2UT config/dictionary')
        copy_asset(p, directory/p.name, metadata[str(p)])
    # Preserve all verified side assets (e.g. CMVN stats); never copy train/test TSVs.
    for name, sha in metadata.items():
        p = Path(name)
        rel = p.relative_to(original)
        if p.suffix in ('.npy', '.npz', '.model', '.vocab'):
            copy_asset(p, inside(directory,directory/rel), sha)
    for value in path_values(config):
        if not value:
            continue
        path = Path(value)
        if path.is_absolute():
            if str(path) not in metadata:
                raise ValueError('unbound absolute feature asset in S2UT config')
        else:
            source = original/path
            if str(source) not in metadata:
                raise ValueError('unbound relative feature asset in S2UT config')
            copy_asset(source,inside(directory,directory/path),metadata[str(source)])
    tasks = read_yaml(original/'config_multitask.yaml')
    if set(tasks) != {'source_letter','target_letter','decoder_target_ctc'}:
        raise ValueError('8E requires the saved S2UT three-task configuration')
    for task, cfg in tasks.items():
        source = Path(cfg['dict'])
        if str(source) not in metadata:
            raise ValueError('unbound saved auxiliary dictionary')
        copy_asset(source, directory/task/'dict.txt', metadata[str(source)])
        # Only file locations change, never attachment/loss/model settings.
        cfg['data'], cfg['dict'] = (directory/task).as_posix(), (directory/task/'dict.txt').as_posix()
        atomic_write_text(directory/task/'test.tsv', 'id\ttgt_text\n'+''.join(
            r['pair_id']+'\t<unk>\n' for r in plan['source']['samples']), resume=True)
    atomic_write_text(directory/'config_multitask.yaml', yaml.safe_dump(tasks), resume=True)
    lines = ['id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n']
    for row in track(plan['source']['samples'], '8E: copy/reuse verified natural WAVs'):
        destination = local/'inputs'/(row['pair_id']+'.wav')
        if destination.is_file():
            if raw_sha(destination) != row['source_sha256']:
                raise ValueError('different external input WAV retained')
        else:
            copy_verified(row['source_audio'], destination, row['source_sha256'])
        info = audio_audit(destination)
        if info['sample_rate'] != 16000 or info['sha256'] != row['source_sha256']:
            raise ValueError('external WAV SHA/sample rate mismatch')
        frames = int(round(info['duration']*16000))//160
        if frames < 1:
            raise ValueError('empty external source frames')
        # Fixed dummy Unit, NOT FLORES English/reference units. Fairseq needs
        # these columns for collation, not decoding (prefix_size remains 0).
        lines.append(f"{row['pair_id']}\t{destination.as_posix()}\t{frames}\t0\t1\n")
    atomic_write_text(directory/'test.tsv', ''.join(lines), resume=True)
    receipt = dict(plan=plan['identity'], files={str(p.relative_to(local)):raw_sha(p)
                   for p in sorted(directory.rglob('*')) if p.is_file()},
                   inputs={r['pair_id']:r['source_sha256'] for r in plan['source']['samples']},
                   references_in_generation_input=False, dummy_target_unit=0, dummy_auxiliary='<unk>')
    atomic_write_json(Path(plan['output'])/'external-input-audit.json', receipt, resume=True)
    return directory


def read_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def path_values(config):
    """Check feature-file paths without interpreting transform names as files."""
    if isinstance(config, dict):
        for key, value in config.items():
            if key.endswith(('_path', '_file', '_filename', '_root')) and isinstance(value, str):
                yield value
            else:
                yield from path_values(value)


@operation('8E: verify prepared external inputs before decoding')
def verify_external(plan):
    local, output = Path(plan['local']), Path(plan['output'])
    audit = read(output/'external-input-audit.json')
    expected = {r['pair_id']:r['source_sha256'] for r in plan['source']['samples']}
    if (audit.get('plan') != plan['identity'] or audit.get('inputs') != expected
            or audit.get('references_in_generation_input') is not False
            or audit.get('dummy_target_unit') != 0 or audit.get('dummy_auxiliary') != '<unk>'):
        raise ValueError('external input audit differs from the selected plan')
    for name, sha in track(audit['files'].items(), '8E: verify external prepared files'):
        path = inside(local/'prepared', local/name)
        if raw_sha(path) != sha:
            raise ValueError('external prepared file SHA mismatch: '+name)
    from .multitask import read_tsv
    rows = read_tsv(local/'prepared/test.tsv',('id','src_audio','src_n_frames','tgt_audio','tgt_n_frames'))
    if set(rows) != set(expected):
        raise ValueError('external prepared sample set changed')
    for key, sha in track(expected.items(), '8E: verify decoded source WAV hashes'):
        path = local/'inputs'/(key+'.wav')
        if (rows[key]['src_audio'] != path.as_posix() or rows[key]['tgt_audio'] != '0'
                or rows[key]['tgt_n_frames'] != '1' or raw_sha(path) != sha):
            raise ValueError('external source WAV/placeholder mismatch: '+key)


def parse_units(path, ids):
    from .reduce_units import validate_units
    result = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.startswith('D-'):
            continue
        fields = line.split('\t')
        if len(fields) != 3:
            raise ValueError('malformed fairseq D-line; not a sample-level generation failure')
        index = int(fields[0][2:])
        if not 0 <= index < len(ids) or ids[index] in result:
            raise ValueError('duplicate/out-of-range external fairseq ID')
        try:
            units = validate_units([int(x) for x in fields[2].split()], clusters=100)
            result[ids[index]] = dict(status='success', error=None, units=units)
        except ValueError as error:
            result[ids[index]] = dict(status='failed', error=str(error), units=[])
    return {key:result.get(key, dict(status='failed', error='no decoded hypothesis for external sample', units=[]))
            for key in ids}


def shard_records(path, ids):
    decoded = parse_units(path,ids)
    present = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith(('S-','D-')):
            index = int(line.split('\t')[0][2:])
            if not 0 <= index < len(ids):
                raise ValueError('out-of-range external fairseq source ID')
            present[ids[index]] = decoded[ids[index]]
    return present


@operation('8E: generate/reuse durable Unit shards')
def units(plan, *, runner=subprocess.run):
    from .fairseq_infer import generation_command
    verify_plan(plan)
    verify_external(plan)
    local, output = Path(plan['local']), Path(plan['output'])
    records = []
    directory = local/'prepared'
    ids = plan['ids']
    # Use the existing numeric-ID/sharded fairseq path and generation settings.
    count = max(1, math.ceil(len(ids)/32))
    for shard in track(range(count), '8E: generate/restore Unit shards'):
        label = f'part-{shard:05d}'
        target = local/'generation'/label
        identity = dict(plan=plan['identity'], shard=shard, num_shards=count)
        if (output/'generation'/(label+'.json')).is_file():
            restore_tree(output/'generation', label, target, identity)
        saved = target/'records.json'
        if saved.is_file():
            doc = read(saved)
            if (doc.get('identity') != identity
                    or raw_sha(target/'generate-test.txt') != doc.get('log_sha256')
                    or doc.get('records') != shard_records(target/'generate-test.txt',ids)
                    or not isinstance(doc.get('seconds'),(int,float))
                    or not math.isfinite(doc['seconds']) or doc['seconds'] < 0):
                raise ValueError('external Unit shard identity mismatch')
        else:
            command = generation_command(directory, plan['checkpoint'], target, 'test', plan['beam'], 'cuda')
            command[2] = 'direct_s2st.s2ut.external_generate'
            command += ['--distributed-world-size', str(count), '--distributed-rank', str(shard),
                        '--prefix-size', '0']
            start = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                runner(command, check=True, stdout=sys.stderr, stderr=sys.stderr)
            # Pinned fairseq shards its length-sorted iterator, not contiguous ID ranges.
            present = shard_records(target/'generate-test.txt',ids)
            doc = dict(identity=identity, records=present, seconds=time.perf_counter()-start,
                       log_sha256=raw_sha(target/'generate-test.txt'))
            atomic_write_json(saved, doc, resume=True)
        publish_tree(target, output/'generation', label, identity)
        for key, value in doc['records'].items():
            if any(r['pair_id'] == key for r in records):
                raise ValueError('duplicate external ID across shards')
            records.append(dict(pair_id=key, **value, inference_seconds=doc['seconds']/max(1,len(doc['records']))))
    by_id = {r['pair_id']:r for r in records}
    common = {r['pair_id']:r for r in plan['source']['samples']}
    if by_id.keys()-common.keys():
        raise ValueError('unexpected generated external sample')
    predictions = []
    for key in ids:
        value = by_id.get(key, dict(pair_id=key,status='failed',error='fairseq filtered/missing sample',
                                   units=[],inference_seconds=0.0))
        row = common[key]
        predictions.append(dict(value, system_id='s2ut', run_id=plan['run']['name'], split='test',
                                source_audio=str(local/'inputs'/(key+'.wav')), reference_audio='',
                                reference_text=row['en_reference'], units_sha256=digest(value['units'])))
    atomic_write_jsonl(local/'units/predictions.jsonl', predictions, overwrite=True)
    atomic_write_jsonl(output/'units.jsonl', predictions, overwrite=True)
    return predictions


@operation('8E: reconstruct or restore the completed waveform stage')
def audio(plan):
    verify_plan(plan)
    output = Path(plan['output'])
    # The shared worker publishes this manifest only after all waveform groups
    # have durable ZIPs. Once ASR has begun, keep this stage immutable, including
    # reported sample failures, so its journal identity does not change on resume.
    if (output/'predictions.jsonl').is_file():
        return restore_audio(plan)
    from .benchmark_worker import audio as reconstruct
    return reconstruct(plan)


def restore_audio(plan):
    """External equivalent of benchmark restore; no training test references."""
    from .benchmark_worker import ordered
    local, output = Path(plan['local']), Path(plan['output'])
    document = read(output/'audio-index.json')
    body = document['body']
    if document['sha256'] != digest(body) or body['identity'] != plan['identity']:
        raise ValueError('external audio index identity/SHA mismatch')
    samples = ordered(body['samples'], plan['ids'])
    unit_rows = ordered(list(read_jsonl(output/'units.jsonl')), plan['ids'])
    from ..vocoders.reference import CHECKPOINT_SHA256, CONFIG_SHA256
    for offset in range(0,len(unit_rows),32):
        batch = unit_rows[offset:offset+32]
        label = f'part-{offset//32:05d}'
        expected = dict(plan=plan['identity'],inputs=digest(batch),vocoder=CHECKPOINT_SHA256,config=CONFIG_SHA256)
        if any(r['part'] != label or r['part_identity'] != expected for r in samples[offset:offset+32]):
            raise ValueError('external Unit manifest differs from WAV archive inputs')
    for label in track(sorted({r['part'] for r in samples}), '8E: restore/verify WAV ZIPs'):
        group = [r for r in samples if r['part'] == label]
        restore_tree(output/'audio', label, local/'audio'/label, group[0]['part_identity'])
    predictions = ordered(list(read_jsonl(output/'predictions.jsonl')), plan['ids'])
    references = {r['pair_id']:r['en_reference'] for r in plan['source']['samples']}
    for row, item, unit in zip(predictions, samples, unit_rows):
        if (row['reference_text'] != references[row['pair_id']] or row['status'] != item['status']
                or row.get('units') != unit['units'] or unit['units_sha256'] != digest(unit['units'])
                or row['system_id'] != 's2ut' or row['run_id'] != plan['run']['name']):
            raise ValueError('external prediction reference/status mismatch')
        if row['status'] == 'success':
            path = local/'audio'/item['part']/item['member']
            if str(path) != row['output_audio'] or raw_sha(path) != item['output_sha256']:
                raise ValueError('external generated WAV SHA mismatch')
    atomic_write_jsonl(local/'predictions.jsonl', predictions, overwrite=True)
    return predictions


def verify_asr_runtime(plan, runtime):
    # Same pinned 8C numeric/audio/ASR dependencies. GPU identity may differ.
    keys = ('torch', 'torchaudio', 'transformers', 'tokenizers', 'safetensors', 'numpy',
            'librosa', 'soundfile', 'soxr', 'sacrebleu', 'accelerate', 'huggingface-hub')
    saved, current = plan['source']['environment'], runtime
    saved_packages = canonical_packages(saved['packages'])
    current_packages = canonical_packages(current['packages'])
    for key in keys:
        if key not in saved_packages or key not in current_packages or saved_packages[key] != current_packages[key]:
            raise ValueError('8E evaluation dependency differs from saved 8C: '+key+
                             f' (saved={saved_packages.get(key)!r}, actual={current_packages.get(key)!r})')
    if saved['python'] != current['python']:
        raise ValueError('8E evaluation Python differs from 8C')
    for key in ('torch_cuda','cudnn','matmul_allow_tf32','cudnn_allow_tf32'):
        if saved.get(key) != current.get(key):
            raise ValueError('8E evaluation backend setting differs from 8C: '+key)


def canonical_packages(packages):
    """Distribution names use case-insensitive [-_.]+ aliases; versions do not."""
    result = {}
    for name, version in packages.items():
        key = re.sub(r'[-_.]+','-',name).lower()
        if key in result and result[key] != version:
            raise ValueError('conflicting versions for canonical distribution name: '+key)
        result[key] = version
    return result


def verify_saved_asr_dependencies(ctx):
    """Reject incompatible saved 8C versions before expensive model inference."""
    repo = Path(__file__).resolve().parents[3]
    expected = {}
    for line in (repo/'requirements/diagnostic-flores.txt').read_text(encoding='utf-8').splitlines():
        if '==' in line and not line.startswith('#'):
            name, version = line.split(';')[0].strip().split('==')
            expected[name] = version
    expected = canonical_packages(expected)
    keys = ('torch','torchaudio','transformers','tokenizers','safetensors','numpy',
            'librosa','soundfile','soxr','sacrebleu','accelerate','huggingface-hub')
    expected.update(torch='2.7.1+cu126',torchaudio='2.7.1+cu126')
    saved = ctx['environment']
    verify_asr_runtime({'source':{'environment':saved}},
                       dict(saved, python='3.10.18', packages={key:expected[key] for key in keys}))


@operation('8E: merge verified durable and newer local ASR progress')
def restore_asr_progress(plan, identity):
    """A same-VM journal can be newer than the last 16-item ZIP.

    Restore into a temporary directory, then merge compatible journal rows.
    Never extract the older per_sample.jsonl over a newer local journal.
    """
    local, output = Path(plan['local']), Path(plan['output'])
    scores = local/'scores'
    if not (output/'progress/asr.json').is_file():
        return
    from ..journal import Journal
    with tempfile.TemporaryDirectory(prefix='asr-progress-',dir=local) as name:
        saved = Path(name)
        restore_tree(output/'progress','asr',saved,identity)
        expected = read(saved/'per_sample.lock.json')
        if expected.get('evaluation') != identity:
            raise ValueError('saved ASR journal evaluation identity mismatch')
        durable = Journal(saved/'per_sample.jsonl',expected,resume=True)
        current = Journal(scores/'per_sample.jsonl',expected,resume=True)
        allowed = {r['pair_id']:r['en_reference'] for r in plan['source']['samples']}
        for row in [*durable.rows.values(),*current.rows.values()]:
            if (row.get('pair_id') not in allowed or row.get('reference_raw') != allowed[row['pair_id']]
                    or row.get('system_id') != 's2ut' or row.get('run_id') != plan['run']['name']):
                raise ValueError('saved/local ASR journal sample/reference mismatch')
        merged = dict(durable.rows)
        for key,row in current.rows.items():
            old = merged.get(key)
            if old and old.get('evaluation_status') == row.get('evaluation_status') == 'success' and old != row:
                raise ValueError('conflicting successful ASR journal rows; both retained')
            if not old or old.get('evaluation_status') != 'success':
                merged[key] = row
        # Preserve local progress byte-for-byte when it already contains the
        # merged result; otherwise replace only this explicitly resumed journal.
        if merged != current.rows:
            atomic_write_jsonl(current.path,merged.values(),overwrite=True)


@operation('8E: English evaluation ASR with durable journals')
def evaluate(plan, *, asr_factory=None):
    from ..evaluation.run import evaluate_predictions
    verify_plan(plan)
    local, output = Path(plan['local']), Path(plan['output'])
    restore_audio(plan)
    identity = dict(plan=plan['identity'], asr=plan['asr'], normalization=NORMALIZATION,
                    bleu_denominator=plan['bleu_denominator'])
    scores = local/'scores'
    restore_asr_progress(plan,identity)
    model = None
    prior_rows = list(read_jsonl(scores/'per_sample.jsonl')) if (scores/'per_sample.jsonl').is_file() else []
    prior = indexed(prior_rows, 'pair_id') if prior_rows else {}
    predictions = list(read_jsonl(local/'predictions.jsonl'))
    # Dependency/model/audit errors are fatal BEFORE evaluate_predictions' per-file
    # error handling. They must never be misreported as hundreds of ASR failures.
    if any(r['status']=='success' and prior.get(r['pair_id'],{}).get('evaluation_status')!='success'
           for r in predictions):
        from ..evaluation.transcribe import EvaluationASR
        model = (asr_factory or EvaluationASR)(**plan['asr'], device=0)
        if asr_factory is None:
            pipeline = model.model.pipeline
            observed = dict(dtype=str(next(pipeline.model.parameters()).dtype),
                            generation_config=pipeline.model.generation_config.to_dict(),
                            adapter='EvaluationASR English transcribe')
            if observed != plan['source']['asr_audit']:
                raise RuntimeError('8E actual Whisper dtype/generation config differs from saved 8C')
            atomic_write_json(output/'evaluation-asr-audit.json', observed, resume=True)
    def transcribe(path):
        if model is None:
            raise RuntimeError('unexpected missing external ASR model')
        text = model(path)
        if not isinstance(text, str) or not text.strip():
            raise ValueError('empty ASR output')
        return text
    def persist(count, total):
        if count % 16 == 0 or count == total:
            publish_tree(scores, output/'progress', 'asr', identity)
    metrics = evaluate_predictions(local/'predictions.jsonl', scores, transcribe=transcribe,
        resume=True, evaluation_identity=identity, include_failed_in_bleu=True, on_record=persist)
    publish_tree(scores, output/'progress', 'asr', identity)
    return save_report(plan, metrics, list(read_jsonl(scores/'per_sample.jsonl')))


@operation('8E: persist external comparison metrics and reports')
def save_report(plan, metrics, scores):
    from .benchmark_worker import ordered
    local, output = Path(plan['local']), Path(plan['output'])
    scores = ordered(scores, plan['ids'])
    predictions = indexed(list(read_jsonl(output/'predictions.jsonl')), 'pair_id')
    generated = indexed(list(read_jsonl(output/'units.jsonl')), 'pair_id')
    audio_index = indexed(read(output/'audio-index.json')['body']['samples'], 'pair_id')
    rows = []
    for item, score in zip(plan['source']['samples'], scores):
        key = item['pair_id']
        unit, audio = generated[key], predictions[key]
        saved_audio = audio_index[key]
        hyp = score['hypothesis_raw'] if score['evaluation_status'] == 'success' else ''
        rows.append(dict(item, output_asr_en_text=hyp,
            sentence_bleu=sentence_bleu(normalize_english(hyp),normalize_english(item['en_reference'])),
            units=unit['units'], units_sha256=unit['units_sha256'],
            generated_english_wav=audio.get('output_audio'), output_sha256=audio.get('output_sha256'),
            audio_index=str(output/'audio-index.json'),
            audio_archive=str(output/'audio'/saved_audio['archive']), audio_member=saved_audio['member'],
            audio_archive_sha256=saved_audio['archive_sha256'],
            generation_status=unit['status'], vocoder_status=audio['status'],
            evaluation_asr_status=score['evaluation_status'],
            status='success' if score['evaluation_status']=='success' else 'failed',
            error=score.get('evaluation_error') or audio.get('error') or unit.get('error')))
    unit_failures = sum(r['generation_status'] != 'success' for r in rows)
    metrics.update(purpose=PURPOSE, condition='natural', sample_count=len(rows),
        generation_failures=unit_failures,
        vocoder_failures=sum(r['generation_status']=='success' and r['vocoder_status']!='success' for r in rows),
        evaluation_asr_failures=metrics['asr_failures'], s2ut_checkpoint=plan['checkpoint'],
        checkpoint_sha256=plan['checkpoint_sha256'], best_update=plan['best_update'],
        training_updates=plan['run']['updates'], code_revision=plan['execution_revision'], beam=plan['beam'],
        vocoder=plan['vocoder'], evaluation_asr=plan['asr'], normalization=NORMALIZATION,
        dataset_revisions=plan['source']['dataset_revisions'],
        cascade_8c_identity=plan['source']['identity'], cascade_8c_root=plan['source']['root'],
        cascade_natural_asr_bleu=plan['source']['cascade_natural_bleu'],
        difference=metrics['bleu']-plan['source']['cascade_natural_bleu'],
        sample_set_sha256=digest([{k:r[k] for k in SAMPLE_FIELDS} for r in plan['source']['samples']]))
    text = ('# 8E: FLEURS/FLORES Natural S2UT external evaluation\n\n'
        'External diagnostic; original training/test/8C artifacts unchanged. No retraining or test selection.\n\n'
        f"Samples: {len(rows)}; generation failures: {unit_failures}; vocoder failures: {metrics['vocoder_failures']}; "
        f"evaluation ASR failures: {metrics['evaluation_asr_failures']}. All samples retained; failures are empty hypotheses.\n\n"
        f"Cascade Natural ASR-BLEU (saved 8C): {metrics['cascade_natural_asr_bleu']:.4f}\n\n"
        f"S2UT ASR-BLEU: {metrics['bleu']:.4f}\n\nDifference (S2UT - Cascade): {metrics['difference']:+.4f}\n\n"
        f"Run: `{plan['run']['name']}`; dev-best update: {plan['best_update']}; training reached: {plan['run']['updates']}.\n\n"
        f"Checkpoint SHA256: `{plan['checkpoint_sha256']}`; beam: {plan['beam']}.\n\n"
        f"8C identity: `{plan['source']['identity']}`. Code: `{plan['execution_revision']}`.\n\n"
        'Official Unit HiFi-GAN; same saved English Whisper, dtype/decoding configuration and normalization as 8C. '
        'English references are scoring-only. WAVs are retained in `audio/` ZIPs, indexed by `audio-index.json`.\n')
    atomic_write_json(output/'metrics.json', metrics, overwrite=True)
    atomic_write_jsonl(output/'per_sample.jsonl', rows, overwrite=True)
    stream = io.StringIO(newline='')
    fields = ['sample_id','flores_id','fleurs_id','ja_text','en_reference','output_asr_en_text',
              'sentence_bleu','status','error','units_sha256','output_sha256','generated_english_wav',
              'audio_archive','audio_member','audio_archive_sha256']
    writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    atomic_write_text(output/'per_sample.csv', stream.getvalue(), overwrite=True)
    atomic_write_text(output/'report.md', text, overwrite=True)
    return dict(output_root=str(output), metrics=metrics, display=text)
