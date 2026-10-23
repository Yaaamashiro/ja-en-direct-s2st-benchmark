"""Normal cells 6..9: single-system full test ASR-BLEU and saved audio."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from inference_session import saved_run, checked_metadata, environment
from direct_s2st.benchmark_storage import guarded, read, restore_tree
from direct_s2st.drive_staging import raw_sha, register, stage_audio, ensure_local
from direct_s2st.io import atomic_write_json, read_jsonl
from direct_s2st.journal import digest
from direct_s2st.progress import operation, track

BENCHMARK_BACKEND_VERSION = 5


def load_plan(persistent, target='s2ut'):
    if target not in ('s2ut', 'cascade'):
        raise ValueError('unsupported single-system benchmark target')
    pointer = read(persistent/'benchmarks'/target/'latest.json')
    path = Path(pointer['plan'])
    if raw_sha(path) != pointer['sha256']:
        raise ValueError('benchmark plan checksum changed; rerun normal cell 6')
    plan = read(path)
    if (plan.get('version') != 1 or plan.get('split') != 'test'
            or plan.get('system_id', 's2ut') != target):
        raise ValueError('not a matching full-test benchmark plan')
    return path, plan


@operation('cell 6: prepare cascade-only full-test benchmark')
def prepare_cascade(persistent, data, local, revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('cascade evaluation requires an immutable execution revision')
    from direct_s2st.config import load_config
    configuration = load_config(ROOT/'configs/cascade/default.yaml')
    # Only model settings: profile max_samples/training knobs must never limit
    # this explicitly confirmed full-test evaluation.
    pipeline = {key: configuration[key] for key in ('asr', 'mt', 'tts')}
    from direct_s2st.cascade.qwen_generation import generation_policy
    pipeline['tts']['generation'] = generation_policy(pipeline['tts'].get('generation'))
    for key in ('asr', 'mt', 'tts'):
        if not re.fullmatch('[0-9a-f]{40}', pipeline[key]['revision']):
            raise ValueError('cascade models require immutable revisions')
    manifest = data/'common/test.jsonl'
    rows = list(read_jsonl(manifest))
    ids = [r['pair_id'] for r in rows]
    if (not ids or len(set(ids)) != len(ids) or any(r['split'] != 'test' for r in rows)
            or any(not re.fullmatch('[A-Za-z0-9_-]+', key) for key in ids)
            or any(not isinstance(r.get('en_text'), str) or not r['en_text'].strip() for r in rows)):
        raise ValueError('cascade evaluation requires all unique test IDs and English references')
    body = dict(version=1, system_id='cascade', purpose='cascade_full_test_asr_bleu_not_four_system_acceptance',
                split='test', ids=ids, common_root=str(data/'common'), common_sha256=raw_sha(manifest),
                dataset_lock_sha256=raw_sha(data/'common/dataset-lock.json'),
                pipeline=pipeline, execution_revision=revision,
                asr=dict(model_id='openai/whisper-large-v3-turbo',
                         revision='60be3615a4d667e1258e8ad29130467587c489aa'),
                bleu_denominator='all_samples_failed_as_empty')
    reused, source = None, None
    pointer_path = persistent/'benchmarks/cascade/latest.json'
    previous_pointer = pointer_path.read_bytes() if pointer_path.is_file() else None
    if pointer_path.is_file():
        old_path, old = load_plan(persistent, 'cascade')
        if all(old.get(key) == value for key, value in body.items()):
            reused = old
            if old.get('text_source'):
                from direct_s2st.cascade.benchmark_worker import text_source_plan
                source = text_source_plan(old)
        elif (all(old.get(key) == body[key] for key in
                  ('ids', 'common_root', 'common_sha256', 'dataset_lock_sha256', 'asr', 'bleu_denominator'))
              and all(old['pipeline'][key] == pipeline[key] for key in ('asr', 'mt'))):
            # Explicit policy fork: do NOT adopt an overlay that would mix
            # greedy audio with sampled audio. Completed text remains read-only.
            from direct_s2st.cascade.benchmark_worker import text_source_plan
            source = text_source_plan(old)
            source_path = Path(old['text_source']['plan']) if old.get('text_source') else old_path
            root = Path(source['output'])
            files = {name: raw_sha(root/name) for name in ('text/asr.json', 'text/mt.json', 'runtime.json')}
            for phase in ('asr', 'mt'):
                receipt = read(root/'text'/(phase+'.json'))
                if receipt['sha256'] != digest(receipt['body']) or receipt['body']['plan'] != source['identity']:
                    raise ValueError('saved text pointer corrupt; no ASR/MT recomputation')
            body.update(text_source=dict(plan=str(source_path), plan_sha256=raw_sha(source_path), files=files),
                        dependency_revision=source['execution_revision'])
            print('Cascade policy fork: reuse completed ASR/MT; old WAVs preserved in their original evaluation',
                  file=sys.stderr, flush=True)
    token = digest(body)[:24]
    owned = ensure_local(Path(reused['local']) if reused else local/'cascade'/token)
    output = guarded(Path(reused['output']) if reused else persistent/'benchmarks/cascade'/token)
    input_owner = Path(source['local']) if source else owned
    mapping = stage_audio(data/'common', data/'.drive-audio-packs', input_owner/'inputs',
                          splits=('test',), languages=('ja', 'en'))
    atomic_write_json(owned/'input-map.json', dict(rows=mapping, sha256=digest(mapping)), overwrite=True)
    if reused:
        print(json.dumps(dict(target='cascade', test_samples=len(ids), output=str(output),
                              resume_saved_asr_mt=True, original_plan_preserved=True,
                              execution_revision=revision)), flush=True)
        return reused
    plan = dict(body, local=str(owned), output=str(output), identity=digest(body),
                run=dict(name='cascade-test-'+token))
    if source and previous_pointer is not None:
        from direct_s2st.io import atomic_write_bytes
        atomic_write_bytes(output/'previous-latest.json', previous_pointer, resume=True)
    atomic_write_json(output/'plan.json', plan, resume=True)
    atomic_write_json(persistent/'benchmarks/cascade/latest.json',
                      dict(plan=str(output/'plan.json'), sha256=raw_sha(output/'plan.json')), overwrite=True)
    print(json.dumps(dict(target='cascade', test_samples=len(ids), output=str(output)), ensure_ascii=False), flush=True)
    return plan


@operation('cascade benchmark: restore measured inference environment')
def cascade_environment(persistent, plan, invoke, *, prefix=Path('/content/s2st-runtime'), build=None):
    from environment_recovery import ensure_environment
    from direct_s2st.io import atomic_write_bytes
    path = Path(plan['output'])/'runtime.json'
    previous = read(path) if path.is_file() else None
    if previous and (previous['sha256'] != digest(previous['body'])
                     or previous['body']['plan'] != plan['identity']):
        raise ValueError('cascade dependency receipt differs from saved evaluation')
    expected = previous['body']['environment_sha256'] if previous else None
    if plan.get('text_source'):
        from direct_s2st.cascade.benchmark_worker import text_source_plan
        source = text_source_plan(plan)
        receipt = read(Path(source['output'])/'runtime.json')
        if receipt['sha256'] != digest(receipt['body']) or receipt['body']['plan'] != source['identity']:
            raise ValueError('cascade source dependency receipt changed')
        saved = receipt['body']['environment_sha256']
        if expected and expected != saved:
            raise ValueError('cascade policy fork dependency environment changed')
        expected = saved
    python = ensure_environment(build or ROOT, persistent, invoke, expected=expected, prefix=prefix)
    checksum = raw_sha(prefix/'environment.json')
    if expected and checksum != expected:
        raise ValueError('cascade runtime differs; saved outputs retained')
    body = dict(plan=plan['identity'], environment_sha256=checksum)
    atomic_write_json(path, dict(body=body, sha256=digest(body)), resume=True)
    atomic_write_bytes(Path(plan['output'])/'environment.json', (prefix/'environment.json').read_bytes(), resume=True)
    os.environ.update(PYTHONPATH=str(ROOT/'src')+os.pathsep+str(prefix/'fairseq'),
                      S2ST_CONFIG_ROOT=str(ROOT/'configs'),
                      S2ST_BENCHMARK_ENVIRONMENT_SHA256=checksum)
    return python


@operation('cascade benchmark: execute without training checkpoints')
def execute_cascade(persistent, action, invoke, revision):
    path, plan = load_plan(persistent, 'cascade')
    if action == 'run' and not plan['pipeline']['tts'].get('generation'):
        raise ValueError('legacy greedy TTS requires a separate policy evaluation; run normal cell 6 first')
    overlay = None
    if plan['execution_revision'] != revision:
        from cascade_loader_overlay import approve
        overlay = approve(ROOT, path, plan, revision)
    local = Path(plan['local'])
    if not (local/'input-map.json').is_file():
        raise ValueError('local test inputs lost; normal cell 1 -> 6 -> 7 -> 8 restores saved work')
    common = Path(plan['common_root'])
    if (raw_sha(common/'test.jsonl') != plan['common_sha256']
            or raw_sha(common/'dataset-lock.json') != plan['dataset_lock_sha256']):
        raise ValueError('cascade test metadata changed')
    os.environ['S2ST_INPUT_MAP'] = str(local/'input-map.json')
    if overlay or plan.get('dependency_revision'):
        from cascade_loader_overlay import original_build
        dependency = plan.get('dependency_revision', plan['execution_revision'])
        if dependency == revision:
            build = ROOT
        else:
            build = original_build(ROOT, dependency, invoke)
        python = cascade_environment(persistent, plan, invoke, build=build)
    else:
        python = cascade_environment(persistent, plan, invoke)
    bf16_check = ('assert torch.cuda.is_bf16_supported(), "The pinned Qwen adapter requires a BF16-capable GPU"; '
                  if action == 'run' else '')
    invoke(python, '-c', 'from direct_s2st.inference_logging import configure_logging; configure_logging(); '
           'import torch, torchaudio, soundfile, sacrebleu, accelerate, sentencepiece; '
           'assert torch.cuda.is_available(), "Cascade/ASR evaluation requires a GPU"; '
           +bf16_check+
           'from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer; '
           'from qwen_tts import Qwen3TTSModel; print("Cascade dependencies checked")')
    if action == 'run':
        invoke(python, '-c', 'from direct_s2st.cascade.tts import check_runtime; check_runtime()')
        # A previous/adopted run is read-only for ASR/MT. Missing/corrupt results
        # must fail, never cause implicit Whisper/NLLB recomputation.
        output = Path(plan['output'])
        existing_text = (overlay or plan.get('text_source') or any((output/'text').glob('*.json'))
                         or any((output/'progress').glob('*.json')))
        phases = ('verify-text', 'tts') if existing_text else ('asr', 'mt', 'verify-text', 'tts')
    else:
        phases = ('evaluate',)
    for phase in phases:
        invoke(python, '-m', 'direct_s2st.cascade.benchmark_worker', '--plan', path, '--action', phase)
    if action == 'evaluate':
        if plan.get('text_source'):
            from direct_s2st.io import atomic_write_text
            results = Path(plan['output'])/'results'
            metrics = read(results/'metrics.json')
            metrics.update(text_source=plan['text_source'], tts_generation_policy=plan['pipeline']['tts']['generation'],
                           old_audio_reused=False)
            atomic_write_json(results/'metrics.json', metrics, overwrite=True)
            report = (results/'report.md').read_text(encoding='utf-8')
            report += ('\nTTS policy: `official_sampling_eos_v1`; per-text seed fixed. '
                       'Completed ASR/MT reused read-only from `'+plan['text_source']['plan']+'`. '
                       'Old greedy WAVs are preserved separately and are NOT used in this evaluation.\n')
            atomic_write_text(results/'report.md', report, overwrite=True)
        if overlay:
            from direct_s2st.io import atomic_write_text
            results = Path(plan['output'])/'results'
            metrics = read(results/'metrics.json')
            metrics['execution_overlay'] = overlay
            atomic_write_json(results/'metrics.json', metrics, overwrite=True)
            report = (results/'report.md').read_text(encoding='utf-8')
            report += ('\nReviewed TTS loader execution: `'+revision+'`; original plan/dependencies: `'
                       +plan['execution_revision']+'`. ASR/MT history is unchanged.\n')
            atomic_write_text(results/'report.md', report, overwrite=True)
        print((Path(plan['output'])/'results/report.md').read_text(encoding='utf-8'), flush=True)


def validate_test(run, data):
    from direct_s2st.s2ut.multitask import read_tsv, tokenize
    from direct_s2st.vocoders.reference import validate_reference_units
    command = run['config']['command']
    root = Path(command[3])
    if '{' in str(root):
        raise ValueError('saved S2UT prepared data path is unresolved')
    metadata = {}
    for name, checksum in track(run['identity']['inputs'].items(), 'benchmark: verify bound prepared metadata'):
        path = Path(name)
        if path.is_relative_to(root):
            checked_metadata(run, path)
            metadata[name] = checksum
    for relative in ('test.tsv', 'data-lock.json', 'target_letter/test.tsv'):
        checked_metadata(run, root/relative)
    lock = read(root/'data-lock.json')
    if lock['common_dataset_lock_sha256'] != raw_sha(data/'common/dataset-lock.json'):
        raise ValueError('test corpus differs from saved training dataset lock')
    validate_reference_units(lock['unit_configuration'])
    rows = list(read_jsonl(data/'common/test.jsonl'))
    ids = [r['pair_id'] for r in rows]
    if (not ids or len(set(ids)) != len(ids) or any(r['split'] != 'test' for r in rows)
            or any(not re.fullmatch('[A-Za-z0-9_-]+', x) for x in ids)):
        raise ValueError('test IDs must be unique, safe, and nonempty')
    prepared = read_tsv(root/'test.tsv', ('id', 'src_audio', 'src_n_frames', 'tgt_audio', 'tgt_n_frames'))
    labels = read_tsv(root/'target_letter/test.tsv', ('id', 'tgt_text'))
    if set(prepared) != set(ids) or set(labels) != set(ids):
        raise ValueError('full common/prepared test ID mismatch; no subset evaluation')
    for row in rows:
        if labels[row['pair_id']]['tgt_text'].split() != tokenize(row['en_text']):
            raise ValueError('test reference differs from checkpoint-bound target labels')
    return root, metadata, rows, prepared


@operation('cell 6: prepare S2UT full-test benchmark')
def prepare(persistent, data, local, requested, revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('benchmark execution requires an immutable revision')
    local = ensure_local(local)
    run = saved_run(persistent, 's2ut', requested, local)
    root, metadata, rows, prepared = validate_test(run, data)
    from direct_s2st.translatotron2.monitor import select_checkpoint
    checkpoint = select_checkpoint(run['work'], 'best')
    monitor = read(Path(run['work'])/'training-monitor.json')
    identity = dict(version=1, purpose='s2ut_full_test_asr_bleu_not_four_system_acceptance',
                    split='test', ids=[r['pair_id'] for r in rows], run=run,
                    metadata=metadata, data_root=str(root), common_root=str(data/'common'),
                    common_sha256=raw_sha(data/'common/test.jsonl'),
                    checkpoint=str(checkpoint), checkpoint_sha256=raw_sha(checkpoint),
                    best_update=monitor['best_update'], execution_revision=revision,
                    beam=10, vocoder='official_hubert100_ljspeech',
                    asr=dict(model_id='openai/whisper-large-v3-turbo',
                             revision='60be3615a4d667e1258e8ad29130467587c489aa'),
                    bleu_denominator='all_samples_failed_as_empty')
    token = digest(identity)[:24]
    owned = ensure_local(local/token)
    output = guarded(persistent/'benchmarks/s2ut'/token)
    mapping = stage_audio(data/'common', data/'.drive-audio-packs', owned/'inputs',
                          splits=('test',), languages=('ja', 'en'))
    from direct_s2st.drive_staging import logical
    for row in rows:
        # Portable aliases are already registered by stage_audio. Avoid old
        # absolute Drive WAV stats while registering the actual prepared TSV.
        aliases = [Path(row['ja_audio'])]
        if row.get('ja_audio_corpus_relative') and os.environ.get('CORPUS_ROOT'):
            aliases.append(Path(os.environ['CORPUS_ROOT'])/row['ja_audio_corpus_relative'])
        entry = next((mapping[logical(p)] for p in aliases if logical(p) in mapping), None)
        if entry is None:
            raise ValueError('test Japanese WAV mapping missing')
        register(mapping, prepared[row['pair_id']]['src_audio'], Path(entry['local']), entry['source_stamp'])
    atomic_write_json(owned/'input-map.json', dict(rows=mapping, sha256=digest(mapping)), overwrite=True)
    plan = dict(identity, local=str(owned), output=str(output), identity=digest(identity))
    atomic_write_json(output/'plan.json', plan, resume=True)
    atomic_write_json(persistent/'benchmarks/s2ut/latest.json',
                      dict(plan=str(output/'plan.json'), sha256=raw_sha(output/'plan.json')), overwrite=True)
    print(json.dumps(dict(run=run['name'], training_updates=run['updates'],
                          selected_best_update=plan['best_update'], test_samples=len(rows),
                          output=str(output)), ensure_ascii=False), flush=True)
    return plan


@operation('benchmark: verify saved plan and dependency environment')
def execute(repo, persistent, action, invoke, revision):
    path, plan = load_plan(persistent)
    if plan['execution_revision'] != revision:
        raise ValueError('benchmark execution revision changed; rerun normal cell 6')
    local = Path(plan['local'])
    if not (local/'input-map.json').is_file():
        raise ValueError('local test inputs lost; same settings, normal cell 1 -> 6 -> 7 -> 8')
    if raw_sha(Path(plan['common_root'])/'test.jsonl') != plan['common_sha256']:
        raise ValueError('benchmark test manifest changed')
    for name, checksum in track(plan['metadata'].items(), 'benchmark: recheck prepared metadata'):
        if raw_sha(name) != checksum:
            raise ValueError('checkpoint-bound prepared metadata changed')
    if raw_sha(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('benchmark dev-best checkpoint changed')
    os.environ['S2ST_INPUT_MAP'] = str(local/'input-map.json')
    python = environment(repo, persistent, plan['run'], invoke)
    invoke(python, '-c', 'import torch, torchaudio, soundfile, sacrebleu, accelerate; '
           'assert torch.cuda.is_available(), "Full-test generation/ASR requires a GPU"; '
           'from fairseq_cli import generate; '
           'from fairseq.models.text_to_speech.codehifigan import CodeGenerator; '
           'from transformers import pipeline; print("Benchmark dependencies checked")')
    # Each subprocess releases its GPU models before the next phase.
    actions = ('units', 'audio') if action == 'run' else ('evaluate',)
    for phase in actions:
        invoke(python, '-m', 'direct_s2st.s2ut.benchmark_worker', '--plan', path, '--action', phase)
    if action == 'evaluate':
        print((Path(plan['output'])/'results/report.md').read_text(encoding='utf-8'), flush=True)


@operation('cell 9: listen to saved benchmark WAV without GPU or models')
def listen(persistent, local, pair_id='', index=0, *, target='s2ut'):
    _, plan = load_plan(persistent, target)
    output = Path(plan['output'])
    document = read(output/'audio-index.json')
    if document['sha256'] != digest(document['body']) or document['body']['identity'] != plan['identity']:
        raise ValueError('saved benchmark audio index mismatch')
    rows = document['body']['samples']
    if not pair_id and not 0 <= index < len(rows):
        raise ValueError('BENCHMARK_LISTEN_INDEX outside saved sample list')
    selected = next((r for r in rows if r['pair_id'] == pair_id), None) if pair_id else rows[index]
    if selected is None:
        raise ValueError('BENCHMARK_LISTEN_ID not present in saved index')
    owned = ensure_local(local/plan['identity'][:24]/'listening')
    audio = {}
    if selected['status'] == 'success':
        target = owned/selected['part']
        restore_tree(output/'audio', selected['part'], target, selected['part_identity'],
                     members=[selected['member']])
        if raw_sha(target/selected['member']) != selected['output_sha256']:
            raise ValueError('saved listening WAV differs from audio index')
        label = 'カスケード翻訳' if plan.get('system_id') == 'cascade' else 'S2UT翻訳'
        audio[label] = str(target/selected['member'])
    # Original corpus is retained separately, not duplicated in every archive.
    from direct_s2st.drive_staging import read_common_manifest_row, copy_verified
    common = next(r for r in read_jsonl(Path(plan['common_root'])/'test.jsonl') if r['pair_id'] == selected['pair_id'])
    if raw_sha(Path(plan['common_root'])/'test.jsonl') != plan['common_sha256']:
        raise ValueError('saved listening references changed')
    original_errors = {}
    for language, label in [('ja', '日本語原音'), ('en', '英語正解')]:
        target = owned/(selected['pair_id']+'-'+language+'.wav')
        try:
            if not target.is_file() or raw_sha(target) != common[language+'_sha256']:
                row = read_common_manifest_row(common, language)
                copy_verified(row[language+'_audio'], target, row[language+'_sha256'])
            audio[label] = str(target)
        except (OSError, ValueError) as error:
            # Missing original corpus must not prevent playing the saved translation.
            original_errors[label] = f'{type(error).__name__}: {error}'
    result = dict(pair_id=selected['pair_id'], ja_text=common['ja_text'], en_text=common['en_text'],
                  status=selected['status'], error=selected.get('error'), audio=audio,
                  original_errors=original_errors)
    scores = output/'results/per_sample.jsonl'
    if scores.is_file():
        result['evaluation'] = next((r for r in read_jsonl(scores) if r['pair_id'] == result['pair_id']), None)
    target = output/'listening.json'
    atomic_write_json(target, result, overwrite=True)
    return result


def main():
    from direct_s2st.inference_logging import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'run', 'evaluate', 'listen'), required=True)
    for key in ('repo', 'persistent', 'data'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--local', type=Path, default=Path('/content/s2st-benchmark'))
    parser.add_argument('--run-name', default='')
    parser.add_argument('--target', choices=('s2ut', 'cascade'), default='s2ut')
    parser.add_argument('--pair-id', default='')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    guarded(args.persistent)
    def invoke(*command):
        print('実行:', *map(str, command), flush=True)
        subprocess.run(list(map(str, command)), check=True)
    if args.action == 'listen':
        listen(args.persistent, args.local, args.pair_id, args.index, target=args.target)
        return
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if args.action == 'prepare':
        if args.target == 'cascade':
            prepare_cascade(args.persistent, args.data, args.local, revision)
        else:
            prepare(args.persistent, args.data, args.local, args.run_name, revision)
    elif args.target == 'cascade':
        execute_cascade(args.persistent, args.action, invoke, revision)
    else:
        execute(args.repo, args.persistent, args.action, invoke, revision)


if __name__ == '__main__':
    main()
