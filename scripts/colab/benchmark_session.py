"""Normal cells 6..9: one S2UT run, full test ASR-BLEU, saved audio."""
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

BENCHMARK_BACKEND_VERSION = 1


def load_plan(persistent):
    pointer = read(persistent/'benchmarks/s2ut/latest.json')
    path = Path(pointer['plan'])
    if raw_sha(path) != pointer['sha256']:
        raise ValueError('benchmark plan checksum changed; rerun normal cell 6')
    plan = read(path)
    if plan.get('version') != 1 or plan.get('split') != 'test':
        raise ValueError('not a full-test S2UT benchmark plan')
    return path, plan


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
def listen(persistent, local, pair_id='', index=0):
    _, plan = load_plan(persistent)
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
        audio['S2UT翻訳'] = str(target/selected['member'])
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
    parser.add_argument('--pair-id', default='')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    guarded(args.persistent)
    def invoke(*command):
        print('実行:', *map(str, command), flush=True)
        subprocess.run(list(map(str, command)), check=True)
    if args.action == 'listen':
        listen(args.persistent, args.local, args.pair_id, args.index)
        return
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if args.action == 'prepare':
        prepare(args.persistent, args.data, args.local, args.run_name, revision)
    else:
        execute(args.repo, args.persistent, args.action, invoke, revision)


if __name__ == '__main__':
    main()
