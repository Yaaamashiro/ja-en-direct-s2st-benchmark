"""Resume a preparation-only experiment after upgrading the code checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from direct_s2st.hashing import sha256_file
from direct_s2st.io import atomic_write_text, read_jsonl
from direct_s2st.progress import operation, track


@operation('recovery: check saved phonemization')
def check_saved_phonemes(data):
    common, phones = data / 'common', data / 'translatotron2/phonemes'
    metadata = json.loads((phones / 'metadata.json').read_text(encoding='utf-8'))
    if (metadata.get('engine'), metadata.get('version'), metadata.get('num_shards')) != ('espeak-ng', '1.52.0', 1):
        raise ValueError('expected complete, unsharded eSpeak NG 1.52.0 phonemization')
    tokens = (phones / 'inventory.txt').read_text(encoding='utf-8').splitlines()
    if not tokens or len(tokens) != len(set(tokens)) or any(
            not token or any(char.isspace() for char in token) for token in tokens):
        raise ValueError('invalid saved phoneme inventory')
    if sha256_file(phones / 'inventory.txt') != metadata.get('inventory_sha256'):
        raise ValueError('saved phoneme inventory does not match its metadata')
    for split in ('train', 'dev', 'test'):
        manifest = common / f'{split}.jsonl'
        if sha256_file(manifest) != metadata.get('common_manifests', {}).get(split):
            raise ValueError(f'saved phonemization has stale {split} input')
        ids = [row['pair_id'] for row in track(read_jsonl(manifest), f'recovery: check {split} IDs')]
        labels = set()
        with (phones / f'{split}.tsv').open(encoding='utf-8') as stream:
            for line in stream:
                pair_id, separator, sequence = line.rstrip('\n').partition('\t')
                if not separator or not pair_id or not sequence or pair_id in labels:
                    raise ValueError(f'invalid saved phoneme row in {split}')
                labels.add(pair_id)
        if (not ids or len(ids) != len(set(ids)) or set(ids) != set(labels)
                or metadata.get('counts', {}).get(split) != len(ids)):
            raise ValueError(f'saved {split} phonemization is incomplete')
    return metadata


def update_revision(persistent, revision, *, overwrite=False):
    # A dictionary/code change cannot migrate an already-started training run.
    if (list(persistent.glob('*.json')) or any((persistent / 'checkpoints').glob('*'))
            or any((persistent / 'runs').rglob('*.pt'))):
        raise ValueError('training configuration/checkpoints exist; use a separate run migration')
    pin = persistent / 'repository-revision.txt'
    old = pin.read_text(encoding='utf-8').strip()
    if old == revision:
        return
    if len(old) != 40 or any(c not in '0123456789abcdef' for c in old):
        raise ValueError('invalid saved repository revision')
    if not overwrite:
        raise ValueError('revision migration requires explicit --overwrite; prepared data is retained')
    backup = persistent / f'repository-revision.before-{revision[:12]}.txt'
    atomic_write_text(backup, old + '\n', resume=True)
    atomic_write_text(pin, revision + '\n', overwrite=True)
    print(f'[recovery] revision={old} -> {revision}; previous pin={backup}', flush=True)


@operation('recovery: resume Mel preparation')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persistent', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--profile', default='smoke')
    parser.add_argument('--overwrite', action='store_true', help='Back up and update only the repository revision pin')
    args = parser.parse_args()
    persistent = args.persistent.resolve()
    corpus = Path(os.environ['CORPUS_ROOT']).resolve()
    if persistent.is_relative_to(corpus) or corpus.is_relative_to(persistent):
        raise ValueError('experiment output must be separate from CORPUS_ROOT')
    actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != args.revision or len(actual) != 40:
        raise ValueError('checkout does not match the requested full revision')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('code checkout has local changes')
    data = persistent / 'data'
    check_saved_phonemes(data)  # Check before changing the pin or building anything.
    update_revision(persistent, actual, overwrite=args.overwrite)
    import runpy
    from direct_s2st.preparation_workflow import run_stage
    def run(*command):
        print('Recovery:', ' '.join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), check=True)
    python = runpy.run_path(str(ROOT / 'scripts/colab/runtime.py'))['ensure_runtime'](
        ROOT, persistent, run, require_gpu=False)
    common, phones, prepared = data / 'common', data / 'translatotron2/phonemes', data / 'translatotron2/fairseq'
    configuration = {p.relative_to(ROOT).as_posix(): sha256_file(p)
                     for p in sorted((ROOT / 'configs').rglob('*')) if p.is_file()}
    for action, inputs in [('prepare', [common, phones]), ('validate', [prepared])]:
        command = [python, '-m', 'direct_s2st.cli', 'translatotron2', action,
                   '--profile', args.profile, '--resume']
        identity = dict(revision=actual, configuration=configuration, corpus=str(corpus),
                        command=list(map(str, command)))
        run_stage(data / '.prep-checkpoints/stages' / f'translatotron2-{action}.json',
                  identity, inputs, [prepared], lambda: run(*command),
                  force=os.environ.get('S2ST_PREP_RECHECK') == '1')
    print('CPU準備完了。GPUへ切り替え、同じ設定でセル1 → 4Bを実行してください。', flush=True)


if __name__ == '__main__':
    main()
