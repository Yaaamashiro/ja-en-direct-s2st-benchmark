"""Upgrade a preparation-only experiment and finish audio ZIPs on a CPU VM.

Unlike resume_preparation.py this does not redo phonemes/Mel validation. The
audio namespace is unchanged, so all previously published 4B ZIPs are reused.
"""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from direct_s2st.progress import operation


@operation('recovery: CPU audio ZIP preparation')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persistent', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--profile', default='smoke')
    parser.add_argument('--overwrite', action='store_true',
                        help='Retain the old revision pin and explicitly update it; never overwrite WAV ZIPs')
    args = parser.parse_args()
    persistent = args.persistent.resolve()
    corpus = Path(os.environ['CORPUS_ROOT']).resolve()
    if persistent.is_relative_to(corpus) or corpus.is_relative_to(persistent):
        raise ValueError('experiment output must be separate from CORPUS_ROOT')
    actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if (actual != args.revision or len(actual) != 40
            or any(c not in '0123456789abcdef' for c in actual)):
        raise ValueError('checkout does not match the requested full revision')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('code checkout has local changes')
    common = persistent / 'data/common'
    if not (common / 'dataset-lock.json').is_file() or not all(
            (common / f'{split}.jsonl').is_file() for split in ('train', 'dev', 'test')):
        raise ValueError('先に4Aのコーパス取込を完了してください')
    # Reuse the conservative preparation-only guard and backed-up revision pin.
    update_revision = runpy.run_path(str(ROOT / 'scripts/colab/resume_preparation.py'))['update_revision']
    update_revision(persistent, actual, overwrite=args.overwrite)

    def run(*command):
        print('CPU audio preparation:', ' '.join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), check=True)

    python = runpy.run_path(str(ROOT / 'scripts/colab/runtime.py'))['ensure_runtime'](
        ROOT, persistent, run, require_gpu=False)
    run(python, '-m', 'direct_s2st.preparation_workflow', '--stage', 'audio-packs',
        '--profile', args.profile)


if __name__ == '__main__':
    main()
