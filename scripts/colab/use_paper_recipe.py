"""Explicitly upgrade a legacy experiment's code pin; retain data/old snapshots.

This never migrates model/optimizer state. New paper run names must be used.
Plan by default, --overwrite authorizes only the backed-up repository pin change.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from direct_s2st.io import atomic_write_text
from direct_s2st.progress import operation


@operation('migration: legacy experiment to separate paper runs')
def migrate(persistent, revision, *, overwrite=False, separate_recipe_v2_runs=False,
            separate_recipe_v3_runs=False):
    persistent = Path(persistent).resolve()
    corpus = Path(os.environ['CORPUS_ROOT']).resolve()
    if persistent.is_relative_to(corpus) or corpus.is_relative_to(persistent):
        raise ValueError('experiment must be separate from CORPUS_ROOT')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('full immutable revision required')
    actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision or subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('checkout must be clean and match the requested revision')
    pin = persistent/'repository-revision.txt'
    old = pin.read_text(encoding='utf-8').strip()
    if len(old) != 40 or any(c not in '0123456789abcdef' for c in old):
        raise ValueError('invalid previous revision')
    if old == revision:
        return dict(status='ALREADY_CURRENT', revision=revision)
    for path in persistent.glob('*.json'):
        config = json.loads(path.read_text(encoding='utf-8'))
        if config.get('research_metadata', {}).get('reproduction_mode') in ('paper_exact', 'paper_practical'):
            if not separate_recipe_v2_runs and not separate_recipe_v3_runs:
                raise ValueError('paper training configuration already exists; explicitly pass --separate-recipe-v2-runs to preserve it and use new recipe-v2 run names')
            if '-recipe-v3' in path.stem:
                raise ValueError('recipe-v3 configuration already exists; use a separate code/run migration')
            if '-recipe-v2' in path.stem and not separate_recipe_v3_runs:
                raise ValueError('recipe-v2 configuration already exists; use a separate code/run migration')
    result = dict(status='PLAN_ONLY', previous=old, revision=revision,
                  prepared_data_preserved=True, old_snapshots_preserved=True,
                  old_checkpoint_resume=False)
    if overwrite:
        atomic_write_text(persistent/f'repository-revision.before-paper-{revision[:12]}.txt', old+'\n', resume=True)
        atomic_write_text(pin, revision+'\n', overwrite=True)
        result['status'] = 'APPLIED'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persistent', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--separate-recipe-v2-runs', action='store_true')
    parser.add_argument('--separate-recipe-v3-runs', action='store_true',
                        help='Preserve older runs; use a new S2UT recipe-v3 run for the character vocabulary change')
    args = parser.parse_args()
    print(json.dumps(migrate(args.persistent, args.revision, overwrite=args.overwrite,
                             separate_recipe_v2_runs=args.separate_recipe_v2_runs,
                             separate_recipe_v3_runs=args.separate_recipe_v3_runs)))


if __name__ == '__main__':
    main()
