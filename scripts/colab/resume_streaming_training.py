"""Resume a reviewed TT2 run without rewriting its base revision/environment.

The dependency environment stays at its original revision. A separate, clean
Git worktree supplies the reviewed execution-only changes through PYTHONPATH.
Both revisions are explicitly recorded; snapshot identities are not rewritten.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


BASE_REVISION = '48fcdb7eb023cf35860fe63716dd7dacdd277be5'
EXECUTION_FILES = {
    'src/direct_s2st/colab.py', 'src/direct_s2st/drive_staging.py',
    'src/direct_s2st/hashing.py', 'src/direct_s2st/session_runtime.py',
    'src/direct_s2st/stream_inputs.py', 'src/direct_s2st/translatotron2/batching.py',
    'src/direct_s2st/translatotron2/data.py', 'src/direct_s2st/translatotron2/train.py',
    'src/direct_s2st/translatotron2/validation.py',
    'src/direct_s2st/translatotron2/monitor.py',
    'src/direct_s2st/translatotron2/infer.py',
    'src/direct_s2st/translatotron2/engine.py', 'src/direct_s2st/train_runtime.py',
}


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def validate_revision(repo, revision):
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('overlay revision must be a full lowercase commit SHA')
    if git(repo, 'rev-parse', 'HEAD') != BASE_REVISION:
        raise ValueError('this reviewed resume overlay requires base revision '+BASE_REVISION)
    if git(repo, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('base checkout has local changes; retain them and inspect before resuming')
    if git(repo, 'rev-parse', revision+'^{commit}') != revision:
        raise ValueError('overlay revision is not the requested commit')
    if git(repo, 'merge-base', BASE_REVISION, revision) != BASE_REVISION:
        raise ValueError('overlay is not a descendant of the reviewed base')
    changed = set(git(repo, 'diff', '--name-only', BASE_REVISION, revision, '--', 'src').splitlines())
    if not changed <= EXECUTION_FILES:
        raise ValueError('unreviewed model/optimizer/other source changes: '+str(sorted(changed-EXECUTION_FILES)))
    protected = ('configs', 'requirements', 'patches', 'third_party', 'pyproject.toml')
    if git(repo, 'diff', '--name-only', BASE_REVISION, revision, '--', *protected):
        raise ValueError('dependency/scientific settings changed; do not use this execution-only overlay')
    if 'src/direct_s2st/stream_inputs.py' not in changed:
        raise ValueError('requested overlay does not contain streaming inputs')


def validate_snapshot(config, backup, environment, work):
    """Cheap provenance preflight; run_session still verifies snapshot bytes."""
    if config.get('kind') != 'tt2':
        raise ValueError('reviewed streaming resume currently supports TT2 only')
    if not config.get('runtime'):
        raise ValueError('reviewed streaming resume requires the resident training runtime')
    if str(environment) not in config.get('identity_files', []):
        raise ValueError('original config does not use this base environment lock')
    actual = hashlib.sha256(environment.read_bytes()).hexdigest()
    expected_config = config
    if config.get('batch_calibration'):
        from direct_s2st.batch_calibration import selected_config
        selected = json.loads((backup/'batch-calibration.json').read_text(encoding='utf-8'))['selected']
        if type(selected) is not int or selected < 1:
            raise ValueError('no saved TT2 batch selection in this backup; check the actual run name '
                             '(including any suffix). Refusing to restart calibration/training')
        expected_config = selected_config(config, selected)
    markers = sorted(backup.glob('*/snapshot.json'), reverse=True)
    if not markers:
        raise FileNotFoundError('no saved snapshot; refusing to start from update zero')
    update = None
    for marker in markers:
        try:
            saved = json.loads(marker.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue  # run_session checks retained earlier recovery points.
        identity = saved['identity']
        if identity['inputs'].get(str(environment)) != actual:
            raise ValueError('rebuilt dependency environment differs from the saved run; '
                             'do not edit environment.json to bypass this check')
        if identity['work'] != str(work) or identity['config']['kind'] != 'tt2':
            raise ValueError('snapshot belongs to a different work path/trainer')
        if identity['config'] != expected_config:
            raise ValueError('original training config differs from the saved snapshot')
        if identity['config'].get('research_metadata', {}).get('repository_revision') != BASE_REVISION:
            raise ValueError('snapshot provenance is not the reviewed base revision')
        update = max(update or 0, int(saved['updates']))
    if update is None or update < 1:
        raise FileNotFoundError('no readable saved snapshot; refusing to restart')
    return actual, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/content/ja-en-direct-s2st-benchmark'))
    parser.add_argument('--persistent', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--overlay-revision', required=True)
    parser.add_argument('--total-updates', type=int, default=2)
    parser.add_argument('--confirm-training', action='store_true')
    parser.add_argument('--monitor', action='store_true', help='Explicitly adopt checkpointed dev-loss monitoring; no optimizer/learning-rate change')
    parser.add_argument('--monitor-interval', type=int, default=10)
    parser.add_argument('--monitor-patience', type=int, default=10)
    parser.add_argument('--monitor-min-relative-improvement', type=float, default=.001)
    parser.add_argument('--practical-warmup-updates', type=int,
                        help='Explicit practical continuation: shorten warmup using absolute update count')
    args = parser.parse_args()
    if args.total_updates < 1 or (args.total_updates > 10 and not args.confirm_training):
        parser.error('long training requires --confirm-training')
    if args.practical_warmup_updates is not None:
        if args.practical_warmup_updates < 1 or not args.monitor:
            parser.error('practical warmup requires a positive endpoint and --monitor')
        os.environ['S2ST_TT2_PRACTICAL_WARMUP'] = str(args.practical_warmup_updates)
    if args.monitor:
        import math
        if min(args.monitor_interval, args.monitor_patience) < 1 or not math.isfinite(args.monitor_min_relative_improvement) or not 0 <= args.monitor_min_relative_improvement < 1:
            parser.error('invalid monitor interval/patience/relative improvement')
        os.environ['S2ST_TT2_MONITOR_POLICY'] = json.dumps(dict(interval=args.monitor_interval,
            patience=args.monitor_patience, min_relative_improvement=args.monitor_min_relative_improvement))
    repo, persistent = args.repo.resolve(), args.persistent.resolve()
    work, backup = args.work_root.resolve(), args.backup_root.resolve()
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and any(p.is_relative_to(Path(corpus).resolve()) for p in (persistent, work, backup)):
        raise ValueError('resume outputs must be outside CORPUS_ROOT')
    if not args.config.resolve().is_relative_to(persistent) or not backup.is_relative_to(persistent):
        raise ValueError('config and backup must belong to the selected experiment')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_revision(repo, args.overlay_revision)
    pinned = persistent/'repository-revision.txt'
    if pinned.read_text().strip() != BASE_REVISION:
        raise ValueError('experiment is not pinned to the reviewed base; do not rewrite its revision')
    sys.path.insert(0, str(repo/'src'))
    from direct_s2st.progress import operation
    from direct_s2st.io import atomic_write_json
    from direct_s2st.journal import digest
    spec = importlib.util.spec_from_file_location('base_colab_runtime', repo/'scripts/colab/runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    @operation('training recovery: reviewed execution overlay')
    def resume():
        def run(*command):
            print('実行:', *map(str, command), flush=True)
            subprocess.run(list(map(str, command)), check=True)
        prefix = Path('/content/s2st-runtime')
        python = runtime.ensure_runtime(repo, persistent, run, require_gpu=True, prefix=prefix)
        environment = prefix/'environment.json'
        environment_sha, saved_updates = validate_snapshot(config, backup, environment, work)
        overlay = prefix/'execution-overlays'/args.overlay_revision
        if overlay.exists():
            if git(overlay, 'rev-parse', 'HEAD') != args.overlay_revision or git(overlay, 'status', '--porcelain'):
                raise ValueError('existing execution overlay is incomplete/changed; preserve and inspect it')
        else:
            overlay.parent.mkdir(parents=True, exist_ok=True)
            run('git', '-C', repo, 'worktree', 'add', '--detach', overlay, args.overlay_revision)
        body = dict(base_revision=BASE_REVISION, execution_revision=args.overlay_revision,
                    base_environment_sha256=environment_sha,
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    work=str(work), backup=str(backup), snapshot_identity_unchanged=True)
        monitor_policy = os.environ.get('S2ST_TT2_MONITOR_POLICY')
        if monitor_policy:
            body['tt2_monitor_policy'] = json.loads(monitor_policy)
        if os.environ.get('S2ST_TT2_PRACTICAL_WARMUP'):
            body['practical_warmup_updates'] = int(os.environ['S2ST_TT2_PRACTICAL_WARMUP'])
            body['effective_reproduction_mode'] = 'paper_practical'
        audit_name = args.overlay_revision + '-policy-' + digest(body)[:12]
        audit = persistent/'execution-overlays'/f'{audit_name}.json'
        atomic_write_json(audit, dict(body=body, sha256=digest(body)), resume=True)
        os.environ['PYTHONPATH'] = str(overlay/'src')+os.pathsep+os.environ.get('PYTHONPATH', '')
        os.environ.update(S2ST_DRIVE_SAFE='1', S2ST_STREAM_INPUTS='1')
        print(f'[recovery-overlay] base={BASE_REVISION} execution={args.overlay_revision} '
              f'saved_at_least={saved_updates} audit={audit} time_limit=none', flush=True)
        # No make_config or calibration rewrite; --resume refuses a missing
        # verified recovery point. Both trainer and backup verify identities.
        command = [python, '-m', 'direct_s2st.colab', '--config', args.config,
                   '--work-root', work, '--backup-root', backup,
                   '--total-updates', str(args.total_updates), '--session-seconds', '0', '--resume']
        if args.confirm_training:
            command.append('--confirm-training')
        run(*command)
    resume()


if __name__ == '__main__':
    main()
