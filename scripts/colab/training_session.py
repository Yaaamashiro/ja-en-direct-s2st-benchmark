"""Cell 5: choose a durable run, restore its code/environment, then train."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from direct_s2st.io import atomic_write_json
from direct_s2st.progress import operation, track
from environment_recovery import ensure_environment
import resume_streaming_training as legacy

LEGACY_EXECUTION = 'd70516a221199262a0fbd26637c6c5a8b86f73df'
WORK_ROOT = Path('/content/s2st-work')
RUNTIME_ROOT = Path('/content/s2st-runtime')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


@operation('cell 5: inspect saved training runs')
def choose_run(persistent, kind, default_name, requested=''):
    persistent = Path(persistent)
    if requested and not re.fullmatch('[A-Za-z0-9_-]+', requested):
        raise ValueError('invalid TRAIN_RUN_NAME')
    candidates = []
    for path in track(sorted(persistent.glob(kind+'-*.json')), 'recovery: inspect run configs'):
        if path.name.endswith('.launch.json'):
            continue
        config = json.loads(path.read_text(encoding='utf-8'))
        if config.get('kind') != ('vocoder' if kind in ('unit', 'mel') else kind):
            continue
        backup = persistent/'checkpoints'/path.stem
        markers = sorted(backup.glob('*/snapshot.json'), reverse=True)
        # These are candidates, not verified checkpoints. run_session verifies
        # complete bytes and may fall back to an older retained recovery point.
        for marker in markers:
            try:
                saved = json.loads(marker.read_text(encoding='utf-8'))
                if type(saved.get('updates')) is not int or saved['updates'] < 1:
                    raise ValueError('invalid updates')
                if saved['identity']['config']['kind'] != config['kind']:
                    raise ValueError('trainer differs')
                candidates.append((path.stem, saved))
                break
            except (ValueError, KeyError):
                continue
        else:
            if markers:
                raise ValueError(f'no readable recovery marker in {backup}; refusing a fresh restart')
    if requested:
        matches = [item for item in candidates if item[0] == requested]
        if matches:
            return matches[0]
        # An explicit name never falls through to update zero if another run exists.
        if candidates or (persistent/(requested+'.json')).exists():
            raise ValueError('TRAIN_RUN_NAME has no readable snapshot; retained runs: '+str([v[0] for v in candidates]))
        return requested, None
    if len(candidates) > 1:
        raise ValueError('multiple trained runs; select TRAIN_RUN_NAME in cell 1: '+str([v[0] for v in candidates]))
    return candidates[0] if candidates else (default_name, None)


def checkout(repo, revision, run):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('execution code requires an immutable full SHA')
    root = repo.parent/'s2st-execution'/revision
    if root.exists():
        if git(root, 'rev-parse', 'HEAD') != revision or git(root, 'status', '--porcelain', '--untracked-files=no'):
            raise ValueError('saved execution checkout is changed; preserve it and inspect')
    else:
        run('git', '-C', repo, 'fetch', 'origin', revision)
        root.parent.mkdir(parents=True, exist_ok=True)
        run('git', '-C', repo, 'worktree', 'add', '--detach', root, revision)
    return root


@operation('cell 5: restore and continue selected training')
def launch(repo, persistent, data, kind, default_name, model_size, options, total, seconds,
           *, requested='', confirm=False, run=None):
    repo, persistent, data = Path(repo), Path(persistent), Path(data)
    if total < 1 or seconds < 0 or (total > 10 and not confirm):
        raise ValueError('long training requires CONFIRM_TRAINING=True')
    practical_warmup = options.get('tt2_warmup_updates') if kind == 'tt2' else None
    if practical_warmup is not None and (type(practical_warmup) is not int or practical_warmup not in (400, 500, 1000, 10000)):
        raise ValueError('TT2 warmup must be 400, 500, 1000 or 10000')
    s2ut_warmup = options.get('s2ut_warmup_updates') if kind == 's2ut' else None
    if s2ut_warmup is not None and (type(s2ut_warmup) is not int or s2ut_warmup not in (400, 500, 1000, 10000)):
        raise ValueError('S2UT warmup must be 400, 500, 1000 or 10000')
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and any(p.resolve().is_relative_to(Path(corpus).resolve()) for p in (persistent, data)):
        raise ValueError('training outputs must stay outside CORPUS_ROOT')
    name, snapshot = choose_run(persistent, kind, default_name, requested)
    config_path = persistent/(name+'.json')
    backup = persistent/'checkpoints'/name
    work = WORK_ROOT/name
    route_path = persistent/(name+'.launch.json')
    config = json.loads(config_path.read_text()) if config_path.is_file() else None
    if snapshot and s2ut_warmup is not None:
        from direct_s2st.recipes import option
        saved_warmup = (config or {}).get('research_metadata', {}).get('warmup_updates')
        if saved_warmup is None:
            saved_warmup = option((config or {}).get('command', []), '--warmup-updates')
        if str(saved_warmup) != str(s2ut_warmup):
            raise ValueError(f'S2UT warmup differs from saved run ({saved_warmup} vs {s2ut_warmup}); '
                             'retain its cell-1 warmup to resume, or use a new experiment for a different schedule; '
                             'saved checkpoints are preserved')
    route = json.loads(route_path.read_text()) if route_path.is_file() else None
    if route and (route.get('version') != 1 or route['run_name'] != name or route['kind'] != kind or route['work'] != str(work)):
        raise ValueError('saved launch route belongs to a different training run')
    expected = snapshot['identity']['inputs'].get('/content/s2st-runtime/environment.json') if snapshot else None
    if snapshot and not expected:
        raise ValueError('saved snapshot does not contain the runtime environment identity')
    if snapshot and snapshot['identity']['work'] != str(work):
        raise ValueError('saved work path differs; refusing to restart')
    base = (config or {}).get('research_metadata', {}).get('repository_revision')
    legacy_run = kind == 'tt2' and base == legacy.BASE_REVISION
    if route and route.get('environment_sha256'):
        if expected and expected != route['environment_sha256']:
            raise ValueError('launch environment differs from snapshot')
        expected = route['environment_sha256']
    if route is None:
        if snapshot and not legacy_run:
            # A pre-launcher run must resume its recorded code, never current main.
            if not base:
                raise ValueError('legacy run has no recorded code revision; cannot infer its trainer')
            build_revision = execution_revision = base
        elif legacy_run:
            build_revision, execution_revision = legacy.BASE_REVISION, LEGACY_EXECUTION
        else:
            build_revision = execution_revision = git(ROOT, 'rev-parse', 'HEAD')
        route = dict(version=1, run_name=name, kind=kind, work=str(work),
                     build_revision=build_revision, execution_revision=execution_revision,
                     editable_repo='primary' if snapshot or legacy_run else 'worktree')
    if route['editable_repo'] == 'primary' and git(repo, 'rev-parse', 'HEAD') != route['build_revision']:
        if git(repo, 'status', '--porcelain', '--untracked-files=no'):
            raise ValueError('base checkout has local changes; cannot restore its original editable path')
        run('git', '-C', repo, 'fetch', 'origin', route['build_revision'])
        run('git', '-C', repo, 'checkout', '--detach', route['build_revision'])
    if route['editable_repo'] not in ('primary', 'worktree'):
        raise ValueError('invalid saved editable repository location')
    build = repo if route['editable_repo'] == 'primary' else checkout(repo, route['build_revision'], run)
    execution = build if route['execution_revision'] == route['build_revision'] else checkout(repo, route['execution_revision'], run)
    if legacy_run:
        legacy.validate_revision(build, route['execution_revision'])
    print(f'[cell5-recovery] run={name} mode={"resume" if snapshot else "new"} '
          f'candidate_updates={snapshot["updates"] if snapshot else 0} '
          f'build={route["build_revision"]} execution={route["execution_revision"]}', flush=True)
    python = ensure_environment(build, persistent, run, expected=expected, prefix=RUNTIME_ROOT)
    environment = RUNTIME_ROOT/'environment.json'
    route['environment_sha256'] = hashlib.sha256(environment.read_bytes()).hexdigest()
    if legacy_run and snapshot:
        legacy.validate_snapshot(config, backup, environment, work)
        # This reviewed continuation uses the original optimizer/config. Saved
        # schedule/monitor state is inherited and checked by the trainer.
        if practical_warmup is None and os.environ.get('S2ST_TT2_MONITOR_POLICY'):
            os.environ.setdefault('S2ST_TT2_PRACTICAL_WARMUP', '1000')
    if kind == 'tt2':
        if practical_warmup is not None:
            # 10000 selects the saved schedule on resume, never lengthens it.
            if practical_warmup == 10000:
                os.environ.pop('S2ST_TT2_PRACTICAL_WARMUP', None)
            else:
                os.environ['S2ST_TT2_PRACTICAL_WARMUP'] = str(practical_warmup)
        if practical_warmup is not None or os.environ.get('S2ST_TT2_PRACTICAL_WARMUP') or os.environ.get('S2ST_TT2_MONITOR_POLICY'):
            from tt2_schedule_overlay import prepare
            execution = prepare(execution, ROOT, persistent, name)
    if kind == 's2ut':
        from validation_dependencies import ensure
        ensure(python, RUNTIME_ROOT, persistent, run)
        from s2ut_runtime_overlay import prepare
        execution = prepare(execution, ROOT, persistent, name)
    builder_options = {k: v for k, v in options.items() if k != 'tt2_warmup_updates'}
    os.environ['PYTHONPATH'] = str(execution/'src')+os.pathsep+os.environ.get('PYTHONPATH', '')
    os.environ['S2ST_CONFIG_ROOT'] = str(execution/'configs')
    if kind == 's2ut' and snapshot and (work/'checkpoints/checkpoint_last.pt').is_file():
        run(python, ROOT/'scripts/colab/rescue_local_s2ut.py', '--config', config_path,
            '--work', work, '--backup', backup, '--total', str(total))
    if snapshot:
        print('[cell5-recovery] saved config/batch retained; cell-1 throughput options do not replace them', flush=True)
    else:
        if kind == 's2ut':
            run(python, '-m', 'direct_s2st.s2ut.migrate_labels', '--common-root', data/'common', '--data-root', data/'s2ut/fairseq')
        builder = ('import json,sys\nfrom pathlib import Path\nfrom direct_s2st.colab import make_config\n'
                   'cfg=make_config(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],**json.loads(sys.argv[6]))\n'
                   'print(json.dumps(cfg))')
        # Generate in the actual fixed Python, not the notebook kernel.
        cfg = json.loads(subprocess.check_output([str(python), '-c', builder, str(execution), str(data),
            str(environment), kind, model_size, json.dumps(builder_options)], text=True))
        atomic_write_json(config_path, cfg, resume=True)
        config = cfg
    atomic_write_json(route_path, route, resume=True)
    atomic_write_json(persistent/f'last-training-run-{kind}.json', dict(run_name=name, kind=kind,
        config=str(config_path), backup=str(backup), work=str(work)), overwrite=True)
    command = [python, '-m', 'direct_s2st.colab', '--config', config_path, '--work-root', work,
               '--backup-root', backup, '--total-updates', str(total), '--session-seconds', str(seconds),
               '--resume' if snapshot else '--auto-resume']
    if confirm:
        command.append('--confirm-training')
    run(*command)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('repo', 'persistent', 'data'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--kind', choices=('tt2', 's2ut', 'unit', 'mel'), required=True)
    parser.add_argument('--default-name', required=True)
    parser.add_argument('--run-name', default='')
    parser.add_argument('--model-size', default='smoke')
    parser.add_argument('--options', default='{}')
    parser.add_argument('--total-updates', type=int, default=2)
    parser.add_argument('--session-seconds', type=float, default=0)
    parser.add_argument('--confirm-training', action='store_true')
    args = parser.parse_args()
    log = args.persistent/(args.kind+'-session.log')
    def run(*command):
        print('実行:', *map(str, command), flush=True)
        with log.open('a', encoding='utf-8') as stream:
            process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                    stream.flush()
                status = process.wait()
            except BaseException:
                process.terminate()
                raise
        if status:
            raise RuntimeError(f'command failed ({status}); saved checkpoints retained. See {log}')
    launch(args.repo, args.persistent, args.data, args.kind, args.default_name, args.model_size,
           json.loads(args.options), args.total_updates, args.session_seconds,
           requested=args.run_name, confirm=args.confirm_training, run=run)


if __name__ == '__main__':
    main()
