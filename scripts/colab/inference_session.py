"""Cells 6/7: bounded dev listening, using verified saved runs, never training."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from direct_s2st.colab import latest, restore
from direct_s2st.drive_staging import copy_verified, raw_sha, register
from direct_s2st.io import atomic_write_json, read_jsonl
from direct_s2st.journal import digest
from direct_s2st.preparation import file_stamp
from direct_s2st.progress import operation, track
from direct_s2st.recipes import option, TT2_VOCODERS, S2UT_VOCODERS
from direct_s2st.preview import settings, vocoder_settings

KINDS = ('tt2', 's2ut', 'unit', 'mel')
PREVIEW_BACKEND_VERSION = 2


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_config(config, saved, backup):
    expected = config
    if saved != config:
        from direct_s2st.batch_calibration import selected_config
        measured = read(backup/'batch-calibration.json')
        size = measured.get('selected')
        if type(size) is not int:
            raise ValueError('saved inference run has no valid batch calibration')
        expected = selected_config(config, size)
    if saved != expected:
        raise ValueError('snapshot does not match saved training configuration')


@operation('preview: discover actual durable run names')
def choose(persistent, kind, requested=''):
    if requested and not re.fullmatch('[A-Za-z0-9_-]+', requested):
        raise ValueError('invalid preview run name')
    names = []
    for path in track(sorted(persistent.glob(kind+'-*.json')), 'preview: inspect saved runs'):
        if path.name.endswith('.launch.json'):
            continue
        cfg = read(path)
        command = cfg.get('command', [])
        if cfg.get('kind') != ('vocoder' if kind in ('unit', 'mel') else kind):
            continue
        if kind in ('unit', 'mel') and option(command, '--kind') != kind:
            continue
        if any((persistent/'checkpoints'/path.stem).glob('*/snapshot.json')):
            names.append(path.stem)
    pointer = persistent/f'last-training-run-{kind}.json'
    preferred = requested or (read(pointer).get('run_name') if pointer.is_file() else None)
    if preferred:
        if preferred not in names:
            raise ValueError(f'{kind}: selected run has no durable snapshot: {preferred}; candidates={names}')
        return preferred
    if len(names) != 1:
        raise ValueError(f'{kind}: select PREVIEW_RUN_NAMES in cell 1; durable candidates={names}')
    return names[0]


@operation('preview: restore checkpoints without staging training inputs')
def saved_run(persistent, kind, requested, local):
    name = choose(persistent, kind, requested)
    config = read(persistent/(name+'.json'))
    backup = persistent/'checkpoints'/name
    markers = sorted(backup.glob('*/snapshot.json'), reverse=True)
    identity = None
    for marker in markers:
        try:
            candidate = read(marker)
            if type(candidate.get('updates')) is not int or candidate['updates'] < 1:
                continue
            identity = candidate['identity']
            break
        except (ValueError, KeyError, OSError):
            continue
    if identity is None:
        raise ValueError(f'no readable trained snapshot for {name}')
    verify_config(config, identity['config'], backup)
    if identity['work'] != '/content/s2st-work/'+name:
        raise ValueError('saved training work path does not match run name')
    snapshot = latest(backup, identity)
    if snapshot is None:
        raise ValueError(f'no verified snapshot for {name}; original backups preserved')
    directory, state = snapshot
    # Separate from training work; same basename retains the true run ID.
    work = local/'models'/directory.name/name
    if not work.is_dir() or any(not (work/p).is_file() or raw_sha(work/p) != sha
                                for p, sha in state['files'].items()):
        restore(snapshot, work)
    route_path = persistent/(name+'.launch.json')
    route = read(route_path) if route_path.is_file() else None
    if route:
        if (route.get('version') != 1 or route['run_name'] != name or route['kind'] != kind
                or route['work'] != identity['work']):
            raise ValueError('saved launch route differs from training run')
    else:
        base = config.get('research_metadata', {}).get('repository_revision')
        if not base:
            raise ValueError(f'{name}: saved build revision missing; cannot approximate dependencies')
        route = dict(build_revision=base, editable_repo='primary')
    expected = identity['inputs'].get('/content/s2st-runtime/environment.json')
    if not expected or route.get('environment_sha256', expected) != expected:
        raise ValueError('saved runtime identity missing or conflicting')
    return dict(name=name, config=config, work=str(work), snapshot=str(directory),
                updates=state['updates'], identity=identity, files=state['files'],
                route=route, environment_sha256=expected)


def checked_metadata(run, path):
    path = Path(path)
    expected = run['identity']['inputs'].get(str(path))
    if expected is None or raw_sha(path) != expected:
        raise ValueError(f'prepared inference metadata changed or unbound: {path}')


@operation('cell 6: prepare small dev listening preview')
def prepare(persistent, data, local, requested, count, revision, *,
            tt2_vocoder_mode='hifigan', s2ut_vocoder_mode='trained'):
    if not 1 <= count <= 10:
        raise ValueError('PREVIEW_SAMPLES must be 1..10')
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('preview execution requires an immutable revision')
    vocoders = vocoder_settings(tt2_vocoder_mode, s2ut_vocoder_mode)
    from direct_s2st.drive_staging import ensure_local, read_common_manifest_row
    local = ensure_local(local)
    required = ['tt2', 's2ut']
    if s2ut_vocoder_mode == 'trained':
        required.append('unit')
    if tt2_vocoder_mode == 'hifigan':
        required.append('mel')
    runs = {kind: saved_run(persistent, kind, requested.get(kind, ''), local) for kind in required}
    common = data/'common/dev.jsonl'
    # Vocoder fitting identities include common manifests. Verify their source,
    # without hashing the full corpus, feature ZIPs or training audio.
    for kind in ('unit', 'mel'):
        if kind in runs:
            checked_metadata(runs[kind], common)
    raw = list(read_jsonl(common))
    if len({r['pair_id'] for r in raw}) != len(raw) or len(raw) < count or any(r['split'] != 'dev' for r in raw):
        raise ValueError('invalid common dev metadata')
    ids = sorted(r['pair_id'] for r in raw)[:count]
    if any(not re.fullmatch('[A-Za-z0-9_-]+', pair_id) for pair_id in ids):
        raise ValueError('unsafe preview sample ID')
    selected = {r['pair_id']: r for r in raw if r['pair_id'] in ids}
    for kind in ('tt2', 's2ut'):
        cmd = runs[kind]['config']['command']
        root = option(cmd, '--data-root')
        if root is None and kind == 's2ut':
            # Pinned fairseq trainer takes the prepared data root positionally.
            root = cmd[3]
        if not root or '{' in root:
            raise ValueError('cannot determine saved prepared data root')
        runs[kind]['data_root'] = root
        runs[kind]['metadata'] = {}
        directory = Path(root)
        for path in track(sorted(directory.rglob('*')), f'preview: verify {kind} metadata'):
            if path.is_file() and path.suffix in ('.json', '.tsv', '.yaml', '.txt', '.model', '.vocab'):
                # Check only files originally bound to the run (not later receipts).
                if str(path) in runs[kind]['identity']['inputs']:
                    checked_metadata(runs[kind], path)
                    runs[kind]['metadata'][str(path)] = runs[kind]['identity']['inputs'][str(path)]
        checked_metadata(runs[kind], directory/'dev.tsv')
    if tt2_vocoder_mode == 'griffin_lim':
        from direct_s2st.vocoders.griffin_lim import validate_spec
        spec_path = Path(runs['tt2']['data_root'])/'mel-spec.json'
        checked_metadata(runs['tt2'], spec_path)
        validate_spec(read(spec_path))
    if s2ut_vocoder_mode == 'paper':
        from direct_s2st.vocoders.reference import validate_reference_units
        lock_path = Path(runs['s2ut']['data_root'])/'data-lock.json'
        checked_metadata(runs['s2ut'], lock_path)
        validate_reference_units(read(lock_path)['unit_configuration'])
    body = dict(version=2, purpose='dev_listening_only_not_benchmark', split='dev', ids=ids,
                common_root=str(data/'common'), common_sha256=raw_sha(common),
                execution_revision=revision, runs=runs, vocoders=vocoders)
    token = digest(body)[:24]
    output = persistent/'preview'/token
    owned = local/token
    rows, mapping = [], {}
    for pair_id in track(ids, 'preview: copy only selected Japanese/English WAVs'):
        row = selected[pair_id]
        for language in ('ja', 'en'):
            row = read_common_manifest_row(row, language)
            source = Path(row[f'{language}_audio'])
            checksum = row[f'{language}_sha256']
            target = owned/'originals'/f'{pair_id}-{language}.wav'
            if not target.is_file() or raw_sha(target) != checksum:
                copy_verified(source, target, checksum)
            register(mapping, source, target, file_stamp(source))
            # Original prepared TSV may retain an older absolute mount path.
            if language == 'ja':
                import csv
                for kind in ('tt2', 's2ut'):
                    with (Path(runs[kind]['data_root'])/'dev.tsv').open(encoding='utf-8') as stream:
                        match = next(r for r in csv.DictReader(stream, delimiter='\t', quoting=csv.QUOTE_NONE) if r['id'] == pair_id)
                    register(mapping, match['src_audio'], target, file_stamp(source))
            row[f'{language}_local'] = str(target)
        rows.append(row)
    body.update(output=str(output), local=str(owned), rows=rows)
    owned.mkdir(parents=True, exist_ok=True)
    atomic_write_json(owned/'input-map.json', dict(rows=mapping, sha256=digest(mapping)), overwrite=True)
    atomic_write_json(owned/'sample-ids.json', ids, resume=True)
    atomic_write_json(output/'plan.json', body, resume=True)
    atomic_write_json(persistent/'preview/latest.json', dict(plan=str(output/'plan.json'), sha256=raw_sha(output/'plan.json')), overwrite=True)
    print(json.dumps(dict(plan=str(output/'plan.json'), ids=ids,
                          trained_runs={k: v['name'] for k,v in runs.items()}, vocoders=vocoders)), flush=True)
    return body


@operation('preview: restore one saved dependency environment')
def environment(repo, persistent, run, invoke):
    from training_session import checkout, git
    from environment_recovery import ensure_environment
    route = run['route']
    revision = route['build_revision']
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('saved build revision is not immutable')
    if route['editable_repo'] == 'primary':
        if git(repo, 'rev-parse', 'HEAD') != revision:
            if git(repo, 'status', '--porcelain', '--untracked-files=no'):
                raise ValueError('original repository has local changes; cannot restore editable path')
            invoke('git', '-C', repo, 'fetch', 'origin', revision)
            invoke('git', '-C', repo, 'checkout', '--detach', revision)
        build = repo
    elif route['editable_repo'] == 'worktree':
        build = checkout(repo, revision, invoke)
    else:
        raise ValueError('unsupported saved editable repository location')
    python = ensure_environment(build, persistent, invoke, expected=run['environment_sha256'])
    os.environ['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+'/content/s2st-runtime/fairseq'
    os.environ['S2ST_CONFIG_ROOT'] = str(ROOT/'configs')
    # Read the saved generator configuration from its actual original checkout.
    if run['config']['kind'] == 'vocoder':
        config = Path(option(run['config']['command'], '--config'))
        checked_metadata(run, config)
    return python


@operation('cell 7: translate and reconstruct selected dev examples')
def execute(repo, persistent, invoke, *, tt2_vocoder_mode=None, s2ut_vocoder_mode=None):
    pointer = read(persistent/'preview/latest.json')
    path = Path(pointer['plan'])
    if raw_sha(path) != pointer['sha256']:
        raise ValueError('preview plan changed; rerun cell 6')
    plan = read(path)
    selected = settings(plan)
    if ((tt2_vocoder_mode is not None and tt2_vocoder_mode != selected['mel']['mode'])
            or (s2ut_vocoder_mode is not None and s2ut_vocoder_mode != selected['unit']['mode'])):
        raise ValueError('preview vocoder settings changed; rerun normal cell 6 before cell 7')
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if plan['execution_revision'] != revision:
        raise ValueError('preview execution revision changed; rerun cell 6')
    owned = Path(plan['local'])
    if not (owned/'input-map.json').is_file():
        raise ValueError('local preview copies were lost; rerun normal cell 6')
    if raw_sha(Path(plan['common_root'])/'dev.jsonl') != plan['common_sha256']:
        raise ValueError('common dev metadata changed; preview retained')
    os.environ['S2ST_INPUT_MAP'] = str(owned/'input-map.json')
    for kind in KINDS:
        owner = kind if kind in plan['runs'] else ('tt2' if kind == 'mel' else 's2ut')
        run = plan['runs'][owner]
        for name, sha in track(run.get('metadata', {}).items(), f'preview: recheck {kind} metadata'):
            if raw_sha(Path(name)) != sha:
                raise ValueError('prepared metadata changed; preview retained')
        for relative, sha in track(run['files'].items(), f'preview: verify restored {kind}'):
            if raw_sha(Path(run['work'])/relative) != sha:
                raise ValueError('restored checkpoint changed; rerun cell 6')
        python = environment(repo, persistent, run, invoke)
        invoke(python, '-m', 'direct_s2st.preview', '--plan', path, '--kind', kind)
    # Publish only tiny preview files; originals, training outputs and test
    # benchmark namespaces are never changed.
    from direct_s2st.preview import publish
    publish(plan)


def main():
    from direct_s2st.inference_logging import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'run'), required=True)
    for key in ('repo', 'persistent', 'data'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--local', type=Path, default=Path('/content/s2st-preview'))
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--run-names', default='{}')
    parser.add_argument('--tt2-vocoder-mode', choices=TT2_VOCODERS, default='griffin_lim')
    parser.add_argument('--s2ut-vocoder-mode', choices=S2UT_VOCODERS, default='trained')
    args = parser.parse_args()
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and args.persistent.resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('preview outputs must be outside CORPUS_ROOT')
    def invoke(*command):
        print('実行:', *map(str, command), flush=True)
        subprocess.run(list(map(str, command)), check=True)
    if args.action == 'prepare':
        revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        prepare(args.persistent, args.data, args.local, json.loads(args.run_names), args.samples, revision,
                tt2_vocoder_mode=args.tt2_vocoder_mode, s2ut_vocoder_mode=args.s2ut_vocoder_mode)
    else:
        execute(args.repo, args.persistent, invoke, tt2_vocoder_mode=args.tt2_vocoder_mode,
                s2ut_vocoder_mode=args.s2ut_vocoder_mode)


if __name__ == '__main__':
    main()
