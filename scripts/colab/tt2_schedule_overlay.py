"""Apply only reviewed schedule/monitor functions to a pinned TT2 trainer."""
import ast
import hashlib
import json
from pathlib import Path

from direct_s2st.io import atomic_write_bytes, atomic_write_json
from direct_s2st.progress import operation, track

MONITOR = 'direct_s2st/translatotron2/monitor.py'
REVIEWED = {
    '97b631236281ba33b8a5520736e80ba0c6031f4cfe20ac676a839d4072a95a79': True,  # d70516a
    '7510b3efec45dab148b2f83a108e2e22d07cb07fddb68970756764a01124542d': False,  # 6e311c4
    '6648895bb16a4f102020355f83b930c290050b6442f0b94b81b9a72106cf92fb': False,  # d55102b..41c6ca2
    '70e99bba5091e503a2901fe8521825d45c746d16687e33420fbf75c574af197d': False,  # 334c14b
}


def functions(source):
    tree = ast.parse(source)
    monitor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Monitor')
    nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
             for name in ('resume_schedule', 'validate_policy')]
    nodes += [next(n for n in monitor.body if isinstance(n, ast.FunctionDef) and n.name == name)
              for name in ('observe', 'state', 'restore')]
    return {n.name: (min([n.lineno] + [d.lineno for d in n.decorator_list])-1, n.end_lineno) for n in nodes}


def patched_monitor(original, reviewed):
    """All other source bytes, including model/Adam/batching, stay unchanged."""
    source, replacement = original.decode(), reviewed.decode()
    old, new = functions(source), functions(replacement)
    lines, replacements = source.splitlines(keepends=True), replacement.splitlines(keepends=True)
    if all(''.join(lines[a:b]) == ''.join(replacements[new[name][0]:new[name][1]])
           for name, (a, b) in old.items()):
        return original
    digest = hashlib.sha256(original).hexdigest()
    if digest not in REVIEWED:
        raise ValueError('TT2 schedule migration requires a reviewed monitor implementation; saved trainer preserved')
    for name, (a, b) in sorted(old.items(), key=lambda item: item[1][0], reverse=True):
        value = ''.join(replacements[new[name][0]:new[name][1]])
        if REVIEWED[digest]:
            value = value.replace('cls.version', 'VERSION').replace('self.version', 'VERSION')
            value = value.replace('self.metrics', 'METRICS').replace('self.score_key', "'loss'")
            value = value.replace('self.metric', "'teacher_forced_dev_loss'")
        lines[a:b] = [value]
    result = ''.join(lines).encode()
    ast.parse(result)
    return result


@operation('cell 5: reviewed TT2 warmup continuation')
def prepare(execution, launcher, persistent, name):
    execution, launcher, persistent = map(Path, (execution, launcher, persistent))
    original = (execution/'src'/MONITOR).read_bytes()
    changed = patched_monitor(original, (launcher/'src'/MONITOR).read_bytes())
    if changed == original:
        return execution
    files = {p.relative_to(execution).as_posix(): p.read_bytes()
             for scope in ('src', 'configs')
             for p in track(sorted((execution/scope).rglob('*')), 'schedule overlay: inspect pinned '+scope)
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    files['src/'+MONITOR] = changed
    hashes = {p: hashlib.sha256(data).hexdigest() for p, data in files.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    # Remain within the verified Git worktree, so recorded Git provenance still
    # resolves to the original execution SHA. Configs are exact pinned copies.
    root = execution/'.schedule-overlays'/identity
    # Interrupted copies resume missing files. Changed copies are never replaced.
    for scope in ('src', 'configs'):
        extras = [p for p in (root/scope).rglob('*') if p.is_file() and
                  '__pycache__' not in p.parts and p.suffix != '.pyc' and p.relative_to(root).as_posix() not in files]
        if extras:
            raise ValueError('unexpected file in schedule overlay; preserve and inspect it')
    for path, data in track(files.items(), 'schedule overlay: verify/copy source'):
        target = root/path
        atomic_write_bytes(target, data, resume=True)
        if target.read_bytes() != data:
            raise ValueError('schedule overlay source verification failed')
    audit = dict(version=1, pinned_execution=str(execution), source_sha256=hashes,
                 overlay=str(root), modified_files=['src/'+MONITOR],
                 modified_functions=['resume_schedule', 'validate_policy', 'Monitor.observe', 'Monitor.state', 'Monitor.restore'],
                 snapshot_identity_unchanged=True)
    atomic_write_json(persistent/'execution-overlays'/f'{name}-schedule-{identity}.json', audit, resume=True)
    print(f'[training-schedule-overlay] verified={identity} model_optimizer_unchanged=True', flush=True)
    return root
