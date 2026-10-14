"""Reviewed FP16 retry fix for a pinned S2UT execution, without recipe changes."""
import ast
import hashlib
import json
from pathlib import Path

from direct_s2st.io import atomic_write_bytes, atomic_write_json
from direct_s2st.progress import operation, track

PERFORMANCE = 'src/direct_s2st/s2ut/performance.py'
AUDIT = 'src/direct_s2st/s2ut/fairseq_train.py'
REVIEWED = {  # 2486164: the reported run's immutable execution code.
    ('eb384f0900fb47f2cc8def3b08d3cb6a4eae9b675332f3696edddabdc92469a1',
     '1235756223b300578acde2a081de116105fadb001cb383417fe4685b93ca28a3'),
}


def audit_span(source):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'TrainingAudit')
    return node.lineno-1, node.end_lineno


def replacements(execution, launcher):
    old = {name: (execution/name).read_bytes() for name in (PERFORMANCE, AUDIT)}
    new = {name: (launcher/name).read_bytes() for name in old}
    start, end = audit_span(old[AUDIT])
    a, b = audit_span(new[AUDIT])
    lines = old[AUDIT].decode().splitlines(keepends=True)
    lines[start:end] = new[AUDIT].decode().splitlines(keepends=True)[a:b]
    changed = {PERFORMANCE: new[PERFORMANCE], AUDIT: ''.join(lines).encode()}
    if changed == old:
        return {}
    if tuple(hashlib.sha256(old[name]).hexdigest() for name in (PERFORMANCE, AUDIT)) not in REVIEWED:
        raise ValueError('S2UT FP16 repair requires reviewed pinned runtime sources; saved run preserved')
    for value in changed.values():
        ast.parse(value)
    return changed


@operation('cell 5: reviewed S2UT FP16 retry repair')
def prepare(execution, launcher, persistent, name):
    execution, launcher, persistent = map(Path, (execution, launcher, persistent))
    changed = replacements(execution, launcher)
    if not changed:
        return execution
    files = {p.relative_to(execution).as_posix(): p.read_bytes()
             for scope in ('src', 'configs')
             for p in track(sorted((execution/scope).rglob('*')), 'FP16 overlay: inspect pinned '+scope)
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    files.update(changed)
    hashes = {path: hashlib.sha256(value).hexdigest() for path, value in files.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    # Inside the original Git checkout: provenance and scientific config remain pinned.
    root = execution/'.fp16-overlays'/identity
    for scope in ('src', 'configs'):
        if any(p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc' and
               p.relative_to(root).as_posix() not in files for p in (root/scope).rglob('*')):
            raise ValueError('unexpected file in FP16 overlay; preserve and inspect it')
    for path, value in track(files.items(), 'FP16 overlay: verify/copy source'):
        target = root/path
        atomic_write_bytes(target, value, resume=True)
        if target.read_bytes() != value:
            raise ValueError('FP16 overlay verification failed')
    atomic_write_json(persistent/'execution-overlays'/f'{name}-fp16-{identity}.json',
        dict(version=1, pinned_execution=str(execution), overlay=str(root), source_sha256=hashes,
             modified_files=list(changed), modified_scope=['FP16 runtime retry', 'TrainingAudit'],
             snapshot_identity_unchanged=True, model_optimizer_recipe_unchanged=True), resume=True)
    print(f'[s2ut-fp16-overlay] verified={identity} config_environment_checkpoint_identity_unchanged=true', flush=True)
    return root
