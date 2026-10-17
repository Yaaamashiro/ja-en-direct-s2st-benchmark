"""Reviewed vocoder-only monitor overlay for normal-cell legacy resume.

Immutable configs, model factories, GAN objectives and both Adam states remain
unchanged. Unknown old implementations fail closed instead of being patched.
"""
import ast
import hashlib
import json
from pathlib import Path

from direct_s2st.io import atomic_write_bytes, atomic_write_json
from direct_s2st.progress import operation, track

MODULES = {
    'src/direct_s2st/session_runtime.py': '6ba54624e2bd4f9529fb887eb452ad242f2245a802dd1e17a3b7d0dd484e46f0',
    'src/direct_s2st/vocoders/train.py': '785c463f26bb41a51e7924f6316bd2f981d3da92349ed6c6e7746785d411bb0a',
    'src/direct_s2st/vocoders/verification.py': '11e7694b46ae4cf07e247686102272a08ad3d6b37156083816be7997b5626649',
}
FUNCTIONS = {
    'src/direct_s2st/colab.py': {
        'stage_training': '9bb1e9d447be6e4a1d7da83858f382e0248f4f3792c3e898f0b60725cbebd6d3',
        'publish_packed': 'a209889492b52d080b602083639afbae5f36c74ed5c253b71294fdc5d5d6eef7'},
    'src/direct_s2st/translatotron2/monitor.py': {
        'early_stop_result': '62f2e488a453916852a50cf22fd3fd0c8668627ffa4d348f344f7de7ef36702b'},
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def replacements(execution, launcher):
    changed = {}
    for name, expected in MODULES.items():
        original, new = (execution/name).read_bytes(), (launcher/name).read_bytes()
        if original != new:
            if sha(original) != expected:
                raise ValueError('unreviewed vocoder resume implementation: '+name)
            changed[name] = new
    for name, functions in FUNCTIONS.items():
        old, new = (execution/name).read_text(encoding='utf-8'), (launcher/name).read_text(encoding='utf-8')
        nodes = {n.name: n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
        reviewed = {n.name: n for n in ast.parse(new).body if isinstance(n, ast.FunctionDef)}
        lines = old.splitlines(keepends=True)
        for function in sorted(functions, key=lambda k: nodes[k].lineno, reverse=True):
            a, b = nodes[function], reviewed[function]
            original, replacement = ast.get_source_segment(old, a), ast.get_source_segment(new, b)
            if original != replacement:
                if sha(original.encode()) != functions[function]:
                    raise ValueError('unreviewed vocoder recovery function: '+function)
                lines[a.lineno-1:a.end_lineno] = [replacement+'\n']
        result = ''.join(lines).encode()
        if result != old.encode():
            changed[name] = result
    name = 'src/direct_s2st/vocoders/monitor.py'
    new = (launcher/name).read_bytes()
    if (execution/name).is_file() and (execution/name).read_bytes() != new:
        raise ValueError('saved vocoder monitor implementation changed; preserve its policy')
    if not (execution/name).is_file():
        changed[name] = new
    for data in changed.values():
        ast.parse(data)
    return changed


@operation('cell 5: reviewed vocoder reconstruction monitor')
def prepare(execution, launcher, persistent, name):
    execution, launcher, persistent = map(Path, (execution, launcher, persistent))
    changed = replacements(execution, launcher)
    if not changed:
        return execution
    files = {p.relative_to(execution).as_posix(): p.read_bytes()
             for scope in ('src', 'configs') for p in track(sorted((execution/scope).rglob('*')), 'vocoder overlay: inspect '+scope)
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    files.update(changed)
    hashes = {p: sha(v) for p, v in files.items()}
    identity = sha(json.dumps(hashes, sort_keys=True).encode())
    root = execution/'.vocoder-overlays'/identity
    for scope in ('src', 'configs'):
        if any(p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
               and p.relative_to(root).as_posix() not in files for p in (root/scope).rglob('*')):
            raise ValueError('unexpected files in vocoder overlay')
    for path, data in track(files.items(), 'vocoder overlay: verify/copy source'):
        atomic_write_bytes(root/path, data, resume=True)
        if (root/path).read_bytes() != data:
            raise ValueError('vocoder overlay copy verification failed')
    atomic_write_json(persistent/'execution-overlays'/f'{name}-vocoder-{identity}.json',
        dict(version=1, pinned_execution=str(execution), overlay=str(root), source_sha256=hashes,
             modified_files=sorted(changed), snapshot_identity_unchanged=True,
             generator_discriminator_objectives_optimizers_unchanged=True), resume=True)
    return root
