"""Reviewed S2UT runtime/recovery fixes, preserving pinned scientific inputs."""
import ast
import hashlib
import json
from pathlib import Path

from direct_s2st.io import atomic_write_bytes, atomic_write_json
from direct_s2st.progress import operation, track

PERFORMANCE = 'src/direct_s2st/s2ut/performance.py'
AUDIT = 'src/direct_s2st/s2ut/fairseq_train.py'
RUNTIME = 'src/direct_s2st/train_runtime.py'
SESSION = 'src/direct_s2st/session_runtime.py'
CHECKPOINT = 'src/direct_s2st/s2ut/checkpoint.py'
RECOVERY = 'src/direct_s2st/recovery.py'
RECIPE = 'src/direct_s2st/s2ut/recipe_identity.py'
PREFLIGHT = 'src/direct_s2st/s2ut/preflight.py'
MONITOR = 'src/direct_s2st/s2ut/monitor.py'
SHARED_MONITOR = 'src/direct_s2st/translatotron2/monitor.py'
REVIEWED = {  # 2486164: the reported run's immutable execution code.
    ('eb384f0900fb47f2cc8def3b08d3cb6a4eae9b675332f3696edddabdc92469a1',
     '1235756223b300578acde2a081de116105fadb001cb383417fe4685b93ca28a3'),
    ('c2d257ab42b41546bb307e4ca1b62219b835468d20cd39d5c454922f92d5de56',
     'd53124076a1804e9e82f1e90f3e6403d1a8bb5295b71829b2ee2ffe3f8933f16'),  # e56ca1c
    ('ee51261415ccdbd26b7f609212eb60199d40f20b4309e33194abfab9fad501a7',
     '0a6cb3ea1cf20906637eafd5adcc53e6beb7e50a886bdcd956724de691403863'),  # 39b41c3
    ('ee51261415ccdbd26b7f609212eb60199d40f20b4309e33194abfab9fad501a7',
     '4c2d4e610a631054c3069aa1782eb74ec4cbfac34270b55b3f00b9d182f3c1a1'),  # 761a87f
    ('ee51261415ccdbd26b7f609212eb60199d40f20b4309e33194abfab9fad501a7',
     '90aaf2300c18ded5899f9d6ba1fb22d0b26fbbba1c3f5ea826fe82eb40ae441b'),  # f9cbe6d
}

EXTRA_REVIEWED = {
    RUNTIME: '51a8f053a7d88d19239840b8d67f419f9d1d97d85bf8eaf46ae8f2a0661c30a6',
    SESSION: ('d70b530541b60d39da4f2ad270b339bef24af8cf903de393cd874c1ee4c60f18',
              '32072cd96da2a6e81ec1275f5cf11b71a20ed42da1c35a72f3d091b59b9d743a',
              '6ba54624e2bd4f9529fb887eb452ad242f2245a802dd1e17a3b7d0dd484e46f0'),
    CHECKPOINT: 'd9672003743cd110e215eeabddfb6140e0a806d26a1e005f88e9f6b92c487496',
    PREFLIGHT: '6a14765746c3b871a80fc0be869b892d9d81ec2d7da224191c7d48fa4d706ece',
    MONITOR: ('85fffccf90b5b9173c15695d96b9303c174bdcb18dc9490f845b185d915d2c1b',
              '4e3f91b025aacfdc166261a0106c1d0833eeb9c27881a8e8b9c5f1a2c000a3fc'),
}


def audit_span(source):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'TrainingAudit')
    return node.lineno-1, node.end_lineno


def metadata_span(source):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_research_metadata')
    return min([node.lineno]+[d.lineno for d in node.decorator_list])-1, node.end_lineno


def main_span(source):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    return min([node.lineno]+[d.lineno for d in node.decorator_list])-1, node.end_lineno


def replacements(execution, launcher):
    old = {name: (execution/name).read_bytes() for name in (PERFORMANCE, AUDIT)}
    new = {name: (launcher/name).read_bytes() for name in old}
    lines = old[AUDIT].decode().splitlines(keepends=True)
    # Descending spans keep original line positions valid.
    for span in sorted((audit_span, metadata_span, main_span), key=lambda f: f(old[AUDIT])[0], reverse=True):
        start, end = span(old[AUDIT])
        a, b = span(new[AUDIT])
        lines[start:end] = new[AUDIT].decode().splitlines(keepends=True)[a:b]
    changed = {PERFORMANCE: new[PERFORMANCE], AUDIT: ''.join(lines).encode()}
    if changed != old and tuple(hashlib.sha256(old[name]).hexdigest() for name in (PERFORMANCE, AUDIT)) not in REVIEWED:
        raise ValueError('S2UT FP16 repair requires reviewed pinned runtime sources; saved run preserved')
    changed = {name: value for name, value in changed.items() if value != old[name]}
    for name in (RUNTIME, SESSION, CHECKPOINT, PREFLIGHT, MONITOR):
        original, reviewed = (execution/name).read_bytes(), (launcher/name).read_bytes()
        replacement = reviewed
        if name == RUNTIME:
            def span(source):
                node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'checkpoint_saved')
                return min([node.lineno]+[d.lineno for d in node.decorator_list])-1, node.end_lineno
            a, b = span(original)
            c, d = span(reviewed)
            lines = original.decode().splitlines(keepends=True)
            lines[a:b] = reviewed.decode().splitlines(keepends=True)[c:d]
            replacement = ''.join(lines).encode()
        if replacement != original:
            expected = EXTRA_REVIEWED[name]
            if hashlib.sha256(original).hexdigest() not in (expected if isinstance(expected, tuple) else (expected,)):
                raise ValueError('S2UT recovery repair requires reviewed pinned sources: '+name)
            changed[name] = replacement
    from tt2_schedule_overlay import patched_monitor
    original = (execution/SHARED_MONITOR).read_bytes()
    replacement = patched_monitor(original, (launcher/SHARED_MONITOR).read_bytes())
    if replacement != original:
        changed[SHARED_MONITOR] = replacement
    for name in (RECOVERY, RECIPE):
        if (execution/name).is_file():
            if (execution/name).read_bytes() != (launcher/name).read_bytes():
                raise ValueError('S2UT recovery helper differs from reviewed source: '+name)
        else:
            changed[name] = (launcher/name).read_bytes()
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
             modified_files=list(changed), modified_scope=['FP16 runtime retry', 'TrainingAudit',
                 'checkpoint validation/freezing', 'resident error-exit backup',
                 'recipe argv[0] canonical comparison; original saved fingerprint retained',
                 'metadata-only predownload validation; training audio checks unchanged',
                 'future-only all-loss stopping v3; selection metrics unchanged; S2UT absolute update cap 3000'],
             snapshot_identity_unchanged=True, model_optimizer_recipe_unchanged=True), resume=True)
    print(f'[s2ut-fp16-overlay] verified={identity} config_environment_checkpoint_identity_unchanged=true', flush=True)
    return root
