"""Reviewed TTS recovery continuation, preserving original text and audio.

Keep the original plan, ASR/MT journals and dependency identity byte-for-byte.
Only the source overlay is new; never repin a run or reinterpret generated audio.
"""
from pathlib import Path
import ast
import re
import subprocess

from direct_s2st.benchmark_storage import read
from direct_s2st.drive_staging import raw_sha
from direct_s2st.io import atomic_write_json, read_jsonl
from direct_s2st.journal import digest
from direct_s2st.progress import operation, track

BASE_REVISION = 'f9f1520047bca2c0e977b581cab70168892091e2'
PINNED_LOADER_REVISION = 'c741fca0c8408d67750a480cb989370fbcab9071'
REVIEWED_FILES = {
    'src/direct_s2st/cascade/tts.py',
    'src/direct_s2st/cascade/benchmark_worker.py',
    'scripts/colab/benchmark_session.py',
    'scripts/colab/cascade_loader_overlay.py',
    'tests/unit/test_qwen_tts_loading.py',
    'tests/unit/test_cascade_benchmark.py',
    'tests/unit/test_cascade_tts_resume.py',
    'tests/gpu/test_cascade_qwen_preflight.py',
    'notebooks/colab_training.ipynb', 'docs/COLAB.md', 'DESIGN.md',
}


def verify_text_code(root, base):
    """TTS-only worker changes cannot alter old text identity/decoding/history."""
    name = 'src/direct_s2st/cascade/benchmark_worker.py'
    previous = ast.parse(git(root, 'show', base+':'+name))
    current = ast.parse((Path(root)/name).read_text(encoding='utf-8'))
    protected = ('runtime_identity', 'common_rows', 'phase_identity', 'loader',
                 'read_phase', 'text_phase', 'valid_saved')
    def functions(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    before, after = functions(previous), functions(current)
    if any(key not in before or before[key] != after.get(key) for key in protected):
        raise ValueError('unreviewed worker ASR/MT/identity changes; saved results retained')


def verify_tts_contract(root):
    """Only the known, complete-snapshot loader's generated WAVs may migrate."""
    name = 'src/direct_s2st/cascade/tts.py'
    before = ast.parse(git(root, 'show', PINNED_LOADER_REVISION+':'+name))
    after = ast.parse((Path(root)/name).read_text(encoding='utf-8'))
    def contract(tree):
        bundle = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pinned_snapshot')
        adapter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenTTS')
        calls = [ast.dump(n, include_attributes=False) for n in ast.walk(adapter)
                 if isinstance(n, ast.Call) and ast.unparse(n.func) in
                 ('Qwen3TTSModel.from_pretrained', 'self.model.generate_custom_voice')]
        return ast.dump(bundle, include_attributes=False), calls
    if contract(before) != contract(after):
        raise ValueError('Qwen snapshot/model loading/decoding contract changed; retain existing WAVs separately')


def verified_parent(root, plan_path, plan, revision):
    path = Path(plan['output'])/'execution-overlays'/(PINNED_LOADER_REVISION+'.json')
    if not path.is_file():
        return None
    receipt = read(path)
    body = receipt['body']
    expected = dict(version=1, reason='complete_pinned_qwen_snapshot_before_local_sdk_load',
                    plan=plan['identity'], original_plan_sha256=raw_sha(plan_path),
                    base_revision=BASE_REVISION, execution_revision=PINNED_LOADER_REVISION,
                    changed_files=sorted(git(root, 'diff', '--name-only', BASE_REVISION,
                                             PINNED_LOADER_REVISION, '--').splitlines()))
    if (receipt['sha256'] != digest(body) or body != expected
            or set(body['changed_files']) - REVIEWED_FILES
            or git(root, 'merge-base', PINNED_LOADER_REVISION, revision) != PINNED_LOADER_REVISION):
        raise ValueError('previous pinned-loader adoption receipt/lineage changed; saved data retained')
    verify_tts_contract(root)
    return receipt['sha256']


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def validate_revision(root, base, revision):
    if base != BASE_REVISION or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('this loader repair supports only the reviewed original cascade revision')
    if git(root, 'rev-parse', 'HEAD') != revision or git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('cascade loader overlay must be the requested clean commit')
    if git(root, 'merge-base', base, revision) != base:
        raise ValueError('cascade loader repair is not a descendant of its reviewed base')
    changed = set(git(root, 'diff', '--name-only', base, revision, '--').splitlines())
    if not changed or changed - REVIEWED_FILES or 'src/direct_s2st/cascade/tts.py' not in changed:
        raise ValueError('unreviewed ASR/MT, worker, models, dependencies or other changes; cannot reuse old evaluation')
    if 'src/direct_s2st/cascade/benchmark_worker.py' in changed:
        verify_text_code(root, base)
    return sorted(changed)


def require_no_tts_outputs(plan):
    output, local = Path(plan['output']), Path(plan['local'])
    index = output/'audio-index.json'
    if index.is_file():
        document = read(index)
        if document['sha256'] != digest(document['body']) or document['body']['identity'] != plan['identity']:
            raise ValueError('old cascade audio index is damaged')
        if any(r['status'] == 'success' for r in document['body']['samples']):
            raise ValueError('old TTS already generated audio; loader repair cannot silently mix models')
    known_archives = set()
    for path in track(sorted((output/'audio').glob('*.json')), 'cascade repair: inspect saved WAV receipts'):
        receipt = read(path)
        if receipt['sha256'] != digest(receipt['body']):
            raise ValueError('old cascade WAV receipt is damaged')
        if any(name.endswith('.wav') for name in receipt['body']['files']):
            raise ValueError('old TTS has published WAVs; preserve them and use a separate evaluation')
        known_archives.add(receipt['body']['archive'])
    if {p.name for p in (output/'audio').glob('*.zip')} - known_archives:
        raise ValueError('uncommitted old TTS archives exist; retain and review before loader repair')
    if any((local/'audio').rglob('*.wav')):
        raise ValueError('old local TTS WAVs exist; loader repair requires review, not automatic reuse')


@operation('cascade repair: verify and adopt TTS-only execution overlay')
def approve(root, plan_path, plan, revision, *, adopt=False):
    changed = validate_revision(root, plan['execution_revision'], revision)
    body = dict(version=1, reason='complete_pinned_qwen_snapshot_before_local_sdk_load',
                plan=plan['identity'], original_plan_sha256=raw_sha(plan_path),
                base_revision=plan['execution_revision'], execution_revision=revision, changed_files=changed)
    parent = verified_parent(root, plan_path, plan, revision)
    if parent:
        body.update(version=2, reason='verified_pinned_loader_audio_with_isolated_preflight',
                    parent_revision=PINNED_LOADER_REVISION, parent_receipt_sha256=parent)
    path = Path(plan['output'])/'execution-overlays'/(revision+'.json')
    if path.is_file():
        receipt = read(path)
        if receipt['sha256'] != digest(receipt['body']) or receipt['body'] != body:
            raise ValueError('cascade loader execution receipt changed')
        # Audio produced AFTER adoption is safe to resume under this same code.
        return receipt
    if not adopt:
        raise ValueError('cascade loader repair not adopted; rerun normal cell 6 first')
    if not parent:
        require_no_tts_outputs(plan)
    receipt = dict(body=body, sha256=digest(body))
    atomic_write_json(path, receipt)
    return receipt


@operation('cascade repair: restore original editable build location')
def original_build(root, revision, invoke):
    if revision != BASE_REVISION:
        raise ValueError('unsupported original cascade dependency build')
    # This exact path was installed editable and is part of the old pip freeze.
    build = Path('/content/s2st-launchers')/revision
    if not build.exists():
        invoke('git', '-C', root, 'fetch', 'origin', revision)
        invoke('git', '-C', root, 'worktree', 'add', '--detach', build, revision)
    if git(build, 'rev-parse', 'HEAD') != revision or git(build, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('original cascade build changed; dependency identity must not be forged')
    return build
