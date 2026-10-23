"""Reviewed loader-only continuation of a cascade stopped before any TTS WAV.

Keep the original plan, ASR/MT journals and dependency identity byte-for-byte.
Only the source overlay is new; never repin a run or reinterpret generated audio.
"""
from pathlib import Path
import re
import subprocess

from direct_s2st.benchmark_storage import read
from direct_s2st.drive_staging import raw_sha
from direct_s2st.io import atomic_write_json, read_jsonl
from direct_s2st.journal import digest
from direct_s2st.progress import operation, track

BASE_REVISION = 'f9f1520047bca2c0e977b581cab70168892091e2'
REVIEWED_FILES = {
    'src/direct_s2st/cascade/tts.py',
    'scripts/colab/benchmark_session.py',
    'scripts/colab/cascade_loader_overlay.py',
    'tests/unit/test_qwen_tts_loading.py',
    'tests/unit/test_cascade_benchmark.py',
    'notebooks/colab_training.ipynb', 'docs/COLAB.md', 'DESIGN.md',
}


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


@operation('cascade repair: verify and adopt loader-only execution overlay')
def approve(root, plan_path, plan, revision, *, adopt=False):
    changed = validate_revision(root, plan['execution_revision'], revision)
    body = dict(version=1, reason='complete_pinned_qwen_snapshot_before_local_sdk_load',
                plan=plan['identity'], original_plan_sha256=raw_sha(plan_path),
                base_revision=plan['execution_revision'], execution_revision=revision, changed_files=changed)
    path = Path(plan['output'])/'execution-overlays'/(revision+'.json')
    if path.is_file():
        receipt = read(path)
        if receipt['sha256'] != digest(receipt['body']) or receipt['body'] != body:
            raise ValueError('cascade loader execution receipt changed')
        # Audio produced AFTER adoption is safe to resume under this same code.
        return receipt
    if not adopt:
        raise ValueError('cascade loader repair not adopted; rerun normal cell 6 first')
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
