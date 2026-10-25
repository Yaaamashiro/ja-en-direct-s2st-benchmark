"""Reviewed package-name comparison repair; retain completed 4e18c8f artifacts.

Original plans and archives stay byte-identical. Only an explicit execution
receipt is added; changing model/data/scoring code cannot adopt these artifacts.
"""
import ast
import hashlib
from pathlib import Path
import sys

from ..benchmark_storage import guarded, read
from ..io import atomic_write_json, atomic_write_text
from ..journal import digest
from ..progress import operation, track

BASE_REVISION = '4e18c8f872561c57a6cb32f115164f5c331db972'
REPAIR = 'canonical_distribution_names_preserve_exact_versions_v1'
DIAGNOSTIC = 'src/direct_s2st/s2ut/flores_fleurs_diagnostic.py'
LAUNCHER = 'scripts/colab/s2ut_flores_fleurs_diagnostic.py'
SELF = 'src/direct_s2st/s2ut/flores_fleurs_resume.py'
BASE_FILES = {
    DIAGNOSTIC:'2eac4a757b56da50efb88e152171f3f0940ca2702cfec0c127711a6721c2b6e0',
    LAUNCHER:'4aa44424e8fa7151d5d90cc860199fa7b550cb80f413efef21c69c8f547d5dc1',
}
SCIENCE = ('source_context','prepare_external','verify_external','parse_units','shard_records',
           'units','audio','restore_audio','evaluate','restore_asr_progress','save_report',
           'VERSION','PURPOSE','NORMALIZATION','ASR','SAMPLE_FIELDS','report_rows')
LAUNCH_SCOPES = ('invoke','ensure_host_yaml','training_context','main')
REVIEWED_SCOPES = {
    # 8F adds a purpose-specific report dispatch and extracts the same row
    # renderer. Natural inputs, inference/vocoding/ASR and metric definitions
    # are unchanged; regression tests retain the released Natural behavior.
    DIAGNOSTIC:(SCIENCE,'ad29deb89a59253d510df3a7463306dd521a86d6043a057cfa939e82fef5f471'),
    LAUNCHER:(LAUNCH_SCOPES,'8d5745e1ea96bab9051a401587ea199b1004e464a7e15f457d25248b051e01db'),
}


def scientific_body(body):
    return {k:v for k,v in body.items() if k not in
            ('execution_files','execution_revision','identity','local','output')}


def relative_files(files):
    launcher = [Path(p) for p in files if Path(p).as_posix().endswith('/'+LAUNCHER)]
    if len(launcher) != 1:
        raise ValueError('8E reviewed repair requires one original launcher root')
    root = launcher[0].parents[2]
    result = {}
    for path,sha in files.items():
        try:relative=Path(path).relative_to(root).as_posix()
        except ValueError as error:raise ValueError('8E code files span incompatible roots') from error
        result[relative]=(Path(path),sha)
    return result


def scope_sha(path, names):
    source=Path(path).read_text(encoding='utf-8')
    lines=source.splitlines(); selected={}
    for node in ast.parse(source).body:
        name=(node.name if isinstance(node,ast.FunctionDef) else
              node.targets[0].id if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) else None)
        if name in names:
            start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
            selected[name]='\n'.join(lines[start:node.end_lineno])
    if set(selected)!=set(names):
        raise ValueError('8E reviewed scientific scope is incomplete')
    return digest(selected)


def review(plan, current_files):
    if plan.get('execution_revision') != BASE_REVISION:
        raise ValueError('8E repair only supports the reviewed 4e18c8f plan')
    old,new=relative_files(plan['execution_files']),relative_files(current_files)
    if set(new)!=set(old)|{SELF}:
        raise ValueError('8E repair implementation file set changed')
    for relative,(_,sha) in old.items():
        if relative in BASE_FILES:
            if sha!=BASE_FILES[relative]:
                raise ValueError('8E original implementation is not the reviewed release: '+relative)
        elif sha not in {new[relative][1],hashlib.sha256(new[relative][0].read_bytes().replace(b'\r\n',b'\n')).hexdigest()}:
            raise ValueError('8E inference/model/dependency code changed: '+relative)
    for relative,(names,sha) in REVIEWED_SCOPES.items():
        if scope_sha(new[relative][0],names)!=sha:
            raise ValueError('8E scientific/worker scope changed; cannot reuse artifacts: '+relative)


def receipt_path(plan,current_files):
    return Path(plan['output'])/'execution-overlays'/(digest(current_files)+'.json')


def verify_execution(plan,current_files):
    review(plan,current_files)
    document=read(receipt_path(plan,current_files)); body=document['body']
    if (document.get('sha256')!=digest(body) or body.get('base_identity')!=plan['identity']
            or body.get('base_revision')!=BASE_REVISION or body.get('repair')!=REPAIR
            or body.get('base_execution_files_sha256')!=digest(plan['execution_files'])
            or body.get('execution_files')!=current_files
            or body.get('scientific_settings_unchanged') is not True):
        raise ValueError('8E reviewed execution receipt differs from the saved plan/code')
    return body


@operation('8E: reuse reviewed saved preparation without changing its identity')
def adopt(body,base,current_files):
    matches=[]
    for path in track(sorted(Path(base).glob('*/plan.json')),'8E: inspect retained preparation plans'):
        previous=read(path)
        if (previous.get('execution_revision')!=BASE_REVISION
                or scientific_body(previous)!=scientific_body(body)):
            continue
        # The reported failure occurred before ASR. Never reinterpret an old
        # already-scored evaluation or merge another scientific configuration.
        already_adopted=receipt_path(previous,current_files).is_file()
        if already_adopted:
            verify_execution(previous,current_files)
        if not already_adopted and ((path.parent/'completion.json').exists() or (path.parent/'progress/asr.json').exists()):
            continue
        parts=list((path.parent/'generation').glob('part-*.json'))
        if not parts and not (path.parent/'predictions.jsonl').is_file():
            continue
        old_body={k:v for k,v in previous.items() if k not in ('identity','local','output')}
        if (digest(old_body)!=previous.get('identity') or path.parent.name!=previous['identity'][:24]
                or Path(previous['output'])!=path.parent
                or Path(previous['local'])!=Path(previous['workspace_base'])/previous['identity'][:24]):
            raise ValueError('8E retained plan identity/path mismatch')
        review(previous,current_files)
        matches.append(((path.parent/'predictions.jsonl').is_file(),len(parts),path.stat().st_mtime_ns,previous))
    if not matches:
        return None
    previous=max(matches,key=lambda item:item[:3])[-1]
    guarded(previous['local']);guarded(previous['output'])
    receipt=dict(version=1,repair=REPAIR,base_identity=previous['identity'],base_revision=BASE_REVISION,
        base_execution_files_sha256=digest(previous['execution_files']),
        execution_revision=body['execution_revision'],execution_files=current_files,
        scientific_settings_unchanged=True)
    atomic_write_json(receipt_path(previous,current_files),dict(body=receipt,sha256=digest(receipt)),resume=True)
    print('[8E-resume] retained_identity='+previous['identity']+
          ' repair='+REPAIR+' artifacts=verify_and_reuse_saved_Unit_WAV_ZIPs',file=sys.stderr,flush=True)
    return previous


def annotate_report(plan,current_files):
    if plan['execution_files']==current_files:
        return
    overlay=verify_execution(plan,current_files)
    root=Path(plan['output']);metrics=read(root/'metrics.json')
    metrics.update(code_revision=overlay['execution_revision'],original_code_revision=BASE_REVISION,
                   execution_overlay=overlay)
    atomic_write_json(root/'metrics.json',metrics,overwrite=True)
    report=(root/'report.md').read_text(encoding='utf-8')
    report+='\nReviewed dependency-name repair: `'+overlay['execution_revision']+'`; original preparation: `'+BASE_REVISION+'`. '
    report+='Saved plan/Unit/WAV identities are retained. Models, references, decoding and exact package versions are unchanged.\n'
    atomic_write_text(root/'report.md',report,overwrite=True)
