"""Cell 10: one synthetic Japanese utterance, two independent S2ST branches."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from direct_s2st.benchmark_storage import guarded, read
from direct_s2st.config import load_config
from direct_s2st.drive_staging import ensure_local, raw_sha
from direct_s2st.io import atomic_write_json
from direct_s2st.journal import digest
from direct_s2st.interactive_comparison import PURPOSE, PHASES, validate_request, read_stage
from direct_s2st.progress import operation, track

INTERACTIVE_BACKEND_VERSION = 1
RUNTIME = Path('/content/s2st-runtime')


def restored_metadata(run, kind):
    from direct_s2st.recipes import option
    from inference_session import checked_metadata
    command = run['config']['command']
    root = option(command, '--data-root')
    if root is None and kind == 's2ut':
        root = command[3]
    if not root or '{' in root:
        raise ValueError('cannot determine saved inference metadata root')
    root = Path(root)
    required = ('config.yaml', 'config_multitask.yaml', 'data-lock.json') if kind == 's2ut' else ('mel-spec.json',)
    for name in required:
        checked_metadata(run, root/name)
    metadata = {}
    for name, sha in track(run['identity']['inputs'].items(), 'interactive: verify bound frontend metadata only'):
        path = Path(name)
        if path.is_relative_to(root) and path.suffix in ('.json', '.yaml', '.txt', '.model', '.vocab', '.npz'):
            checked_metadata(run, path)
            metadata[name] = sha
    for relative, sha in track(run['files'].items(), 'interactive: verify restored model snapshot'):
        if raw_sha(Path(run['work'])/relative) != sha:
            raise ValueError('restored inference checkpoint changed; originals retained')
    from direct_s2st.translatotron2.monitor import select_checkpoint
    checkpoint = select_checkpoint(Path(run['work']), 'best')
    checksum = raw_sha(checkpoint)
    relative = checkpoint.relative_to(Path(run['work'])).as_posix()
    if run['files'].get(relative) != checksum:
        raise ValueError('selected dev-best is not bound to restored snapshot')
    metadata[str(checkpoint)] = checksum
    if kind == 's2ut':
        from direct_s2st.vocoders.reference import validate_reference_units
        validate_reference_units(read(root/'data-lock.json')['unit_configuration'])
    return dict(name=run['name'], updates=run['updates'], work=run['work'], data_root=str(root),
                checkpoint=str(checkpoint), checkpoint_sha256=checksum), metadata


def saved_plan(persistent):
    pointer = persistent/'interactive/latest.json'
    if not pointer.is_file():
        return None
    receipt = read(pointer)
    path = Path(receipt['plan'])
    if not path.resolve().is_relative_to((persistent/'interactive').resolve()) or raw_sha(path) != receipt['sha256']:
        raise ValueError('interactive plan pointer changed; original retained')
    return read(path)


@operation('cell 10: synthetic Japanese two-model listening and performance comparison')
def launch(repo, persistent, local, text, model, requested, revision, invoke, *, remeasure=False):
    from inference_session import saved_run, environment, choose
    text = validate_request(text, model)
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('interactive comparison requires immutable execution revision')
    persistent, local = guarded(persistent), ensure_local(local)
    selected = choose(persistent, model, requested) if model in ('tt2', 's2ut') else ''
    request = dict(text=text, model=model, run_name=selected, execution_revision=revision)
    previous = saved_plan(persistent)
    if not remeasure and previous and previous.get('request') == request:
        stages = {p: read_stage(previous, p) for p in PHASES}
        evaluation_complete = stages['evaluate'] and all(v.get('status') == 'success'
                              for v in stages['evaluate'].get('data', {}).get('branches', {}).values())
        evaluation_complete = evaluation_complete and len(stages['evaluate'].get('data', {}).get('branches', {})) == 2
        if evaluation_complete and all(stages[p] and stages[p]['status'] == 'success' for p in PHASES):
            report = Path(previous['output'])/'comparison.json'
            pointer = read(persistent/'interactive/latest.json')
            if report.is_file() and pointer.get('report_sha256') == raw_sha(report) and read(report).get('plan') == previous['identity']:
                print('保存済み比較を再表示します。速度・メモリは前回の測定値です。再測定する場合はINTERACTIVE_REMEASURE=True。', flush=True)
                return previous
    # Restore only the selected trained model. No corpus staging or preparation.
    original = saved_run(persistent, model, selected, local) if selected else None
    if original:
        run, metadata = restored_metadata(original, model)
        python = environment(repo, persistent, original, invoke)
    else:
        from environment_recovery import ensure_environment
        python = ensure_environment(ROOT, persistent, invoke)
        run, metadata = None, {}
    os.environ.update(PYTHONPATH=str(ROOT/'src')+os.pathsep+'/content/s2st-runtime/fairseq',
                      S2ST_CONFIG_ROOT=str(ROOT/'configs'), S2ST_INTERACTIVE_MEASUREMENT_SESSION=uuid.uuid4().hex)
    env_path = RUNTIME/'environment.json'
    body = dict(version=1, purpose=PURPOSE, request=request, text=text, model=model, run=run,
                metadata=metadata, execution_revision=revision, environment_path=str(env_path),
                environment_sha256=raw_sha(env_path),
                cascade=load_config(ROOT/'configs/cascade/default.yaml'),
                s2t=load_config(ROOT/'configs/s2t_tts/default.yaml'),
                evaluation=load_config(ROOT/'configs/evaluation/default.yaml'),
                modes=dict(unit='official_ljspeech', mel='griffin_lim', gpu_execution='sequential',
                           sentence_bleu_reference='cascade_mt_pseudo_reference'),
                remeasurement=uuid.uuid4().hex if remeasure else
                    (previous.get('remeasurement') if previous and previous.get('request') == request else None))
    token = digest(body)[:24]
    plan = dict(body, identity=digest(body), local=str(local/token), output=str(persistent/'interactive'/token))
    guarded(plan['local'])
    guarded(plan['output'])
    plan_path = Path(plan['output'])/'plan.json'
    atomic_write_json(plan_path, plan, resume=True)
    atomic_write_json(persistent/'interactive/latest.json', dict(plan=str(plan_path), sha256=raw_sha(plan_path)), overwrite=True)
    invoke(python, '-c', 'from direct_s2st.cascade.tts import check_runtime; check_runtime(); '
           'import scipy.signal, soundfile, sacrebleu; from qwen_tts import Qwen3TTSModel; '
           'print("Interactive dependencies checked")')
    # Each GPU stage is a subprocess, so models are not simultaneously resident.
    # Failed candidate inference does not suppress the independent Cascade result.
    for phase in PHASES:
        invoke(python, '-m', 'direct_s2st.interactive_comparison', '--plan', plan_path, '--phase', phase)
    invoke(python, '-m', 'direct_s2st.interactive_comparison', '--plan', plan_path, '--phase', 'report')
    atomic_write_json(persistent/'interactive/latest.json', dict(plan=str(plan_path), sha256=raw_sha(plan_path),
                      report_sha256=raw_sha(Path(plan['output'])/'comparison.json')), overwrite=True)
    print('比較結果:', Path(plan['output'])/'comparison.json', flush=True)
    return plan


def main():
    from direct_s2st.inference_logging import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--persistent', type=Path, required=True)
    parser.add_argument('--local', type=Path, default=Path('/content/s2st-interactive'))
    parser.add_argument('--text', required=True)
    parser.add_argument('--model', choices=('tt2', 's2ut', 's2t_tts'), default='s2ut')
    parser.add_argument('--run-name', default='')
    parser.add_argument('--remeasure', action='store_true')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    def invoke(*command):
        print('実行:', *map(str, command), flush=True)
        subprocess.run(list(map(str, command)), check=True)
    launch(args.repo, args.persistent, args.local, args.text, args.model, args.run_name, revision, invoke, remeasure=args.remeasure)


if __name__ == '__main__':
    main()
