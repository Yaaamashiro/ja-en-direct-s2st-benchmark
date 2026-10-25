"""Normal Cell 8E: frozen 8C natural inputs and existing dev-best S2UT."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
S2UT_FLORES_FLEURS_VERSION = 2


def invoke(*command, env=None):
    from direct_s2st.progress import Progress
    from cascade_diagnostic_runtime import stop_child
    print('8E:', ' '.join(map(str, command)), file=sys.stderr, flush=True)
    with Progress('8E: dependency restoration / sequential inference subprocess'):
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', stdin=subprocess.DEVNULL, env=env)
        try:
            with process.stdout:
                for line in process.stdout:
                    print(line, end='', file=sys.stderr, flush=True)
            status = process.wait()
        except BaseException:
            stop_child(process)
            raise
        if status:
            raise RuntimeError(f'8E subprocess failed ({status}). Successful stages are saved; rerun Cell 8E.')


def ensure_host_yaml(local):
    """Only the launcher needs YAML; never pip-install into the notebook kernel."""
    try:
        import yaml
    except ImportError:
        target = Path(local)/'launcher-dependencies/pyyaml-6.0.2'
        from direct_s2st.benchmark_storage import guarded
        guarded(target)
        if not (target/'yaml/__init__.py').is_file():
            invoke(sys.executable, '-B', '-m', 'pip', '--isolated', 'install', '--no-deps',
                   '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                   '--target', target, 'PyYAML==6.0.2')
        sys.path.insert(0, str(target))
        import yaml
    if yaml.safe_load('external: true') != {'external':True}:
        raise RuntimeError('8E launcher YAML dependency is unusable')


def training_context(persistent, requested, local):
    from inference_session import saved_run, checked_metadata
    from direct_s2st.benchmark_storage import read
    from direct_s2st.progress import track
    from direct_s2st.recipes import option
    from direct_s2st.translatotron2.monitor import select_checkpoint
    from direct_s2st.vocoders.reference import validate_reference_units
    run = saved_run(persistent, 's2ut', requested, local)
    command = run['config']['command']
    root = Path(option(command, '--data-root') or command[3])
    if not root.is_absolute() or '{' in str(root):
        raise ValueError('unresolved saved S2UT prepared data root')
    metadata = {}
    for name, sha in track(run['identity']['inputs'].items(), '8E: verify checkpoint-bound prepared metadata'):
        path = Path(name)
        if path.is_relative_to(root):
            checked_metadata(run, path)
            metadata[name] = sha
    for name in ('config.yaml', 'config_multitask.yaml', 'dict.txt', 'data-lock.json'):
        checked_metadata(run, root/name)
    validate_reference_units(read(root/'data-lock.json')['unit_configuration'])
    run['data_root'] = str(root)
    checkpoint = select_checkpoint(run['work'], 'best')
    monitor = read(Path(run['work'])/'training-monitor.json')
    return run, metadata, checkpoint, monitor['best_update']


def launch(repo, persistent, local, requested='', cascade_id='', *, resume=True, execute=True):
    from direct_s2st.s2ut.flores_fleurs_diagnostic import (
        choose_cascade, source_context, prepare_plan, verify_asr_runtime, verify_plan,
        verify_saved_asr_dependencies, implementation_files)
    from direct_s2st.benchmark_storage import read
    from direct_s2st.io import atomic_write_json
    from direct_s2st.progress import operation
    import flores_fleurs_runtime
    repo, persistent, local = map(lambda p:Path(p).absolute(), (repo,persistent,local))
    protected = [persistent]+([Path(os.environ['CORPUS_ROOT'])] if os.environ.get('CORPUS_ROOT') else [])
    if (any(local.resolve().is_relative_to(p.resolve()) for p in protected)
            or any(p.is_symlink() for p in (local,*local.parents))):
        raise ValueError('8E local/runtime files must remain outside experiments and CORPUS_ROOT')
    root = choose_cascade(persistent, cascade_id)
    # Stdlib preflight first; no GPU/package/model downloads for wrong paths.
    ctx = source_context(root, verify_audio=False)
    print(f"8E: 8C={root.name} samples={len(ctx['samples'])} "
          f"saved Natural ASR-BLEU={ctx['cascade_natural_bleu']:.4f}", file=sys.stderr, flush=True)
    if not execute:
        return dict(cascade_root=str(root), samples=len(ctx['samples']),
                    display='8E実行はOFFです。保存済み8Cの入力確認まで完了しました。')
    verify_saved_asr_dependencies(ctx)
    ensure_host_yaml(local)
    from inference_session import environment
    if not repo.is_dir():
        invoke('git', 'clone', '--recurse-submodules',
               'https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark.git', repo)
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    run, metadata, checkpoint, best = training_context(persistent, requested, local)
    # Complete model-free resume doesn't require reinstalling CUDA/Whisper.
    # Source SHA verification and saved checkpoint selection still run first.
    plan = prepare_plan(persistent, local, ctx, run, metadata, checkpoint, best, revision, resume=resume)
    destination = Path(plan['output'])
    if (destination/'completion.json').is_file():
        if not resume:
            raise ValueError('8E result exists; enable resume. Originals are never overwritten')
        verify_plan(plan)
        receipt = read(destination/'completion.json')
        from direct_s2st.drive_staging import raw_sha
        if (receipt['identity'] != plan['identity'] or any(raw_sha(destination/name) != sha
                for name, sha in receipt['files'].items())):
            raise ValueError('8E completed report checksum mismatch')
        return dict(output_root=str(destination), metrics=read(destination/'metrics.json'),
                    display=(destination/'report.md').read_text(encoding='utf-8'))
    if not resume and any(destination.iterdir()):
        # plan.json is just created above; other artifacts require explicit resume.
        if any(p.name != 'plan.json' for p in destination.iterdir()):
            raise ValueError('8E in-progress result exists; enable resume')
    # Restore EXACTLY the saved fairseq environment through the existing Cell 6 path.
    python = environment(repo, persistent, run, invoke)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src')+os.pathsep+'/content/s2st-runtime/fairseq',
               PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    env.pop('S2ST_INPUT_MAP', None)
    env.pop('HF_HUB_OFFLINE', None)
    env.pop('TRANSFORMERS_OFFLINE', None)
    invoke(python, '-c', 'import torch, torchaudio, soundfile, yaml, sacrebleu; '
           'from fairseq_cli import generate; '
           'from fairseq.models.text_to_speech.codehifigan import CodeGenerator; '
           'assert torch.cuda.is_available(), "8E requires a CUDA GPU"; print("8E S2UT dependencies checked")', env=env)
    helper = ROOT/'scripts/colab/s2ut_flores_fleurs_diagnostic.py'
    for stage in ('prepare', 'units', 'audio'):
        invoke(python, '-B', helper, '--worker-stage', stage, '--plan', destination/'plan.json', env=env)
    # Keep evaluation dependencies separate: 8C numpy 2.2.6 vs fairseq numpy 1.26.4.
    # Reuse the tested 8C runtime/bootstrap, without running any Qwen/NLLB synthesis.
    asr_python, asr_env, runtime = flores_fleurs_runtime.ensure(
        (), sys.executable, Path('/content/s2st-flores-fleurs-runtime'),
        Path('/content/s2st-flores-fleurs-cache'), [persistent, local], allow_install=True)
    verify_asr_runtime(plan, runtime)
    asr_env.update(PYTHONPATH=str(ROOT/'src'))
    atomic_write_json(destination/'evaluation-runtime.json', runtime, overwrite=True)
    invoke(asr_python, '-B', helper, '--worker-stage', 'evaluate', '--plan', destination/'plan.json', env=asr_env)
    from direct_s2st.s2ut.flores_fleurs_resume import annotate_report
    annotate_report(plan, implementation_files())
    from direct_s2st.drive_staging import raw_sha
    atomic_write_json(destination/'completion.json', dict(identity=plan['identity'],
        files={name:raw_sha(destination/name) for name in
               ('metrics.json','per_sample.jsonl','per_sample.csv','report.md','audio-index.json',
                'units.jsonl','predictions.jsonl','external-input-audit.json','evaluation-runtime.json')}), resume=True)
    return dict(output_root=str(destination), metrics=read(destination/'metrics.json'),
                display=(destination/'report.md').read_text(encoding='utf-8'))


def main():
    from direct_s2st.inference_logging import configure_logging
    from direct_s2st.progress import Progress
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--persistent', type=Path)
    parser.add_argument('--local', type=Path, default=Path('/content/s2st-flores-fleurs-s2ut'))
    parser.add_argument('--run-name', default='')
    parser.add_argument('--cascade-id', default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--worker-stage', choices=('prepare','units','audio','evaluate'))
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    with Progress('8E: saved natural FLEURS external S2UT evaluation'):
        with contextlib.redirect_stdout(sys.stderr):
            if args.worker_stage:
                if not args.plan:
                    parser.error('--plan is required for worker')
                from direct_s2st.s2ut import flores_fleurs_diagnostic as diag
                from direct_s2st.benchmark_storage import read
                plan = read(args.plan)
                diag.verify_plan(plan)
                {'prepare':diag.prepare_external,'units':diag.units,'audio':diag.audio,
                 'evaluate':diag.evaluate}[args.worker_stage](plan)
                return
            if not args.repo or not args.persistent:
                parser.error('--repo and --persistent are required')
            result = launch(args.repo, args.persistent, args.local, args.run_name, args.cascade_id,
                            resume=args.resume, execute=not args.inspect_only)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
