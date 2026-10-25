"""Normal Cell 8F: reuse 8C Synthetic sources and the completed Natural 8E model."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
S2UT_FLORES_FLEURS_SYNTHETIC_VERSION = 1


def launch(repo,persistent,local,requested='',*,resume=True,execute=True):
    from direct_s2st.s2ut import flores_fleurs_synthetic as synthetic
    from direct_s2st.s2ut import flores_fleurs_diagnostic as core
    from direct_s2st.benchmark_storage import read
    from direct_s2st.io import atomic_write_json
    from direct_s2st.drive_staging import raw_sha
    from s2ut_flores_fleurs_diagnostic import invoke,ensure_host_yaml,training_context
    import flores_fleurs_runtime
    repo,persistent,local=map(lambda p:Path(p).absolute(),(repo,persistent,local))
    protected=[persistent]+([Path(os.environ['CORPUS_ROOT'])] if os.environ.get('CORPUS_ROOT') else [])
    if (any(local.resolve().is_relative_to(p.resolve()) for p in protected)
            or any(p.is_symlink() for p in (local,*local.parents))):
        raise ValueError('8F runtime/local files must remain outside experiment/corpus roots')
    natural=synthetic.natural_context(persistent,requested)
    revision=(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
              if execute else None)
    if execute and resume:
        complete=synthetic.completed_result(persistent,natural,revision)
        if complete is not None:return complete
    ctx=synthetic.source_context(natural,verify_audio=execute)
    print(f"8F: Natural 8E={natural['plan']['identity']} 8C={ctx['identity']} samples={len(ctx['samples'])} "
          f"Natural S2UT BLEU={natural['metrics']['bleu']:.6f} checkpoint={natural['plan']['checkpoint_sha256']}",
          file=sys.stderr,flush=True)
    if not execute:
        return dict(cascade_root=ctx['root'],samples=len(ctx['samples']),
                    display='8F実行はOFFです。保存済み8C Synthetic・8E Naturalの入力確認が完了しました。')
    ensure_host_yaml(local)
    from inference_session import environment
    if not repo.is_dir():
        invoke('git','clone','--recurse-submodules','https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark.git',repo)
    run,metadata,checkpoint,best=training_context(persistent,natural['plan']['run']['name'],local)
    if best!=natural['plan']['best_update']:
        raise ValueError('8F restored dev-best update differs from completed Natural 8E')
    plan=synthetic.prepare_plan(persistent,local,natural,ctx,run,metadata,checkpoint,revision)
    destination=Path(plan['output'])
    if (destination/'completion.json').is_file():
        if not resume:raise ValueError('8F result exists; enable resume. Originals are retained')
        synthetic.verify_plan(plan)
        synthetic.checked_completion(destination,expected_identity=plan['identity'])
        return dict(output_root=str(destination),metrics=read(destination/'metrics.json'),
                    display=(destination/'report.md').read_text(encoding='utf-8'))
    if not resume and any(p.name!='plan.json' for p in destination.iterdir()):
        raise ValueError('8F preparation exists; enable resume')
    python=environment(repo,persistent,run,invoke)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+'/content/s2st-runtime/fairseq',
             PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    for key in ('S2ST_INPUT_MAP','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):env.pop(key,None)
    invoke(python,'-c','import torch, torchaudio, soundfile, yaml, sacrebleu; '
           'from fairseq_cli import generate; from fairseq.models.text_to_speech.codehifigan import CodeGenerator; '
           'assert torch.cuda.is_available(), "8F requires a CUDA GPU"; print("8F S2UT dependencies checked")',env=env)
    helper=ROOT/'scripts/colab/s2ut_flores_fleurs_synthetic.py'
    for stage in ('prepare','units','audio'):
        invoke(python,'-B',helper,'--worker-stage',stage,'--plan',destination/'plan.json',env=env)
    asr_python,asr_env,runtime=flores_fleurs_runtime.ensure((),sys.executable,
        Path('/content/s2st-flores-fleurs-runtime'),Path('/content/s2st-flores-fleurs-cache'),[persistent,local],allow_install=True)
    core.verify_asr_runtime(plan,runtime)
    core.verify_asr_runtime({'source':{'environment':plan['natural_asr_runtime']}},runtime)
    asr_env.update(PYTHONPATH=str(ROOT/'src'))
    atomic_write_json(destination/'evaluation-runtime.json',runtime,overwrite=True)
    invoke(asr_python,'-B',helper,'--worker-stage','evaluate','--plan',destination/'plan.json',env=asr_env)
    names=list(synthetic.RESULT_FILES)
    if (destination/'evaluation-asr-audit.json').is_file():names.append('evaluation-asr-audit.json')
    atomic_write_json(destination/'completion.json',dict(identity=plan['identity'],
        files={name:raw_sha(destination/name) for name in names}),resume=True)
    return dict(output_root=str(destination),metrics=read(destination/'metrics.json'),
                display=(destination/'report.md').read_text(encoding='utf-8'))


def main():
    from direct_s2st.inference_logging import configure_logging
    from direct_s2st.progress import Progress
    configure_logging()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--persistent',type=Path)
    parser.add_argument('--local',type=Path,default=Path('/content/s2st-flores-fleurs-s2ut-synthetic'))
    parser.add_argument('--natural-id',default='')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--worker-stage',choices=('prepare','units','audio','evaluate'))
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    with Progress('8F: saved Synthetic FLEURS external S2UT evaluation'):
        with contextlib.redirect_stdout(sys.stderr):
            if args.worker_stage:
                if not args.plan:parser.error('--plan is required for worker')
                from direct_s2st.s2ut import flores_fleurs_diagnostic as core
                from direct_s2st.s2ut.flores_fleurs_synthetic import verify_plan
                from direct_s2st.benchmark_storage import read
                plan=read(args.plan);verify_plan(plan)
                {'prepare':core.prepare_external,'units':core.units,'audio':core.audio,'evaluate':core.evaluate}[args.worker_stage](plan)
                return
            if not args.repo or not args.persistent:parser.error('--repo and --persistent are required')
            result=launch(args.repo,args.persistent,args.local,args.natural_id,
                          resume=args.resume,execute=not args.inspect_only)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
