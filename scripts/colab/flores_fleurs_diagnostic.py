"""Cell 8C launcher and isolated sequential inference workers."""
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
FLORES_FLEURS_VERSION = 1


def source_code(proven,cache,explicit=None):
    from direct_s2st.cascade.source_corpus_tts import verify_code,git,REPOSITORY
    if explicit:
        verify_code(explicit,proven)
        return Path(explicit)
    path=Path(cache)/'corpus-code'/proven['receipt']['code_revision']
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        from flores_fleurs_runtime import logged
        with contextlib.redirect_stdout(sys.stderr):
            logged(['git','clone','--no-checkout','--filter=blob:none','https://github.com/'+REPOSITORY+'.git',path],os.environ)
            git(path,'checkout','--detach',proven['receipt']['code_revision'])
    verify_code(path,proven)
    return path


def main():
    from direct_s2st.inference_logging import configure_logging
    configure_logging()
    from direct_s2st.cascade.source_corpus_tts import proof,verify_code,CorpusSource
    from direct_s2st.cascade.flores_fleurs_diagnostic import context,prepare,worker,report,STAGES
    from direct_s2st.cascade.flores_fleurs_data import OfficialData
    from direct_s2st.cascade.corpus_diagnostic import locate,read
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-provenance',type=Path)
    parser.add_argument('--preflight-source',action='store_true')
    parser.add_argument('--source-code',type=Path)
    parser.add_argument('--persistent',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--target',choices=('cascade','all'),default='cascade')
    parser.add_argument('--evaluation-root',type=Path)
    parser.add_argument('--comparison-config',type=Path)
    parser.add_argument('--runs-root',type=Path)
    parser.add_argument('--limit',type=int,default=300)
    parser.add_argument('--overwrite',action='store_true',help='refresh owned 8C report views after retry; never official data or WAVs')
    parser.add_argument('--worker-stage',choices=STAGES)
    parser.add_argument('--diagnostic-root',type=Path)
    args=parser.parse_args()
    if args.worker_stage:
        if not args.diagnostic_root:
            parser.error('--diagnostic-root is required for worker')
        plan=read(args.diagnostic_root/'identity.json')
        # Every worker rechecks historical evidence and original source code.
        for path,sha in plan['source_proof']['inputs_sha256'].items():
            from direct_s2st.cascade.corpus_diagnostic import checksum
            if checksum(path)!=sha:
                raise RuntimeError('8C historical source evidence changed')
        verify_code(plan['source_code']['root'],plan['source_proof'])
        factory=lambda p,d:CorpusSource(p['source_proof'],p['source_code']['root'],d)
        worker(args.diagnostic_root,args.worker_stage,data=OfficialData(args.cache),source_factory=factory)
        return
    proven=proof(args.source_provenance)  # First: never spend CU inferring guessed source conditions.
    if args.preflight_source:
        print(json.dumps(dict(proven=True,receipt_sha256=proven['receipt_sha256'])))
        return
    if not all((args.persistent,args.manifest,args.cache)):
        parser.error('--persistent, --manifest, --cache are required')
    from flores_fleurs_runtime import probe,environment,logged
    env=environment(args.cache)
    runtime=probe(sys.executable,env)
    for path in (args.cache,):
        if path.resolve().is_relative_to(args.persistent.resolve()) or (os.environ.get('CORPUS_ROOT') and
            path.resolve().is_relative_to(Path(os.environ['CORPUS_ROOT']).resolve())):
            raise RuntimeError('8C runtime caches must remain local, outside persistent data and corpus')
    code=source_code(proven,args.cache,args.source_code)
    baseline=locate(args.persistent,args.target,comparison_config=args.comparison_config,
                    runs_root=args.runs_root,evaluation_root=args.evaluation_root)
    ctx=context(args.manifest,baseline)
    data=OfficialData(args.cache)
    samples,audit=data.load(args.limit)
    protected=[code]+([os.environ['CORPUS_ROOT']] if os.environ.get('CORPUS_ROOT') else [])
    root=prepare(args.persistent,ctx,samples,audit,proven,verify_code(code,proven),runtime,protected)
    with contextlib.redirect_stdout(sys.stderr):
        for stage in STAGES:
            logged([sys.executable,'-B',Path(__file__),'--worker-stage',stage,'--diagnostic-root',root,'--cache',args.cache],env)
    print(json.dumps(report(root,overwrite=args.overwrite),ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
