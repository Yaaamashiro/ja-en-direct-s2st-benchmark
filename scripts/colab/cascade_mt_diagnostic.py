"""Cell 8B entry point: GPU NLLB only, diagnostic outputs separate from Cell 8."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
MT_DIAGNOSTIC_VERSION = 1


def main():
    from direct_s2st.cascade.corpus_diagnostic import locate
    from direct_s2st.cascade.mt_diagnostic import saved_context, run
    from mt_diagnostic_runtime import probe, environment
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('persistent','manifest','cache'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--target',choices=('cascade','all'),default='cascade')
    parser.add_argument('--evaluation-root',type=Path)
    parser.add_argument('--comparison-config',type=Path)
    parser.add_argument('--runs-root',type=Path)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--label',default='default')
    args = parser.parse_args()
    from direct_s2st.inference_logging import configure_logging
    configure_logging()
    root = locate(args.persistent,args.target,comparison_config=args.comparison_config,
                  runs_root=args.runs_root,evaluation_root=args.evaluation_root)
    context = saved_context(args.manifest,root,args.limit)
    manifest = probe(sys.executable,environment(args.cache))
    report = run(context,args.persistent,manifest,limit=args.limit,label=args.label,
                 corpus=__import__('os').environ.get('CORPUS_ROOT'))
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
