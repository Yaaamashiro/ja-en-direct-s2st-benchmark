"""Read-only CPU entry point. No setup, dependency mutation, audio IO or model imports."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
CASCADE_DIAGNOSTIC_VERSION = 1


def main():
    from direct_s2st.cascade.corpus_diagnostic import diagnose, locate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persistent', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--target', choices=('cascade', 'all', 's2ut'), required=True)
    parser.add_argument('--comparison-config', type=Path)
    parser.add_argument('--runs-root', type=Path)
    parser.add_argument('--evaluation-root', type=Path)
    args = parser.parse_args()
    try:
        root = locate(args.persistent, args.target, comparison_config=args.comparison_config,
                      runs_root=args.runs_root, evaluation_root=args.evaluation_root)
        result = diagnose(args.manifest, root)
    except (ImportError, KeyError, OSError, ValueError, RuntimeError) as error:
        raise RuntimeError('8A diagnostic stopped without inference or result writes: '+str(error)) from error
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
