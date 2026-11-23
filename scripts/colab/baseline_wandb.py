"""CPU-only inspect/sync/watch CLI; never imports the training runtime."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from direct_s2st.baseline_wandb import PROJECT, collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'sync', 'watch'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--entity', default='')
    parser.add_argument('--project', default=PROJECT)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    if args.action == 'inspect':
        bundle = collect(args.source, snapshot=args.snapshot)
        print(json.dumps({k: bundle[k] for k in ('durable_update', 'counts', 'summary', 'sources', 'conditions')}, ensure_ascii=False))
        return
    if not args.state or not args.entity or args.interval < 1:
        parser.error('sync/watch require --state and explicit --entity; interval must be >=1')
    from direct_s2st.baseline_wandb_delivery import synchronize
    if args.action == 'sync':
        print(json.dumps(synchronize(args.source, args.state, args.entity,
                                    project=args.project, snapshot=args.snapshot)))
        return
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous = None
    while True:
        final = stopping or bool(args.stop_file and args.stop_file.exists())
        try:
            # Avoid opening large report JSONs on every heartbeat when no
            # published dashboard changed. Failed syncs are always retried.
            stamps = [(p.name, p.stat().st_mtime_ns, p.stat().st_size)
                      for p in args.source.glob('training-*') if p.is_file()]
            if final or stamps != previous:
                result = synchronize(args.source, args.state, args.entity,
                                     project=args.project, snapshot=args.snapshot)
                print('[baseline-wandb] ' + json.dumps(result), file=sys.stderr, flush=True)
                previous = stamps
        except Exception as error:
            previous = None
            # Do not print exception bodies containing credentials/request URLs.
            print('[baseline-wandb] deferred=' + type(error).__name__ +
                  '; training unaffected; rerun sync to retry', file=sys.stderr, flush=True)
        if final:
            return
        deadline = time.monotonic() + args.interval
        while time.monotonic() < deadline and not stopping and not (args.stop_file and args.stop_file.exists()):
            time.sleep(min(1, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
