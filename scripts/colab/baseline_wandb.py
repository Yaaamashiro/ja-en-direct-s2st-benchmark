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
    parser.add_argument('action', choices=('inspect', 'sync', 'watch', 'audit-v3'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--registration', type=Path, help='Published Baseline registration snapshot.json')
    parser.add_argument('--baseline-run-key', default='', help='Original MyDrive-relative publication directory')
    parser.add_argument('--existing-run-id', default='', help='Verified existing W&B Run ID')
    parser.add_argument('--schema-v2-only', action='store_true',
                        help='Append only the new schema-v2 keys to an existing W&B run')
    parser.add_argument('--metric-schema', type=int, choices=(2, 3), default=2)
    parser.add_argument('--group', default='')
    parser.add_argument('--audit-journals', action='store_true')
    parser.add_argument('--state', type=Path)
    parser.add_argument('--entity', default='')
    parser.add_argument('--project', default=PROJECT)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    if args.action == 'inspect':
        bundle = collect(args.source, snapshot=args.snapshot, registration=args.registration,
                         audit_journals=args.audit_journals, metric_schema=args.metric_schema,
                         baseline_run_key=args.baseline_run_key)
        print(json.dumps({k: bundle[k] for k in ('durable_update', 'training_fingerprint', 'counts', 'coverage',
                                               'summary', 'sources', 'conditions', 'schema_identity') if k in bundle},
                         ensure_ascii=False))
        return
    if not args.state or not args.entity or args.interval < 1:
        parser.error('sync/watch/audit require --state and explicit --entity; interval must be >=1')
    from direct_s2st.baseline_wandb_delivery import audit_v3, synchronize
    if args.action == 'audit-v3':
        if args.metric_schema != 3 or args.existing_run_id or args.schema_v2_only:
            parser.error('audit-v3 requires --metric-schema 3 without any legacy override')
        print(json.dumps(audit_v3(args.source, args.state, args.entity, project=args.project, group=args.group,
                                  snapshot=args.snapshot, registration=args.registration,
                                  baseline_run_key=args.baseline_run_key, audit_journals=args.audit_journals)))
        return
    if args.action == 'watch' and args.metric_schema == 3:
        parser.error('Schema v3 is for the completed 400k Baseline import')
    options = dict(project=args.project, snapshot=args.snapshot, registration=args.registration,
                   baseline_run_key=args.baseline_run_key, existing_run_id=args.existing_run_id,
                   group=args.group, audit_journals=args.audit_journals,
                   schema_v2_only=args.schema_v2_only, metric_schema=args.metric_schema)
    if args.action == 'sync':
        print(json.dumps(synchronize(args.source, args.state, args.entity, **options)))
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
            stamps.append(str(args.snapshot or max(args.source.glob('*/snapshot.json'), default='')))
            if final or stamps != previous:
                result = synchronize(args.source, args.state, args.entity, **options)
                print('[baseline-wandb] ' + json.dumps(result), file=sys.stderr, flush=True)
                previous = stamps if result['status'] == 'SYNCED' else None
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
