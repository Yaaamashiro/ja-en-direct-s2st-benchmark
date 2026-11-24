"""Optional network adapter, imported only by the isolated CPU reporter."""
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import time
from collections import Counter

from . import baseline_wandb_v3 as v3
from .baseline_wandb import (API_BASE, PROJECT, SCHEMA_V2_METRICS, atomic, binding, canonical, collect,
                             digest, plan, run_id)
from .progress import operation


class HistoryNotVisible(RuntimeError):
    """W&B has advanced past an outbox event its read API does not show."""


def _wait_for_history(remote, bundle, events, history=None):
    """Poll the read API after a write without submitting the same steps again."""
    expected = {event['id'] for event in events}
    for delay in (0, 2, 4, 8, 15, 30):
        if delay:
            time.sleep(delay)
        if delay or history is None:
            history = remote.history()
        _, acknowledged = plan(bundle, history, adopted=remote.contract['adopted'])
        missing = expected - acknowledged
        if not missing:
            return history, acknowledged
        print(f'[baseline-wandb] waiting for history: confirmed={len(expected)-len(missing)}/{len(expected)}',
              file=sys.stderr, flush=True)
    return history, acknowledged


class WandbRemote:
    def __init__(self, contract, logs):
        import wandb
        self.wandb, self.contract, self.logs = wandb, contract, Path(logs)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.api = wandb.Api(overrides={'base_url': API_BASE}, timeout=20)
        # Do not silently create another workspace/project after a typo.
        if not any(p.name == contract['project'] for p in self.api.projects(entity=contract['entity'])):
            raise ValueError('existing destination W&B project was not found')
        self.path = '/'.join((contract['entity'], contract['project'], run_id(contract)))
        self.existing = False
        self.progress = 0

    def _matches(self, config):
        saved = config.get('baseline_contract') or {}
        if self.contract.get('metric_schema') == 3:
            return saved == self.contract and config.get('metric_schema') == 3
        identity = ('baseline_run_key', 'training_fingerprint')
        return (saved == self.contract
                or (self.contract['adopted'] and config.get('condition', saved.get('condition')) == 'baseline'
                    and (all(config.get(key, saved.get(key)) == self.contract[key] for key in identity)
                         or self._matches_pre_review_import(config))))

    def _matches_pre_review_import(self, config):
        saved = config.get('baseline_contract') or {}
        expected = getattr(self, 'source_identity_sha256', None)
        return (saved.get('version') == 1 and expected is not None
                and saved.get('source_identity_sha256') == expected
                and digest(config.get('source_identity')) == expected
                and all(saved.get(key) == self.contract[key] for key in ('entity', 'project', 'condition')))

    def history(self):
        self.api.flush()
        try:
            remote = self.api.run(self.path)
        except self.wandb.errors.CommError as error:
            # Only a definite missing run permits creation; auth/network errors
            # are not evidence that a new run is needed.
            response = getattr(error, 'response', None)
            # Public API raises ValueError("Could not find run ...") wrapped as
            # CommError, without an HTTP response, for its GraphQL null run.
            if (getattr(response, 'status_code', None) == 404
                    or str(error).startswith('Could not find run ')):
                if self.contract['adopted'] or self.existing or getattr(self, 'require_existing', False):
                    raise ValueError('selected existing W&B Run ID was not found') from error
                return []
            raise
        if not self._matches(remote.config):
            raise ValueError('W&B run identity/destination differs; refusing to write')
        require_idle = (self.contract['adopted'] or ((getattr(self, 'require_existing', False)
                        or getattr(self, 'require_idle', False))
                        and not self.existing and not getattr(self, 'started_writer', False)))
        if require_idle and remote.state not in {'finished', 'failed', 'crashed', 'killed'}:
            raise ValueError('selected existing W&B run is active; use one writer')
        self.existing = True
        group = getattr(self, 'group', '')
        self.needs_group_update = bool(group and getattr(remote, 'group', None) != group)
        if self.contract.get('metric_schema') == 3 and self.needs_group_update:
            raise ValueError('existing Schema v3 Run has the wrong group')
        if self.contract.get('metric_schema') == 3:
            self.remote_summary = {key: remote.summary.get(key) for key in v3.SUMMARY}
            if any(value is not None and (type(value) not in (float, int) or not 0 <= value <= v3.MAX_UPDATE)
                   for value in self.remote_summary.values()):
                raise ValueError('existing Schema v3 Run has invalid optimizer progress')
        # The initial PR used the planned budget as max_update. Only its
        # separately recorded durable watermark counts as verified progress.
        progress_key = ('train/durable_update' if self._matches_pre_review_import(remote.config)
                        else 'train/max_update')
        self.progress = max(self.progress, int(remote.summary.get(progress_key, 0) or 0))
        return list(remote.scan_history(page_size=500, use_cache=False))

    def send(self, events, bundle):
        options = {}
        if getattr(self, 'group', ''):
            options['group'] = self.group
        existing = getattr(self, 'existing', False) or self.contract['adopted']
        config = {'condition': 'baseline', 'baseline_contract': self.contract,
                  'baseline_run_key': self.contract['baseline_run_key'],
                  'training_fingerprint': self.contract['training_fingerprint'],
                  'source_identity': bundle['identity'], 'loss_mapping_version': 2}
        if self.contract.get('metric_schema') == 3:
            config.update(metric_schema=3, loss_mapping_version=3, baseline_origin=self.contract['origin'],
                          baseline_snapshot_lineage=self.contract['lineage'])
        run = self.wandb.init(
            entity=self.contract['entity'], project=self.contract['project'],
            id=run_id(self.contract), name='baseline', resume='must' if existing else 'never',
            dir=str(self.logs), save_code=False,
            config=None if existing else config,
            settings=self.wandb.Settings(console='off', disable_git=True, disable_code=True,
                base_url=API_BASE,
                x_disable_meta=True, x_disable_stats=True, x_service_wait=10,
                init_timeout=20, finish_timeout=20, finish_timeout_raises=True), **options)
        self.started_writer = True
        try:
            if not self._matches(run.config):
                raise ValueError('W&B initialization returned another experiment')
            if options.get('group') and run.group != options['group']:
                raise ValueError('W&B did not apply the selected group; verify the existing run group')
            if events and run.step > events[0]['step']:
                raise HistoryNotVisible('server advanced beyond pending event; retry after history becomes visible')
            run.define_metric('update', summary='none', overwrite=True)
            for prefix in ('baseline', 'train', 'dev', 'eval', 'alignment'):
                run.define_metric(prefix + '/*', step_metric='update')
            for item in events:
                event = item['event']
                payload = dict(event['values'], update=event['update'])
                payload.update({'baseline/event_id': item['id'],
                                'baseline/fields': canonical(sorted(event['values'])),
                                'baseline/provenance': canonical(event['provenance'])})
                run.log(payload, step=item['step'], commit=True)
            maximum = max(bundle['durable_update'], getattr(self, 'progress', 0))
            if getattr(self, 'require_existing', False):
                if maximum > getattr(self, 'progress', 0):
                    run.summary['train/max_update'] = maximum
            else:
                for key, value in bundle['summary'].items():
                    run.summary[key] = maximum if key == 'train/max_update' else value
                run.summary['baseline/sources'] = bundle['sources']
                run.summary['baseline/receipt'] = bundle['receipt']
                run.summary['baseline/evaluation_conditions'] = bundle['conditions']
        finally:
            run.finish()


@operation('baseline W&B: reconcile and publish measured metrics')
def synchronize(source, state, entity, *, project=PROJECT, snapshot=None, registration=None,
                baseline_run_key='', existing_run_id='', group='', audit_journals=False,
                schema_v2_only=False, metric_schema=2, remote_factory=WandbRemote):
    """Persist an outbox first. Remote event hashes, not local watermarks, dedupe.

    The source is always read-only. State/logs must live outside it. A successful
    SDK finish is still verified against remote history before acknowledging.
    """
    from filelock import FileLock
    source, state = Path(source).resolve(), Path(state).resolve()
    if state == source or state.is_relative_to(source) or source.is_relative_to(state):
        raise ValueError('reporter state must be separate from read-only source data')
    if group and (group != group.strip() or re.search(r'[\x00-\x1f\x7f]', group)):
        raise ValueError('invalid W&B group')
    if metric_schema == 3:
        _validate_v3_state(source, state, entity, project, group, existing_run_id, schema_v2_only)
    elif project == v3.PROJECT:
        raise ValueError('the Schema v3 project rejects legacy Baseline writes')
    bundle = collect(source, snapshot=snapshot, registration=registration, audit_journals=audit_journals,
                     metric_schema=metric_schema, baseline_run_key=baseline_run_key)
    state.mkdir(parents=True, exist_ok=True)
    with FileLock(str(state / 'writer.lock'), timeout=0):
        owner = state / 'binding.json'
        saved = json.loads(owner.read_text(encoding='utf-8')) if owner.exists() else {}
        contract = binding(bundle, entity, project,
                           key=baseline_run_key or saved.get('baseline_run_key') or source.as_posix(),
                           adopted_run_id=existing_run_id or (saved.get('run_id', '') if saved.get('adopted') else ''))
        if saved and saved != contract:
            raise ValueError('reporter state belongs to another experiment/entity/project')
        group_file = state / 'group.json'
        if not group and group_file.is_file():
            group = json.loads(group_file.read_bytes())
        outbox_path = state / 'outbox.json'
        old = json.loads(outbox_path.read_text(encoding='utf-8')) if outbox_path.exists() else None
        if old and old['contract'] != contract:
            raise ValueError('outbox belongs to another experiment')
        for item in (old or {}).get('events', []):
            event = item['event']
            if digest(event) != item['id'] or event['update'] > bundle['durable_update']:
                raise ValueError('invalid or stale pending event')
            for key, value in event['values'].items():
                if bundle['rows'].get(str(event['update']), {}).get(key) != value:
                    raise ValueError('pending event no longer matches confirmed source')
        remote = remote_factory(contract, state / 'sdk')
        remote.group = group
        remote.require_existing = schema_v2_only
        remote.require_idle = metric_schema == 3
        remote.source_identity_sha256 = digest(bundle['identity'])
        history = remote.history()
        _, acknowledged = plan(bundle, history, adopted=contract['adopted'])
        if schema_v2_only and getattr(remote, 'needs_group_update', False):
            raise ValueError('schema-v2 backfill requires the existing W&B Run group')
        if not saved:
            atomic(owner, contract)
        if group:
            atomic(group_file, group)
        pending = [e for e in (old or {}).get('events', []) if e['id'] not in acknowledged]
        result = dict(run_id=run_id(contract), name='baseline', entity=entity, project=project,
                      sent_rows=0, durable_update=bundle['durable_update'], group=group, counts=bundle['counts'])

        def awaiting(events):
            return dict(result, status='ACK_PENDING',
                        pending_rows=sum(event['id'] not in acknowledged for event in events))

        if pending:
            # Older outboxes predate the finish marker. Give the read API a
            # chance to expose their rows before opening an existing writer.
            finished = old.get('awaiting_ack', False)
            if finished or getattr(remote, 'existing', False):
                history, acknowledged = _wait_for_history(remote, bundle, pending, history=history)
                pending = [event for event in pending if event['id'] not in acknowledged]
            if finished and pending:
                return awaiting(pending)
            if pending:
                if schema_v2_only and any(not SCHEMA_V2_METRICS.issuperset(event['event']['values'])
                                          for event in pending):
                    raise ValueError('previous sync still has unconfirmed non-schema-v2 rows; finish it first')
                try:
                    remote.send(pending, old['bundle'])
                except HistoryNotVisible:
                    return awaiting(pending)
                result['sent_rows'] += len(pending)
                atomic(outbox_path, dict(contract=contract, bundle=old['bundle'], events=pending, awaiting_ack=True))
                history, acknowledged = _wait_for_history(remote, bundle, pending)
                if any(event['id'] not in acknowledged for event in pending):
                    return awaiting(pending)
        events, _ = plan(bundle, history, adopted=contract['adopted'],
                         eligible=SCHEMA_V2_METRICS if schema_v2_only else None)
        # An unchanged retry can finish by reading history alone. Preserve
        # summary refresh for newer confirmed updates or an explicit new group.
        same_summary = old and all(old['bundle'][key] == bundle[key] for key in
                                  ('summary', 'sources', 'receipt', 'conditions'))
        if metric_schema == 3:
            summary_refresh = getattr(remote, 'remote_summary', {}) != v3.SUMMARY
        else:
            summary_refresh = not schema_v2_only and (not same_summary or getattr(remote, 'needs_group_update', False))
        if events or summary_refresh:
            atomic(outbox_path, dict(contract=contract, bundle=bundle, events=events))
            try:
                remote.send(events, bundle)
            except HistoryNotVisible:
                return awaiting(events)
            result['sent_rows'] += len(events)
            if events:
                atomic(outbox_path, dict(contract=contract, bundle=bundle, events=events, awaiting_ack=True))
                _, acknowledged = _wait_for_history(remote, bundle, events)
                if any(event['id'] not in acknowledged for event in events):
                    return awaiting(events)
        atomic(outbox_path, dict(contract=contract, bundle=bundle, events=[], awaiting_ack=False))
        result.update(status='SYNCED', pending_rows=0)
        atomic(state / 'last-success.json', result)
        return result


def _validate_v3_state(source, state, entity, project, group, existing_run_id='', schema_v2_only=False):
    v3.validate_destination(entity, project, group, state, existing_run_id=existing_run_id,
                            schema_v2_only=schema_v2_only)
    if source == state or state.parent != source.parent:
        raise ValueError('the Schema v3 state must be a separate sibling of the Baseline source')


@operation('baseline W&B: read-only Schema v3 history audit')
def audit_v3(source, state, entity, *, project, group, snapshot=None, registration=None,
             baseline_run_key='', audit_journals=False, remote_factory=WandbRemote):
    source, state = Path(source).resolve(), Path(state).resolve()
    _validate_v3_state(source, state, entity, project, group)
    bundle = collect(source, snapshot=snapshot, registration=registration,
                     audit_journals=audit_journals, metric_schema=3, baseline_run_key=baseline_run_key)
    contract = binding(bundle, entity, project, key=bundle['schema_identity']['origin']['baseline_run_key'])
    owner = state/'binding.json'
    if owner.is_file() and json.loads(owner.read_bytes()) != contract:
        raise ValueError('reporter state belongs to another experiment/entity/project')
    with TemporaryDirectory(prefix='baseline-v3-audit-') as temp:
        remote = remote_factory(contract, Path(temp))
        remote.group = group
        history = remote.history()
    missing_events, _ = plan(bundle, history, check_duplicates=False)
    missing = Counter(key for event in missing_events for key in event['event']['values'])
    keys = Counter((row['update'], key) for row in history for key in json.loads(row['baseline/fields']))
    duplicates = Counter()
    for (_, key), count in keys.items():
        if count > 1:
            duplicates[key] += count - 1
    summary = getattr(remote, 'remote_summary', {})
    verdict = ('DUPLICATES' if duplicates else 'NOT_IMPORTED' if not getattr(remote, 'existing', bool(history))
               else 'INCOMPLETE' if missing or summary != v3.SUMMARY else 'VERIFIED')
    return dict(status=verdict, run_id=run_id(contract), expected_by_metric=bundle['counts'],
                missing_by_metric=dict(missing), duplicate_by_metric=dict(duplicates),
                summary_expected=v3.SUMMARY, summary_observed=summary)
