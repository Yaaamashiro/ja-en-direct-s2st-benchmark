"""Optional network adapter, imported only by the isolated CPU reporter."""
import json
from pathlib import Path

from .baseline_wandb import (API_BASE, PROJECT, atomic, binding, canonical, collect,
                             digest, plan, run_id)
from .progress import operation


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

    def history(self):
        self.api.flush()
        try:
            remote = self.api.run(self.path)
        except self.wandb.errors.CommError as error:
            # Only a definite missing run permits creation; auth/network errors
            # are not evidence that a new run is needed.
            response = getattr(error, 'response', None)
            if getattr(response, 'status_code', None) == 404:
                return []
            # Public API raises ValueError("Could not find run ...") wrapped as
            # CommError, without an HTTP response, for its GraphQL null run.
            if str(error).startswith('Could not find run '):
                return []
            raise
        if remote.config.get('baseline_contract') != self.contract:
            raise ValueError('W&B run identity/destination differs; refusing to write')
        return list(remote.scan_history(page_size=500, use_cache=False))

    def send(self, events, bundle):
        run = self.wandb.init(
            entity=self.contract['entity'], project=self.contract['project'],
            id=run_id(self.contract), name='baseline', resume='allow',
            dir=str(self.logs), save_code=False,
            config={'condition': 'baseline', 'baseline_contract': self.contract,
                    'source_identity': bundle['identity'],
                    'loss_mapping_version': 1},
            settings=self.wandb.Settings(console='off', disable_git=True, disable_code=True,
                base_url=API_BASE,
                x_disable_meta=True, x_disable_stats=True, x_service_wait=10,
                init_timeout=20, finish_timeout=20, finish_timeout_raises=True))
        try:
            if run.config.get('baseline_contract') != self.contract:
                raise ValueError('W&B initialization returned another experiment')
            run.define_metric('update', summary='none')
            for prefix in ('baseline', 'train', 'dev', 'eval'):
                run.define_metric(prefix + '/*', step_metric='update')
            for item in events:
                # Reuse assigned W&B steps after an uncertain acknowledgement.
                # A later server step without a visible event is NOT a license
                # to append the same event at a new step.
                if run.step > item['step']:
                    raise RuntimeError('server advanced beyond pending event; retry after history becomes visible')
                event = item['event']
                payload = dict(event['values'], update=event['update'])
                payload.update({'baseline/event_id': item['id'],
                                'baseline/fields': canonical(sorted(event['values'])),
                                'baseline/provenance': canonical(event['provenance'])})
                run.log(payload, step=item['step'], commit=True)
            for key, value in bundle['summary'].items():
                run.summary[key] = value
            run.summary['baseline/sources'] = bundle['sources']
            run.summary['baseline/receipt'] = bundle['receipt']
            run.summary['baseline/evaluation_conditions'] = bundle['conditions']
        finally:
            run.finish()


@operation('baseline W&B: reconcile and publish measured metrics')
def synchronize(source, state, entity, *, project=PROJECT, snapshot=None, remote_factory=WandbRemote):
    """Persist an outbox first. Remote event hashes, not local watermarks, dedupe.

    The source is always read-only. State/logs must live outside it. A successful
    SDK finish is still verified against remote history before acknowledging.
    """
    from filelock import FileLock
    source, state = Path(source).resolve(), Path(state).resolve()
    if state == source or state.is_relative_to(source) or source.is_relative_to(state):
        raise ValueError('reporter state must be separate from read-only source data')
    bundle = collect(source, snapshot=snapshot)
    contract = binding(bundle, entity, project)
    state.mkdir(parents=True, exist_ok=True)
    with FileLock(str(state / 'writer.lock'), timeout=0):
        owner = state / 'binding.json'
        if owner.exists() and json.loads(owner.read_text(encoding='utf-8')) != contract:
            raise ValueError('reporter state belongs to another experiment/entity/project')
        if not owner.exists():
            atomic(owner, contract)
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
        if old is None:
            # Preserve measured data even if authentication/network fails before
            # the first remote read. Steps are assigned only after reconciliation.
            atomic(outbox_path, dict(contract=contract, bundle=bundle, events=[]))
        remote = remote_factory(contract, state / 'sdk')
        history = remote.history()
        _, acknowledged = plan(bundle, history)
        pending = [e for e in (old or {}).get('events', []) if e['id'] not in acknowledged]
        if pending:
            # Finish the frozen old batch before adding late metric cells.
            remote.send(pending, old['bundle'])
            history = remote.history()
            _, acknowledged = plan(bundle, history)
            if any(e['id'] not in acknowledged for e in pending):
                raise RuntimeError('W&B acknowledgement pending; outbox retained for retry')
        events, _ = plan(bundle, history)
        atomic(outbox_path, dict(contract=contract, bundle=bundle, events=events))
        # Also refresh measured summary/provenance on a no-op history sync.
        remote.send(events, bundle)
        _, acknowledged = plan(bundle, remote.history())
        if any(e['id'] not in acknowledged for e in events):
            raise RuntimeError('W&B acknowledgement pending; outbox retained for retry')
        atomic(outbox_path, dict(contract=contract, bundle=bundle, events=[]))
        result = dict(run_id=run_id(contract), name='baseline', entity=entity, project=project,
                      sent_rows=len(events) + len(pending), durable_update=bundle['durable_update'],
                      counts=bundle['counts'])
        atomic(state / 'last-success.json', result)
        return result
