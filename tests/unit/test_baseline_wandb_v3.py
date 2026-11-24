"""Exercise the published lineage, schema isolation, and offline history reconciliation."""
import copy
from hashlib import sha256
import json
import math
from pathlib import Path
import types
from zipfile import ZipFile

import pytest

from direct_s2st import baseline_wandb_v3 as v3
from direct_s2st.baseline_wandb import atomic, binding, collect, digest, run_id
from direct_s2st import baseline_wandb_delivery as delivery
from direct_s2st.s2ut.monitor import LossSums
from test_baseline_wandb import Server, source, seal, _matched_bleu


@pytest.fixture
def registered(tmp_path):
    root = source(tmp_path, durable=400000, update=100)
    monitor_path = root/'training-monitor.json'
    monitor = json.loads(monitor_path.read_bytes())
    points = monitor['history']
    points[0]['dev'] = None
    points += [{'update': 1000, 'train': points[0]['train'], 'learning_rate': 0.0002,
                'dev': dict(loss=9, unit_loss=2, source_letter=5, target_letter=6, decoder_target_ctc=7)},
               {'update': 1001, 'train': points[0]['train'], 'dev': {'unit_loss': 99}}]
    monitor['update'] = 400000
    atomic(monitor_path, monitor)
    (root/'training-monitor.csv').unlink()
    (root/'training-bleu.csv').unlink()
    atomic(root/'training-bleu.json', {'results': [_matched_bleu(1000), _matched_bleu(2000)]})
    seal(root, 400000)
    snapshot = json.loads((root/'snapshot.json').read_bytes())
    snapshot.pop('s2ut_lineage')
    snapshot['files']['checkpoints/checkpoint_best.pt'] = 'a'*64
    snapshot['identity']['config']['research_metadata'] = dict(s2ut_recipe='paper_fisher_v1',
                                                               fairseq_revision=v3.FAIRSEQ_REVISION)
    atomic(root/'snapshot.json', snapshot)
    record = dict(format='native-fairseq-s2ut-comparison-v1', method='baseline',
                  checkpoint='/original/baseline/checkpoints/checkpoint_best.pt',
                  checkpoint_hash='a'*64, checkpoint_updates=400000, baseline_fingerprint=v3.FINGERPRINT,
                  metadata=dict(training_fingerprint=v3.FINGERPRINT),
                  provenance=dict(recipe_lock_hash='b'*64, metadata_hash='b'*64))
    record['record_hash'] = sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    registration = tmp_path/'registration'
    registration.mkdir()
    payload = json.dumps(record).encode()
    archive = registration/'snapshot.zip'
    with ZipFile(archive, 'w') as package:
        package.writestr('baseline.json', payload)
    marker = registration/'snapshot.json'
    atomic(marker, dict(storage='zip-v1', zip_sha256=sha256(archive.read_bytes()).hexdigest(),
                        files={'baseline.json': sha256(payload).hexdigest()}))
    return root, marker


def common(registered):
    return dict(registration=registered[1], baseline_run_key=v3.BASELINE_KEY,
                entity=v3.ENTITY, project=v3.PROJECT, group=v3.GROUP)


def write(registered, state, server, **overrides):
    options = dict(common(registered), metric_schema=3, remote_factory=server.factory)
    options.update(overrides)
    return delivery.synchronize(registered[0], state, **options)


def test_unequal_microbatches_use_weighted_sums_and_not_nll():
    fixture = json.loads((Path(__file__).resolve().parents[1]/'fixtures/wandb_metric_schema_v3.json').read_bytes())
    collector = LossSums()  # Exercise the existing Baseline monitor's reducer.
    for batch in fixture['microbatches']:
        task = batch['main_sum'] + sum(fixture['weights'][key] * n for key, n in batch['aux_sum'].items())
        collector.record((task, batch['main_count'], {'multitask': {
            key: dict(loss=n, sample_size=batch['aux_count'][key]) for key, n in batch['aux_sum'].items()
        }}), batch['main_sum'])
    bits = collector.result()
    row = {f'baseline/train/{key}_bits': value for key, value in bits.items()}
    rows = v3.measurements({'rows': {'100': row}})
    assert rows['100'] == pytest.approx({f'train/{key}': value for key, value in fixture['expected_nats'].items()})
    assert rows['100']['train/main_loss'] != fixture['expected_nll_nats']


def test_v3_source_cadence_exact_conversion_and_bleu_qualification(registered):
    source, receipt = registered
    legacy = collect(source, registration=receipt)
    bundle = collect(source, registration=receipt, metric_schema=3, baseline_run_key=v3.BASELINE_KEY)
    row = bundle['rows']['1000']
    assert row['dev/main_loss'] == legacy['rows']['1000']['dev/main_loss'] * math.log(2)
    assert row['dev/total_loss'] == 9 * math.log(2)
    assert [row[f'dev/aux/{key}_loss'] for key in v3.AUX] == [n * math.log(2) for n in (5, 6, 7)]
    assert row['train/total_loss'] == row['train/task_loss'] == 12 * math.log(2)
    assert 'dev/main_loss' not in bundle['rows']['100']
    assert set(bundle['rows']) == {'100', '1000'}
    assert row['eval/asr_bleu'] == 8 and row['eval/asr_failures'] == 0
    assert not any(key.startswith(('eval/aux_', 'train/window/')) for key in row)
    assert bundle['summary'] == v3.SUMMARY
    assert bundle['schema_identity']['origin']['run_id'] == '54aa109cb906644038c08f91'
    assert bundle['schema_identity']['lineage']['checkpoints/checkpoint_best.pt'] == 'a'*64


def test_v3_rejects_wrong_scope_before_touching_legacy_binding(registered, tmp_path):
    old = tmp_path/'wandb-baseline'
    atomic(old/'binding.json', {'version': 2, 'project': 'direct-s2st'})
    server = Server()
    before = (old/'binding.json').read_bytes()
    with pytest.raises(ValueError, match='own wandb-baseline-v3 state'):
        write(registered, old, server)
    state = tmp_path/v3.STATE_NAME
    for override in ({'project': 'direct-s2st'}, {'entity': 'wrong'}, {'group': 'wrong'},
                     {'existing_run_id': '54aa109cb906644038c08f91'}, {'schema_v2_only': True},
                     {'metric_schema': 2}, {'baseline_run_key': 'wrong/baseline'},
                     {'registration': None}):
        with pytest.raises(ValueError):
            write(registered, state, server, **override)
    assert list(old.iterdir()) == [old/'binding.json']
    assert (old/'binding.json').read_bytes() == before
    assert not state.exists() and server.contract is None


def test_v3_partial_retry_idempotence_and_read_only_audit(registered, tmp_path):
    class Partial(Server):
        def send(self, events, bundle):
            if self.fail == 'partial':
                super().send(events[:1], bundle)
                raise ConnectionError('interrupted')
            super().send(events, bundle)
            self.remote_summary = bundle['summary']

    state, server = tmp_path/v3.STATE_NAME, Partial()
    before = delivery.audit_v3(registered[0], state, **common(registered), remote_factory=server.factory)
    assert before['status'] == 'NOT_IMPORTED' and not state.exists()
    server.fail = 'partial'
    with pytest.raises(ConnectionError):
        write(registered, state, server)
    server.fail = None
    partial = delivery.audit_v3(registered[0], state, **common(registered), remote_factory=server.factory)
    assert partial['status'] == 'INCOMPLETE' and partial['missing_by_metric']['dev/main_loss'] == 1
    result = write(registered, state, server)
    assert result['status'] == 'SYNCED' and result['sent_rows'] == 1
    assert result['run_id'] == '5db658f8baac28cbe211f3b8'
    assert [row['_step'] for row in server.rows] == [0, 1]
    assert [row['update'] for row in server.rows] == [100, 1000]
    assert json.loads((state/'binding.json').read_bytes())['metric_schema'] == 3
    assert write(registered, state, server)['sent_rows'] == 0 and server.calls == 2
    local = (state/'last-success.json').read_bytes()
    after = delivery.audit_v3(registered[0], state, **common(registered), remote_factory=server.factory)
    assert after['status'] == 'VERIFIED' and not after['missing_by_metric']
    assert server.calls == 2 and (state/'last-success.json').read_bytes() == local
    state.rename(tmp_path/'archived-state')
    assert write(registered, state, server)['sent_rows'] == 0 and server.calls == 2

    duplicate = dict(server.rows[0], _step=2)
    server.rows.append(duplicate)
    report = delivery.audit_v3(registered[0], state, **common(registered), remote_factory=server.factory)
    assert report['status'] == 'DUPLICATES' and report['duplicate_by_metric']['train/main_loss'] == 1
    with pytest.raises(ValueError, match='duplicate Schema v3 history'):
        write(registered, state, server)
    server.rows.pop()
    row = server.rows[0]
    row['train/main_loss'] += 1
    bits = {key: row[key] for key in json.loads(row['baseline/fields'])}
    row['baseline/event_id'] = digest(dict(update=row['update'], values=bits,
                                           provenance=json.loads(row['baseline/provenance'])))
    with pytest.raises(ValueError, match='conflicting duplicate metric: train/main_loss'):
        write(registered, state, server)
    assert server.calls == 2


def test_v3_waits_for_history_visibility_without_resending(registered, tmp_path, monkeypatch):
    class Delayed(Server):
        visible = False

        def send(self, events, bundle):
            super().send(events, bundle)
            self.remote_summary = bundle['summary']

        def history(self):
            return super().history() if self.visible else []
    monkeypatch.setattr(delivery.time, 'sleep', lambda _seconds: None)
    state, server = tmp_path/v3.STATE_NAME, Delayed()
    result = write(registered, state, server)
    assert result['status'] == 'ACK_PENDING' and server.calls == 1
    assert write(registered, state, server)['status'] == 'ACK_PENDING' and server.calls == 1
    server.visible = True
    assert write(registered, state, server)['status'] == 'SYNCED' and server.calls == 1


def test_v3_remote_rejects_schema_v2_or_wrong_group(registered, tmp_path):
    bundle = collect(registered[0], registration=registered[1], metric_schema=3, baseline_run_key=v3.BASELINE_KEY)
    contract = binding(bundle, v3.ENTITY, v3.PROJECT, key=v3.BASELINE_KEY)
    assert run_id(contract) != contract['origin']['run_id']
    remote = delivery.WandbRemote.__new__(delivery.WandbRemote)
    remote.contract, remote.path = contract, 'run'
    remote.group, remote.progress, remote.existing = v3.GROUP, 0, False
    remote.wandb = types.SimpleNamespace(errors=types.SimpleNamespace(CommError=ConnectionError))
    target = types.SimpleNamespace(config={'baseline_contract': copy.deepcopy(contract), 'metric_schema': 2},
                                   state='finished', group=v3.GROUP, summary=v3.SUMMARY)
    remote.api = types.SimpleNamespace(flush=lambda: None, run=lambda _path: target)
    with pytest.raises(ValueError, match='identity/destination'):
        remote.history()
    target.config['metric_schema'] = 3
    target.group = 'experiment-001/transfer-fixed-v1'
    with pytest.raises(ValueError, match='wrong group'):
        remote.history()
