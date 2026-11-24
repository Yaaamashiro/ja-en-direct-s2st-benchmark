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
    train = monitor['history'][0]['train']
    dev = dict(loss=9, unit_loss=2, source_letter=5, target_letter=6, decoder_target_ctc=7)
    monitor['history'] = [
        dict(update=update, train=train, dev=dev if update % 1000 == 0 else None, learning_rate=0.0002)
        for update in range(100, 400001, 100)
    ]
    monitor['update'] = 400000
    atomic(monitor_path, monitor)
    (root/'training-monitor.csv').unlink()
    (root/'training-bleu.csv').unlink()
    atomic(root/'training-bleu.json', {'results': [
        _matched_bleu(update) for update in (1000, 5000, 10000, 25000, 50000, 100000, 200000, 400000)
    ]})
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


def collect_v3(registered):
    return collect(registered[0], registration=registered[1], metric_schema=3, baseline_run_key=v3.BASELINE_KEY)


def replace_report(registered, name, change):
    root = registered[0]
    report, marker = root/name, root/'snapshot.json'
    value = json.loads(report.read_bytes())
    change(value)
    report.write_bytes(json.dumps(value).encode('utf-8'))
    receipt = json.loads(marker.read_bytes())
    receipt['files'][name] = sha256(report.read_bytes()).hexdigest()
    atomic(marker, receipt)


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
    assert len(bundle['rows']) == 4000 and sum(bundle['counts'].values()) == 30024
    for key, count in bundle['counts'].items():
        expected = (range(100, 400001, 100) if key.startswith('train/') else
                    range(1000, 400001, 1000) if key.startswith('dev/') else
                    (1000, 5000, 10000, 25000, 50000, 100000, 200000, 400000))
        assert count == len(expected)
        assert [int(update) for update, values in bundle['rows'].items() if key in values] == list(expected)
    assert row['eval/asr_bleu'] == 8 and row['eval/asr_failures'] == 0
    assert not any(key.startswith(('eval/aux_', 'train/window/')) for key in row)
    assert bundle['summary'] == v3.SUMMARY
    assert bundle['schema_identity']['origin']['run_id'] == '54aa109cb906644038c08f91'
    assert bundle['schema_identity']['lineage']['checkpoints/checkpoint_best.pt'] == 'a'*64


@pytest.mark.parametrize('split, index, field, metric, update', [
    ('train', 1, 'unit_loss', 'train/main_loss', 200),
    ('dev', 19, 'target_letter', 'dev/aux/target_letter_loss', 2000),
])
def test_v3_rejects_a_missing_required_loss(registered, split, index, field, metric, update):
    replace_report(registered, 'training-monitor.json', lambda report: report['history'][index][split].pop(field))
    with pytest.raises(ValueError, match=f'{metric} missing updates: {update}'):
        collect_v3(registered)


def test_v3_rejects_missing_asr_milestone(registered):
    replace_report(registered, 'training-bleu.json', lambda report: report['results'].pop(1))
    with pytest.raises(ValueError, match=r'eval/asr_bleu missing updates: 5000'):
        collect_v3(registered)


def test_v3_rejects_an_asr_measurement_at_an_unscheduled_update(registered):
    def change(report):
        report['results'][1] = _matched_bleu(6000)

    replace_report(registered, 'training-bleu.json', change)
    with pytest.raises(ValueError, match=r'eval/asr_bleu at unexpected update 6000'):
        collect_v3(registered)


@pytest.mark.parametrize('field', ('bleu', 'failures', 'samples'))
def test_v3_rejects_partial_asr_measurement(registered, field):
    replace_report(registered, 'training-bleu.json', lambda report: report['results'][1].pop(f'asr_{field}'))
    with pytest.raises(ValueError, match=f'eval/asr_{field} missing at update 5000'):
        collect_v3(registered)


def test_v3_only_adopts_a_qualified_complete_aux_group(registered):
    def change(report):
        for index in (0, 1):
            item = report['results'][index]
            item.update(status='completed', aux_failures=0)
            item['aux_scoring']['failures'] = 0
        report['results'][1].pop('aux_samples')

    replace_report(registered, 'training-bleu.json', change)
    rows = collect_v3(registered)['rows']
    assert {key: rows['1000'][key] for key in v3.AUX_BLEU_METRICS} == {
        'eval/aux_bleu': 12, 'eval/aux_failures': 0, 'eval/aux_samples': 500,
    }
    assert not (rows['5000'].keys() & set(v3.AUX_BLEU_METRICS))


def test_v3_detects_wrong_coordinates_even_with_the_same_point_count(registered):
    replace_report(registered, 'training-monitor.json', lambda report: report['history'][1].update(update=201))
    legacy = collect(registered[0], registration=registered[1])
    assert legacy['counts']['train/learning_rate'] == 4000
    with pytest.raises(ValueError, match=r'train/learning_rate missing updates: 200; train/learning_rate unexpected updates: 201'):
        collect_v3(registered)


@pytest.mark.parametrize('update', (0, 100.5, 400100))
def test_v3_rejects_invalid_monitor_coordinates(registered, update):
    replace_report(registered, 'training-monitor.json', lambda report: report['history'][0].update(update=update))
    with pytest.raises(ValueError) as error:
        collect_v3(registered)
    assert str(update) in str(error.value)
    assert 'train/main_loss' in str(error.value)
    assert 'unexpected updates' in str(error.value) or 'invalid optimizer update' in str(error.value)


def test_v3_reports_nonfinite_required_metric_and_update(registered):
    replace_report(registered, 'training-monitor.json',
                   lambda report: report['history'][0]['train'].update(unit_loss=math.nan))
    with pytest.raises(ValueError, match=r'train/main_loss at update 100: nonfinite'):
        collect_v3(registered)


def test_v3_reports_nonfinite_asr_scoring_and_update(registered):
    replace_report(registered, 'training-bleu.json',
                   lambda report: report['results'][1]['asr_scoring'].update(bleu=math.inf))
    with pytest.raises(ValueError, match=r'eval/asr_bleu scoring at update 5000: nonfinite'):
        collect_v3(registered)


def test_v3_invalid_source_cannot_verify_or_open_writer(registered, tmp_path):
    replace_report(registered, 'training-monitor.json', lambda report: report['history'][0].pop('learning_rate'))
    state, server = tmp_path/v3.STATE_NAME, Server()
    with pytest.raises(ValueError, match=r'train/learning_rate missing updates: 100'):
        delivery.audit_v3(registered[0], state, **common(registered), remote_factory=server.factory)
    with pytest.raises(ValueError, match=r'train/learning_rate missing updates: 100'):
        write(registered, state, server)
    assert server.contract is None and server.calls == 0 and not state.exists()


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
    assert partial['status'] == 'INCOMPLETE' and partial['missing_by_metric']['dev/main_loss'] == 400
    result = write(registered, state, server)
    assert result['status'] == 'SYNCED' and result['sent_rows'] == 3999
    assert result['run_id'] == '5db658f8baac28cbe211f3b8'
    assert [row['_step'] for row in server.rows] == list(range(4000))
    assert [row['update'] for row in server.rows] == list(range(100, 400001, 100))
    assert json.loads((state/'binding.json').read_bytes())['metric_schema'] == 3
    assert write(registered, state, server)['sent_rows'] == 0 and server.calls == 2
    local = (state/'last-success.json').read_bytes()
    after = delivery.audit_v3(registered[0], state, **common(registered), remote_factory=server.factory)
    assert after['status'] == 'VERIFIED' and not after['missing_by_metric']
    assert server.calls == 2 and (state/'last-success.json').read_bytes() == local
    state.rename(tmp_path/'archived-state')
    assert write(registered, state, server)['sent_rows'] == 0 and server.calls == 2

    duplicate = dict(server.rows[0], _step=4000)
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
