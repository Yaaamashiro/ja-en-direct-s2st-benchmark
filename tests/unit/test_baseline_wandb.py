"""CPU fixtures exercise measured values, checkpoint gating and uncertain acks."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from direct_s2st.baseline_wandb import (FILES, atomic, binding, canonical, collect,
                                       digest, plan, run_id)
from direct_s2st.baseline_wandb_delivery import synchronize


def source(tmp_path, *, durable=100, update=100):
    root = tmp_path / 'source'
    root.mkdir(exist_ok=True)
    monitor = dict(metric='teacher_forced_dev_unit_loss', update=update,
                   policy=dict(max_update=400000), warmup=10000,
                   best_value=3., best_update=100, best_sha256='a'*64,
                   history=[dict(update=update, learning_rate=.0001,
                       train=dict(loss=12., unit_loss=3., source_letter=5.,
                                  target_letter=6., decoder_target_ctc=7.),
                       dev=dict(loss=10., unit_loss=2.5))])
    atomic(root/'training-monitor.json', monitor)
    (root/'training-monitor.csv').write_text(
        f'update,learning_rate,train_loss,train_unit_loss,dev_unit_loss\n{update},0.0001,12,3,2.5\n', encoding='utf-8')
    atomic(root/'training-bleu.json', dict(purpose='dev_diagnostics_not_test', results=[
        dict(update=update, aux_bleu=0., aux_samples=500, aux_failures=500,
             asr_bleu=None, asr_samples=0, asr_failures=0, seconds=7.,
             identity=dict(policy=dict(beam=10), ids=['private-id'], references={'id': 'private text'}))]))
    (root/'training-bleu.csv').write_text(
        f'update,aux_bleu,asr_bleu,aux_samples,asr_samples,aux_failures,asr_failures\n{update},0,,500,0,500,0\n', encoding='utf-8')
    seal(root, durable)
    return root


def seal(root, durable=100):
    files = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
             for name in FILES if (root/name).exists()}
    atomic(root/'snapshot.json', dict(updates=durable, files=files,
           identity=dict(config=dict(kind='s2ut'), inputs={'manifest': 'abc'}, work='/original/baseline')))


class Server:
    def __init__(self):
        self.rows = []
        self.contract = None
        self.fail = None
        self.calls = 0
        self.summary = {}
    def factory(self, contract, logs):
        if self.contract is None:
            self.contract = contract
        if self.contract != contract:
            raise ValueError('foreign remote run')
        return self
    def history(self):
        return copy.deepcopy(self.rows)
    def send(self, events, bundle):
        self.calls += 1
        if self.fail == 'before':
            raise ConnectionError('no network')
        for event in events:
            if any(row['_step'] == event['step'] for row in self.rows):
                raise ValueError('step reused')
            self.rows.append(dict(event['event']['values'], update=event['event']['update'],
                _step=event['step'], **{'baseline/event_id': event['id'],
                                      'baseline/fields': canonical(sorted(event['event']['values'])),
                                      'baseline/provenance': canonical(event['event']['provenance'])}))
        self.summary = bundle['summary']
        if self.fail == 'after':
            raise ConnectionError('ack lost after successful delivery')


def sync(root, state, server, **kwargs):
    return synchronize(root, state, 'owner', remote_factory=server.factory, **kwargs)


def test_measured_zero_missing_asr_and_coalesced_json_csv(tmp_path):
    root = source(tmp_path)
    bundle = collect(root)
    assert list(bundle['rows']) == ['100']
    row = bundle['rows']['100']
    assert row['eval/aux_bleu'] == 0 and row['eval/aux_failures'] == 500
    assert not any('asr' in key for key in row)
    assert row['dev/main_loss'] == 2.5
    assert row['baseline/train/unit_loss_bits'] == 3
    assert 'train/main_loss' not in row and 'train/total_loss' not in row
    assert bundle['summary']['train/max_update'] == 400000
    assert 'private text' not in canonical(bundle)
    assert 'private-id' not in canonical(bundle)


def test_idempotence_and_lost_local_state(tmp_path):
    root, server = source(tmp_path), Server()
    assert sync(root, tmp_path/'state', server)['sent_rows'] == 1
    assert sync(root, tmp_path/'state', server)['sent_rows'] == 0
    assert sync(root, tmp_path/'new-local-state', server)['sent_rows'] == 0
    assert len(server.rows) == 1
    assert server.summary['baseline/best_sha256'] == 'a'*64


@pytest.mark.parametrize('failure', ['before', 'after'])
def test_communication_failure_replay_without_duplicate(tmp_path, failure):
    root, server, state = source(tmp_path), Server(), tmp_path/'state'
    server.fail = failure
    with pytest.raises(ConnectionError):
        sync(root, state, server)
    assert json.loads((state/'outbox.json').read_text())['events']
    server.fail = None
    sync(root, state, server)
    assert len(server.rows) == 1
    assert json.loads((state/'outbox.json').read_text())['events'] == []


def test_resume_new_updates_and_late_bleu_same_update(tmp_path):
    root, server, state = source(tmp_path), Server(), tmp_path/'state'
    sync(root, state, server)
    p = root/'training-bleu.json'
    data = json.loads(p.read_text())
    data['results'][0].update(asr_bleu=1.5, asr_samples=100, asr_failures=0)
    atomic(p, data)
    # CSV may be absent; do not fabricate measurements to fill it.
    (root/'training-bleu.csv').unlink()
    seal(root)
    sync(root, state, server)
    assert len(server.rows) == 2
    assert 'eval/aux_bleu' not in server.rows[-1]
    assert server.rows[-1]['eval/asr_failures'] == 0
    source(tmp_path, durable=200, update=200)
    sync(root, state, server)
    assert [r['update'] for r in server.rows] == [100, 100, 200]


def test_source_artifacts_remain_byte_identical(tmp_path):
    root, server = source(tmp_path), Server()
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    sync(root, tmp_path/'state', server)
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError, match='separate'):
        sync(root, root/'wandb', server)


def test_no_unconfirmed_updates_or_best(tmp_path):
    root = source(tmp_path, durable=100, update=200)
    b = collect(root)
    assert not b['rows'] and 'baseline/best_update' not in b['summary']


def test_receipt_required_and_report_hash_checked(tmp_path):
    root = source(tmp_path)
    (root/'training-monitor.csv').write_text('corrupt', encoding='utf-8')
    with pytest.raises(ValueError, match='published snapshot'):
        collect(root)
    (root/'snapshot.json').unlink()
    with pytest.raises(ValueError, match='receipt'):
        collect(root)


def test_duplicate_conflicts_and_nonfinite_rejected(tmp_path):
    root = source(tmp_path)
    p = root/'training-monitor.json'
    data = json.loads(p.read_text())
    data['history'].append(copy.deepcopy(data['history'][0]))
    atomic(p, data); seal(root)
    assert len(collect(root)['rows']) == 1
    data['history'][-1]['train']['loss'] = 13
    atomic(p, data); seal(root)
    with pytest.raises(ValueError, match='conflicting'):
        collect(root)
    data['history'][-1]['train']['loss'] = 'NaN'
    atomic(p, data); seal(root)
    with pytest.raises(ValueError, match='nonfinite'):
        collect(root)


def test_cross_experiment_and_destination_misrouting(tmp_path):
    root, server, state = source(tmp_path), Server(), tmp_path/'state'
    sync(root, state, server)
    with pytest.raises(ValueError):
        synchronize(root, state, 'different-owner', remote_factory=server.factory)
    with pytest.raises(ValueError):
        sync(root, state, server, project='typo')
    receipt = json.loads((root/'snapshot.json').read_text())
    receipt['identity']['inputs']['manifest'] = 'other-dataset'
    atomic(root/'snapshot.json', receipt)
    with pytest.raises(ValueError, match='another experiment'):
        sync(root, state, server)


def test_remote_conflicting_or_foreign_history_is_never_overwritten(tmp_path):
    b = collect(source(tmp_path))
    with pytest.raises(ValueError, match='foreign'):
        plan(b, [dict(update=100, _step=0, loss=3.)])
    server = Server()
    sync(tmp_path/'source', tmp_path/'state', server)
    server.rows[0]['eval/aux_bleu'] = 7
    with pytest.raises(ValueError, match='content differs'):
        sync(tmp_path/'source', tmp_path/'state', server)


def test_no_model_imports_in_inspection(tmp_path):
    import subprocess
    root = source(tmp_path)
    code = ('import sys; from direct_s2st.baseline_wandb import collect; collect(sys.argv[1]); '
            'assert not set(sys.modules).intersection({"torch","fairseq","transformers","wandb","numpy"})')
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-c', code, str(root)], cwd=repo/'src', capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_notebooks_compile_and_default_reporting_disabled():
    import ast
    repo = Path(__file__).resolve().parents[2]
    for name in ('baseline_wandb_import.ipynb', 'colab_training.ipynb'):
        notebook = json.loads((repo/'notebooks'/name).read_text(encoding='utf-8'))
        for c in notebook['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(c['source']), feature_version=(3, 10))
        config = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'WANDB_ENTITY =' in ''.join(c['source']))
        defaults = {n.targets[0].id: n.value.value for n in ast.parse(config).body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and isinstance(n.value, ast.Constant)}
        assert defaults['WANDB_ENTITY'] == ''
        assert defaults['WANDB_PROJECT'] == 'alignment-transfer-s2ut'
    normal = (repo/'notebooks/colab_training.ipynb').read_text(encoding='utf-8')
    assert 'BASELINE_WANDB = False' in normal
    importer = (repo/'notebooks/baseline_wandb_import.ipynb').read_text(encoding='utf-8')
    assert 'SYNC_TO_WANDB = False' in importer


def test_scientific_notebook_cells_unchanged_outside_optional_reporter():
    import subprocess
    from notebook_baseline_wandb import without_wandb
    repo = Path(__file__).resolve().parents[2]
    before = json.loads(subprocess.check_output(['git', '-c', 'safe.directory='+str(repo),
        '-C', str(repo), 'show', '31ae800317d13e0644f3623b219adab6757ac3e4:notebooks/colab_training.ipynb'], encoding='utf-8'))
    after = json.loads((repo/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    assert [without_wandb(c) for c in after['cells']] == before['cells']


def test_reporter_failure_preserves_training_exit(tmp_path, monkeypatch):
    import importlib.util
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('wb_runtime_test', repo/'scripts/colab/wandb_runtime.py')
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(OSError('offline')))
    ran = False
    with runtime.reporter(repo, tmp_path/'source', tmp_path/'state', 'python', 'owner', 'alignment-transfer-s2ut'):
        ran = True
    assert ran
    with pytest.raises(RuntimeError, match='original trainer error'):
        with runtime.reporter(repo, tmp_path/'source', tmp_path/'state', 'python', 'owner', 'alignment-transfer-s2ut'):
            raise RuntimeError('original trainer error')


def test_copied_sources_have_the_same_identity_and_no_new_history(tmp_path):
    import shutil
    root, server = source(tmp_path), Server()
    sync(root, tmp_path/'state', server)
    copied = tmp_path/'copy'
    shutil.copytree(root, copied)
    result = sync(copied, tmp_path/'copy-state', server)
    assert result['sent_rows'] == 0 and len(server.rows) == 1


def test_multiple_receipts_at_same_update_choose_matching_bytes(tmp_path):
    root = source(tmp_path)
    good = json.loads((root/'snapshot.json').read_text())
    stale = copy.deepcopy(good)
    stale['files']['training-monitor.json'] = 'f'*64
    (root/'snapshot.json').unlink()
    atomic(root/'000100-a'/'snapshot.json', good)
    atomic(root/'000100-z'/'snapshot.json', stale)
    assert collect(root)['durable_update'] == 100
    # A higher published checkpoint never falls back silently to an old mirror.
    stale['updates'] = 200
    atomic(root/'000200-a'/'snapshot.json', stale)
    with pytest.raises(ValueError, match='latest published'):
        collect(root)


def test_remote_stale_source_and_modified_pending_event_rejected(tmp_path):
    root, server, state = source(tmp_path, durable=200, update=200), Server(), tmp_path/'state'
    sync(root, state, server)
    source(tmp_path, durable=100, update=100)
    with pytest.raises(ValueError, match='older'):
        sync(root, state, server)
    server.rows = []
    server.fail = 'before'
    with pytest.raises(ConnectionError):
        sync(root, state, server)
    outbox = json.loads((state/'outbox.json').read_text())
    outbox['events'][0]['event']['values']['eval/aux_bleu'] = 999
    atomic(state/'outbox.json', outbox)
    server.fail = None
    with pytest.raises(ValueError, match='invalid or stale'):
        sync(root, state, server)


def test_adapter_rejects_wrong_project_foreign_run_and_auth_error(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from direct_s2st.baseline_wandb import API_BASE, PROJECT
    from direct_s2st.baseline_wandb_delivery import WandbRemote
    contract = binding(collect(source(tmp_path)), 'owner')
    class CommError(Exception):
        pass
    class Api:
        projects_found = [NS(name=PROJECT)]
        config = {'baseline_contract': contract}
        error = None
        flushed = 0
        def __init__(self, **kwargs):
            assert kwargs['overrides']['base_url'] == API_BASE
        def projects(self, **kwargs):
            return self.projects_found
        def flush(self):
            type(self).flushed += 1
        def run(self, path):
            if self.error:
                raise self.error
            def scan(**kwargs):
                assert kwargs == dict(page_size=500, use_cache=False)
                return []
            return NS(config=self.config, scan_history=scan)
    monkeypatch.setitem(sys.modules, 'wandb', NS(Api=Api, errors=NS(CommError=CommError)))
    remote = WandbRemote(contract, tmp_path/'logs')
    assert remote.history() == [] and Api.flushed == 1
    Api.error = CommError('authentication failed')
    with pytest.raises(CommError):
        remote.history()
    Api.error = CommError('Could not find run missing')
    assert remote.history() == []
    Api.error, Api.config = None, {'baseline_contract': {'wrong': 'experiment'}}
    with pytest.raises(ValueError, match='identity/destination'):
        remote.history()
    Api.projects_found = []
    with pytest.raises(ValueError, match='project was not found'):
        WandbRemote(contract, tmp_path/'logs')


def test_adapter_never_reassigns_uncertain_event_to_new_server_step(tmp_path):
    from types import SimpleNamespace as NS
    from direct_s2st.baseline_wandb_delivery import WandbRemote
    b = collect(source(tmp_path))
    r = WandbRemote.__new__(WandbRemote)
    r.contract, r.logs = binding(b, 'owner'), tmp_path
    calls = []
    run = NS(config={'baseline_contract': r.contract}, step=1,
             define_metric=lambda *a, **k: None, finish=lambda: calls.append('finish'),
             log=lambda *a, **k: calls.append('log'))
    r.wandb = NS(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
    events, _ = plan(b, [])
    with pytest.raises(RuntimeError, match='server advanced'):
        r.send(events, b)
    assert calls == ['finish']


def test_reporter_shutdown_race_cannot_change_success_or_training_error(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace as NS
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('wb_runtime_race', repo/'scripts/colab/wandb_runtime.py')
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **k: NS())
    monkeypatch.setattr(runtime, '_stop', lambda *a: (_ for _ in ()).throw(ProcessLookupError('already exited')))
    with runtime.reporter(repo, tmp_path/'source', tmp_path/'state', 'python', 'owner', 'test-project'):
        pass
    with pytest.raises(RuntimeError, match='trainer'):
        with runtime.reporter(repo, tmp_path/'source', tmp_path/'state', 'python', 'owner', 'test-project'):
            raise RuntimeError('trainer')


def test_extra_measured_metric_is_preserved_and_summary_nan_rejected(tmp_path):
    root = source(tmp_path)
    p = root/'training-monitor.json'; data = json.loads(p.read_text())
    data['history'][0]['train']['gradient_norm'] = 1.25
    atomic(p, data); seal(root)
    assert collect(root)['rows']['100']['baseline/train/observed/gradient_norm'] == 1.25
    data['best_value'] = 'NaN'
    atomic(p, data); seal(root)
    with pytest.raises(ValueError, match='nonfinite'):
        collect(root)
