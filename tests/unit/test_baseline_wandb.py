"""CPU fixtures exercise measured values, checkpoint gating and uncertain acks."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

from direct_s2st.baseline_wandb import (FILES, atomic, binding, canonical, collect,
                                       digest, plan, run_id)
from direct_s2st.baseline_wandb_delivery import synchronize
from direct_s2st.baseline_wandb_evidence import EXPECTED_SIGNATURE, NORMALIZATION

KEY = 'alignment-transfer-s2ut-data/experiment-001/baseline'
FINGERPRINT = '0'*64
AUX_LEGACY = 'eval/aux_bleu_legacy_unclassified'


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
    evidence = {name: 'b'*64 for name in (
        'checkpoints/checkpoint_last.pt', 'research-metadata.json', 's2ut-recipe-lock.json')}
    files.update(evidence)
    atomic(root/'snapshot.json', dict(updates=durable, files=files,
           s2ut_lineage=dict(evidence, training_fingerprint=FINGERPRINT, updates=durable),
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
    kwargs.setdefault('baseline_run_key', KEY)
    return synchronize(root, state, 'owner', remote_factory=server.factory, **kwargs)


def test_measured_zero_missing_asr_and_coalesced_json_csv(tmp_path):
    root = source(tmp_path)
    bundle = collect(root)
    assert list(bundle['rows']) == ['100']
    row = bundle['rows']['100']
    assert row[AUX_LEGACY] == 0 and row['eval/aux_failures_legacy_unclassified'] == 500
    assert 'eval/aux_bleu' not in row and 'eval/aux_failures' not in row
    assert not any(key.startswith('eval/asr_') for key in row)
    assert row['dev/main_loss'] == 2.5
    assert row['baseline/train/unit_loss_bits'] == 3
    assert row['train/main_loss'] == 3 * math.log(2)
    assert 'train/total_loss' not in row
    assert bundle['summary']['train/max_update'] == 100
    assert bundle['summary']['train/target_update'] == 400000
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


def test_delayed_history_after_sdk_finish_never_resends_rows(tmp_path, monkeypatch):
    from direct_s2st import baseline_wandb_delivery as delivery

    class DelayedHistory(Server):
        visible = False
        def history(self):
            return super().history() if self.visible else []

    monkeypatch.setattr(delivery.time, 'sleep', lambda seconds: None)
    root, server, state = source(tmp_path), DelayedHistory(), tmp_path/'state'
    result = sync(root, state, server)
    assert (result['status'], result['sent_rows'], result['pending_rows']) == ('ACK_PENDING', 1, 1)
    assert json.loads((state/'outbox.json').read_bytes())['awaiting_ack'] is True
    assert not (state/'last-success.json').exists()
    assert sync(root, state, server)['status'] == 'ACK_PENDING'
    assert server.calls == 1
    server.visible = True
    assert sync(root, state, server)['status'] == 'SYNCED'
    assert server.calls == 1
    assert json.loads((state/'outbox.json').read_bytes())['events'] == []


def test_existing_outbox_confirms_on_read_without_reopening_run(tmp_path, monkeypatch):
    from direct_s2st import baseline_wandb_delivery as delivery

    class DelayedHistory(Server):
        existing = False
        scans = 0
        def history(self):
            self.scans += 1
            return super().history() if self.scans >= 3 else []

    monkeypatch.setattr(delivery.time, 'sleep', lambda seconds: None)
    root, server, state = source(tmp_path), DelayedHistory(), tmp_path/'state'
    server.fail = 'after'  # Old outbox has no persisted SDK-finish marker.
    with pytest.raises(ConnectionError):
        sync(root, state, server)
    server.existing, server.fail = True, None
    assert sync(root, state, server)['status'] == 'SYNCED'
    assert server.calls == 1


def test_snapshot_discovery_reads_only_newest_published_update(tmp_path, monkeypatch):
    root = source(tmp_path)
    receipt = json.loads((root/'snapshot.json').read_bytes())
    (root/'snapshot.json').unlink()
    for step in (10, 20, 30, 100):
        atomic(root/f'{step:012d}-published'/'snapshot.json', dict(receipt, updates=step))
    (root/'000000000200-pending').mkdir()
    original = Path.read_text
    opened = []

    def read_text(path, *args, **kwargs):
        opened.append(path.parent.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    assert collect(root)['durable_update'] == 100
    assert opened == ['000000000200-pending', '000000000100-published']


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
    assert server.rows[-1]['eval/asr_failures_unmatched'] == 0
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
    receipt['s2ut_lineage']['training_fingerprint'] = '1'*64
    atomic(root/'snapshot.json', receipt)
    with pytest.raises(ValueError, match='another experiment'):
        sync(root, state, server)


def test_remote_conflicting_or_foreign_history_is_never_overwritten(tmp_path):
    b = collect(source(tmp_path))
    with pytest.raises(ValueError, match='foreign'):
        plan(b, [dict(update=100, _step=0, loss=3.)])
    server = Server()
    sync(tmp_path/'source', tmp_path/'state', server)
    server.rows[0][AUX_LEGACY] = 7
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


def test_cpu_reporter_recovers_from_partial_install_and_reuses_verified_environment(tmp_path, monkeypatch, capsys):
    import runpy
    from types import SimpleNamespace
    repo = Path(__file__).resolve().parents[2]
    runtime = runpy.run_path(str(repo/'scripts/colab/wandb_runtime.py'))
    monkeypatch.setenv('WANDB_API_KEY', 'test-key')
    monkeypatch.setenv('PYTHONPATH', 'unrelated-training-code')
    monkeypatch.setenv('PIP_EXTRA_INDEX_URL', 'https://example.invalid')
    calls = []
    failed_once = False

    def subprocess_run(command, *, env, **kwargs):
        nonlocal failed_once
        assert 'WANDB_API_KEY' not in env and 'PYTHONPATH' not in env
        assert 'PIP_EXTRA_INDEX_URL' not in env
        calls.append(command)
        if 'venv' in command:
            assert '--without-pip' in command
        if 'install' in command:
            assert '--python' in command
            if not failed_once:
                failed_once = True
                return SimpleNamespace(returncode=1, stdout='', stderr='No matching distribution at https://secret@example.invalid/')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime['subprocess'], 'run', subprocess_run)
    prepare = runtime['prepare']
    with pytest.raises(RuntimeError, match='Install pinned W&B wheels failed') as failure:
        prepare(repo, root=tmp_path, required=True)
    assert 'No matching distribution' in str(failure.value)
    assert 'secret@example' not in str(failure.value)
    assert not list(tmp_path.rglob('.ready'))

    result = prepare(repo, root=tmp_path, required=True)
    assert result['key'] == 'test-key'
    assert len(list(tmp_path.rglob('.ready'))) == 1
    assert len([call for call in calls if 'install' in call]) == 2
    assert len({call[-1] for call in calls if 'venv' in call}) == 2
    assert prepare(repo, root=tmp_path, required=True) == result
    assert len([call for call in calls if 'install' in call]) == 2
    assert 'test-key' not in capsys.readouterr().err


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
    outbox['events'][0]['event']['values'][AUX_LEGACY] = 999
    atomic(state/'outbox.json', outbox)
    server.fail = None
    with pytest.raises(ValueError, match='invalid or stale'):
        sync(root, state, server)


def test_adapter_rejects_wrong_project_foreign_run_and_auth_error(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from direct_s2st.baseline_wandb import API_BASE, PROJECT
    from direct_s2st.baseline_wandb_delivery import WandbRemote
    contract = binding(collect(source(tmp_path)), 'owner', key=KEY)
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
            return NS(config=self.config, scan_history=scan, summary={}, state='finished')
    monkeypatch.setitem(sys.modules, 'wandb', NS(Api=Api, errors=NS(CommError=CommError)))
    remote = WandbRemote(contract, tmp_path/'logs')
    assert remote.history() == [] and Api.flushed == 1
    Api.error = CommError('authentication failed')
    with pytest.raises(CommError):
        remote.history()
    Api.error = CommError('Could not find run missing')
    with pytest.raises(ValueError, match='was not found'):
        remote.history()
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
    r.contract, r.logs = binding(b, 'owner', key=KEY), tmp_path
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


def test_pr8_run_id_vectors_and_binding_ignores_group_and_launcher(tmp_path):
    from direct_s2st.baseline_wandb_evidence import baseline_wandb_run_id, publication_key
    assert baseline_wandb_run_id(KEY, FINGERPRINT) == '047024bc03277652fa2aa9fe'
    assert baseline_wandb_run_id('benchmark-data/実験/baseline', FINGERPRINT) == '22eba1499911c50df601ae81'
    assert publication_key('/content/drive/MyDrive/'+KEY) == KEY
    for key in ('/absolute', '../baseline', 'runs//baseline', 'runs/./baseline', 'runs\\baseline', 'runs/'):
        with pytest.raises(ValueError, match='publication key'):
            baseline_wandb_run_id(key, FINGERPRINT)
    with pytest.raises(ValueError, match='training fingerprint'):
        baseline_wandb_run_id(KEY, 'F'*64)
    root, server, state = source(tmp_path), Server(), tmp_path/'state'
    first = sync(root, state, server, group='experiment-001/transfer-fixed-v1')
    assert first['run_id'] == '047024bc03277652fa2aa9fe'
    assert server.group == 'experiment-001/transfer-fixed-v1'
    marker = json.loads((root/'snapshot.json').read_bytes())
    marker['identity']['config']['command'] = ['new-launcher']
    atomic(root/'snapshot.json', marker)
    second = sync(root, state, server, group='experiment-001/another-group')
    assert second['run_id'] == first['run_id'] and len(server.rows) == 1
    assert server.group == 'experiment-001/another-group'
    # A new runtime or relocated mirror reuses the verified, frozen original key.
    assert sync(root, state, server, baseline_run_key='')['run_id'] == first['run_id']


def _matched_bleu(update):
    ids = sorted((f'pair-{i}' for i in range(500)), key=lambda key: hashlib.sha256(f'1:{key}'.encode()).hexdigest())
    def scoring(bleu, samples, failures):
        return dict(bleu=bleu, samples=samples, failures=failures,
                    signature=EXPECTED_SIGNATURE, normalization=NORMALIZATION)
    return dict(update=update, status='completed_with_failures', aux_bleu=12., aux_samples=500,
                aux_failures=1, aux_scoring=scoring(12., 500, 1),
                asr_bleu=8., asr_samples=100, asr_failures=0, asr_scoring=scoring(8., 100, 0),
                identity=dict(update=update, weights='a'*64, ids=ids, references=dict.fromkeys(ids, 'ref'),
                    subset='fixed_hash_sample_of_dev', generation='source_audio_only',
                    auxiliary_decoding='greedy encoder-attached target-letter decoder',
                    unit_vocoder='official LJSpeech (not Fisher-trained)',
                    asr='benchmark Whisper (not original-paper wav2vec2)',
                    policy=dict(seed=1, aux_samples=500, asr_samples=100, max_aux_tokens=256,
                                beam=10, max_len_a=1, max_len_b=200, asr_model='openai/whisper-large-v3-turbo',
                                asr_revision='60be3615a4d667e1258e8ad29130467587c489aa')))


@pytest.mark.parametrize('failure, corrupt, shared', [
    ('ValueError: auxiliary decoding reached max_aux_tokens without EOS', False, True),
    ('TimeoutError: evaluation time budget reached', False, False),
    ('ValueError: auxiliary decoding reached max_aux_tokens without EOS', True, False),
])
def test_pr8_aux_journal_classifies_each_point_and_keeps_clean_asr(tmp_path, failure, corrupt, shared):
    root = source(tmp_path)
    report = _matched_bleu(100)
    atomic(root/'training-bleu.json', dict(results=[report]))
    (root/'training-bleu.csv').write_text(
        'update,aux_bleu,aux_samples,aux_failures,asr_bleu,asr_samples,asr_failures\n100,12,500,1,8,100,0\n',
        encoding='utf-8')
    seal(root)
    marker = root/'snapshot.json'
    receipt = json.loads(marker.read_bytes())
    stem = 'training-bleu/000000100/aux'
    rows = [dict(pair_id=key, status='success', hypothesis='text', error=None)
            for key in report['identity']['ids']]
    rows[0].update(status='failed', hypothesis='', error=failure)
    for name, data in (
        (stem+'.lock.json', canonical(report['identity'])),
        (stem+'.jsonl', '\n'.join(map(canonical, rows))),
    ):
        path = root/'files'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode())
        receipt['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if corrupt and name.endswith('.jsonl'):
            path.write_bytes(b'corrupt')
    atomic(marker, receipt)
    unaudited = collect(root)['rows']['100']
    assert unaudited[AUX_LEGACY] == 12
    assert 'eval/aux_bleu' not in unaudited and 'eval/aux_failures' not in unaudited
    audited = collect(root, audit_journals=True)['rows']['100']
    assert ('eval/aux_bleu' in audited) == shared
    assert ('eval/aux_failures' in audited) == shared
    assert audited['eval/asr_bleu'] == 8 and audited['eval/asr_failures'] == 0


def test_pr8_existing_run_is_verified_and_resumed_must(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from direct_s2st.baseline_wandb_delivery import WandbRemote
    root, state = source(tmp_path), tmp_path/'state'
    server = Server()
    server.rows = [{'update': 100, '_step': 0, 'dev/main_loss': 2.5}]
    result = sync(root, state, server, existing_run_id='existing-id')
    assert result['run_id'] == 'existing-id'
    assert 'dev/main_loss' not in server.rows[1]
    assert sync(root, state, server)['sent_rows'] == 0

    contract = server.contract
    candidate = dict(condition='baseline', baseline_run_key=KEY, training_fingerprint=FINGERPRINT)
    requested = []
    class FakeRun:
        config, step, summary = candidate, 1, {}
        group = 'experiment-001/transfer-fixed-v1'
        def define_metric(self, *args, **kwargs):
            pass
        def finish(self):
            pass
    remote = WandbRemote.__new__(WandbRemote)
    remote.contract, remote.logs, remote.group = contract, tmp_path, 'experiment-001/transfer-fixed-v1'
    remote.progress, remote.existing, remote.path = 0, False, 'owner/project/existing-id'
    class Missing(Exception):
        pass
    snapshot = NS(config=candidate, state='finished', summary={'train/max_update': 200},
                  scan_history=lambda **kwargs: server.rows)
    remote.api = NS(flush=lambda: None, run=lambda path: snapshot)
    remote.wandb = NS(init=lambda **kwargs: requested.append(kwargs) or FakeRun(),
                      Settings=lambda **kwargs: kwargs, errors=NS(CommError=Missing))
    assert remote.history()
    remote.send([], collect(root))
    assert requested[0]['resume'] == 'must'
    assert requested[0]['id'] == 'existing-id'
    assert requested[0]['group'] == 'experiment-001/transfer-fixed-v1'
    assert FakeRun.summary['train/max_update'] == 200
    assert FakeRun.summary['train/target_update'] == 400000
    FakeRun.group = 'another-group'
    with pytest.raises(ValueError, match='did not apply'):
        remote.send([], collect(root))
    FakeRun.group = remote.group
    legacy_bundle = collect(root)
    legacy_identity = legacy_bundle['identity']
    remote.source_identity_sha256 = digest(legacy_identity)
    FakeRun.config = snapshot.config = dict(condition='baseline', source_identity=legacy_identity,
        baseline_contract=dict(version=1, condition='baseline', entity='owner',
            project=contract['project'], source_identity_sha256=digest(legacy_identity)))
    remote.progress = 0
    snapshot.summary = {'train/max_update': 400000, 'train/durable_update': 100}
    remote.history()
    remote.send([], legacy_bundle)
    assert FakeRun.summary['train/max_update'] == 100
    snapshot.config = dict(candidate, training_fingerprint='b'*64)
    with pytest.raises(ValueError, match='identity/destination'):
        remote.history()
    snapshot.config, snapshot.state = candidate, 'running'
    with pytest.raises(ValueError, match='active'):
        remote.history()
    remote.api.run = lambda path: (_ for _ in ()).throw(Missing('Could not find run missing'))
    with pytest.raises(ValueError, match='was not found'):
        remote.history()


def test_pr8_registration_binds_measured_snapshot_to_verified_fingerprint(tmp_path):
    from zipfile import ZipFile
    root = source(tmp_path)
    marker = root/'snapshot.json'
    receipt = json.loads(marker.read_bytes())
    receipt.pop('s2ut_lineage')
    atomic(marker, receipt)
    with pytest.raises(ValueError, match='verified training fingerprint'):
        binding(collect(root), 'owner', key=KEY)
    record = dict(format='native-fairseq-s2ut-comparison-v1', method='baseline',
                  checkpoint='/original/baseline/checkpoints/checkpoint_last.pt',
                  checkpoint_hash='b'*64, checkpoint_updates=100, baseline_fingerprint=FINGERPRINT,
                  metadata=dict(training_fingerprint=FINGERPRINT),
                  provenance=dict(recipe_lock_hash='b'*64, metadata_hash='b'*64))
    record['record_hash'] = hashlib.sha256(canonical(record).encode()).hexdigest()
    registration = tmp_path/'registration'
    registration.mkdir()
    archive, published = registration/'snapshot.zip', registration/'snapshot.json'
    body = canonical(record).encode()
    with ZipFile(archive, 'w') as package:
        package.writestr('baseline.json', body)
    atomic(published, dict(storage='zip-v1', zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                           files={'baseline.json': hashlib.sha256(body).hexdigest()}))
    assert collect(root, registration=published)['training_fingerprint'] == FINGERPRINT
    receipt['files']['research-metadata.json'] = 'c'*64
    atomic(marker, receipt)
    with pytest.raises(ValueError, match='registration does not bind'):
        collect(root, registration=published)
