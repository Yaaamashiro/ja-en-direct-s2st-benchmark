import builtins
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from direct_s2st.diagnostics import text_pair_audit as audit
from direct_s2st.diagnostics.text_pair_audit_ui import AuditUI

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values), encoding='utf-8')


def context(tmp_path, n=6):
    experiment = tmp_path/'training/experiment-001'
    common = experiment/'data/common'
    lock = dict(schema_version=1, source_manifest=str(tmp_path/'corpus/accepted.jsonl'),
                source_manifest_sha256='c'*64, splits={})
    all_rows = []
    for population in audit.POPULATIONS:
        values = []
        for corpus in audit.CORPORA:
            for i in range(n):
                # Intentionally misleading IDs: metadata is the authority.
                identifier = ('jesc' if corpus == 'kftt' else 'kftt')+'_'+population+'_'+str(i)
                values.append(dict(pair_id=identifier, split=population, corpus=corpus, source_id='source:'+identifier,
                    ja_text='彼は京都へ行った。 '+str(i), en_text='He went to Kyoto. '+str(i),
                    ja_text_raw='raw 日本語 '+str(i), en_text_raw='raw English '+str(i),
                    ja_tts_text='彼は京都へ行った。 '+str(i), en_tts_text='He went to Kyoto. '+str(i),
                    # These must never be opened, resolved or imported by 8D.
                    ja_audio='/nonexistent/audio.wav', en_audio='/nonexistent/english.wav',
                    sentence_bleu=0.1, asr_ja_text='hidden automatic hypothesis'))
        lines(common/(population+'.jsonl'), values)
        lock['splits'][population] = {'pairs': len(values)}
        all_rows.extend(values)
    write(common/'dataset-lock.json', lock)
    return experiment, all_rows


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file() and 'diagnostics' not in p.parts}


def finish(store, label='correct'):
    for row in store.queue:
        if row['audit_id'] not in store.annotations:
            store.save(row['audit_id'], label)


def test_only_actual_experiment_rows_and_explicit_corpus_not_prefix(tmp_path):
    experiment, values = context(tmp_path)
    store = audit.prepare(experiment, per_stratum=3)
    assert {r['pair_id'] for r in store.queue} <= {r['pair_id'] for r in values}
    assert all(not r['pair_id'].startswith(r['corpus']) for r in store.queue)
    assert len(store.queue) == 12
    assert all('sentence_bleu' not in r and 'asr_ja_text' not in r for r in store.queue)
    assert audit.read(store.root/'sampling-audit.json')['metadata_source'].startswith('explicit common')


def test_deterministic_sampling_blind_order_and_input_order_independence(tmp_path):
    experiment, _ = context(tmp_path, n=11)
    rows, _, _ = audit.collect(experiment)
    left, strata = audit.select(rows, per_stratum=4)
    right, _ = audit.select(list(reversed(rows)), per_stratum=4)
    assert left == right and len(left) == 16
    assert [audit.priority(audit.AUDIT_VERSION, 'blind', audit.AUDIT_SEED, r['pair_id']) for r in left] == sorted(
        audit.priority(audit.AUDIT_VERSION, 'blind', audit.AUDIT_SEED, r['pair_id']) for r in left)
    for group in strata:
        expected = sorted([r for r in rows if r['population'] == group['population'] and r['corpus'] == group['corpus']],
            key=lambda r: audit.priority(audit.AUDIT_VERSION, audit.AUDIT_SEED, r['population'], r['corpus'], r['pair_id']))[:4]
        assert {r['pair_id'] for r in expected} <= {r['pair_id'] for r in left}


def test_small_strata_and_available_counts(tmp_path):
    experiment, _ = context(tmp_path, n=2)
    store = audit.prepare(experiment)
    assert len(store.queue) == 8
    assert all((r['requested'], r['available'], r['selected']) == (100, 2, 2)
               for r in audit.read(store.root/'sampling-audit.json')['strata'])


@pytest.mark.parametrize('damage', ['missing_corpus', 'unknown_corpus'])
def test_missing_provenance_is_reported_and_all_unproven_stops(tmp_path, damage):
    experiment, values = context(tmp_path)
    for population in audit.POPULATIONS:
        for r in values:
            r['corpus'] = None if damage == 'missing_corpus' else 'unsupported'
        lines(experiment/'data/common'/(population+'.jsonl'), [r for r in values if r['split'] == population])
    with pytest.raises(audit.ProvenanceError, match='provenance'):
        audit.prepare(experiment)
    failures = list((experiment/'diagnostics/text-pair-audit/provenance-failures').glob('*.json'))
    assert len(failures) == 1
    report = audit.read(failures[0])
    assert len(report['provenance_audit']['excluded']) == len(values)
    assert not list(experiment.rglob('annotations.jsonl'))


def test_individual_missing_provenance_and_text_not_silently_dropped(tmp_path):
    experiment, values = context(tmp_path)
    train = [r for r in values if r['split'] == 'train']
    train[0]['corpus'] = None
    train[1]['ja_text'] = ''
    lines(experiment/'data/common/train.jsonl', train)
    store = audit.prepare(experiment)
    excluded = audit.read(store.root/'sampling-audit.json')['excluded']
    assert len(excluded) == 2 and {r['reason'] for r in excluded} == {
        'missing_canonical_text', 'missing_explicit_corpus_and_locked_accepted_manifest_unavailable'}
    assert not {r['pair_id'] for r in excluded} & {r['pair_id'] for r in store.queue}


@pytest.mark.parametrize('damage', [None, 'hash', 'text', 'duplicate', 'split', 'source_id'])
def test_strict_accepted_metadata_join_never_samples_raw_extra_rows(tmp_path, damage):
    experiment, values = context(tmp_path)
    source = tmp_path/'corpus/accepted.jsonl'
    raw = json.loads(json.dumps(values))
    if damage == 'text': raw[0]['en_text'] = 'Unrelated content.'
    if damage == 'duplicate': raw.append(raw[0])
    if damage == 'split': raw[0]['split'] = 'dev'
    if damage == 'source_id': raw[0]['source_id'] = 'different'
    raw.append(dict(pair_id='only-in-raw', corpus='kftt', split='train', ja_text='原文', en_text='Raw only'))
    lines(source, raw)
    lock_path = experiment/'data/common/dataset-lock.json'
    lock = audit.read(lock_path); lock['source_manifest_sha256'] = 'f'*64 if damage == 'hash' else audit.checksum(source)
    write(lock_path, lock)
    values[0]['corpus'] = None
    lines(experiment/'data/common/train.jsonl', [r for r in values if r['split'] == 'train'])
    if damage == 'hash':
        with pytest.raises(ValueError, match='SHA mismatch'): audit.prepare(experiment, corpus_root=source.parent)
        return
    store = audit.prepare(experiment, corpus_root=source.parent)
    assert 'only-in-raw' not in store.by_id
    assert not any(r['pair_id'] == 'only-in-raw' for r in store.queue)
    assert any(r['pair_id'] == values[0]['pair_id'] for r in store.queue) == (damage is None)
    assert len(audit.read(store.root/'sampling-audit.json')['excluded']) == (0 if damage is None else 1)


@pytest.mark.parametrize('damage', ['count', 'duplicate', 'split', 'prepared_lock'])
def test_population_and_usage_identity_validation(tmp_path, damage):
    experiment, values = context(tmp_path)
    path = experiment/'data/common/dataset-lock.json'
    if damage == 'count':
        lock = audit.read(path); lock['splits']['train']['pairs'] = 1; write(path, lock)
    elif damage == 'prepared_lock':
        write(experiment/'data/s2ut/fairseq/data-lock.json', {'common_dataset_lock_sha256': '0'*64})
    else:
        train = [r for r in values if r['split'] == 'train']
        if damage == 'duplicate': train[1]['pair_id'] = train[0]['pair_id']
        else: train[0]['split'] = 'test'
        lines(experiment/'data/common/train.jsonl', train)
    with pytest.raises(ValueError): audit.prepare(experiment)


def test_every_answer_is_durable_blind_and_resume_preserves_exact_bytes(tmp_path):
    experiment, _ = context(tmp_path)
    store = audit.prepare(experiment)
    store.save(store.queue[0]['audit_id'], 'acceptable', flags=['fragment'], note='human only')
    saved = (store.root/'annotations.jsonl').read_bytes()
    record = list(audit.rows(store.root/'annotations.jsonl'))[0]
    assert set(record) == {'audit_id', 'label', 'flags', 'confidence', 'note', 'timestamp'}
    resumed = audit.prepare(experiment)
    assert resumed.root == store.root and resumed.annotations == store.annotations
    assert resumed.first_pending() == 1
    assert (resumed.root/'annotations.jsonl').read_bytes() == saved
    with pytest.raises(ValueError, match='Existing annotation preserved'):
        resumed.save(store.queue[0]['audit_id'], 'wrong_alignment')
    assert (resumed.root/'annotations.jsonl').read_bytes() == saved
    resumed.save(store.queue[0]['audit_id'], 'partial_mismatch', allow_update=True)
    assert audit.AuditStore(store.root).annotations[record['audit_id']]['label'] == 'partial_mismatch'


@pytest.mark.parametrize('change', ['seed', 'limit', 'text', 'raw', 'extra_manifest_field', 'version', 'experiment'])
def test_identity_changes_do_not_reuse_annotations(tmp_path, monkeypatch, change):
    experiment, values = context(tmp_path)
    original = audit.prepare(experiment, per_stratum=2)
    original.save(original.queue[0]['audit_id'], 'correct')
    old_bytes = (original.root/'annotations.jsonl').read_bytes()
    options = dict(per_stratum=2)
    if change == 'seed': options['seed'] = audit.AUDIT_SEED+1
    elif change == 'limit': options['per_stratum'] = 3
    elif change == 'version': monkeypatch.setattr(audit, 'AUDIT_VERSION', 'new-policy')
    elif change == 'experiment': experiment, _ = context(tmp_path/'other')
    else:
        values[0][{'text': 'ja_text', 'raw': 'ja_text_raw', 'extra_manifest_field': 'extra'}[change]] = 'changed'
        lines(experiment/'data/common/train.jsonl', [r for r in values if r['split'] == 'train'])
    newer = audit.prepare(experiment, **options)
    assert newer.root != original.root and not newer.annotations
    assert (original.root/'annotations.jsonl').read_bytes() == old_bytes


@pytest.mark.parametrize('damage', ['text', 'mapping', 'identity', 'annotation_id'])
def test_tampered_queue_identity_or_annotations_rejected(tmp_path, damage):
    experiment, _ = context(tmp_path)
    store = audit.prepare(experiment)
    if damage == 'identity':
        identity = audit.read(store.root/'identity.json'); identity['sampling_seed'] = -1; write(store.root/'identity.json', identity)
    elif damage == 'annotation_id':
        store.save(store.queue[0]['audit_id'], 'correct')
        records = list(audit.rows(store.root/'annotations.jsonl')); records[0]['audit_id'] = 'foreign'; lines(store.root/'annotations.jsonl', records)
    else:
        queue = store.queue
        queue[0]['ja_text' if damage == 'text' else 'corpus'] = 'changed'
        lines(store.root/'audit_queue.jsonl', queue)
    with pytest.raises(ValueError): audit.AuditStore(store.root)


def test_no_unblinded_report_or_export_until_every_answer_exists(tmp_path):
    experiment, _ = context(tmp_path, n=2)
    before = snapshot(experiment)
    store = audit.prepare(experiment)
    for row in store.queue[:-1]: store.save(row['audit_id'], 'correct')
    state = audit.read(store.root/'summary.json')
    assert set(state) == {'complete', 'annotated', 'remaining', 'total'}
    assert (store.root/'report.md').read_text().startswith('Annotated: 7 / 8')
    assert not (store.root/'per_sample.csv').exists() and not (store.root/'per_stratum.csv').exists()
    store.save(store.queue[-1]['audit_id'], 'uncertain')
    state = audit.read(store.root/'summary.json')
    assert state['complete'] and len(state['strata']) == 4
    with (store.root/'per_sample.csv').open(encoding='utf-8', newline='') as f: exported = list(csv.DictReader(f))
    assert len(exported) == len(store.queue)
    for left, right in zip(exported, store.queue):
        assert all(left[k] == right[k] for k in ('audit_id', 'pair_id', 'corpus', 'population', 'original_split', *audit.TEXT_FIELDS))
    assert snapshot(experiment) == before  # no official manifests or outputs touched


def test_uncertain_denominators_rates_and_wilson(tmp_path):
    experiment, _ = context(tmp_path, n=5)
    store = audit.prepare(experiment)
    for population in audit.POPULATIONS:
        for corpus in audit.CORPORA:
            group = [r for r in store.queue if r['population'] == population and r['corpus'] == corpus]
            for row, label in zip(group, audit.LABELS):
                store.save(row['audit_id'], label, flags=['boundary_mismatch'] if label == 'partial_mismatch' else [])
    summary = audit.read(store.root/'summary.json')
    for group in summary['strata']:
        assert group['N'] == 5 and group['decided'] == 4
        assert group['rates']['strict_correct_rate']['estimate'] == .25
        assert group['rates']['usable_rate']['estimate'] == .5
        assert group['rates']['serious_error_rate']['estimate'] == .5
        assert group['rates']['uncertain_rate']['estimate'] == .2
    assert summary['jesc_minus_kftt_serious_error_pp'] == {'train': 0, 'test': 0}
    assert summary['flags_by_corpus']['kftt']['boundary_mismatch'] == 2
    ci = audit.wilson(50, 100)
    assert ci['lower'] == pytest.approx(.4038315303659956)
    assert ci['upper'] == pytest.approx(.5961684696340044)
    assert audit.wilson(0, 0) == dict(estimate=None, lower=None, upper=None)
    for r in store.annotations.values():
        if r['label'] == 'uncertain': assert r['confidence'] == 'low'


def test_all_uncertain_produces_na_not_zero_quality(tmp_path):
    experiment, _ = context(tmp_path, n=1)
    store = audit.prepare(experiment)
    finish(store, 'uncertain')
    assert all(r['rates']['usable_rate']['estimate'] is None for r in audit.read(store.root/'summary.json')['strata'])


def test_atomic_failed_write_retains_old_answers_and_can_resume(tmp_path, monkeypatch):
    from direct_s2st import io as atomic_io
    experiment, _ = context(tmp_path)
    store = audit.prepare(experiment)
    store.save(store.queue[0]['audit_id'], 'correct')
    prior = (store.root/'annotations.jsonl').read_bytes()
    replace = atomic_io.os.replace
    def fail(src, dest):
        if Path(dest).name == 'annotations.jsonl': raise OSError('simulated disconnected Drive')
        return replace(src, dest)
    with monkeypatch.context() as m:
        m.setattr(atomic_io.os, 'replace', fail)
        with pytest.raises(OSError): store.save(store.queue[1]['audit_id'], 'acceptable')
    assert (store.root/'annotations.jsonl').read_bytes() == prior
    resumed = audit.prepare(experiment)
    assert resumed.first_pending() == 1
    resumed.save(resumed.queue[1]['audit_id'], 'acceptable')
    assert len(audit.AuditStore(store.root).annotations) == 2


def test_real_widgets_blinding_click_save_previous_explicit_edit_and_resume(tmp_path):
    pytest.importorskip('ipywidgets')
    experiment, _ = context(tmp_path)
    store = audit.prepare(experiment)
    ui = AuditUI(store)
    def visible(widget):
        result = []
        for name in ('value', 'description', 'tooltip'):
            v = getattr(widget, name, None)
            if isinstance(v, str): result.append(v)
        for child in getattr(widget, 'children', ()): result.extend(visible(child))
        return result
    text = '\n'.join(visible(ui.widget))
    for forbidden in ('kftt', 'jesc', 'train', 'test', 'pair_id', 'sentence_bleu', 'hidden automatic hypothesis'):
        assert not __import__('re').search(r'\b'+forbidden+r'\b', text, flags=__import__('re').I)
    assert store.queue[0]['pair_id'] not in text and store.queue[0]['audit_id'] not in text
    ui.next.click()  # no selected primary label: don't save or advance
    assert ui.position == 0 and not store.annotations
    ui.flags['missing_content'].value = True
    ui.buttons['partial_mismatch'].click()
    assert ui.position == 1 and len(audit.AuditStore(store.root).annotations) == 1
    assert ui.flags['missing_content'].value is False
    ui.previous.click()
    ui.buttons['acceptable'].click()
    assert store.annotations[store.queue[0]['audit_id']]['label'] == 'acceptable'
    resumed = AuditUI(audit.prepare(experiment))
    assert resumed.position == 1
    ui.close(); ui.buttons['correct'].click()
    assert len(store.annotations) == 1


def test_core_runs_with_stdlib_only_and_forbids_model_imports(tmp_path):
    experiment, _ = context(tmp_path)
    code = '''import sys,builtins
sys.path.insert(0,sys.argv[1])
original=builtins.__import__
def checked(name,*a,**k):
    assert name.split('.')[0] not in {'torch','torchaudio','transformers','qwen_tts','numpy','soundfile','librosa','huggingface_hub','fairseq','sacrebleu'}, name
    return original(name,*a,**k)
builtins.__import__=checked
from direct_s2st.diagnostics.text_pair_audit import prepare
s=prepare(sys.argv[2],per_stratum=2)
assert len(s.queue)==8
'''
    subprocess.run([sys.executable, '-S', '-B', '-c', code, str(ROOT/'src'), str(experiment)], check=True, capture_output=True)


def test_source_outputs_untouched_and_output_cannot_be_in_corpus(tmp_path):
    experiment, _ = context(tmp_path)
    before = snapshot(experiment)
    with pytest.raises(ValueError, match='CORPUS_ROOT'): audit.prepare(experiment, corpus_root=tmp_path)
    assert snapshot(experiment) == before and not (experiment/'diagnostics').exists()


def test_launcher_missing_widgets_fails_before_any_output(tmp_path, monkeypatch):
    launcher = runpy.run_path(str(ROOT/'scripts/colab/text_pair_audit.py'))
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'ipywidgets': raise ImportError('fixture missing widget dependency')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(RuntimeError, match='standard Colab'):
        launcher['launch'](tmp_path/'experiment')
    assert not list(tmp_path.iterdir())


def test_cell1_then_normal_8d_without_cell2_gpu_hf_or_other_diagnostics(tmp_path, monkeypatch):
    pytest.importorskip('ipywidgets')
    experiment, _ = context(tmp_path)
    nb = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    config = next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('#@title 1.'))
    code = next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('#@title 8D.'))
    # No REPO/PERSISTENT/DATA pre-created by Cell 2. Only an existing code checkout.
    code = code.replace("'/content/ja-en-direct-s2st-benchmark'", repr(str(ROOT)))
    code = code.replace("'/content/drive/MyDrive'", repr(str(tmp_path)))
    namespace = {}
    exec(config, namespace)
    namespace.update(OUTPUT_BASE=str(tmp_path), EXPERIMENT='experiment-001')
    assert not any(k in namespace for k in ('REPO', 'DATA', 'PERSISTENT'))
    original = subprocess.run
    calls = []
    def git_only(command, **kwargs):
        calls.append(command)
        assert command[0] == 'git' and not any(k in command for k in ('fetch', 'checkout', 'reset'))
        return SimpleNamespace(returncode=0, stdout='' if 'status' in command else 'a'*40, stderr='')
    monkeypatch.setattr(subprocess, 'run', git_only)
    exec(code, namespace)
    ui = namespace['text_pair_audit_ui']
    assert len(ui.store.queue) == 24 and ui.position == 0
    ui.buttons['correct'].click()
    exec(code, namespace)
    assert namespace['text_pair_audit_ui'].position == 1 and ui.closed


def test_existing_notebook_cells_unchanged_and_8d_follows_8c():
    current = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    prior = json.loads(subprocess.check_output(['git', '-c', 'safe.directory='+str(ROOT), '-C', str(ROOT),
        'show', '837f379f2a6f37e4c28f64f685d0a677090de9b4:notebooks/colab_training.ipynb'], encoding='utf-8'))
    new = [c for c in current['cells'] if ''.join(c['source']).startswith('#@title 8D.')]
    assert len(new) == 1
    assert [c for c in current['cells'] if c not in new and not ''.join(c['source']).startswith(('#@title 8E.','#@title 8F.'))] == prior['cells']
    position = current['cells'].index(new[0])
    assert ''.join(current['cells'][position-1]['source']).startswith('#@title 8C.')
    assert ''.join(current['cells'][position+1]['source']).startswith(('#@title 8E.', '#@title 9.'))


def test_normal_8d_fetches_code_from_blank_runtime_without_touching_training_or_drive(tmp_path, monkeypatch):
    pytest.importorskip('ipywidgets')
    experiment, _ = context(tmp_path)
    source = tmp_path/'source'; source.mkdir()
    actual = subprocess.run
    def git(*args):
        return actual(['git', '-c', 'safe.directory='+str(source), '-c', 'user.name=Test',
            '-c', 'user.email=test@example.invalid', '-C', str(source), *args],
            check=True, capture_output=True, text=True, encoding='utf-8').stdout.strip()
    git('init')
    for relative in ('scripts/colab/text_pair_audit.py', 'src/direct_s2st/__init__.py',
                     'src/direct_s2st/progress.py', 'src/direct_s2st/io.py',
                     'src/direct_s2st/diagnostics/__init__.py',
                     'src/direct_s2st/diagnostics/text_pair_audit.py',
                     'src/direct_s2st/diagnostics/text_pair_audit_ui.py'):
        import shutil
        (source/relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, source/relative)
    git('add', '.'); git('commit', '-m', 'test: text audit bootstrap fixture')
    git('branch', '-M', 'main')
    revision = git('rev-parse', 'HEAD')
    training = tmp_path/'old-training'; training.mkdir(); (training/'preserved').write_text('original')
    nb = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    code = next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('#@title 8D.'))
    code = code.replace("'/content/s2st-text-audit-code'", repr(str(tmp_path/'code-cache')))
    code = code.replace("'/content/drive/MyDrive'", repr(str(tmp_path)))
    calls = []
    def local_git(command, **kwargs):
        calls.append(command)
        assert command[0] == 'git'
        directory = command[command.index('-C')+1]
        command = [str(source) if arg == 'https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark.git' else arg for arg in command]
        return actual(['git', '-c', 'safe.directory='+str(directory), *command[1:]], **kwargs)
    monkeypatch.setattr(subprocess, 'run', local_git)
    before = snapshot(experiment)
    ns = dict(OUTPUT_BASE=str(tmp_path), MODE='training', EXPERIMENT='experiment-001',
              CORPUS_PATH=str(tmp_path/'corpus'), REPO=training)
    exec(code, ns)
    assert ns['_ta_code'].name == revision and len(ns['text_pair_audit_ui'].store.queue) == 24
    assert (training/'preserved').read_text() == 'original' and not (training/'.git').exists()
    assert not any('reset' in call or 'checkout' in call for call in calls)
    assert snapshot(experiment) == before
    assert not (experiment/'session.log').exists() and not (experiment/'repository-revision.txt').exists()


def test_snapshot_commit_before_report_failure_can_resume_without_losing_answer(tmp_path, monkeypatch):
    experiment, _ = context(tmp_path)
    store = audit.prepare(experiment)
    real = audit.atomic_write_json
    def fail(path, value, **kwargs):
        if Path(path).name == 'summary.json': raise OSError('simulated report failure after annotation commit')
        return real(path, value, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(audit, 'atomic_write_json', fail)
        with pytest.raises(OSError): store.save(store.queue[0]['audit_id'], 'correct')
    resumed = audit.prepare(experiment)
    assert resumed.first_pending() == 1 and len(resumed.annotations) == 1


def test_missing_saved_answers_are_not_silently_reset(tmp_path):
    experiment, _ = context(tmp_path)
    store = audit.prepare(experiment)
    store.save(store.queue[0]['audit_id'], 'correct')
    lines(store.root/'annotations.jsonl', [])
    with pytest.raises(ValueError, match='missing compared with saved progress'):
        audit.prepare(experiment)


def test_stale_explicit_edit_does_not_overwrite_another_ui(tmp_path):
    experiment, _ = context(tmp_path)
    first = audit.prepare(experiment)
    key = first.queue[0]['audit_id']
    first.save(key, 'correct')
    second = audit.prepare(experiment)
    second.save(key, 'acceptable', allow_update=True)
    with pytest.raises(ValueError, match='another UI'):
        first.save(key, 'wrong_alignment', allow_update=True)


def test_save_failure_does_not_advance_widget(tmp_path, monkeypatch):
    pytest.importorskip('ipywidgets')
    experiment, _ = context(tmp_path)
    store = audit.prepare(experiment)
    ui = AuditUI(store)
    def fail(*args, **kwargs): raise OSError('simulated Drive write failure')
    monkeypatch.setattr(store, 'save', fail)
    ui.buttons['correct'].click()
    assert ui.position == 0 and not store.annotations and not ui.busy
    assert 'simulated Drive write failure' in ui.message.value


def test_complete_resume_and_previous_edit_preserve_other_answers(tmp_path):
    pytest.importorskip('ipywidgets')
    experiment, _ = context(tmp_path, n=1)
    store = audit.prepare(experiment)
    finish(store)
    saved = dict(store.annotations)
    resumed = audit.prepare(experiment)
    ui = AuditUI(resumed)
    assert ui.position == len(store.queue)
    ui.previous.click(); ui.buttons['uncertain'].click()
    last = store.queue[-1]['audit_id']
    assert resumed.annotations[last]['label'] == 'uncertain'
    assert all(resumed.annotations[key] == value for key, value in saved.items() if key != last)


def test_os_save_lock_released_when_process_is_killed(tmp_path):
    experiment, _ = context(tmp_path)
    store = audit.prepare(experiment)
    script = '''import sys,time
sys.path.insert(0,sys.argv[1])
from direct_s2st.diagnostics.text_pair_audit import AuditStore
s=AuditStore(sys.argv[2])
with s.writing():
    print('LOCKED',flush=True)
    time.sleep(60)
'''
    child = subprocess.Popen([sys.executable, '-B', '-c', script, str(ROOT/'src'), str(store.root)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'LOCKED'
        with pytest.raises(RuntimeError, match='save lock unavailable'):
            store.save(store.queue[0]['audit_id'], 'correct')
    finally:
        child.kill(); child.communicate(timeout=10)
    store.save(store.queue[0]['audit_id'], 'correct')
    assert len(audit.AuditStore(store.root).annotations) == 1
