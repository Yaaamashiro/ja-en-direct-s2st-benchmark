import ast
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf

from direct_s2st.cascade import corpus_diagnostic as diagnostic
from direct_s2st.benchmark_storage import publish_tree
from direct_s2st.evaluation.bleu import corpus_bleu, normalize_english
from direct_s2st.evaluation.run import evaluate_predictions
from direct_s2st.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from direct_s2st.journal import digest

ROOT = Path(__file__).resolve().parents[2]


def fixture(tmp_path, monkeypatch, *, benchmark=True, shuffled=False, generation_failure=False,
            asr_failure=False, no_jesc=False):
    for key in ('CORPUS_ROOT', 'S2ST_INPUT_MAP'):
        monkeypatch.delenv(key, raising=False)
    data, output = tmp_path/'data', tmp_path/'cascade-evaluation'
    references = ['the very old book is on the table', 'short phrase', 'a different sentence is here now', 'last sample text']
    hypotheses = ['the old book on table', 'short phrase', 'totally wrong output', 'last text']
    ids = ['jesc_misleading_0', 'jesc_misleading_1', 'kftt_misleading_2', 'kftt_misleading_3']
    ja = ['Ａ　あ。', 'い', '日本', '文字']
    asr_ja = ['A あ。', 'いう', '日本', '誤り']
    manifest, predictions = [], []
    for i, key in enumerate(ids):
        wave = tmp_path/'audio'/(key+'.wav')
        wave.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wave, np.sin(np.arange(1600)/10)*.1, 16000)
        corpus = 'KFTT' if i == 0 else ('kftt' if i == 1 else ('other' if no_jesc else 'jesc'))
        manifest.append(dict(pair_id=key, split='test', corpus=corpus, ja_text=ja[i], en_text=references[i],
                             ja_audio=str(wave), en_audio=str(wave)))
        failed = generation_failure and i == 0
        predictions.append(dict(pair_id=key, system_id='cascade', run_id='saved-cascade', split='test',
                                source_audio=str(wave), reference_audio=str(wave), reference_text=references[i],
                                output_audio=None if failed else str(wave), output_duration=None if failed else .1,
                                inference_seconds=1., real_time_factor=None if failed else 10.,
                                status='failed' if failed else 'success', error='TTS failed' if failed else None,
                                asr_ja_text=asr_ja[i], mt_en_text=references[i]))
    if shuffled:
        random.Random(19).shuffle(manifest)
        random.Random(73).shuffle(predictions)
    common = data/'common/test.jsonl'
    atomic_write_jsonl(common, manifest)
    atomic_write_json(common.parent/'dataset-lock.json', dict(test='frozen'))
    predicted = output/'predictions.jsonl' if benchmark else output/'predictions/predictions.jsonl'
    atomic_write_jsonl(predicted, predictions)
    plan = dict(version=1, system_id='cascade', split='test', ids=ids, run={'name':'saved-cascade'},
                common_sha256=diagnostic.checksum(common),
                dataset_lock_sha256=diagnostic.checksum(common.parent/'dataset-lock.json'),
                bleu_denominator=diagnostic.ALL_FAILED_EMPTY)
    plan['identity'] = digest(plan)
    identity = (dict(plan=plan['identity'], predictions=diagnostic.checksum(predicted),
                     bleu_denominator=diagnostic.ALL_FAILED_EMPTY, normalization=diagnostic.NORMALIZATION,
                     asr=dict(model_id='pinned-evaluation-asr', revision='a'*40), optional_metrics=False)
                if benchmark else dict(run_id='saved-cascade', asr={'model':'pinned-evaluation-asr', 'revision':'a'*40}))
    if benchmark:
        atomic_write_json(output/'plan.json', plan)
    else:
        atomic_write_json(predicted.with_suffix('.lock.json'), dict(common=diagnostic.checksum(common)))
    def transcribe(path):
        index = ids.index(path.stem)
        if asr_failure and index == 1:
            raise ValueError('saved ASR failure')
        return hypotheses[index]
    scores = output/'scores' if benchmark else output/'metrics'
    metrics = evaluate_predictions(predicted, scores, transcribe=transcribe, evaluation_identity=identity,
                                   include_failed_in_bleu=benchmark)
    if benchmark:
        (output/'results').mkdir()
        for name in ('metrics.json', 'per_sample.jsonl'):
            shutil.copyfile(scores/name, output/'results'/name)
        publish_tree(scores, output/'progress', 'asr-'+digest(identity)[:24], identity)
    return common, output, manifest, predictions, metrics


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def test_kftt_two_jesc_two_real_cell8_schema_and_corpus_bleu_not_sentence_mean(tmp_path, monkeypatch):
    common, output, manifest, predicted, metrics = fixture(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    result = diagnostic.diagnose(common, output)
    assert result['groups']['All']['samples'] == 4
    assert result['groups']['KFTT']['samples'] == 2 and result['groups']['JESC']['samples'] == 2
    assert {r['pair_id'] for r in result['kftt_samples']} == {'jesc_misleading_0', 'jesc_misleading_1'}
    assert result['groups']['KFTT']['ja_asr_cer'] == .25  # Micro CER, not mean of per-utterance CERs (.5).
    assert result['groups']['KFTT']['mt_bleu'] == 100
    rows = result['kftt_samples']
    expected = corpus_bleu([normalize_english(r['generated_en_asr_text']) for r in rows],
                           [normalize_english(r['en_text']) for r in rows])
    assert result['groups']['KFTT']['final_asr_bleu'] == expected
    assert abs(expected-sum(r['sentence_bleu'] for r in rows)/len(rows)) > .001
    assert result['groups']['All']['final_asr_bleu'] == metrics['bleu']
    assert snapshot(tmp_path) == before  # Includes result JSON, receipts, WAVs, ZIPs and mtimes.


def test_shuffled_metadata_and_scores_have_identical_diagnostics(tmp_path, monkeypatch):
    first = fixture(tmp_path/'first', monkeypatch)
    second = fixture(tmp_path/'second', monkeypatch, shuffled=True)
    a, b = diagnostic.diagnose(*first[:2]), diagnostic.diagnose(*second[:2])
    assert a['groups'] == b['groups'] and a['kftt_samples'] == b['kftt_samples']


@pytest.mark.parametrize('benchmark', [True, False])
def test_recorded_failures_match_each_unchanged_cell8_failure_policy(tmp_path, monkeypatch, benchmark):
    common, output, _, _, metrics = fixture(tmp_path, monkeypatch, benchmark=benchmark,
                                           generation_failure=True, asr_failure=True)
    result = diagnostic.diagnose(common, output)
    kftt = result['groups']['KFTT']
    assert kftt['samples'] == 2 and kftt['failed_samples'] == 2
    assert kftt['generation_failures'] == 1 and kftt['asr_failures'] == 1
    assert kftt['mt_bleu'] == 100  # TTS/English-ASR failures must not discard completed MT text.
    assert kftt['final_bleu_samples'] == (2 if benchmark else 0)
    assert kftt['final_asr_bleu'] == (0 if benchmark else None)
    assert result['groups']['All']['final_asr_bleu'] == metrics['bleu']


def test_missing_kftt_evaluation_row_is_incomplete_not_a_fabricated_failed_sample(tmp_path, monkeypatch):
    common, output, _, _, _ = fixture(tmp_path, monkeypatch, benchmark=False)
    path = output/'metrics/per_sample.jsonl'
    atomic_write_jsonl(path, [r for r in read_jsonl(path) if r['pair_id'] != 'jesc_misleading_0'], overwrite=True)
    with pytest.raises(RuntimeError, match='incomplete pair_id join'):
        diagnostic.diagnose(common, output)


@pytest.mark.parametrize('collection', ['manifest', 'predictions', 'scores'])
def test_duplicate_pair_ids_are_rejected(collection):
    with pytest.raises(RuntimeError, match='duplicate pair_id'):
        diagnostic.index([{'pair_id':'a'}, {'pair_id':'a'}], collection)


def test_jesc_absent_is_allowed_and_does_not_change_kftt(tmp_path, monkeypatch):
    common, output, *_ = fixture(tmp_path, monkeypatch, no_jesc=True)
    result = diagnostic.diagnose(common, output)
    assert result['groups']['JESC'] is None and result['groups']['KFTT']['samples'] == 2
    assert 'JESC: not present' in result['display']


@pytest.mark.parametrize('damage', ['formal_score', 'reference', 'normalization', 'status', 'system', 'corpus', 'split', 'no_kftt'])
def test_wrong_results_or_schema_do_not_silently_score(tmp_path, monkeypatch, damage):
    common, output, *_ = fixture(tmp_path, monkeypatch, benchmark=False)
    score_path, manifest_path = output/'metrics/per_sample.jsonl', common
    if damage == 'formal_score':
        path = output/'metrics/metrics.json'
        value = diagnostic.read(path)
        value['bleu'] += 1
        atomic_write_json(path, value, overwrite=True)
    elif damage in ('corpus', 'split', 'no_kftt'):
        values = list(read_jsonl(manifest_path))
        if damage == 'no_kftt':
            for row in values: row['corpus'] = 'jesc'
        else:
            values[0][damage] = None if damage == 'corpus' else 'dev'
        atomic_write_jsonl(manifest_path, values, overwrite=True)
        atomic_write_json(output/'predictions/predictions.lock.json', dict(common=diagnostic.checksum(common)), overwrite=True)
    else:
        values = list(read_jsonl(score_path))
        field, value = {'reference': ('reference_raw', 'wrong pair'), 'normalization': ('hypothesis_normalized', 'WRONG'),
                        'status': ('generation_status', 'failed'), 'system': ('system_id', 's2ut')}[damage]
        values[0][field] = value
        atomic_write_jsonl(score_path, values, overwrite=True)
    with pytest.raises(RuntimeError):
        diagnostic.diagnose(common, output)


def test_score_receipt_corruption_stops_before_scoring(tmp_path, monkeypatch):
    common, output, *_ = fixture(tmp_path, monkeypatch)
    path = output/'results/per_sample.jsonl'
    path.write_text(path.read_text(encoding='utf-8')+'\n', encoding='utf-8')
    monkeypatch.setattr(diagnostic, 'aggregate', lambda *args: pytest.fail('must verify evidence first'))
    with pytest.raises(RuntimeError, match='score receipt/checksum'):
        diagnostic.diagnose(common, output)


def test_no_wav_or_zip_reads_or_model_imports_in_cpu_subprocess(tmp_path, monkeypatch):
    common, output, *_ = fixture(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    code = '''
import pathlib, sys, runpy
class BlockModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','fairseq','qwen_tts','huggingface_hub'}:
            raise AssertionError('diagnostic attempted model import: '+fullname)
sys.meta_path.insert(0, BlockModels())
original = pathlib.Path.open
def guarded(self, *args, **kwargs):
    mode = args[0] if args else kwargs.get('mode', 'r')
    assert not any(flag in mode for flag in 'wax+'), 'diagnostic attempted file write'
    assert self.suffix not in ('.wav','.zip','.pt','.npy'), 'diagnostic attempted audio/checkpoint/archive access'
    return original(self, *args, **kwargs)
pathlib.Path.open = guarded
sys.argv = ARGS
runpy.run_path(ARGS[0], run_name='__main__')
'''
    args = [str(ROOT/'scripts/colab/cascade_corpus_diagnostic.py'), '--persistent', str(tmp_path), '--manifest', str(common),
            '--target', 'cascade', '--evaluation-root', str(output)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    completed = subprocess.run([sys.executable, '-B', '-c', 'ARGS='+repr(args)+'\n'+code],
                               capture_output=True, encoding='utf-8', env=env, timeout=45)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)['groups']['KFTT']['samples'] == 2
    assert snapshot(tmp_path) == before


def test_existing_cells_1_to_8_and_other_cells_are_unchanged_and_8a_is_immediately_after_8():
    old = json.loads(subprocess.check_output(['git', '-c', 'safe.directory='+str(ROOT).replace('\\','/'), '-C', str(ROOT),
                                               'show', 'HEAD:notebooks/colab_training.ipynb'], encoding='utf-8'))
    new = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    added = [c for c in new['cells'] if ''.join(c['source']).startswith('#@title 8A.')]
    # Compare other cells with the current base, without requiring deep Git history
    # or permanently freezing unrelated notebook cells to this feature's commit.
    previous = [c for c in old['cells'] if not ''.join(c['source']).startswith('#@title 8A.')]
    assert len(added) == 1 and [c for c in new['cells'] if c is not added[0]] == previous
    i = new['cells'].index(added[0])
    assert ''.join(new['cells'][i-1]['source']).startswith('#@title 8. 共通評価')
    assert ''.join(new['cells'][i+1]['source']).startswith('#@title 9.')
    code = ''.join(added[0]['source'])
    ast.parse(code, feature_version=(3,10))
    assert "CUDA_VISIBLE_DEVICES=''" in code and "PYTHONDONTWRITEBYTECODE='1'" in code
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in
                   ('run','cli','ensure_workspace','ensure_runtime','preview_launcher','run_cascade_benchmark') for n in ast.walk(ast.parse(code)))


def test_locator_uses_current_pointer_or_explicit_comparison_config_not_id_prefix_or_mtime(tmp_path, monkeypatch):
    common, output, *_ = fixture(tmp_path, monkeypatch)
    persistent = tmp_path/'persistent'
    pointer = persistent/'benchmarks/cascade/latest.json'
    atomic_write_json(pointer, dict(plan=str(output/'plan.json'), sha256=diagnostic.checksum(output/'plan.json')))
    assert diagnostic.locate(persistent, 'cascade') == output
    config = tmp_path/'configs'
    atomic_write_json(config/'cascade-eval.yaml', {'run_id':'the-actual-current-cascade'})
    assert diagnostic.locate(persistent, 'all', comparison_config=config, runs_root=tmp_path/'runs') == tmp_path/'runs/the-actual-current-cascade'
    assert diagnostic.locate(persistent, 's2ut', evaluation_root=output) == output
    with pytest.raises(RuntimeError, match='not Cascade/all'):
        diagnostic.locate(persistent, 's2ut')
    with pytest.raises(RuntimeError, match='COMPARE_CONFIG'):
        diagnostic.locate(persistent, 'all')
    atomic_write_json(pointer, dict(plan=str(output/'plan.json'), sha256='0'*64), overwrite=True)
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        diagnostic.locate(persistent, 'cascade')


def test_cer_policy_keeps_punctuation_and_can_exceed_one_hundred_percent():
    assert diagnostic.normalize_japanese_cer(' Ａ　あ？\n') == 'Aあ?'
    assert diagnostic.edit_distance('あ。', 'あ') == 1
    assert diagnostic.edit_distance('あ', 'あいうえお') == 4


def test_incompatible_scoring_package_is_rejected_without_install_or_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic.importlib.metadata, 'version', lambda name: '99.0')
    with pytest.raises(RuntimeError, match='existing sacrebleu==2.5.1'):
        diagnostic.diagnose(tmp_path/'unused', tmp_path/'unused')


@pytest.mark.parametrize('legacy_revision', [False, True])
def test_normal_cell_bootstraps_only_local_code_and_keeps_all_experiment_files_unchanged(tmp_path, monkeypatch, legacy_revision):
    common, output, *_ = fixture(tmp_path/'experiment', monkeypatch)
    result = diagnostic.diagnose(common, output)
    result['kftt_samples'][0]['asr_ja_text'] = '<script>untrusted text</script>'
    persistent = tmp_path/'experiment/persistent'
    atomic_write_json(persistent/'benchmarks/cascade/latest.json',
                      dict(plan=str(output/'plan.json'), sha256=diagnostic.checksum(output/'plan.json')))
    before = snapshot(tmp_path/'experiment')
    repo = tmp_path/'code-repo'
    repo.mkdir()
    actual_run = subprocess.run
    def git(*args):
        return actual_run(['git', '-c', 'safe.directory='+str(repo), '-c', 'user.name=Diagnostic Test',
                           '-c', 'user.email=test@example.invalid', '-C', str(repo), *args],
                          capture_output=True, text=True, encoding='utf-8', check=True).stdout.strip()
    git('init')
    (repo/'README').write_text('old source', encoding='utf-8')
    git('add', 'README')
    git('commit', '-m', 'test: initial local source')
    old = git('rev-parse', 'HEAD')
    helper = repo/'scripts/colab/cascade_corpus_diagnostic.py'
    helper.parent.mkdir(parents=True)
    helper.write_text('CASCADE_DIAGNOSTIC_VERSION = 1\n', encoding='utf-8')
    module = repo/'src/direct_s2st/cascade/corpus_diagnostic.py'
    module.parent.mkdir(parents=True)
    module.write_text('# diagnostic code fixture\n', encoding='utf-8')
    git('add', '.')
    git('commit', '-m', 'test: add local diagnostic source')
    new = git('rev-parse', 'HEAD')
    git('branch', '-f', 'main', new)
    git('remote', 'add', 'origin', '.')  # Local-only fetch, no network or user repository changes.
    git('checkout', '--detach', old)
    calls, rendered = [], []
    def invoke(command, **kwargs):
        calls.append(command)
        if command[0] == 'git':
            # Windows sandbox owns temporary checkouts under a different account.
            # Trust only this test's -C path; production Colab commands stay unchanged.
            directory = command[command.index('-C')+1]
            return actual_run(['git', '-c', 'safe.directory='+str(directory), *command[1:]], **kwargs)
        assert command[1] == '-B'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        return subprocess.CompletedProcess(command, 0, json.dumps(result, ensure_ascii=False))
    monkeypatch.setattr(subprocess, 'run', invoke)
    display_module = ModuleType('IPython.display')
    display_module.HTML, display_module.display = lambda html: html, rendered.append
    monkeypatch.setitem(sys.modules, 'IPython.display', display_module)
    notebook = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    code = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('#@title 8A.'))
    # Only redirect the disposable code-cache path in the test; same notebook logic/Git commands.
    code = code.replace("'/content/s2st-diagnostic-code'", repr(str(tmp_path/'code-cache')))
    def forbidden(*args, **kwargs):
        pytest.fail('8A must not call the experiment logger/runtime/model/evaluation helpers')
    namespace = dict(PERSISTENT=persistent, DATA=common.parent.parent, REPO=repo,
                     TRAIN_CODE_REVISION=old if legacy_revision else new, BENCHMARK_TARGET='cascade', PYTHON=sys.executable,
                     run=forbidden, cli=forbidden, ensure_workspace=forbidden, ensure_runtime=forbidden,
                     preview_launcher=forbidden, run_cascade_benchmark=forbidden)
    exec(code, namespace)
    assert namespace['cascade_corpus_diagnostic']['groups']['KFTT']['samples'] == 2
    assert namespace['_diag_revision'] == new
    assert git('rev-parse', 'HEAD') == old
    assert any(command[:1] == ['git'] and 'fetch' in command for command in calls)
    assert any(command[:1] == ['git'] and 'worktree' in command for command in calls)
    assert not any('reset' in command for command in calls)
    assert snapshot(tmp_path/'experiment') == before
    assert len(rendered) == 1 and '<script>' not in rendered[0] and '&lt;script&gt;' in rendered[0]
