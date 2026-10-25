import importlib.util
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cascade_diagnostic_runtime', ROOT/'scripts/colab/cascade_diagnostic_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.mark.parametrize('current,expected', [('s2ut', 'cascade'), ('cascade', 'cascade'), ('all', 'all'),
                                              ('tt2', 'cascade'), (None, 'cascade')])
def test_auto_target_is_always_a_cascade_evaluation(current, expected):
    assert runtime.select_target('auto', current) == expected
    assert runtime.select_target('cascade', current) == 'cascade'
    assert runtime.select_target('all', current) == 'all'
    with pytest.raises(RuntimeError, match='8A target'):
        runtime.select_target('s2ut', current)


def test_cpu_environment_does_not_mutate_parent_or_enable_downloads():
    parent = dict(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='0', CUSTOM='preserve')
    assert runtime.cpu_environment(parent) == dict(parent, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    assert parent['CUDA_VISIBLE_DEVICES'] == '0' and parent['HF_HUB_OFFLINE'] == '0'


def test_dependency_probe_uses_real_existing_cpu_scoring_for_both_targets():
    for target in ('cascade', 'all'):
        assert runtime.select_python([None, sys.executable, sys.executable], target, runtime.cpu_environment()) == str(Path(sys.executable).absolute())


@pytest.mark.parametrize('failure', ['missing', 'version', 'python_version', 'broken_import', 'timeout', 'exec_error'])
def test_dependency_probe_skips_bad_python_and_uses_next_existing_candidate(tmp_path, monkeypatch, failure):
    bad, good = tmp_path/'old-python', tmp_path/'usable-python'
    bad.touch()
    good.touch()
    seen = []
    def probe(command, **kwargs):
        seen.append(command)
        assert command[1:3] == ['-B', '-c'] and kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert 'import sacrebleu' in command[-1] and 'import yaml' in command[-1]
        if command[0] == str(bad):
            if failure == 'timeout':
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            if failure == 'exec_error':
                raise OSError('cannot execute stale Python')
            error = {'missing': 'PackageNotFoundError: sacrebleu', 'version': 'requires sacrebleu==2.5.1',
                     'python_version': 'Python >= 3.10 is required',
                     'broken_import': 'ModuleNotFoundError: portalocker'}[failure]
            return subprocess.CompletedProcess(command, 1, '', error)
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(runtime.subprocess, 'run', probe)
    assert runtime.select_python([tmp_path/'absent', bad, bad, good], 'all', runtime.cpu_environment()) == str(good)
    assert [command[0] for command in seen] == [str(bad), str(good)]
    assert not any('pip' in argument or 'setup.py' in argument for command in seen for argument in command)


def test_no_usable_python_lists_real_dependency_error_without_install(tmp_path, monkeypatch):
    python = tmp_path/'python'
    python.touch()
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a[0], 1, '', 'ModuleNotFoundError: sacrebleu'))
    with pytest.raises(RuntimeError, match='ModuleNotFoundError: sacrebleu') as caught:
        runtime.select_python([python], 'cascade', runtime.cpu_environment())
    assert str(python) in str(caught.value) and 'no automatic install' in str(caught.value)


def test_pyyaml_is_required_only_for_the_four_system_config_reader(tmp_path, monkeypatch):
    python = tmp_path/'python'
    python.touch()
    def probe(command, **kwargs):
        missing = 'import yaml' in command[-1]
        return subprocess.CompletedProcess(command, 1 if missing else 0, '', 'ModuleNotFoundError: yaml' if missing else '')
    monkeypatch.setattr(runtime.subprocess, 'run', probe)
    assert runtime.select_python([python], 'cascade', runtime.cpu_environment()) == str(python)
    with pytest.raises(RuntimeError, match='ModuleNotFoundError: yaml'):
        runtime.select_python([python], 'all', runtime.cpu_environment())


def test_child_progress_and_errors_are_visible_in_notebook_output(capsys):
    code = "import sys; print('specific missing artifact: results/per_sample.jsonl', file=sys.stderr, flush=True); sys.exit(7)"
    with pytest.raises(RuntimeError, match='specific missing artifact') as caught:
        runtime.run_json([sys.executable, '-B', '-c', code], runtime.cpu_environment())
    assert 'code 7' in str(caught.value)
    assert 'specific missing artifact' in capsys.readouterr().out


def test_gpu_diagnostic_error_does_not_claim_no_inference_was_attempted():
    with pytest.raises(RuntimeError,match='8B NLLB GPU diagnostic') as caught:
        runtime.run_json([sys.executable,'-B','-c','raise RuntimeError("CUDA fixture failure")'],
                         runtime.cpu_environment(),name='8B',mode='NLLB GPU diagnostic',
                         failure_note='Saved formal results unchanged; diagnostic chunks resumable.')
    assert 'no inference was attempted' not in str(caught.value)
    assert 'CUDA fixture failure' in str(caught.value)


def test_child_success_keeps_result_json_separate_from_progress(capsys):
    report = dict(all_score_matches_saved=True, display='日本語CER', groups={'KFTT':{'samples':2}})
    code = "import json, sys; print('[progress] checked=2/2', file=sys.stderr, flush=True); print(json.dumps("+repr(report)+", ensure_ascii=False))"
    assert runtime.run_json([sys.executable, '-B', '-c', code], runtime.cpu_environment()) == report
    assert '[progress] checked=2/2' in capsys.readouterr().out


@pytest.mark.parametrize('output', ['not result JSON', '[]', '{}', '{"all_score_matches_saved":false}'])
def test_child_invalid_output_is_not_presented_as_verified_report(output):
    code = 'print('+repr(output)+')'
    with pytest.raises(RuntimeError, match='8A helper'):
        runtime.run_json([sys.executable, '-B', '-c', code], runtime.cpu_environment())


def external_result():
    return dict(output_root='/saved/diagnostics/flores-fleurs-s2ut/identity',
                display='外部評価結果：S2UT ASR-BLEU 12.5',
                metrics=dict(purpose='external_flores_fleurs_s2ut_natural_not_formal_benchmark',
                             condition='natural',sample_count=300,bleu=12.5,
                             cascade_natural_asr_bleu=21.153432814483,difference=12.5-21.153432814483))


@pytest.mark.parametrize('inspect',[False,True])
def test_external_s2ut_real_child_json_has_no_formal_all_score_flag(capsys,inspect):
    report=(dict(cascade_root='/saved/8c',samples=300,display='入力確認完了') if inspect else external_result())
    code="import json,sys; print('[progress] 8E status=completed',file=sys.stderr,flush=True); print(json.dumps("+repr(report)+",ensure_ascii=False))"
    result=runtime.run_json([sys.executable,'-B','-c',code],runtime.cpu_environment(),name='8E',
                            result_kind='external_s2ut_inspect' if inspect else 'external_s2ut')
    assert result==report and 'all_score_matches_saved' not in result
    assert 'status=completed' in capsys.readouterr().out
    with pytest.raises(RuntimeError,match='formal All score'):
        runtime.run_json([sys.executable,'-B','-c',code],runtime.cpu_environment())


@pytest.mark.parametrize('change',['no_metrics','no_output','no_display','wrong_purpose','wrong_condition',
    'empty_samples','bool_samples','nan_bleu','string_bleu','negative_bleu','out_of_range','wrong_difference'])
def test_external_result_contract_rejects_incomplete_or_wrong_score(change):
    report=external_result()
    if change=='no_metrics':report.pop('metrics')
    elif change=='no_output':report.pop('output_root')
    elif change=='no_display':report['display']=''
    else:
        key,value={'wrong_purpose':('purpose','formal benchmark'),'wrong_condition':('condition','synthetic'),
            'empty_samples':('sample_count',0),'bool_samples':('sample_count',True),'nan_bleu':('bleu',float('nan')),
            'string_bleu':('bleu','12.5'),'negative_bleu':('bleu',-1),'out_of_range':('bleu',101),
            'wrong_difference':('difference',0)}[change]
        report['metrics'][key]=value
    code='import json; print(json.dumps('+repr(report).replace('nan','float("nan")')+'))'
    with pytest.raises(RuntimeError,match='valid external S2UT'):
        runtime.run_json([sys.executable,'-B','-c',code],runtime.cpu_environment(),name='8E',result_kind='external_s2ut')


def test_inspection_and_evaluation_contracts_cannot_be_substituted():
    with pytest.raises(RuntimeError,match='inspection'):
        runtime.validate_result(external_result(),'external_s2ut_inspect','8E')
    with pytest.raises(RuntimeError,match='evaluation report'):
        runtime.validate_result(dict(cascade_root='/saved',samples=300,display='ok'),'external_s2ut','8E')


def test_perfect_real_sacrebleu_roundoff_is_accepted_without_rounding_saved_values():
    import sacrebleu
    text='the very old book is on the table'
    score=sacrebleu.corpus_bleu([text],[[text]]).score
    report=external_result();report['metrics'].update(bleu=score,cascade_natural_asr_bleu=score,difference=0.0)
    code='import json; print(json.dumps('+repr(report)+'))'
    assert runtime.run_json([sys.executable,'-B','-c',code],runtime.cpu_environment(),name='8E',
                            result_kind='external_s2ut')==report


def test_unknown_result_kind_stops_before_subprocess(monkeypatch):
    monkeypatch.setattr(runtime.subprocess,'Popen',lambda *a,**kw:pytest.fail('no child for invalid contract'))
    with pytest.raises(ValueError,match='result contract'):
        runtime.run_json(['unused'],{},result_kind='anything')


def test_keyboard_interrupt_stops_only_the_diagnostic_child(monkeypatch):
    class InterruptedOutput(io.StringIO):
        def read(self, *args):
            raise KeyboardInterrupt
    class Child:
        stdout = InterruptedOutput()
        stderr = io.StringIO('')
        terminated = False
        def poll(self): return None
        def terminate(self): self.terminated = True
        def wait(self, timeout=None): return 0
    child = Child()
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **k: child)
    with pytest.raises(KeyboardInterrupt):
        runtime.run_json(['diagnostic-only'], runtime.cpu_environment())
    assert child.terminated and child.stdout.closed and child.stderr.closed


def test_existing_usable_python_never_installs_even_when_permission_is_on(tmp_path, monkeypatch):
    env = runtime.cpu_environment()
    monkeypatch.setattr(runtime, 'select_python', lambda *a, **k: sys.executable)
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **k: pytest.fail('unnecessary install'))
    assert runtime.ensure_python([sys.executable], 'cascade', env,
        dependency_root=tmp_path/'unused', allow_install=True) == (sys.executable, env)
    assert not list(tmp_path.iterdir())


def test_missing_packages_cannot_trigger_install_without_permission(tmp_path, monkeypatch):
    def missing(*a, **k): raise RuntimeError('no usable existing CPU Python')
    monkeypatch.setattr(runtime, 'select_python', missing)
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **k: pytest.fail('install not authorized'))
    with pytest.raises(RuntimeError, match='no usable'):
        runtime.ensure_python([sys.executable], 'cascade', runtime.cpu_environment(),
            dependency_root=tmp_path/'unused', allow_install=False)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('protected', ['experiment', 'corpus'])
def test_cpu_dependency_directory_cannot_be_below_any_protected_root(tmp_path, monkeypatch, protected):
    def missing(*a, **k): raise RuntimeError('missing dependency')
    monkeypatch.setattr(runtime, 'select_python', missing)
    with pytest.raises(RuntimeError, match='outside saved experiments'):
        runtime.ensure_python([sys.executable], 'cascade', runtime.cpu_environment(),
            dependency_root=tmp_path/protected/'packages', protected=[tmp_path/protected], allow_install=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('broken_cache,status', [(False, 0), (True, 0), (False, 1)])
def test_cpu_packages_are_isolated_verified_reused_and_never_overwrite_cache(tmp_path, monkeypatch, broken_cache, status):
    base = tmp_path/'python'
    base.touch()
    cache = tmp_path/'deps'
    fingerprint = '{"cpu":"fake-test-platform"}'
    requirements = (ROOT/'requirements/diagnostic-cpu.txt').read_text(encoding='utf-8')
    key = hashlib.sha256((requirements+fingerprint).encode()).hexdigest()[:20]
    if broken_cache:
        (cache/key).mkdir(parents=True)
        (cache/key/'old.txt').write_text('preserve', encoding='utf-8')
    installs = []
    def choose(candidates, target, env, required_versions=None):
        path = env.get('PYTHONPATH', '').split(runtime.os.pathsep)[0]
        if path and (Path(path)/'verified.txt').is_file():
            assert required_versions['sacrebleu'] == '2.5.1'
            assert len(required_versions) == 8
            return str(base)
        raise RuntimeError('no usable existing CPU Python')
    monkeypatch.setattr(runtime, 'select_python', choose)
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a[0], 0, fingerprint, ''))
    class Install:
        stdout = io.StringIO('installing isolated CPU wheels\n')
        def wait(self): return status
    def install(command, **kwargs):
        installs.append(command)
        assert command[:6] == [str(base), '-B', '-m', 'pip', '--isolated', 'install']
        for option in ('--no-deps', '--only-binary=:all:', '--no-cache-dir', '--target'):
            assert option in command
        packages = Path(command[command.index('--target')+1])
        assert packages.resolve().is_relative_to(cache.resolve())
        packages.mkdir()
        (packages/'verified.txt').touch()
        return Install()
    monkeypatch.setattr(runtime.subprocess, 'Popen', install)
    env = runtime.cpu_environment()
    original = dict(env)
    if status:
        with pytest.raises(RuntimeError, match='dependency install failed'):
            runtime.ensure_python([str(base)], 'cascade', env, dependency_root=cache, allow_install=True)
        assert not list(cache.iterdir())
    else:
        first = runtime.ensure_python([str(base)], 'cascade', env, dependency_root=cache, allow_install=True)
        second = runtime.ensure_python([str(base)], 'cascade', env, dependency_root=cache, allow_install=True)
        assert first == second and len(installs) == 1
        assert Path(first[1]['PYTHONPATH']).is_relative_to(cache)
        if broken_cache:
            assert (cache/key/'old.txt').read_text(encoding='utf-8') == 'preserve'
            assert Path(first[1]['PYTHONPATH']) != cache/key
    assert env == original
