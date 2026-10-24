import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]


def runtime():return runpy.run_path(str(ROOT/'scripts/colab/flores_fleurs_runtime.py'))


def test_environment_preserves_gpu_and_auth_but_clears_cpu_diagnostic_overrides(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setenv('HF_TOKEN','secret-test-only')
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('PIP_REQUIREMENT','old.txt')
    monkeypatch.setenv('PYTHONPATH','old-cpu-packages');monkeypatch.setenv('S2ST_INPUT_MAP','old-map')
    env=runtime()['environment'](tmp_path/'cache')
    assert env['CUDA_VISIBLE_DEVICES']=='0' and env['HF_TOKEN']=='secret-test-only'
    assert env['HF_HOME']==str(tmp_path/'cache/hf') and env['S2ST_MODEL_ROOT']==str(tmp_path/'cache/corpus-models')
    assert not any(key in env for key in ('PIP_REQUIREMENT','PYTHONPATH','HF_HUB_OFFLINE','S2ST_INPUT_MAP'))


def test_probe_requires_audio_executables_before_importing_models(tmp_path,monkeypatch):
    rt=runtime();monkeypatch.setattr(rt['shutil'],'which',lambda _:None)
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('no GPU/import attempted'))
    with pytest.raises(RuntimeError,match='ffmpeg and sox'):rt['probe'](sys.executable,{})


def test_probe_redirects_sdk_stdout_checks_full_closure_and_cuda(tmp_path,monkeypatch):
    rt=runtime();monkeypatch.setattr(rt['shutil'],'which',lambda _:str(tmp_path/'executable'))
    calls=[]
    def run(args,**kwargs):
        calls.append((args,kwargs));return SimpleNamespace(returncode=0,stdout=json.dumps({'gpu':'fixture'}),stderr='SDK warning')
    monkeypatch.setattr(subprocess,'run',run)
    assert rt['probe']('python',{})['gpu']=='fixture'
    code=calls[0][0][-1]
    assert "contextlib.redirect_stdout(sys.stderr)" in code and "torch.cuda.is_bf16_supported()" in code
    assert 'pyarrow' in code and 'onnxruntime' in code and 'librosa' in code and 'qwen-tts' in code
    assert 'EXPECTED=' in code and 'gradio' in code
    compile(code,'<8C runtime probe>','exec')


def test_setup_permission_and_output_protection_precede_installs(tmp_path,monkeypatch):
    rt=runtime()
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('setup not permitted yet'))
    with pytest.raises(RuntimeError,match='install disabled'):
        rt['ensure']([],sys.executable,tmp_path/'env',tmp_path/'cache',[],allow_install=False)
    with pytest.raises(RuntimeError,match='outside experiments'):
        rt['ensure']([],sys.executable,tmp_path/'experiment/env',tmp_path/'cache',[tmp_path/'experiment'])
    assert not list(tmp_path.iterdir())


def test_cell_1_then_8c_missing_historical_receipt_stops_before_gpu_or_drive_writes(tmp_path,monkeypatch):
    nb=json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    code=next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('#@title 8C.'))
    # The real source has the new module even before committing. Emulate a clean
    # Git content check only; execute the real stdlib preflight subprocess.
    real_run=subprocess.run
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='git':
            if 'rev-parse' in command:return SimpleNamespace(returncode=0,stdout='a'*40,stderr='')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        assert '--preflight-source' in command
        return real_run(command,**kwargs)
    monkeypatch.setattr(subprocess,'run',run)
    namespace=dict(REPO=ROOT,PERSISTENT=tmp_path/'experiment',DATA=tmp_path/'experiment/data',
                   TRAIN_CODE_REVISION='a'*40,CORPUS=tmp_path/'corpus',BENCHMARK_TARGET='s2ut')
    code=code.replace("'/content/s2st-flores-fleurs-code'",repr(str(tmp_path/'code-cache')))
    with pytest.raises(RuntimeError,match='Cannot prove exact source-corpus Japanese TTS configuration'):
        exec(code,namespace)
    assert not list(tmp_path.iterdir())
    assert not any('pip' in command or 'nvidia-smi' in command for command in calls)


def test_cell_1_then_8c_dispatches_without_8a_or_8b_variables(tmp_path,monkeypatch):
    from test_flores_fleurs_diagnostic import receipt
    provenance,_=receipt(tmp_path)
    official=tmp_path/'experiment';official.mkdir();(official/'original').write_text('preserved')
    before={str(p):p.read_bytes() for p in official.rglob('*') if p.is_file()}
    nb=json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    code=next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('#@title 8C.'))
    code=code.replace("FLORES_FLEURS_SOURCE_PROVENANCE = ''",'FLORES_FLEURS_SOURCE_PROVENANCE = '+repr(str(provenance)))
    code=code.replace("'/content/s2st-flores-fleurs-code'",repr(str(tmp_path/'code-cache')))
    real_run=subprocess.run;real_path=runpy.run_path;calls=[]
    def run(command,**kwargs):
        if command[0]=='git':
            return SimpleNamespace(returncode=0,stdout='a'*40 if 'rev-parse' in command else '',stderr='')
        calls.append('preflight');return real_run(command,**kwargs)
    def ensure(*args,**kwargs):
        assert calls==['preflight'];calls.append('ensure')
        return sys.executable,dict(os.environ),{'gpu':'fixture'}
    def result(args,env,**kwargs):
        assert calls==['preflight','ensure'];calls.append('inference-dispatch')
        assert args[args.index('--target')+1]=='cascade'
        assert args[args.index('--source-provenance')+1]==str(provenance)
        assert args[args.index('--limit')+1]=='300'
        return dict(all_score_matches_saved=True,display='fixture external diagnostic',per_sample=[])
    def read_path(path):
        if str(path).endswith('flores_fleurs_runtime.py'):return {'ensure':ensure}
        module=real_path(path)
        if str(path).endswith('cascade_diagnostic_runtime.py'):module['run_json']=result
        return module
    monkeypatch.setattr(subprocess,'run',run);monkeypatch.setattr(runpy,'run_path',read_path)
    namespace=dict(REPO=ROOT,PERSISTENT=official,DATA=official/'data',TRAIN_CODE_REVISION='a'*40,BENCHMARK_TARGET='s2ut')
    exec(code,namespace)
    assert namespace['flores_fleurs_diagnostic']['all_score_matches_saved']
    assert {str(p):p.read_bytes() for p in official.rglob('*') if p.is_file()}==before
