import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('mt_diagnostic_runtime',ROOT/'scripts/colab/mt_diagnostic_runtime.py')
runtime=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_mt_env_does_not_inherit_cpu_masking_or_pip_and_staging_overrides(monkeypatch,tmp_path):
    for key in ('PIP_CONSTRAINT','PIP_REQUIREMENT','PIP_BUILD_CONSTRAINT','PYTHONPATH','S2ST_INPUT_MAP','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):
        monkeypatch.setenv(key,'old-setting')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    env=runtime.environment(tmp_path)
    assert env['CUDA_VISIBLE_DEVICES']=='0'
    assert env['HF_HOME']==str(tmp_path/'hf') and env['PYTHONDONTWRITEBYTECODE']=='1'
    assert not any(key in env for key in ('PIP_CONSTRAINT','PIP_REQUIREMENT','PIP_BUILD_CONSTRAINT','S2ST_INPUT_MAP','PYTHONPATH','HF_HUB_OFFLINE'))


def test_existing_gpu_environment_is_verified_and_never_installed_into(tmp_path,monkeypatch):
    python=tmp_path/'python'
    python.touch()
    monkeypatch.setattr(runtime,'probe',lambda py,env:{'gpu':'fixture'})
    monkeypatch.setattr(runtime,'logged',lambda *a:pytest.fail('must not install into an existing usable environment'))
    result=runtime.ensure([python],sys.executable,tmp_path/'runtime',tmp_path/'cache',[tmp_path/'experiment'])
    assert result[0]==str(python) and result[2]['gpu']=='fixture'
    assert not (tmp_path/'runtime').exists() and not (tmp_path/'cache').exists()


@pytest.mark.parametrize('root_kind',['runtime','cache'])
def test_environment_paths_cannot_be_inside_experiment_or_corpus(tmp_path,root_kind):
    paths=dict(root=tmp_path/'runtime',cache=tmp_path/'cache')
    paths['root' if root_kind=='runtime' else 'cache']=tmp_path/'corpus/local'
    with pytest.raises(RuntimeError,match='outside experiments'):
        runtime.ensure([],sys.executable,paths['root'],paths['cache'],[tmp_path/'corpus'])
    assert not list(tmp_path.iterdir())


def test_no_install_permission_stops_before_any_setup(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'logged',lambda *a:pytest.fail('install not authorized'))
    with pytest.raises(RuntimeError,match='no install permitted'):
        runtime.ensure([],sys.executable,tmp_path/'runtime',tmp_path/'cache',[],allow_install=False)
    assert not list(tmp_path.iterdir())


def test_cpu_runtime_never_downloads_gpu_dependencies_or_falls_back(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a[0],1,'','no GPU'))
    monkeypatch.setattr(runtime,'logged',lambda *a:pytest.fail('must reject CPU before downloading torch'))
    with pytest.raises(RuntimeError,match='requires a Colab GPU'):
        runtime.ensure([],sys.executable,tmp_path/'runtime',tmp_path/'cache',[])
    assert not list(tmp_path.iterdir())


def test_probe_checks_real_model_imports_cuda_operation_and_fixed_versions(monkeypatch):
    def invoke(command,**kwargs):
        code=command[-1]
        assert 'M2M100ForConditionalGeneration' in code and 'torch.cuda.is_available()' in code
        assert 'torch.ones(1, device=\'cuda\')' in code and '4.57.3' in code
        assert 'sentencepiece' in code and 'sacrebleu' in code
        return subprocess.CompletedProcess(command,0,json.dumps({'gpu':'fixture'}),'')
    monkeypatch.setattr(runtime.subprocess,'run',invoke)
    assert runtime.probe('python',{})=={'gpu':'fixture'}


def test_new_environment_installs_only_into_owned_prefix_not_training_runtime(tmp_path,monkeypatch):
    root,cache=tmp_path/'mt-runtime',tmp_path/'mt-cache'
    calls=[]
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a[0],0,
        '3.10.18\n' if 'platform.python_version()' in a[0][-1] else 'GPU',''))
    monkeypatch.setattr(runtime.subprocess,'check_output',lambda *a,**k:'3.10.18\n')
    monkeypatch.setattr(runtime,'probe',lambda *a:{'gpu':'fixture'})
    def logged(command,env):
        command=list(map(str,command)); calls.append(command)
        assert '/content/s2st-runtime' not in ' '.join(command)
        if '--target' in command:
            assert Path(command[command.index('--target')+1]).is_relative_to(root)
        if command[0] != sys.executable:
            assert Path(command[0]).is_relative_to(root)
    monkeypatch.setattr(runtime,'logged',logged)
    result=runtime.ensure([],sys.executable,root,cache,[tmp_path/'experiment'])
    assert Path(result[0]).is_relative_to(root)
    assert any('3.10.18' in command and 'venv' in command for command in calls)
    assert any('torch==2.7.1' in command and 'https://download.pytorch.org/whl/cu126' in command for command in calls)
    assert any('--no-deps' in command and str(ROOT/'requirements/diagnostic-mt.txt') in command for command in calls)
    assert calls[-1][-1]=='check'
    assert not any('qwen' in ' '.join(command).lower() or 'torchaudio' in command or 'fairseq' in command for command in calls)


def test_unowned_environment_is_preserved(tmp_path,monkeypatch):
    root=tmp_path/'runtime'
    requirements=ROOT/'requirements/diagnostic-mt.txt'
    key=hashlib.sha256(requirements.read_bytes()+b'python3.10.18-torch2.7.1-cu126-v1').hexdigest()[:20]
    prefix=root/key
    prefix.mkdir(parents=True)
    (prefix/'user.txt').write_text('preserve',encoding='utf-8')
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a[0],0,'GPU',''))
    monkeypatch.setattr(runtime,'logged',lambda *a:pytest.fail('unowned prefix must never be modified'))
    with pytest.raises(RuntimeError,match='not an owned'):
        runtime.ensure([],sys.executable,root,tmp_path/'cache',[])
    assert (prefix/'user.txt').read_text(encoding='utf-8')=='preserve'
