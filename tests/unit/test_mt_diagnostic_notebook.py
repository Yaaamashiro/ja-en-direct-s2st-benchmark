import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('legacy',[True,False])
def test_normal_8b_bootstraps_separate_code_and_runs_gpu_dispatch_without_experiment_writes(tmp_path,monkeypatch,legacy):
    repo=tmp_path/'repo'; repo.mkdir()
    persistent=tmp_path/'experiment'; persistent.mkdir()
    (persistent/'saved.txt').write_text('official results preserved',encoding='utf-8')
    before=(persistent/'saved.txt').read_bytes(),(persistent/'saved.txt').stat().st_mtime_ns
    real_run=subprocess.run
    def git(*args):
        return real_run(['git','-c','safe.directory='+str(repo),'-c','user.name=Diagnostic Test',
            '-c','user.email=test@example.invalid','-C',str(repo),*args],capture_output=True,text=True,encoding='utf-8',check=True).stdout.strip()
    git('init')
    (repo/'README').write_text('old training checkout',encoding='utf-8')
    git('add','.'); git('commit','-m','test: old training source')
    old=git('rev-parse','HEAD')
    files=['scripts/colab/cascade_mt_diagnostic.py','scripts/colab/mt_diagnostic_runtime.py',
           'src/direct_s2st/cascade/mt_diagnostic.py','requirements/diagnostic-mt.txt',
           'scripts/colab/cascade_diagnostic_runtime.py']
    for name in files:
        (repo/name).parent.mkdir(parents=True,exist_ok=True)
        (repo/name).write_text('# immutable fixture source\n',encoding='utf-8')
    (repo/files[0]).write_text("MT_DIAGNOSTIC_VERSION = 1\nimport json,sys,os\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='0'\n"
        "assert sys.argv[sys.argv.index('--target')+1]=='cascade'\n"
        "print('[progress] 8B fixture NLLB dispatch',file=sys.stderr,flush=True)\n"
        "print(json.dumps(dict(all_score_matches_saved=True,display='fixture successful',per_sample=[])))\n",encoding='utf-8')
    (repo/files[1]).write_text('import os,sys\ndef ensure(*args,**kwargs):\n'
        "    assert kwargs['allow_install'] is True\n"
        "    return sys.executable,dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'),{'gpu':'test GPU'}\n",encoding='utf-8')
    shutil.copyfile(ROOT/files[-1],repo/files[-1])
    git('add','.'); git('commit','-m','test: immutable 8B code')
    new=git('rev-parse','HEAD')
    git('branch','-f','main',new); git('remote','add','origin','.'); git('checkout','--detach',old)
    calls=[]
    def invoke(command,**kwargs):
        calls.append(command)
        if command[0]=='git':
            directory=command[command.index('-C')+1]
            return real_run(['git','-c','safe.directory='+str(directory),*command[1:]],**kwargs)
        return real_run(command,**kwargs)
    monkeypatch.setattr(subprocess,'run',invoke)
    notebook=json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    code=next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('#@title 8B.'))
    code=code.replace("'/content/s2st-mt-diagnostic-code'",repr(str(tmp_path/'code-cache')))
    namespace=dict(REPO=repo,PERSISTENT=persistent,DATA=persistent/'data',PYTHON=sys.executable,
                   TRAIN_CODE_REVISION=old if legacy else new,BENCHMARK_TARGET='s2ut')
    exec(code,namespace)
    assert namespace['cascade_mt_diagnostic']['all_score_matches_saved']
    assert namespace['_mt_revision']==new and namespace['_mt_target']=='cascade'
    assert git('rev-parse','HEAD')==old
    assert not any('reset' in command for command in calls)
    assert ((persistent/'saved.txt').read_bytes(),(persistent/'saved.txt').stat().st_mtime_ns)==before
    assert [p.name for p in persistent.iterdir()]==['saved.txt']


def test_8b_code_fetch_permission_is_respected_before_gpu_setup(tmp_path,monkeypatch):
    notebook=json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    code=next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('#@title 8B.'))
    code=code.replace('MT_DIAGNOSTIC_FETCH_CODE = True','MT_DIAGNOSTIC_FETCH_CODE = False')
    code=code.replace("'/content/s2st-mt-diagnostic-code'",repr(str(tmp_path/'cache')))
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('Git or GPU setup not authorized yet'))
    with pytest.raises(AssertionError,match='コード取得の許可'):
        exec(code,dict(REPO=tmp_path/'repo',PERSISTENT=tmp_path/'experiment',DATA=tmp_path/'data',TRAIN_CODE_REVISION='a'*40))
    assert not list(tmp_path.iterdir())
