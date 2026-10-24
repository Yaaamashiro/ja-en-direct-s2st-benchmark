"""8C-only runtime. No fairseq/training/system environment mutation."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import textwrap
import uuid

ROOT=Path(__file__).resolve().parents[2]
BASE=runpy.run_path(str(ROOT/'scripts/colab/mt_diagnostic_runtime.py'))
CORE=dict(BASE['CORE'],numpy='2.2.6',**{'qwen-tts':'0.1.1','torchaudio':'2.7.1','pyarrow':'20.0.0',
                                        'soundfile':'0.13.1','soxr':'1.1.0','jiwer':'4.0.0','accelerate':'1.12.0'})


def environment(cache):
    env=BASE['environment'](cache)
    # Capture an existing HF login before changing HF_HOME; never log/save token.
    if not env.get('HF_TOKEN'):
        try:
            from huggingface_hub import get_token
            token=get_token()
            if token:
                env['HF_TOKEN']=token
        except ImportError:
            pass
    env.update(S2ST_MODEL_ROOT=str(Path(cache)/'corpus-models'))
    return env


def probe(python,env):
    if not shutil.which('ffmpeg') or not shutil.which('sox'):
        raise RuntimeError('8C requires ffmpeg and sox executables for existing audio adapters; install them before rerunning')
    expected=dict(CORE)
    for line in (ROOT/'requirements/diagnostic-flores.txt').read_text(encoding='utf-8').splitlines():
        if '==' not in line or line.startswith('#') or ('sys_platform == "win32"' in line and os.name!='nt'):
            continue
        name,value=line.split(';')[0].strip().split('==')
        expected[name]=value
    code='''import importlib.metadata as m, json, platform, torch
assert platform.python_version()=='3.10.18', '8C requires the reviewed isolated Python 3.10.18 runtime'
for key,value in EXPECTED.items():
    assert m.version(key).split('+')[0]==value, key+' differs: '+m.version(key)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, M2M100ForConditionalGeneration, NllbTokenizerFast
from qwen_tts import Qwen3TTSModel
import pyarrow.parquet, soundfile, soxr, jiwer, librosa, torchaudio, yaml, sacrebleu
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), '8C requires CUDA and BF16 support; no fallback'
assert torch.ones(2,device='cuda',dtype=torch.bfloat16).sum().item()==2
torch.cuda.synchronize()
print(json.dumps(dict(python=platform.python_version(),packages={d.metadata['Name'].lower():d.version for d in m.distributions()},
    torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
    vram_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,bf16_supported=torch.cuda.is_bf16_supported(),
    capability=torch.cuda.get_device_capability(0),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),sort_keys=True))
'''
    imports,output=code.split('print(json.dumps',1)
    code='import contextlib,sys\nwith contextlib.redirect_stdout(sys.stderr):\n'+textwrap.indent(imports,'    ')+'\nprint(json.dumps'+output
    Progress=runpy.run_path(str(ROOT/'src/direct_s2st/progress.py'))['Progress']
    with Progress('8C: verify pinned dependency imports and CUDA/BF16'):
        result=subprocess.run([str(python),'-B','-c','EXPECTED='+repr(expected)+'\n'+code],capture_output=True,
                              text=True,encoding='utf-8',errors='replace',env=env,timeout=90)
    if result.returncode:
        raise RuntimeError('8C runtime check failed: '+(result.stderr or result.stdout)[-5000:])
    return json.loads(result.stdout)


def logged(command,env):
    Progress=runpy.run_path(str(ROOT/'src/direct_s2st/progress.py'))['Progress']
    stop=runpy.run_path(str(ROOT/'scripts/colab/cascade_diagnostic_runtime.py'))['stop_child']
    print('8C:', ' '.join(map(str,command)),flush=True)
    with Progress('8C: isolated setup / sequential stage subprocess'):
        process=subprocess.Popen(list(map(str,command)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,text=True,encoding='utf-8',errors='replace',env=env)
        try:
            with process.stdout:
                for line in process.stdout:
                    print(line,end='',flush=True)
            status=process.wait()
        except BaseException:
            stop(process);raise
        if status:
            raise RuntimeError(f'8C subprocess failed ({status}); see preceding error. Successful stages are resumable.')


def ensure(candidates,host_python,root,cache,protected,*,allow_install=True):
    root,cache=Path(root).absolute(),Path(cache).absolute()
    for path in (root,cache):
        if (any(path.resolve().is_relative_to(Path(p).resolve()) for p in protected)
                or any(p.is_symlink() for p in (path,*path.parents))):
            raise RuntimeError('8C runtime/cache must remain local outside experiments/CORPUS_ROOT')
    requirements=ROOT/'requirements/diagnostic-flores.txt'
    key=hashlib.sha256(requirements.read_bytes()+b'py31018-torch271-cu126-8c-v1').hexdigest()[:20]
    prefix=root/key
    env=environment(cache)
    dedicated=prefix/'venv/bin/python'
    for path in (prefix,prefix/'venv',prefix/'uv'):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError('8C managed runtime redirected outside its local root')
    failures=[]
    for python in dict.fromkeys(str(p) for p in (dedicated,*candidates) if p and Path(p).is_file()):
        try:
            return python,env,probe(python,env)
        except (OSError,RuntimeError,subprocess.TimeoutExpired) as error:
            failures.append(str(error))
    if not allow_install:
        raise RuntimeError('8C compatible CUDA runtime missing; install disabled.\n'+'\n'.join(failures))
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    except (OSError,subprocess.TimeoutExpired) as error:
        raise RuntimeError('8C requires a Colab GPU runtime; no CPU fallback or installation attempted') from error
    if gpu.returncode or not gpu.stdout.strip():
        raise RuntimeError('8C requires a Colab GPU runtime; no CPU fallback')
    if not shutil.which('ffmpeg') or not shutil.which('sox'):
        if not shutil.which('apt-get'):
            raise RuntimeError('8C needs ffmpeg and sox executables; automatic setup requires a Colab apt runtime')
        logged(['apt-get','update','-qq'],env)
        logged(['apt-get','install','-y','ffmpeg','sox','libsndfile1'],env)
    owner=prefix/'owner.json'
    identity=dict(purpose='8c_external_cascade_only',requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest())
    if prefix.exists() and (not owner.is_file() or json.loads(owner.read_text(encoding='utf-8'))!=identity):
        raise RuntimeError('8C dedicated prefix is not an owned matching runtime; preserved')
    prefix.mkdir(parents=True,exist_ok=True)
    runpy.run_path(str(ROOT/'src/direct_s2st/io.py'))['atomic_write_json'](owner,identity,resume=True)
    uv=prefix/'uv'
    if not (uv/'uv').is_dir():
        if uv.exists():
            uv=prefix/('uv-'+uuid.uuid4().hex)
        logged([host_python,'-B','-m','pip','--isolated','install','--no-deps','--only-binary=:all:',
                '--index-url','https://pypi.org/simple','--target',uv,'uv==0.8.22'],env)
    bootstrap=dict(env,PYTHONPATH=str(uv),UV_PYTHON_INSTALL_DIR=str(prefix/'python'),UV_CACHE_DIR=str(prefix/'uv-cache'))
    logged([host_python,'-B','-c',"import importlib.metadata as m; assert m.version('uv')=='0.8.22'"],bootstrap)
    if not dedicated.is_file():
        logged([host_python,'-B','-m','uv','venv','--seed','--python','3.10.18',prefix/'venv'],bootstrap)
    logged([dedicated,'-B','-m','pip','--isolated','install','--index-url','https://pypi.org/simple',
            'pip==24.0','setuptools==80.9.0','wheel==0.45.1'],env)
    logged([dedicated,'-B','-m','pip','--isolated','install','--only-binary=:all:',
            '--index-url','https://download.pytorch.org/whl/cu126','torch==2.7.1','torchaudio==2.7.1'],env)
    # sox 1.5.0 ships only an sdist; build its small pure-Python wheel here.
    logged([dedicated,'-B','-m','pip','--isolated','install','--no-build-isolation','--no-deps',
            '--index-url','https://pypi.org/simple','-r',requirements],env)
    logged([dedicated,'-B','-m','pip','--isolated','check'],env)
    return str(dedicated),env,probe(dedicated,env)
