"""GPU-only 8B bootstrap. Never install into a training/system environment."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]
CORE = {'torch':'2.7.1', 'transformers':'4.57.3', 'tokenizers':'0.22.2',
        'sentencepiece':'0.2.0', 'sacrebleu':'2.5.1'}


def environment(cache):
    env = dict(os.environ)
    for key in ('PIP_CONSTRAINT','PIP_REQUIREMENT','PIP_BUILD_CONSTRAINT','PYTHONPATH',
                'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):
        env.pop(key, None)
    # New models/caches are local to 8B; no Drive model cache or old input map.
    env.pop('S2ST_INPUT_MAP', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', HF_HOME=str(Path(cache)/'hf'),
               HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false')
    return env


def probe(python, env):
    code = '''import importlib.metadata as m, json, platform, torch
for key,value in CORE.items():
    assert m.version(key).split('+')[0] == value, key+' version differs: '+m.version(key)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, NllbTokenizerFast, M2M100ForConditionalGeneration
import sentencepiece, tokenizers, sacrebleu, yaml
assert torch.cuda.is_available(), '8B requires a working CUDA GPU; no CPU fallback'
assert torch.ones(1, device='cuda').sum().item() == 1
torch.cuda.synchronize()
print(json.dumps(dict(python=platform.python_version(), platform=platform.system(),
    packages={d.metadata['Name'].lower():d.version for d in m.distributions()},
    torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
    gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32), sort_keys=True))
'''
    result = subprocess.run([str(python), '-B', '-c', 'CORE='+repr(CORE)+'\n'+code],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90, env=env)
    if result.returncode:
        raise RuntimeError(str(python)+': '+(result.stderr or result.stdout)[-4000:])
    return json.loads(result.stdout)


def logged(command, env):
    Progress = runpy.run_path(str(ROOT/'src/direct_s2st/progress.py'))['Progress']
    stop = runpy.run_path(str(ROOT/'scripts/colab/cascade_diagnostic_runtime.py'))['stop_child']
    print('8B setup:', ' '.join(map(str, command)), flush=True)
    with Progress('8B: NLLB-only GPU environment setup command'):
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace', env=env)
        try:
            with process.stdout:
                for line in process.stdout:
                    print(line, end='', flush=True)
            status = process.wait()
        except BaseException:
            stop(process)
            raise
        if status:
            raise RuntimeError(f'8B GPU setup failed ({status}); see output above. Saved results/training runtime unchanged.')


def ensure(candidates, host_python, root, cache, protected, *, allow_install=True):
    root, cache = Path(root).absolute(), Path(cache).absolute()
    for path in (root, cache):
        if (any(path.resolve().is_relative_to(Path(item).resolve()) for item in protected)
                or any(parent.is_symlink() for parent in (path,*path.parents))):
            raise RuntimeError('8B runtime/cache must be local and outside experiments and CORPUS_ROOT')
    env, failures = environment(cache), []
    requirements = ROOT/'requirements/diagnostic-mt.txt'
    key = hashlib.sha256(requirements.read_bytes()+b'python3.10.18-torch2.7.1-cu126-v1').hexdigest()[:20]
    prefix = root/key
    for path in (prefix,prefix/'venv',prefix/'uv'):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError('8B managed runtime must not redirect to another environment')
    dedicated = prefix/'venv/bin/python'
    for python in dict.fromkeys(str(p) for p in [dedicated,*candidates] if p and Path(str(p)).is_file()):
        print('8B: check existing GPU Python:', python, flush=True)
        try:
            return python, env, probe(python, env)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            failures.append(str(error))
    if not allow_install:
        raise RuntimeError('8B has no compatible GPU environment; no install permitted.\n'+'\n'.join(failures))
    try:
        gpu = subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], capture_output=True,
                             text=True, encoding='utf-8', timeout=15)
        if gpu.returncode or not gpu.stdout.strip():
            raise RuntimeError('no GPU detected')
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        raise RuntimeError('8B requires a Colab GPU runtime. No CPU fallback or package install was attempted.') from error
    owner = prefix/'owner.json'
    identity = dict(purpose='8b_nllb_only', python='3.10.18', torch='2.7.1+cu126', requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest())
    if prefix.exists() and (not owner.is_file() or json.loads(owner.read_text(encoding='utf-8')) != identity):
        raise RuntimeError('8B runtime prefix is not an owned matching environment; preserved without overwrite')
    prefix.mkdir(parents=True, exist_ok=True)
    if not owner.is_file():
        runpy.run_path(str(ROOT/'src/direct_s2st/io.py'))['atomic_write_json'](owner,identity,resume=True)
    bootstrap = None
    for candidate in (prefix/'uv',*sorted(prefix.glob('uv-*'))):
        if not (candidate/'uv').is_dir() or candidate.is_symlink():
            continue
        check_env=dict(env,PYTHONPATH=str(candidate))
        try:
            check=subprocess.run([host_python,'-B','-c',"import importlib.metadata as m; import uv; assert m.version('uv')=='0.8.22'"],
                capture_output=True,text=True,encoding='utf-8',env=check_env,timeout=30)
            if check.returncode==0:
                bootstrap=candidate
                break
        except (OSError,subprocess.TimeoutExpired):
            pass
    if bootstrap is None:
        bootstrap=prefix/('uv' if not (prefix/'uv').exists() else 'uv-'+uuid.uuid4().hex)
        logged([host_python,'-B','-m','pip','--isolated','install','--no-cache-dir','--no-deps','--only-binary=:all:',
                '--index-url','https://pypi.org/simple','--target',bootstrap,'uv==0.8.22'], env)
    bootstrap_env = dict(env, PYTHONPATH=str(bootstrap), UV_PYTHON_INSTALL_DIR=str(prefix/'python'),UV_CACHE_DIR=str(prefix/'uv-cache'))
    logged([host_python,'-B','-c', "import importlib.metadata as m; assert m.version('uv')=='0.8.22'"], bootstrap_env)
    if not dedicated.is_file():
        logged([host_python,'-B','-m','uv','venv','--seed','--python','3.10.18',prefix/'venv'], bootstrap_env)
    check=subprocess.run([str(dedicated),'-B','-c','import platform; print(platform.python_version())'],capture_output=True,
        text=True,encoding='utf-8',errors='replace',env=env,timeout=30)
    if check.returncode or check.stdout.strip()!='3.10.18':
        raise RuntimeError('8B dedicated Python check failed: '+(check.stderr or check.stdout)+'; existing environment preserved')
    logged([dedicated,'-B','-m','pip','--isolated','install','--no-cache-dir','--index-url','https://pypi.org/simple',
            'pip==24.0','setuptools==80.9.0','wheel==0.45.1'],env)
    logged([dedicated,'-B','-m','pip','--isolated','install','--no-cache-dir','--only-binary=:all:',
            'torch==2.7.1','--index-url','https://download.pytorch.org/whl/cu126'],env)
    logged([dedicated,'-B','-m','pip','--isolated','install','--no-cache-dir','--only-binary=:all:',
            '--index-url','https://pypi.org/simple','--no-deps','-r',requirements],env)
    logged([dedicated,'-B','-m','pip','--isolated','check'],env)
    return str(dedicated), env, probe(dedicated, env)
