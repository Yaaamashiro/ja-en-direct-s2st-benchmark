"""A pinned validation-only wheel for legacy environments; never alter their freeze."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from direct_s2st.io import atomic_write_bytes, atomic_write_json
from direct_s2st.progress import operation, track

# Official PyPI 0.8.1 CPython3.10/Linux-x86_64 wheel, pinned including its digest.
WHEEL_SHA = '67d143429a49ab552411505f550a0fb4285a1d4336e096804d233ec495ac20fc'
WHEEL_URL = ('https://files.pythonhosted.org/packages/c2/fb/2940d26ebda12efd280ae939436f17ac482930d862df9e774cb8b771ab03/'
             'editdistance-0.8.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')


@operation('validation dependencies: verify immutable native wheel')
def provision(root):
    root = Path(root)/WHEEL_SHA
    wheel = root/'distribution.whl'
    if not wheel.is_file():
        with urllib.request.urlopen(WHEEL_URL, timeout=60) as response:
            data = response.read(2*1024**2)
        if hashlib.sha256(data).hexdigest() != WHEEL_SHA:
            raise ValueError('validation wheel checksum mismatch')
        atomic_write_bytes(wheel, data, resume=True)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA:
        raise ValueError('saved validation wheel is damaged; preserve and inspect it')
    files = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in track(archive.namelist(), 'validation dependencies: verify package files'):
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('unsafe validation wheel path')
            if name.endswith('/') or not name.startswith('editdistance/'):
                continue
            data = archive.read(name)
            target = root/'imports'/name
            atomic_write_bytes(target, data, resume=True)
            if target.read_bytes() != data:
                raise ValueError('validation dependency readback mismatch')
            files[name] = hashlib.sha256(data).hexdigest()
    if 'editdistance/__init__.py' not in files:
        raise ValueError('validation wheel has no importable package')
    # Distribution metadata remains in the wheel, separately audited. Only its
    # unmodified import package is on PYTHONPATH, not a fake venv installation.
    for path in (root/'imports').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.relative_to(root/'imports').as_posix() not in files:
            raise ValueError('unexpected validation dependency file')
    atomic_write_json(root/'manifest.json', dict(version=1, package='editdistance', package_version='0.8.1',
                      url=WHEEL_URL, wheel_sha256=WHEEL_SHA, files=files), resume=True)
    return root/'imports'


@operation('cell 5: validate train and dev runtime before data staging')
def ensure(python, prefix, persistent, run):
    base = Path(prefix)/'environment.json'
    checksum = hashlib.sha256(base.read_bytes()).hexdigest()
    packages = json.loads(base.read_text())['packages']
    sidecar = None
    if 'editdistance==0.8.1' not in packages.splitlines():
        root = Path(prefix)/'validation-dependencies'
        run(sys.executable, Path(__file__), '--root', root)
        sidecar = root/WHEEL_SHA/'imports'
        parts = [str(sidecar)]+os.environ.get('PYTHONPATH', '').split(os.pathsep)
        os.environ['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(p for p in parts if p))
        atomic_write_json(Path(persistent)/'.runtime-extras'/f'editdistance-{checksum[:16]}-{WHEEL_SHA[:16]}.json',
            dict(version=1, base_environment_sha256=checksum, package='editdistance', package_version='0.8.1',
                 wheel_sha256=WHEEL_SHA, import_root=str(sidecar)), resume=True)
    run(python, ROOT/'scripts/colab/check_s2ut_runtime.py')
    freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    if freeze != packages or hashlib.sha256(base.read_bytes()).hexdigest() != checksum:
        raise ValueError('validation dependencies changed the base environment; refusing to bypass its lock')
    return sidecar


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(provision(parser.parse_args().root))
