import importlib.util
from pathlib import Path
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_overlay():
    spec = importlib.util.spec_from_file_location('compat313', ROOT/'scripts/colab/compat313.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlay_is_scoped_and_idempotent(tmp_path):
    source = ROOT/'third_party/fairseq'
    if not (source/'setup.py').is_file():
        pytest.skip('initialize pinned submodule')
    files = ['setup.py', 'fairseq/dataclass/configs.py', 'fairseq/dataclass/initialize.py',
             'fairseq/models/transformer/transformer_config.py', 'fairseq/data/data_utils.py',
             'fairseq/modules/dynamic_crf_layer.py', 'fairseq/model_parallel/megatron/gpt2_data_loader.py']
    target = tmp_path/'fairseq'
    original = {}
    for name in files:
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, path)
        original[name] = (source/name).read_bytes()
    overlay = load_overlay()
    overlay.apply(target)
    first = {name: (target/name).read_bytes() for name in files}
    overlay.apply(target)
    assert first == {name: (target/name).read_bytes() for name in files}
    assert original == {name: (source/name).read_bytes() for name in files}
    assert (target/'fairseq/dataclass/configs.py').read_text().count('field(default_factory=') >= 11
    assert 'field(default_factory=QuantNoiseConfig)' in (target/'fairseq/models/transformer/transformer_config.py').read_text()
    (target/'.git').write_text('gitdir: protected')
    with pytest.raises(ValueError, match='copy'):
        overlay.apply(target)


def test_setup_pins_python_and_dependency_overlay():
    spec = importlib.util.spec_from_file_location('colab_setup', ROOT/'scripts/colab/setup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PYTHON_VERSION == '3.13.7'
    requirements = (ROOT/'requirements/colab313.txt').read_text()
    assert 'numpy==2.1.3' in requirements and 'hydra-core==1.3.2' in requirements
