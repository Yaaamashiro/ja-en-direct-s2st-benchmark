"""Native four-system inference/evaluation plans; subprocesses release GPU memory."""
import json
from .progress import operation
from pathlib import Path
import os
import shutil
import sys
import tempfile
import yaml
from .config import load_config
from .hashing import sha256_file
from .io import atomic_write_text
from .predictions import SYSTEMS
from .runs import validate_run_id


@operation('direct_s2st/colab_compare: stage_file')
def stage_file(source, target):
    """Copy a trusted restored artifact to Drive without replacing differing data."""
    source, target = Path(source), Path(target)
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and target.resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('outputs must be outside CORPUS_ROOT')
    if not source.is_file():
        raise FileNotFoundError(f'restore/provide the trained artifact first: {source}')
    expected = sha256_file(source)
    if target.exists():
        if sha256_file(target) != expected:
            raise ValueError(f'artifact differs; use a new experiment: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=target.parent, suffix='.partial')
    os.close(descriptor)
    try:
        shutil.copyfile(source, name)
        if sha256_file(Path(name)) != expected:
            raise OSError('artifact copy checksum mismatch')
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)


def stage_direct_artifacts(runs_root, experiment, trained_runs, vocoder_roots):
    validate_run_id(experiment)
    for system, kind in [('s2ut', 'unit'), ('translatotron2', 'mel')]:
        output = Path(runs_root) / f'{experiment}-{system}'
        source = Path(trained_runs[system])
        stage_file(source/'checkpoints/checkpoint_last.pt', output/'checkpoints/checkpoint_last.pt')
        if system == 's2ut':
            stage_file(source/'gradient-audit-rank-0.json', output/'gradient-audit-rank-0.json')
        for name in ('generator.pt', 'config.json'):
            stage_file(Path(vocoder_roots[kind])/name, output/f'vocoder-{kind}'/name)


def build_configs(repository, runs_root, experiment):
    validate_run_id(experiment)
    repository, runs_root = Path(repository), Path(runs_root)
    configs = {}
    for system in SYSTEMS:
        run_id = f'{experiment}-{system}'
        cfg = load_config(repository/'configs'/system/('infer.yaml' if system in SYSTEMS[:2] else 'default.yaml'))
        cfg['run_id'] = run_id
        if system in SYSTEMS[:2]:
            command = cfg['inference']['command']
            command[0] = sys.executable
            if '--device' in command:
                command[command.index('--device')+1] = 'cuda'
        configs[system] = cfg
        evaluation = load_config(repository/'configs/evaluation/default.yaml')
        evaluation['run_id'] = run_id
        configs[system+'-eval'] = evaluation
    for system, kind in [('s2ut', 'unit'), ('translatotron2', 'mel')]:
        cfg = load_config(repository/f'configs/vocoder/{kind}.yaml')
        cfg['run_id'] = f'{experiment}-{system}'
        command = cfg['infer']['command']
        command[0] = sys.executable
        command[command.index('--checkpoint')+1] = '{run_root}/vocoder-'+kind+'/generator.pt'
        command[command.index('--config')+1] = '{run_root}/vocoder-'+kind+'/config.json'
        spec = runs_root/cfg['run_id']/f'vocoder-{kind}/config.json'
        if spec.is_file():
            rate = json.loads(spec.read_text(encoding='utf-8'))['sampling_rate']
            command[command.index('--sample-rate')+1] = str(rate)
        configs[kind] = cfg
    configs['comparison'] = dict(run_ids=[f'{experiment}-{system}' for system in SYSTEMS])
    return configs


def save_configs(configs, root):
    root = Path(root)
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and root.resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('outputs must be outside CORPUS_ROOT')
    for name, config in configs.items():
        atomic_write_text(root/(name+'.yaml'), yaml.safe_dump(config, allow_unicode=True), resume=True)
    return root
