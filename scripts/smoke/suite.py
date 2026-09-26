"""Plan/execute/resume all four real-data baselines on a chosen Docker host."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / 'src'))
from direct_s2st.config import load_config
from direct_s2st.io import atomic_write_json, atomic_write_text
from direct_s2st.journal import digest
from direct_s2st.runs import validate_run_id
from direct_s2st.hashing import sha256_file
from direct_s2st.progress import operation, track


def build_suite(name, env_file, *, limit=5, updates=2, docker_context=None, device='cuda', vocoder_mode='train',
                container_config_root='/workspace/configs', tt2_recipe='benchmark'):
    validate_run_id(name)
    if not 1 <= limit <= 100 or not 1 <= updates <= 10:
        raise ValueError('smoke permits 1..100 samples/split and 1..10 updates')
    if device not in ('cpu', 'cuda') or vocoder_mode not in ('train', 'pretrained'):
        raise ValueError('invalid device/vocoder mode')
    if tt2_recipe not in ('benchmark', 'fisher', 'covost2', 'conversational'):
        raise ValueError('invalid TT2 reference recipe')
    configs, stages = {}, []
    configs['_execution'] = {'env_sha256': sha256_file(Path(env_file)) if Path(env_file).is_file() else None}
    prefix = ['docker'] + (['--context', docker_context] if docker_context else [])
    prefix += ['compose', '--env-file', str(Path(env_file).resolve())]
    config_root = container_config_root.rstrip('/') + f'/local/{name}'
    def stage(label, service, args, *, entrypoint=None):
        command = prefix + ['run', '--rm'] + (['--entrypoint', entrypoint] if entrypoint else [])
        command += [service] + args + ([] if entrypoint else ['--profile', 'smoke'])
        stages.append(dict(stage=label, command=command, status='NOT_RUN'))
    def configuration(key, file, run_id):
        cfg = load_config(REPOSITORY / 'configs' / file)
        cfg['run_id'] = run_id
        configs[key] = cfg
        return cfg
    def reference(key):
        return ['--config', f'{config_root}/{key}.yaml']
    stage('common_import', 'common', ['corpus', 'import', '--limit', str(limit), '--resume'])
    stage('common_validate', 'common', ['corpus', 'validate'])
    for system, kind in [('s2ut', 'unit'), ('translatotron2', 'mel')]:
        run_id = f'{name}-{system}'
        paper = system == 'translatotron2' and tt2_recipe != 'benchmark'
        train_file = system+'/train-'+tt2_recipe+'.yaml' if paper else system+'/train.yaml'
        train = configuration(system+'-train', train_file, run_id)
        train['training']['max_updates'] = updates
        train['training']['save_interval_updates'] = 1
        infer = configuration(system+'-infer', system+'/infer.yaml', run_id)
        vocoder = configuration(kind, 'vocoder/'+kind+'.yaml', run_id)
        evaluation = configuration(system+'-eval', 'evaluation/default.yaml', run_id)
        if system == 's2ut':
            stage('unit_artifact', 'fairseq', ['s2ut', 'fetch-artifacts', '--resume'])
            stage('unit_extraction', 'fairseq', ['s2ut', 'extract-units', '--limit', str(limit), '--resume'])
        else:
            stage('phonemes', 'fairseq', ['translatotron2', 'phonemize', '--resume'])
        prepare_options = []
        if paper:
            configuration('tt2-paper-prepare', 'translatotron2/prepare-paper.yaml', run_id)
            prepare_options = reference('tt2-paper-prepare')
            command = vocoder['train']['command']
            command[command.index('--config')+1] = '{config_root}/vocoder/mel-paper-generator.json'
            command = vocoder['infer']['command']
            command[command.index('--sample-rate')+1] = '24000'
        stage(system+'-prepare', 'fairseq', [system, 'prepare', '--resume'] + prepare_options)
        stage(system+'-validate', 'fairseq', [system, 'validate'])
        for cfg, section in [(train, 'training'), (infer, 'inference'), (vocoder, 'train'), (vocoder, 'infer')]:
            command = cfg[section]['command']
            if '--device' in command:
                command[command.index('--device')+1] = device
        if device == 'cpu' and system == 's2ut':
            train['training']['command'] = [x for x in train['training']['command'] if x != '--fp16'] + ['--cpu']
        stage(system+'-train', 'fairseq', [system, 'train'] + reference(system+'-train'))
        stage(system+'-infer', 'fairseq', [system, 'infer'] + reference(system+'-infer'))
        if vocoder_mode == 'train':
            command = vocoder['train']['command']
            command[command.index('--max-updates')+1] = str(updates)
            command = vocoder['infer']['command']
            command[command.index('--checkpoint')+1] = '{run_root}/vocoder-'+kind+'/generator.pt'
            command[command.index('--config')+1] = '{run_root}/vocoder-'+kind+'/config.json'
            stage(kind+'-fit', 'fairseq', ['vocoder', kind, 'train'] + reference(kind))
        stage(kind+'-vocode', 'fairseq', ['vocoder', kind, 'infer'] + reference(kind))
        stage(system+'-evaluate', 'evaluation', ['evaluate', 'run'] + reference(system+'-eval'))
    configuration('cascade', 'cascade/default.yaml', name+'-cascade')
    configuration('cascade-eval', 'evaluation/default.yaml', name+'-cascade')
    stage('cascade', 'cascade', ['cascade', 'run', '--split', 'test'] + reference('cascade'))
    stage('cascade-evaluate', 'evaluation', ['evaluate', 'run'] + reference('cascade-eval'))
    configuration('s2t-tts', 's2t_tts/default.yaml', name+'-s2t_tts')
    configuration('s2t-tts-eval', 'evaluation/default.yaml', name+'-s2t_tts')
    stage('s2t-tts', 'cascade', ['s2t-tts', 'run', '--split', 'test'] + reference('s2t-tts'))
    stage('s2t-tts-evaluate', 'evaluation', ['evaluate', 'run'] + reference('s2t-tts-eval'))
    from direct_s2st.predictions import SYSTEMS
    configs['comparison'] = {'run_ids': [name+'-'+s for s in SYSTEMS]}
    stage('acceptance', 'evaluation', ['evaluate', 'verify'] + reference('comparison'))
    stage('comparison', 'evaluation', ['evaluate', 'aggregate'] + reference('comparison'))
    return configs, stages


@operation('smoke: execute suite')
def execute(configs, stages, root, *, resume=False, runner=subprocess.run):
    root = Path(root)
    report = root / 'suite-execution.json'
    identity = digest({'configs': configs, 'commands': [s['command'] for s in stages]})
    previous = None
    if report.exists():
        if not resume:
            raise FileExistsError('execution exists; use --resume or a new name/data roots')
        previous = json.loads(report.read_text(encoding='utf-8'))
        if previous['identity'] != identity:
            raise ValueError('suite resume plan differs')
    for name, config in configs.items():
        atomic_write_text(root / (name+'.yaml'), yaml.safe_dump(config, allow_unicode=True), resume=True)
    if previous:
        for old, current in zip(previous['stages'], stages):
            current.update(old)
    def persist(status):
        atomic_write_json(report, dict(identity=identity, status=status, stages=stages), overwrite=report.exists())
    persist('RUNNING')
    for stage in track(stages, 'smoke: stages (execute/reuse)'):
        if stage['status'] == 'PASS':
            continue
        retry = stage['status'] in ('FAIL', 'RUNNING')
        command = list(stage['command'])
        if retry and '--resume' not in command:
            # If the failed train never wrote its first checkpoint, do not invent
            # one or destructively restart: report the missing checkpoint clearly.
            if stage['stage'] in ('acceptance', 'comparison'):
                command += ['--overwrite']
            else:
                command += ['--resume']
        stage['status'] = 'RUNNING'
        persist('RUNNING')
        started = time.perf_counter()
        try:
            result = runner(command, cwd=REPOSITORY, check=False)
            stage['returncode'] = result.returncode
            stage['status'] = 'PASS' if result.returncode == 0 else 'FAIL'
        except OSError as error:
            stage.update(status='FAIL', error=str(error))
        stage['seconds'] = time.perf_counter()-started
        persist('RUNNING' if stage['status'] == 'PASS' else 'FAIL')
        if stage['status'] != 'PASS':
            return 1
    persist('PASS')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--name', default='benchmark-smoke')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--max-updates', type=int, default=2)
    parser.add_argument('--docker-context')
    parser.add_argument('--container-config-root', default='/workspace/configs')
    parser.add_argument('--tt2-recipe', choices=['benchmark', 'fisher', 'covost2', 'conversational'], default='benchmark')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--vocoder-mode', choices=['train', 'pretrained'], default='train')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    configs, stages = build_suite(args.name, args.env_file, limit=args.limit, updates=args.max_updates,
                                  docker_context=args.docker_context, device=args.device, vocoder_mode=args.vocoder_mode,
                                  container_config_root=args.container_config_root, tt2_recipe=args.tt2_recipe)
    if not args.execute:
        print(json.dumps({'status': 'PLAN_ONLY', 'stages': stages}, indent=2))
        return 0
    if not args.env_file.is_file() or not shutil.which('docker'):
        raise RuntimeError('an existing env-file and working Docker CLI/context are required')
    return execute(configs, stages, REPOSITORY / 'configs/local' / args.name, resume=args.resume)


if __name__ == '__main__':
    raise SystemExit(main())
