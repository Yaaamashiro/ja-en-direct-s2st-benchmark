"""Shared, resumable startup calibration. Fixed settings after update zero."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .io import atomic_write_json
from .journal import digest
from .progress import operation, track
from .recipes import option, replace_option, validate_s2ut_paper, validate_s2ut_runtime


def s2ut_exact(config):
    if config['kind'] != 's2ut':
        return False
    metadata_mode = config.get('research_metadata', {}).get('reproduction_mode')
    mode = option(config['command'], '--reproduction-mode', metadata_mode or 'smoke')
    if metadata_mode and metadata_mode != mode:
        raise ValueError('S2UT reproduction mode differs between command and metadata')
    if mode == 'paper_exact':
        validate_s2ut_paper(config['command'])
        runtime = config.get('runtime', {})
        validate_s2ut_runtime(mode, fixed_microbatch=runtime.get('fixed_microbatch', 0),
                             adaptive_batch=runtime.get('adaptive_batch', False),
                             precision=runtime.get('precision', 'default'))
        return True
    return False


def hardware():
    import torch
    if not torch.cuda.is_available():
        raise ValueError('startup VRAM calibration requires CUDA')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('startup calibration supports one GPU/process')
    p = torch.cuda.get_device_properties(0)
    return dict(name=p.name, total_memory=p.total_memory, capability=list(torch.cuda.get_device_capability()),
                torch=torch.__version__, cuda=torch.version.cuda)


def candidates(config):
    if s2ut_exact(config):
        # Zero means no cap: test the official batch, never search/split it.
        return [0]
    cap = config['batch_calibration']['max_batch']
    if config['kind'] == 'tt2':
        logical = int(option(config['command'], '--batch-size')) * int(option(config['command'], '--update-freq'))
        return [n for n in range(1, min(logical, cap) + 1) if logical % n == 0]
    values = [2**n for n in range(cap.bit_length()) if 2**n <= cap]
    return values if values[-1] == cap else values + [cap]


def selected_config(config, size):
    cfg = copy.deepcopy(config)
    command = cfg['command']
    exact = s2ut_exact(cfg)
    if exact:
        if size != 0:
            raise ValueError('S2UT paper_exact only checks official batches without a physical cap; use a new paper_practical run')
        cfg['runtime']['fixed_microbatch'] = 0
    elif size < 1:
        raise ValueError('selected physical batch must be positive')
    elif cfg['kind'] == 's2ut':
        # Keep official token-budget batches, update_freq and iterator order.
        cfg['runtime']['fixed_microbatch'] = size
    else:
        if cfg['kind'] == 'tt2':
            logical = int(option(command, '--batch-size')) * int(option(command, '--update-freq'))
            if logical % size:
                raise ValueError('selected physical batch must divide the logical batch')
            replace_option(command, '--update-freq', logical // size)
        replace_option(command, '--batch-size', size)
    metadata = cfg.get('research_metadata')
    if metadata:
        metadata['batch_policy'] = ('official Fisher token-budget batches; startup VRAM fit check only' if exact
                                    else 'startup-calibrated then fixed; microbatch-local statistics')
        if cfg['kind'] == 'tt2':
            metadata['physical_batch_size'] = size
            metadata['update_freq'] = int(option(command, '--update-freq'))
        else:
            metadata['physical_batch_cap'] = None if exact else size
        if cfg['kind'] == 's2ut' and not exact:
            deviation = 'additional physical microbatch splitting differs from official Fisher batch execution'
            if deviation not in metadata.setdefault('deviations', []):
                metadata['deviations'].append(deviation)
    cfg['runtime']['adaptive_batch'] = False
    return cfg


@operation('training: measure physical batch and VRAM')
def trial(config, size, parent):
    """Fresh child process: real forward/backward/optimizer, no training backup."""
    settings = config['batch_calibration']
    cfg = selected_config(config, size)
    with tempfile.TemporaryDirectory(prefix='.batch-probe-', dir=parent) as directory:
        directory = Path(directory)
        result_path = directory / 'measurement.json'
        mapping = dict(python=sys.executable, run_root=str(directory / 'run'),
                       updates=str(settings['steps']), checkpoint='UNUSED')
        command = [v.format(**mapping) for v in cfg['command']]
        expected = {'tt2': ('direct_s2st.translatotron2.train', '--run-root'),
                    's2ut': ('direct_s2st.s2ut.fairseq_train', '--save-dir'),
                    'vocoder': ('direct_s2st.vocoders.train', '--output-root')}
        module, output_flag = expected[cfg['kind']]
        output = option(command, output_flag)
        if (command[:3] != [sys.executable, '-m', module] or output is None
                or not Path(output).resolve().is_relative_to(directory.resolve())
                or '--restore-file' in command or '--resume' in command):
            raise ValueError('calibration requires a supported fresh trainer with private output paths')
        if '--validate-interval' in command:
            replace_option(command, '--validate-interval', 0)
        if '--save-interval-updates' in command:
            replace_option(command, '--save-interval-updates', settings['steps'] + 1)
        if cfg['kind'] == 's2ut':
            command.append('--disable-validation')
        from .vocoders.verification import preflight_command
        command = preflight_command(command, directory)
        env = dict(os.environ, S2ST_BATCH_PROBE=str(result_path),
                   S2ST_BATCH_PROBE_STEPS=str(settings['steps']),
                   S2ST_TRAIN_OPTIMIZE='1', S2ST_TRAIN_ADAPTIVE_BATCH='0',
                   S2ST_TRAIN_FIXED_MICROBATCH=str(cfg['runtime'].get('fixed_microbatch', 0)),
                   S2ST_TRAIN_PRECISION=cfg['runtime']['precision'],
                   S2ST_TRAIN_CACHE=str(directory / 'cache'),
                   S2ST_TRAIN_CACHE_GB=str(cfg['runtime']['cache_gb']),
                   S2ST_TRAIN_READERS=str(cfg['runtime']['readers']))
        for key in ('S2ST_TRAIN_STAGING', 'S2ST_TRAIN_DEADLINE'):
            env.pop(key, None)
        label = 'official_token_batches' if s2ut_exact(cfg) else str(size)
        print(f'[calibration] kind={cfg["kind"]} trial_batch={label} fresh_weights=true', file=sys.stderr, flush=True)
        try:
            result = subprocess.run([sys.executable, '-m', 'direct_s2st.batch_probe', *command[1:]],
                                    env=env, check=False, timeout=settings['trial_seconds'])
        except subprocess.TimeoutExpired:
            raise RuntimeError('VRAM trial timed out; completed measurements retained. Increase trial_seconds explicitly')
        if result.returncode or not result_path.is_file():
            raise RuntimeError('VRAM trial failed for a non-OOM reason; inspect trainer logs')
        return json.loads(result_path.read_text(encoding='utf-8'))


def safe(result, reserve):
    if result.get('status') == 'oom':
        return False
    if result.get('status') != 'ok' or not result.get('steps'):
        raise ValueError('invalid calibration measurement')
    return all(s['seconds'] > 0 and s['samples'] > 0 and 0 < s['used'] <= (1-reserve)*s['total']
               for s in result['steps'])


def throughput(result):
    # Discard optimizer-allocation/first-step warmup from speed scoring, but
    # retain ALL steps for the safety check. Stress workload, not a speed claim.
    steps = result['steps'][1:] or result['steps']
    return sum(s['samples'] for s in steps) / sum(s['seconds'] for s in steps)


@operation('training: select train-only stress batches')
def stress_fairseq_iterator(iterator, steps, frequency):
    batches = list(iterator.frozen_batches)
    if not batches:
        raise ValueError('no legal train batches for calibration')
    sizes = [iterator.dataset.size(i) for i in range(len(iterator.dataset))]
    def attention_cost(batch):
        source = max(1, max(sizes[i][0] for i in batch)//4)
        target = max(sizes[i][1] for i in batch) + 1
        return len(batch)*(source**2 + target**2 + source*target)
    scores = [attention_cost, lambda b: max(max(sizes[i]) for i in b), lambda b: len(b)]
    selected = []
    for step in track(range(steps), 'training: stress batch groups'):
        batch = max(batches, key=scores[step % len(scores)])
        selected.extend([batch]*frequency)
    iterator._frozen_batches = tuple(selected)
    iterator.batch_sampler = tuple(selected)
    iterator.disable_shuffling = True


@operation('training: calibrate once and lock batch')
def calibrate(config, inputs, record_path, parent, *, measure=trial, get_hardware=hardware, inspect_only=False):
    """Record every completed trial durably. Reuse selection, never retune a run."""
    record_path, parent = Path(record_path), Path(parent)
    exact = s2ut_exact(config)
    objective = config['batch_calibration'].get('objective', 'capacity')
    if objective not in ('capacity', 'throughput'):
        raise ValueError('unknown calibration objective')
    request = digest(dict(config=config, inputs=inputs,
                          algorithm='s2ut-official-fit-v1' if exact else
                          'startup-throughput-v2' if objective == 'throughput' else 'startup-batch-v1'))
    record = json.loads(record_path.read_text(encoding='utf-8')) if record_path.is_file() else None
    if record and record['request'] != request:
        raise ValueError('calibration configuration/data changed; use a new run name')
    if inspect_only:
        if not record or record.get('selected') is None:
            raise ValueError('no saved batch selection; cannot restore this run')
        return selected_config(config, record['selected'])
    hw = get_hardware()
    parent.mkdir(parents=True, exist_ok=True)
    reserve = config['batch_calibration']['reserve_ratio']
    if record and record.get('selected') is not None:
        size = record['selected']
        if hw not in record['verified_hardware']:
            # New GPU: validate the SAME batch, do not change checkpoint recipe.
            result = measure(config, size, parent)
            if not safe(result, reserve):
                raise ValueError('saved batch is unsafe on this GPU; use the previous GPU or a new run, not a silent batch change')
            record['hardware_checks'].append(dict(hardware=hw, batch=size, result=result))
            record['verified_hardware'].append(hw)
            atomic_write_json(record_path, record, overwrite=True)
        label = 'official_token_batches' if exact else str(size)
        print(f'[calibration] reused=true selected_batch={label} fixed_for_training=true', file=sys.stderr, flush=True)
        return selected_config(config, size)
    if record and record['hardware'] != hw:
        raise ValueError('unfinished calibration belongs to another GPU; use that GPU or a new run name')
    if not record:
        record = dict(request=request, hardware=hw, trials=[], verified_hardware=[hw], hardware_checks=[],
                      selected=None, weights_reused=False,
                      policy='official_batch_fit_check' if exact else 'physical_batch_search')
        atomic_write_json(record_path, record)
    known = {r['batch']:r['result'] for r in record['trials']}
    def ensure(size):
        if size not in known:
            known[size] = measure(config, size, parent)
            if known[size].get('status') == 'ok' and len(known[size].get('steps', [])) != config['batch_calibration']['steps']:
                raise ValueError('incomplete calibration step count')
            record['trials'].append(dict(batch=size, result=known[size]))
            atomic_write_json(record_path, record, overwrite=True)
        return safe(known[size], reserve)
    accepted = []
    rejected = None
    for size in track(candidates(config), 'training: startup batch candidates'):
        if not ensure(size):
            rejected = size
            break
        accepted.append(size)
    if not accepted:
        if exact:
            raise RuntimeError('S2UT official max_tokens=20000 batch does not fit with the configured VRAM reserve; '
                               'use a larger GPU or explicitly choose paper_practical in a new run. No batch was changed')
        raise RuntimeError('even batch=1 did not fit with the configured VRAM reserve')
    if rejected is not None and config['kind'] != 'tt2' and not exact:
        def refine():
            low, high = max(accepted), rejected
            while high-low > 1:
                size = (low+high)//2
                yield size
                if safe(known[size], reserve):
                    low = size
                    accepted.append(size)
                else:
                    high = size
        for size in track(refine(), 'training: refine physical batch capacity'):
            ensure(size)
    scores = {str(size): throughput(known[size]) for size in accepted}
    if objective == 'throughput' and not exact:
        best = max(scores.values())
        # Prefer larger batches when speeds differ by <=3% (measurement noise).
        record['selected'] = max(size for size in accepted if scores[str(size)] >= best * .97)
    else:
        record['selected'] = max(accepted)
    record['selection_objective'] = 'official_fit_check' if exact else objective
    record['stress_samples_per_second'] = scores
    atomic_write_json(record_path, record, overwrite=True)
    label = 'official_token_batches' if exact else str(record['selected'])
    print(f'[calibration] selected_batch={label} reserve_ratio={reserve} '
          f'logical_update_preserved={config["kind"] != "vocoder"} '
          'probe_weights_discarded=true', file=sys.stderr, flush=True)
    return selected_config(config, record['selected'])
