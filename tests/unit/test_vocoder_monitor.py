import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from direct_s2st.vocoders.monitor import (
    MelMonitor, UnitMonitor, restore_monitor, select_generator, settings_from_env,
    validation_identity, reject_overlap,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = dict(version=3, interval=1, patience=5, min_relative_improvement=.01)


def bind(monitor, work):
    monitor.save_best(work, lambda p: (p.parent.mkdir(parents=True, exist_ok=True), p.write_bytes(b'best')))


@pytest.mark.parametrize('cls', [MelMonitor, UnitMonitor])
def test_vocoder_guard_independent_anchors_resume_and_five_misses(tmp_path, cls):
    monitor = cls(POLICY, warmup=2)
    for update, mel in [(1, 10.), (2, 11.)]:
        values = {key: mel for key in monitor.metrics}
        if monitor.observe(update, values, values, .0002):
            bind(monitor, tmp_path)
    assert all(value == 10 for value in monitor.anchors.values())
    assert monitor.bad_evaluations == 0
    for update in range(3, 8):
        values = {key: 9.99 for key in monitor.metrics}
        if monitor.observe(update, values, values, .0002):
            bind(monitor, tmp_path)
        assert monitor.bad_evaluations == update-2
        assert monitor.stopped == (update == 7)
        assert cls.restore(None, monitor.state(), update, 2, tmp_path).state() == monitor.state()
    monitor.write_reports(tmp_path)
    assert select_generator(tmp_path) == tmp_path/'checkpoints/checkpoint_best.pt'
    csv = (tmp_path/'training-monitor.csv').read_text()
    assert all(f'dev_{key}' in csv for key in monitor.metrics)
    assert '<svg' in (tmp_path/'training-monitor.svg').read_text()


def test_duration_improvement_resets_unit_counter_not_mel_best(tmp_path):
    monitor = UnitMonitor(POLICY, warmup=0)
    for update, dev in enumerate([
            dict(mel_loss=10., duration_loss=80.),
            dict(mel_loss=10.1, duration_loss=79.9),
            dict(mel_loss=10.1, duration_loss=79.8),
            dict(mel_loss=10.1, duration_loss=79.2)], start=1):
        if monitor.observe(update, dev, dev, .0002):
            bind(monitor, tmp_path)
    assert monitor.anchors == dict(mel_loss=10., duration_loss=79.2)
    assert monitor.bad_evaluations == 0 and not monitor.stopped
    assert monitor.best_update == 1 and monitor.best_value == 10.
    assert monitor.best_updates == dict(mel_loss=1, duration_loss=4)
    assert UnitMonitor.restore(None, monitor.state(), 4, 0, tmp_path).state() == monitor.state()


def test_restoration_never_silently_resets_or_changes_dev_policy(tmp_path, monkeypatch):
    monkeypatch.setenv('S2ST_VOCODER_MONITOR_POLICY', json.dumps(dict(POLICY, warmup=2)))
    monitor = restore_monitor('mel', None, 30, tmp_path, 'dev-hash')
    assert monitor.origin == 30 and not monitor.history
    values = dict(mel_loss=10.)
    assert monitor.observe(31, values, values, .0002)
    bind(monitor, tmp_path)
    state = dict(monitor=monitor.state(), monitor_validation='dev-hash')
    monkeypatch.delenv('S2ST_VOCODER_MONITOR_POLICY')
    restored = restore_monitor('mel', state, 31, tmp_path, 'dev-hash')
    assert restored.state() == monitor.state()
    with pytest.raises(ValueError, match='validation inputs'):
        restore_monitor('mel', state, 31, tmp_path, 'changed')
    corrupt = deepcopy(state)
    corrupt['monitor']['anchors']['mel_loss'] = 999
    with pytest.raises(ValueError, match='counters'):
        restore_monitor('mel', corrupt, 31, tmp_path, 'dev-hash')
    monkeypatch.setenv('S2ST_VOCODER_MONITOR_POLICY', json.dumps(dict(POLICY, patience=8, warmup=2)))
    with pytest.raises(ValueError, match='policy changed'):
        restore_monitor('mel', state, 31, tmp_path, 'dev-hash')


@pytest.mark.parametrize('setting', [dict(POLICY, warmup=-1), dict(POLICY, warmup=True),
    dict(POLICY, interval=0), dict(POLICY, version=2), dict(POLICY, min_relative_improvement=float('nan')), []])
def test_bad_settings_rejected(monkeypatch, setting):
    monkeypatch.setenv('S2ST_VOCODER_MONITOR_POLICY', json.dumps(setting))
    with pytest.raises(ValueError):
        settings_from_env()


def test_dev_verification_receipt_binding_and_no_test_or_overlap(tmp_path, monkeypatch):
    from direct_s2st.vocoders.verification import verify_inputs, load_receipt, save_receipt
    from direct_s2st.hashing import sha256_file
    monkeypatch.delenv('CORPUS_ROOT', raising=False)
    monkeypatch.setenv('S2ST_PREP_WORKERS', '1')
    common, units = tmp_path/'common', tmp_path/'units'
    common.mkdir()
    units.mkdir()
    (units/'unit-lock.json').write_text('{"hubert_layer":6,"kmeans_clusters":100}')
    for split in ('train', 'dev'):
        audio = common/(split+'.wav')
        audio.write_bytes(split.encode())
        row = dict(pair_id=split, split=split, en_audio=str(audio), en_sha256=sha256_file(audio))
        (common/(split+'.jsonl')).write_text(json.dumps(row)+'\n')
        directory = units/split/'original'
        directory.mkdir(parents=True)
        (directory/(split+'.units')).write_text('1 1 2 3')
    train = verify_inputs(common, 'unit', units)
    dev = verify_inputs(common, 'unit', units, split='dev')
    assert list(train['sequences']) == ['train'] and list(dev['sequences']) == ['dev']
    reject_overlap(train, dev)
    receipt = tmp_path/'receipt.json'
    save_receipt(receipt, dev)
    assert load_receipt(receipt, common, 'unit', units, split='dev') == dev
    with pytest.raises(ValueError, match='receipt mismatch'):
        load_receipt(receipt, common, 'unit', units)
    with pytest.raises(ValueError, match='train or dev'):
        verify_inputs(common, 'unit', units, split='test')
    identity = validation_identity(dev, 32)
    moved = deepcopy(dev)
    moved['rows'][0]['en_audio'] = '/new/cache.wav'
    assert validation_identity(moved, 32) == identity
    assert validation_identity(moved, 16) != identity
    moved['audio']['dev'] = 'changed'
    assert validation_identity(moved, 32) != identity
    with pytest.raises(ValueError, match='overlap'):
        reject_overlap(train, train)


@pytest.mark.parametrize('kind', ['mel', 'unit'])
def test_real_validation_is_deterministic_no_grad_and_restores_rng_buffers(tmp_path, monkeypatch, kind):
    torch = pytest.importorskip('torch')
    from direct_s2st.vocoders import train
    class Duration(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x).squeeze(-1)
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.9))
            self.dict = torch.nn.Embedding(100, 4)
            self.dur_predictor = Duration(4, 1)
            self.register_buffer('calls', torch.tensor(0))
        def forward(self, x=None, *, code=None, dur_prediction=False):
            self.calls.add_(1)
            torch.rand(1)  # Deliberate side effect must not affect next update.
            value = (code.float()/100)[:, None] if code is not None else x
            return (value*self.weight).repeat_interleave(4, -1)
    model = Generator()
    model.train()
    model.dict.eval()  # Mixed flags must restore individually.
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    flags = [m.training for m in model.modules()]
    row = dict(pair_id='dev', split='dev', en_audio='unused')
    units = [1, 1, 2, 3, 4, 4, 4, 5] if kind == 'unit' else None
    wave = torch.linspace(.1, .8, 32)[None]
    monkeypatch.setattr(train, 'checked_sample', lambda *a: (wave, units))
    mel = lambda x: x.reshape(x.size(0), -1, 4).mean(-1)[:, None]
    args = dict(spec=dict(n_fft=8, sample_rate=16000), config=dict(upsample_rates=[4]),
        segment_frames=4, mel=mel, device='cpu')
    verified = dict(kind=kind, rows=[row, dict(row, pair_id='dev2')], sequences={})
    first = train.validate_dev(model, verified, **args)
    second = train.validate_dev(model, verified, **args)
    assert first == second and set(first) == set(UnitMonitor.metrics if kind == 'unit' else MelMonitor.metrics)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert flags == [m.training for m in model.modules()]


def test_preflight_private_dev_receipt_is_forwarded(tmp_path):
    from direct_s2st.vocoders.verification import preflight_command
    def run(args, **kwargs):
        receipt = Path(args[-1])
        receipt.write_text('{}')
        receipt.with_suffix('.dev.json').write_text('{}')
    args = preflight_command(['python', '-m', 'direct_s2st.vocoders.train'], tmp_path, runner=run)
    assert args[-4:] == ['--verified-inputs', str(tmp_path/'vocoder-inputs.json'),
                           '--verified-dev-inputs', str(tmp_path/'vocoder-inputs.dev.json')]


def test_notebook_defaults_and_normal_training_hook():
    cells = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))['cells']
    source = '\n'.join(''.join(c.get('source', [])) for c in cells if c['cell_type'] == 'code')
    assert 'TRAIN_BACKUP_MAX_GB = 1024' in source
    for setting in ('VOCODER_MONITOR = True', 'VOCODER_MONITOR_INTERVAL = 10', 'VOCODER_MONITOR_PATIENCE = 5',
                    'VOCODER_MONITOR_START_UPDATES = 400', 'VOCODER_MONITOR_MIN_RELATIVE_IMPROVEMENT = 0.01'):
        assert setting in source
    ast.parse(source)
    assert 'from vocoder_monitor_overlay import prepare' in (ROOT/'scripts/colab/training_session.py').read_text(encoding='utf-8')


def test_reviewed_overlay_changes_only_reviewed_sources(tmp_path):
    spec = importlib.util.spec_from_file_location('vocoder_monitor_overlay', ROOT/'scripts/colab/vocoder_monitor_overlay.py')
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    # Fixture source from an immutable pre-feature commit (not moving HEAD).
    baseline = '6716b700bb80832d4cfc2f5d323f6af4133be868'
    git = ['git', '-c', 'safe.directory='+ROOT.as_posix()]
    if subprocess.run(git+['cat-file', '-e', baseline+'^{commit}'], cwd=ROOT, capture_output=True).returncode:
        pytest.skip('reviewed baseline is unavailable in this shallow/archive checkout')
    execution, persistent = tmp_path/'execution', tmp_path/'persistent'
    names = set(overlay.MODULES) | set(overlay.FUNCTIONS)
    for name in names:
        target = execution/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(git+['show', baseline+':'+name], cwd=ROOT))
    original = {name: (execution/name).read_bytes() for name in names}
    changed = overlay.replacements(execution, ROOT)
    assert 'src/direct_s2st/vocoders/monitor.py' in changed
    old, new = map(ast.parse, [original['src/direct_s2st/vocoders/train.py'], changed['src/direct_s2st/vocoders/train.py']])
    for name in ('gan_step', 'finite_step', 'duration_objective', 'run_lengths'):
        assert ast.dump(next(n for n in old.body if getattr(n, 'name', '') == name)) == ast.dump(
            next(n for n in new.body if getattr(n, 'name', '') == name))
    root = overlay.prepare(execution, ROOT, persistent, 'unit-run')
    assert overlay.prepare(execution, ROOT, persistent, 'unit-run') == root
    assert all((execution/name).read_bytes() == data for name, data in original.items())
    assert not overlay.replacements(root, ROOT)
    (execution/'src/direct_s2st/vocoders/train.py').write_bytes(b'# unexpected\n'+original['src/direct_s2st/vocoders/train.py'])
    with pytest.raises(ValueError, match='unreviewed'):
        overlay.replacements(execution, ROOT)


@pytest.mark.parametrize('kind', ['mel', 'unit'])
def test_trainer_cold_resume_is_bitwise_and_stopped_checkpoint_no_more_steps(tmp_path, monkeypatch, kind):
    """Exercise the real CLI loop/save/freeze/report/stop with tiny torch models.

    The fairseq generators and expensive GAN step are fixtures; actual two AdamW
    states, segmentation RNG, monitor, checkpoints and recovery copies are used.
    """
    torch = pytest.importorskip('torch')
    from direct_s2st.vocoders import train
    from direct_s2st.vocoders.monitor import early_stop_result
    from direct_s2st.hashing import sha256_file
    from direct_s2st.recovery import freeze_files
    class Duration(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x).squeeze(-1)
    class Generator(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.1))
            self.dict = torch.nn.Embedding(100, 4)
            self.dur_predictor = Duration(4, 1)
        def forward(self, x=None, *, code=None, dur_prediction=False):
            value = code[:, None].float()/100 if code is not None else x
            value = torch.nn.functional.dropout(value, .1, training=self.training)
            return (self.weight*value).repeat_interleave(320, -1)
    for name, attribute in [('fairseq.models.text_to_speech.codehifigan', 'CodeGenerator'),
                            ('fairseq.models.text_to_speech.hifigan', 'Generator')]:
        module = ModuleType(name)
        setattr(module, attribute, Generator)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(train, 'MultiPeriodDiscriminator', lambda: torch.nn.Linear(1, 1))
    monkeypatch.setattr(train, 'MultiScaleDiscriminator', lambda: torch.nn.Linear(1, 1))
    class Mel(torch.nn.Module):
        def __init__(self, spec):
            super().__init__()
        def forward(self, x):
            return x.reshape(x.size(0), -1, 320).mean(-1)[:, None]
    monkeypatch.setattr(train, 'LogMel', Mel)
    monkeypatch.setattr(train, 'validate_config', lambda *a, **k: None)
    monkeypatch.setattr(train, 'load_wave', lambda *a: torch.linspace(.1, .8, 2560)[None])
    steps = []
    def step(generator, discriminators, optim_g, optim_d, conditioning, real, mel, *, units):
        steps.append(1)
        generator.train()
        optim_d.zero_grad(set_to_none=True)
        d = sum(p.square().sum() for p in discriminators.parameters())
        train.finite_step(d, optim_d, discriminators.parameters())
        optim_g.zero_grad(set_to_none=True)
        fake = generator(code=conditioning) if units is not None else generator(conditioning)
        loss = (fake-real).square().mean()
        duration = train.duration_objective(generator, units[0]) if units is not None else loss*0
        train.finite_step(loss+duration, optim_g, generator.parameters())
        return dict(generator=float(loss.detach()), discriminator=float(d.detach()), adversarial=0.,
                    feature=0., mel=float(loss.detach()), duration=float(duration.detach()))
    monkeypatch.setattr(train, 'gan_step', step)
    monkeypatch.setattr(train, 'validate_dev', lambda *a, **k: dict(mel_loss=10., **(
        dict(duration_loss=10.) if kind == 'unit' else {})))
    monkeypatch.delenv('CORPUS_ROOT', raising=False)
    for name in ('S2ST_TRAIN_STAGING', 'S2ST_BATCH_PROBE', 'S2ST_TRAIN_DEADLINE', 'S2ST_TRAIN_OPTIMIZE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('S2ST_PREP_WORKERS', '1')
    monkeypatch.setenv('S2ST_VOCODER_MONITOR_POLICY', json.dumps(dict(POLICY, patience=2, warmup=1)))
    common, units = tmp_path/'common', tmp_path/'units'
    common.mkdir()
    units.mkdir()
    (units/'unit-lock.json').write_text('{"hubert_layer":6,"kmeans_clusters":100}')
    for split in ('train', 'dev'):
        audio = common/(split+'.wav')
        audio.write_bytes(split.encode())
        row = dict(pair_id=split, split=split, en_audio=str(audio), en_sha256=sha256_file(audio))
        (common/(split+'.jsonl')).write_text(json.dumps(row)+'\n')
        directory = units/split/'original'
        directory.mkdir(parents=True)
        (directory/(split+'.units')).write_text('1 1 1 2 2 2 2 2')
    config = tmp_path/'generator.json'
    config.write_text(json.dumps(dict(upsample_rates=[320], mel=dict(sample_rate=16000,
        n_fft=8, normalize_volume=False, normalization='none', log_transform='natural_log_clamp_eps'))))
    def run(work, limit, resume=False):
        monkeypatch.setattr(sys, 'argv', ['train', '--kind', kind, '--config', str(config),
            '--common-root', str(common), '--units-root', str(units), '--output-root', str(work),
            '--device', 'cpu', '--max-updates', str(limit), '--segment-frames', '4']+(['--resume'] if resume else []))
        train.main()
        return torch.load(work/'generator.pt', weights_only=True)
    continuous, resumed = tmp_path/'continuous', tmp_path/'resumed'
    a = run(continuous, 20)
    assert a['updates'] == 3 and a['monitor']['stopped']
    run(resumed, 2)
    at_cap = len(steps)
    assert run(resumed, 2, resume=True)['updates'] == 2
    assert len(steps) == at_cap  # Absolute max, not two additional updates.
    b = run(resumed, 20, resume=True)
    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                equal(x, y)
        else:
            assert left == right
    for key in ('generator', 'discriminators', 'optimizer_g', 'optimizer_d', 'rng', 'cuda_rng', 'monitor_validation'):
        equal(a[key], b[key])
    # Best full-checkpoint bytes contain RNG; same initial seed/order => same
    # tensor payloads. Zip serialization names can differ, hence compare state.
    left, right = deepcopy(a['monitor']), deepcopy(b['monitor'])
    left.pop('best_sha256')
    right.pop('best_sha256')
    assert left == right
    steps_before = len(steps)
    run(resumed, 20, resume=True)
    assert len(steps) == steps_before
    result = early_stop_result(resumed, 3)
    assert result['status'] == 'EARLY_STOPPED'
    frozen = tmp_path/'frozen'
    files = freeze_files(resumed, resumed/'generator.pt', frozen)
    assert {'generator.pt', 'checkpoints/checkpoint_best.pt', 'training-monitor.json',
            'training-monitor.csv', 'training-monitor.svg'} <= set(files)
    assert early_stop_result(frozen, 3) == result
    (frozen/'checkpoints/checkpoint_best.pt').write_bytes(b'broken')
    with pytest.raises(ValueError, match='best dev checkpoint'):
        early_stop_result(frozen, 3)
    # Adopt monitoring on an actual old optimizer checkpoint, future-only.
    legacy = tmp_path/'legacy'
    monkeypatch.delenv('S2ST_VOCODER_MONITOR_POLICY')
    old = run(legacy, 2)
    assert 'monitor' not in old
    old_g = deepcopy(old['optimizer_g'])
    monkeypatch.setenv('S2ST_VOCODER_MONITOR_POLICY', json.dumps(dict(POLICY, patience=2, warmup=1)))
    adopted = run(legacy, 20, resume=True)
    assert adopted['updates'] == 5 and adopted['monitor']['origin'] == 2
    assert adopted['monitor']['history'][0]['update'] == 3
    assert adopted['monitor']['stopped']
    assert all(int(s['step']) == 2 for s in old_g['state'].values())
    assert all(int(s['step']) == 5 for s in adopted['optimizer_g']['state'].values())


def test_vocoder_staging_includes_dev_not_test(tmp_path, monkeypatch):
    from direct_s2st.colab import stage_training
    from direct_s2st import drive_staging
    common = tmp_path/'data/common'
    common.mkdir(parents=True)
    for split in ('train', 'dev', 'test'):
        (common/f'{split}.jsonl').write_text('{}\n')
    lock = common/'dataset-lock.json'
    lock.write_text('{}')
    captured = []
    monkeypatch.setattr(drive_staging, 'stage_audio', lambda *a, **kw: captured.append(kw) or {})
    stage_training(dict(kind='vocoder', command=[], identity_files=[str(lock)]), tmp_path/'work/run')
    assert captured[0]['splits'] == ('train', 'dev')
    assert captured[0]['languages'] == ('en',)


@pytest.mark.parametrize('kind', ['unit', 'mel'])
def test_pinned_fairseq_generator_validation_then_backward(kind, monkeypatch):
    """Pinned real generator classes at CPU-test dimensions, not quality evidence."""
    torch = pytest.importorskip('torch')
    from argparse import Namespace
    from direct_s2st.vocoders import train
    directory = ROOT/'third_party/fairseq/fairseq/models/text_to_speech'
    if not (directory/'hifigan.py').is_file():
        pytest.skip('initialize pinned fairseq submodule')
    spec = importlib.util.spec_from_file_location('vocoder_monitor_pinned_hifigan', directory/'hifigan.py')
    hifigan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hifigan)
    config = dict(model_in_dim=4, upsample_initial_channel=16,
        upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4], resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]], num_embeddings=100, embedding_dim=4,
        dur_predictor_params=dict(encoder_embed_dim=4, var_pred_hidden_dim=8,
                                 var_pred_kernel_size=3, var_pred_dropout=.5))
    namespace = dict(torch=torch, nn=torch.nn, Namespace=Namespace, Generator=hifigan.Generator)
    if kind == 'unit':
        spec = importlib.util.spec_from_file_location('vocoder_monitor_pinned_dropout',
            ROOT/'third_party/fairseq/fairseq/modules/fairseq_dropout.py')
        dropout = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dropout)
        namespace['FairseqDropout'] = dropout.FairseqDropout
        # Isolate exact classes from the pinned source to avoid importing the
        # unsupported local Python 3.13 full fairseq package/dependency tree.
        for filename, name in [('fastspeech2.py', 'VariancePredictor'), ('codehifigan.py', 'CodeGenerator')]:
            node = next(n for n in ast.parse((directory/filename).read_text(encoding='utf-8')).body
                        if isinstance(n, ast.ClassDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(directory/filename), 'exec'), namespace)
        factory = namespace['CodeGenerator']
    else:
        factory = hifigan.Generator
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        generator = factory(config).train()
        wave = torch.linspace(.1, .8, 256)[None]
        units = [i//2 % 10 for i in range(64)] if kind == 'unit' else None
        monkeypatch.setattr(train, 'checked_sample', lambda *a: (wave, units))
        def mel(x):
            frames = x.reshape(x.size(0), -1, 4).mean(-1)
            return frames[:, None].expand(-1, 4, -1)
        state = {k: v.clone() for k, v in generator.state_dict().items()}
        rng = torch.get_rng_state()
        values = train.validate_dev(generator, dict(kind=kind, rows=[{}], sequences={}),
            spec=dict(n_fft=8, sample_rate=16000), config=config,
            segment_frames=32, mel=mel, device='cpu')
        assert all(torch.isfinite(torch.tensor(v)) for v in values.values())
        assert all(torch.equal(state[k], v) for k, v in generator.state_dict().items())
        assert torch.equal(rng, torch.get_rng_state()) and generator.training
        conditioning, real = train.segment_batch([(wave, units)], spec=dict(n_fft=8), hop=4,
            segment_frames=32, mel=mel, device='cpu')
        fake = generator(code=conditioning, dur_prediction=False) if kind == 'unit' else generator(conditioning)
        loss = (fake-real).square().mean()
        if kind == 'unit':
            loss = loss + train.duration_objective(generator, units)
        loss.backward()  # Inference-mode weight-norm hooks must not poison autograd.
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in generator.parameters())
    finally:
        torch.set_num_threads(old_threads)
