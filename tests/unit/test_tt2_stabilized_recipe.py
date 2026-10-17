"""Physical duration units, versioned fresh runs, and legacy resume contracts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchaudio')
from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
from direct_s2st.translatotron2.recipe import (LEGACY, STABILIZED, recipe_model, validate_recipe)
from direct_s2st.translatotron2.engine import (optimization_step, save_checkpoint, load_checkpoint,
                                             capture_rank_state, restore_rank_state)

ROOT = Path(__file__).resolve().parents[2]
SPEC = dict(sample_rate=16000, hop_length=160, n_mels=80)


@pytest.fixture
def batch():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return dict(source=torch.randn(2, 24, 80), source_lengths=torch.tensor([24, 17]),
                phones=torch.tensor([[3, 4, 5, 2], [4, 3, 2, 0]]), phone_lengths=torch.tensor([4, 3]),
                target=torch.randn(2, 32, 80), target_lengths=torch.tensor([32, 29]))


def test_physical_seconds_initialization_and_frame_api(batch):
    config = recipe_model(ModelConfig.smoke(), STABILIZED, SPEC)
    model = Translatotron2(config, 7).eval()
    output = model(**batch)
    valid = torch.tensor([[8., 8., 8.], [8., 8., 0.]])
    torch.testing.assert_close(output['durations'], valid)
    expected = (((valid.sum(1) - batch['target_lengths']) * .01).square()).mean()
    torch.testing.assert_close(output['duration_loss'], expected)
    torch.testing.assert_close(output['loss'], output['mel_loss'] + 10*output['phone_loss'] + expected)
    conditioning = torch.randn(2, 3, config.linguistic_dim + config.context_dim)
    generated = model.synthesize(conditioning, torch.tensor([3, 2]), max_frames=24)
    assert generated['frame_lengths'].tolist() == [24, 16]
    with pytest.raises(ValueError, match='max_frames'):
        model.synthesize(conditioning, torch.tensor([3, 2]), max_frames=23)


def test_seconds_loss_and_gradient_scale_without_changing_upsampling_or_ranges(batch):
    legacy = Translatotron2(ModelConfig.smoke(), 7).eval()
    config = replace(legacy.config, duration_unit='seconds', duration_output_scale=.01)
    seconds = Translatotron2(config, 7).eval()
    seconds.load_state_dict(legacy.state_dict())
    a, b = legacy(**batch), seconds(**batch)
    for key in ('durations', 'mel', 'post_mel', 'upsampling', 'phone_loss', 'mel_loss'):
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b['duration_loss'], a['duration_loss'] * 1e-4)
    a['duration_loss'].backward()
    b['duration_loss'].backward()
    for (name, p), (_, q) in zip(legacy.named_parameters(), seconds.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad * 1e-4, rtol=2e-4, atol=1e-7, msg=name)


def test_initialization_does_not_change_range_or_other_weights():
    torch.manual_seed(4)
    old = Translatotron2(ModelConfig.smoke(), 7)
    torch.manual_seed(4)
    new = Translatotron2(recipe_model(ModelConfig.smoke(), STABILIZED, SPEC), 7)
    for key, value in old.state_dict().items():
        if not key.startswith('duration.projection.'):
            assert torch.equal(value, new.state_dict()[key]), key


def test_target_hop_is_derived_not_assumed_and_schedule_has_no_boundary_jump():
    from direct_s2st.translatotron2.batching import learning_rate
    config = recipe_model(ModelConfig.smoke(), STABILIZED, dict(sample_rate=24000, hop_length=300))
    assert config.duration_hop_seconds == .0125
    model = Translatotron2(config, 7).eval()
    conditioning = torch.randn(1, 5, config.linguistic_dim + config.context_dim)
    assert model.synthesize(conditioning, torch.tensor([5]))['frame_lengths'].tolist() == [32]
    assert learning_rate(.0003, 400, 400) == .0003
    assert .000299 < learning_rate(.0003, 401, 400) < .0003


def test_new_recipe_exact_optimizer_rng_resume_and_legacy_compatibility(batch, tmp_path):
    for recipe in (LEGACY, STABILIZED):
        model = Translatotron2(recipe_model(ModelConfig.smoke(), recipe, SPEC), 7)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003, weight_decay=1e-6)
        optimization_step(model, optimizer, batch)
        path = tmp_path / (recipe + '.pt')
        identity = dict(fixture=True, recipe=recipe)
        monitor = dict(fixture='preserved monitor and policy history')
        save_checkpoint(path, model, optimizer, 1, list('abcdefg'), identity,
                        rank_states=[capture_rank_state(model)], monitor_state=monitor)
        if recipe == LEGACY:
            # Actual old serialized shape: duration fields were not present.
            payload = torch.load(path, weights_only=True)
            for key in list(payload['model_config']):
                if key.startswith('duration_') and key != 'duration_dim':
                    del payload['model_config'][key]
            torch.save(payload, path)
        before = path.read_bytes()
        expected = optimization_step(model, optimizer, batch)
        restored, state = load_checkpoint(path, expected_fingerprint=identity, restore_rng=True)
        assert state['monitor_state'] == monitor
        assert restored.config == model.config
        opt = torch.optim.Adam(restored.parameters())
        opt.load_state_dict(state['optimizer'])
        restore_rank_state(restored, state['rank_states'][0])
        assert optimization_step(restored, opt, batch) == expected
        for key, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key]), key
        assert before == path.read_bytes()
        with pytest.raises(ValueError, match='fingerprint'):
            load_checkpoint(path, expected_fingerprint=dict(identity, recipe='other'))


@pytest.mark.parametrize('spec', [{}, dict(sample_rate=0, hop_length=160),
                                 dict(sample_rate=16000, hop_length=-1)])
def test_new_recipe_rejects_invalid_hop(spec):
    with pytest.raises(ValueError, match='hop_length/sample_rate'):
        recipe_model(ModelConfig.smoke(), STABILIZED, spec)


def test_new_recipe_is_explicit_and_keeps_paper_exact_unchanged(tmp_path):
    from direct_s2st.colab import make_config
    from direct_s2st.config import load_config
    from direct_s2st.recipes import option
    from direct_s2st.batch_calibration import selected_config
    validate_recipe(LEGACY, 'paper_exact')
    with pytest.raises(ValueError, match='requires paper_practical'):
        validate_recipe(STABILIZED, 'paper_exact')
    root = tmp_path/'data/translatotron2/fairseq'
    root.mkdir(parents=True)
    target = load_config(ROOT/'configs/translatotron2/prepare.yaml')['mel']
    (root/'mel-spec.json').write_text(json.dumps(target))
    (root/'data-lock.json').write_text('{"fixture":true}')
    env = tmp_path/'environment.json'
    env.write_text('{"repository":"fixture"}')
    exact = make_config(ROOT, tmp_path/'data', env, model_size='fisher', performance='gpu80')
    cfg = make_config(ROOT, tmp_path/'data', env, model_size='fisher', performance='gpu80',
                      reproduction_mode='paper_practical', tt2_recipe=STABILIZED, optimize=True,
                      calibrate_batch=True)
    cmd = cfg['command']
    assert option(cmd, '--tt2-recipe') == STABILIZED
    assert option(cmd, '--learning-rate') == '0.0003'
    assert option(cmd, '--warmup-updates') == '400'
    assert cfg['research_metadata']['duration_unit'] == 'seconds'
    assert cfg['research_metadata']['effective_batch_size'] == 1024
    for size in (1, 16, 32, 1024):
        chosen = selected_config(cfg, size)
        assert int(option(chosen['command'], '--update-freq')) * size == 1024
        assert option(chosen['command'], '--tt2-recipe') == STABILIZED
    assert option(exact['command'], '--learning-rate') == '0.0042'
    assert option(exact['command'], '--warmup-updates') == '10000'
    assert '--tt2-recipe' not in exact['command']
    with pytest.raises(ValueError, match='1024'):
        make_config(ROOT, tmp_path/'data', env, model_size='fisher', reproduction_mode='paper_practical',
                    tt2_recipe=STABILIZED, batch_size=8, update_freq=1)


@pytest.mark.parametrize('flag,value', [('--learning-rate', '.0042'), ('--warmup-updates', '500'),
                                       ('--update-freq', '1')])
def test_trainer_rejects_recipe_drift_before_data_access(flag, value, monkeypatch, tmp_path):
    from direct_s2st.translatotron2 import train
    monkeypatch.setattr(sys, 'argv', ['tt2', '--data-root', str(tmp_path/'missing'),
        '--run-root', str(tmp_path/'run'), '--model-size', 'fisher', '--reproduction-mode',
        'paper_practical', '--tt2-recipe', STABILIZED, '--batch-size', '8', flag, value])
    with pytest.raises((SystemExit, ValueError)):
        train._main()
    assert not (tmp_path/'run').exists()


def test_cell5_keeps_legacy_and_stabilized_runs_separate(tmp_path):
    sys.path.insert(0, str(ROOT/'scripts/colab'))
    import training_session as session
    from test_cell5_recovery import snapshot
    _, old_config, old_marker = snapshot(tmp_path, 'tt2-old')
    original = old_config.read_bytes(), old_marker.read_bytes()
    assert session.choose_run(tmp_path, 'tt2', 'tt2-stable', tt2_recipe=STABILIZED) == ('tt2-stable', None)
    with pytest.raises(ValueError, match='recipe differs'):
        session.choose_run(tmp_path, 'tt2', 'tt2-stable', 'tt2-old', tt2_recipe=STABILIZED)
    cfg, path, marker = snapshot(tmp_path, 'tt2-stable')
    cfg['command'] = ['--tt2-recipe', STABILIZED]
    path.write_text(json.dumps(cfg))
    assert session.choose_run(tmp_path, 'tt2', 'tt2-stable', tt2_recipe=STABILIZED)[0] == 'tt2-stable'
    assert session.choose_run(tmp_path, 'tt2', 'tt2-old')[0] == 'tt2-old'
    assert original == (old_config.read_bytes(), old_marker.read_bytes())


def test_notebook_exposes_new_separate_recipe_with_no_s2ut_change():
    notebook = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    source = ''.join(next(c['source'] for c in notebook['cells']
                          if c['cell_type'] == 'code' and 'TT2_RECIPE =' in ''.join(c['source'])))
    # Evaluate only settings/functions before any Colab workspace operations.
    namespace = {}
    exec(source.split('def ensure_workspace():')[0], namespace)
    old_tt2, s2ut = namespace['training_run_name']('tt2'), namespace['training_run_name']('s2ut')
    namespace['TT2_RECIPE'] = STABILIZED
    assert namespace['training_run_name']('tt2') != old_tt2
    assert namespace['training_run_name']('s2ut') == s2ut
    assert namespace['training_mode']('tt2') == 'paper_practical'


def test_normal_cell5_stabilized_launch_and_resume_preserve_config(tmp_path, monkeypatch):
    import os
    from direct_s2st.recipes import option
    sys.path.insert(0, str(ROOT/'scripts/colab'))
    import training_session as session
    import tt2_schedule_overlay
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setenv('S2ST_TT2_PRACTICAL_WARMUP', '1000')
    monkeypatch.setattr(session, 'WORK_ROOT', tmp_path/'work')
    monkeypatch.setattr(session, 'RUNTIME_ROOT', tmp_path/'runtime')
    session.RUNTIME_ROOT.mkdir()
    env = session.RUNTIME_ROOT/'environment.json'
    env.write_text('{"packages":"fixture==1"}')
    import hashlib
    environment_sha = hashlib.sha256(env.read_bytes()).hexdigest()
    code = 'b'*40
    monkeypatch.setattr(session, 'git', lambda *a: code)
    monkeypatch.setattr(session, 'checkout', lambda *a: tmp_path/'execution')
    restored = []
    def ensure(*a, **kw):
        restored.append(kw['expected'])
        return tmp_path/'python'
    monkeypatch.setattr(session, 'ensure_environment', ensure)
    monkeypatch.setattr(tt2_schedule_overlay, 'prepare', lambda *a: pytest.fail('no legacy schedule overlay'))
    name = 'tt2-training-fixture-paper_practical-stabilized-seconds-v1'
    cfg = dict(kind='tt2', command=['--tt2-recipe', STABILIZED, '--learning-rate', '0.0003',
        '--warmup-updates', '400', '--batch-size', '16', '--update-freq', '64'],
        research_metadata=dict(repository_revision=code, tt2_recipe=STABILIZED),
        identity_files=['/content/s2st-runtime/environment.json'])
    def build(command, **kw):
        options = json.loads(command[-1])
        assert options['tt2_recipe'] == STABILIZED
        assert 'tt2_warmup_updates' not in options
        return json.dumps(cfg)
    monkeypatch.setattr(session.subprocess, 'check_output', build)
    options = dict(reproduction_mode='paper_practical', tt2_recipe=STABILIZED, tt2_warmup_updates=400)
    calls = []
    session.launch(tmp_path/'repo', tmp_path, tmp_path/'data', 'tt2', name, 'fisher',
                   options, 1000, 0, confirm=True, run=lambda *a: calls.append(a))
    assert '--auto-resume' in calls[-1]
    assert 'S2ST_TT2_PRACTICAL_WARMUP' not in os.environ
    path = tmp_path/(name+'.json')
    marker = tmp_path/'checkpoints'/name/'000000000010-fixture/snapshot.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(dict(updates=10, identity=dict(config=cfg,
        work=str(session.WORK_ROOT/name), inputs={'/content/s2st-runtime/environment.json':environment_sha}))))
    unchanged = path.read_bytes(), marker.read_bytes()
    session.launch(tmp_path/'repo', tmp_path, tmp_path/'data', 'tt2', name, 'fisher',
                   options, 2000, 0, confirm=True, run=lambda *a: calls.append(a))
    assert '--resume' in calls[-1]
    assert option(calls[-1], '--total-updates') == '1000'
    assert restored == [None, environment_sha]
    assert unchanged == (path.read_bytes(), marker.read_bytes())
    with pytest.raises(ValueError, match='fixed warmup'):
        session.launch(tmp_path/'repo', tmp_path, tmp_path/'data', 'tt2', name, 'fisher',
                       dict(options, tt2_warmup_updates=500), 1000, 0, confirm=True, run=lambda *a: calls.append(a))
    assert restored == [None, environment_sha]  # Reject before setup/staging.


def test_native_cli_stabilized_real_updates_and_resume_with_logical_batch1024(tmp_path, monkeypatch, capsys):
    # Tiny synthetic architecture substitutes only model dimensions/data to
    # exercise the actual CLI, streaming accumulation, Adam, hashes and resume.
    from direct_s2st.translatotron2 import train
    import subprocess
    torch.set_num_threads(1)
    torch.manual_seed(7)
    sample = dict(source=torch.randn(1, 8, 80), source_lengths=torch.tensor([8]),
        phones=torch.tensor([[3, 2]]), phone_lengths=torch.tensor([2]),
        target=torch.randn(1, 4, 80), target_lengths=torch.tensor([4]))
    class Dataset:
        spec = SPEC
        tokens = list('abcdefg')
        rows = [dict(id='fixture', tgt_n_frames=4)]
        labels = {'fixture': 'c'}
        def __init__(self, root, split): assert split == 'train'
        def __len__(self): return 1
        def __getitem__(self, index): return sample
    monkeypatch.setattr(train, 'PreparedDataset', Dataset)
    monkeypatch.setattr(train, 'fingerprint', lambda root: dict(fixture='synthetic'))
    monkeypatch.setattr(ModelConfig, 'fisher', classmethod(lambda cls: cls.smoke()))
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'b'*40)
    for key in ('S2ST_TT2_MONITOR_POLICY', 'S2ST_TT2_PRACTICAL_WARMUP', 'S2ST_BATCH_PROBE'):
        monkeypatch.delenv(key, raising=False)
    for key in ('S2ST_TRAIN_OPTIMIZE', 'S2ST_STREAM_INPUTS', 'S2ST_TRAIN_ADAPTIVE_BATCH'):
        monkeypatch.setenv(key, '0')
    monkeypatch.setenv('S2ST_TRAIN_PRECISION', 'default')
    args = ['tt2', '--data-root', str(tmp_path/'data'), '--run-root', str(tmp_path/'run'),
        '--model-size', 'fisher', '--reproduction-mode', 'paper_practical', '--tt2-recipe', STABILIZED,
        '--batch-size', '16', '--update-freq', '64', '--max-updates', '1']
    monkeypatch.setattr(sys, 'argv', args)
    train.main()
    path = tmp_path/'run/checkpoints/checkpoint_last.pt'
    _, state = load_checkpoint(path)
    assert state['updates'] == 1
    assert state['model_config']['duration_unit'] == 'seconds'
    assert state['data_fingerprint']['training']['tt2_recipe'] == STABILIZED
    monkeypatch.setattr(sys, 'argv', args[:-1] + ['2', '--restore-file', str(path)])
    train.main()
    _, state = load_checkpoint(path)
    assert state['updates'] == 2
    assert state['optimizer']['param_groups'][0]['lr'] == .0003 * 2 / 400
    meta = json.loads((tmp_path/'run/research-metadata.json').read_text(encoding='utf-8'))
    assert meta['effective_batch_size'] == 1024
    assert meta['tt2_recipe'] == STABILIZED
    assert meta['warmup_updates'] == 400
    assert 'schedule_transition' not in state
    assert 'legacy_checkpoint_migration=False' in capsys.readouterr().err
