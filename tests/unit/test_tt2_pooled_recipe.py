"""Evaluation-only train statistics, checkpoint binding and native resume."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from direct_s2st.translatotron2 import postnet_calibration as pooled
from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
from direct_s2st.translatotron2.recipe import POOLED, STABILIZED, LEGACY, recipe_model, validate_recipe
from direct_s2st.translatotron2.engine import (optimization_step, capture_rank_state,
    restore_rank_state, save_checkpoint, load_checkpoint)
from direct_s2st.translatotron2.batching import collate

ROOT = Path(__file__).resolve().parents[2]
SPEC = dict(sample_rate=16000, hop_length=160, n_mels=80)


class Dataset:
    spec, tokens, source_spec = SPEC, list('abcdefg'), SPEC

    def __init__(self, split='train'):
        self.split = split
        self.rows = [dict(id=f'{split}-{i}', tgt_n_frames=12-i) for i in range(6)]
        self.labels = {row['id']: 'c d' for row in self.rows}
        generator = torch.Generator().manual_seed(11)
        self.samples = [dict(source=torch.randn(1, 16-i, 80, generator=generator),
            source_lengths=torch.tensor([16-i]), phones=torch.tensor([[3, 4, 2]]),
            phone_lengths=torch.tensor([3]), target=torch.randn(1, 12-i, 80, generator=generator),
            target_lengths=torch.tensor([12-i])) for i in range(6)]

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.samples[index]


@pytest.fixture
def resources():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    dataset = Dataset()
    model = Translatotron2(recipe_model(replace(ModelConfig.smoke(), dropout=.1), POOLED, SPEC), 7)
    return model, dataset


def assert_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested(x, y)
    else: assert a == b


def test_frame_pooled_moments_exclude_padding_and_are_batch_independent():
    x = torch.tensor([[[1., 2., 999.]], [[3., 8., 11.]]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    one, split = pooled.Moments(), pooled.Moments()
    one.add(x, mask)
    for i in range(2): split.add(x[i:i+1], mask[i:i+1])
    expected = torch.tensor([1., 2., 3., 8., 11.], dtype=torch.float64)
    for accumulator in (one, split):
        mean, variance = accumulator.finish()
        torch.testing.assert_close(mean[0], expected.mean())
        torch.testing.assert_close(variance[0], expected.var(unbiased=True))
        assert accumulator.count == 5
    with pytest.raises(ValueError): pooled.Moments().finish()
    with pytest.raises(ValueError): one.add(x, mask.float())
    x[0, 0, 0] = torch.nan
    with pytest.raises(ValueError): one.add(x, mask)


def test_selection_is_train_only_stable_and_bounded(resources):
    _, dataset = resources
    _, ids = pooled.selection(dataset)
    other = deepcopy(dataset)
    other.rows.reverse()
    assert pooled.selection(other)[1] == ids
    other.rows = [dict(id=f'large-{i}') for i in range(300)]
    assert len(pooled.selection(other)[1]) == 128
    for split in ('dev', 'test', None):
        other.split = split
        with pytest.raises(ValueError, match='train only'): pooled.selection(other)
    dataset.rows[1]['id'] = dataset.rows[0]['id']
    with pytest.raises(ValueError, match='duplicate'): pooled.selection(dataset)


def test_cached_fit_matches_five_full_forwards_and_leaves_nonpostnet_unchanged(resources):
    model, dataset = resources
    naive = deepcopy(model)
    before = deepcopy(model.state_dict())
    with pooled.temporary_eval(model):
        result = pooled.fit(model, dataset, 1)
        actual = deepcopy(model.state_dict())
    indices, _ = pooled.selection(dataset)
    with pooled.temporary_eval(naive):
        for name, module in pooled.layers(naive).items():
            moments = pooled.Moments()
            hook = module.register_forward_pre_hook(lambda module, inputs: moments.add(*inputs))
            try:
                for start in range(0, len(indices), pooled.BATCH_SIZE):
                    naive(**collate([dataset[i] for i in indices[start:start+pooled.BATCH_SIZE]]))
            finally: hook.remove()
            mean, variance = moments.finish()
            module.running_mean.copy_(mean)
            module.running_var.copy_(variance)
        expected = deepcopy(naive.state_dict())
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key
        if not (key.startswith('postnet.') and key.endswith(('running_mean', 'running_var'))):
            assert torch.equal(actual[key], before[key]), key
    assert result['train_ids'] == pooled.selection(dataset)[1]
    assert all(value['valid_frames'] == sum(12-i for i in range(6)) for value in result['layers'].values())
    assert_nested(model.state_dict(), before)


def test_temporary_eval_restores_mixed_flags_all_rng_buffers_gradients_on_failure(resources):
    model, dataset = resources
    model.postnet[0].eval()
    optimizer = torch.optim.Adam(model.parameters())
    optimization_step(model, optimizer, dataset[0])
    model.postnet[0].eval()
    flags = [m.training for m in model.modules()]
    original, rng, adam = deepcopy(model.state_dict()), capture_rank_state(model), deepcopy(optimizer.state_dict())
    grads = [p.grad.clone() for p in model.parameters()]
    with pytest.raises(RuntimeError, match='injected'):
        with pooled.temporary_eval(model):
            pooled.fit(model, dataset, 1)
            random.random(); np.random.rand(); torch.rand(7)
            raise RuntimeError('injected validation failure')
    assert_nested(model.state_dict(), original)
    assert_nested(capture_rank_state(model), rng)
    assert_nested(optimizer.state_dict(), adam)
    assert [m.training for m in model.modules()] == flags
    for p, grad in zip(model.parameters(), grads): assert torch.equal(p.grad, grad)


def test_roundtrip_binding_and_all_layers_validated_before_mutation(resources, tmp_path):
    model, dataset = resources
    optimizer = torch.optim.Adam(model.parameters())
    optimization_step(model, optimizer, dataset[0])
    native = deepcopy(model.state_dict())
    expected_contract = pooled.contract(dataset)
    with pooled.temporary_eval(model):
        overlay = pooled.fit(model, dataset, 1)
        calibrated = deepcopy(model.state_dict())
    identity = dict(training=dict(tt2_recipe=POOLED, postnet_calibration=expected_contract))
    checkpoint = tmp_path/'last.pt'
    save_checkpoint(checkpoint, model, optimizer, 1, dataset.tokens, identity,
                    rank_states=[capture_rank_state(model)], eval_calibration=overlay)
    restored, state = load_checkpoint(checkpoint, restore_rng=True)
    assert_nested(restored.state_dict(), native)  # Training loader never applies eval stats.
    assert_nested(state['optimizer'], optimizer.state_dict())
    pooled.apply_checkpoint(restored, state)
    assert_nested(restored.state_dict(), calibrated)
    pooled.apply(model, json.loads(json.dumps(overlay)), 1, expected_contract)
    assert_nested(model.state_dict(), calibrated)
    model.load_state_dict(native)
    for kind in ('stale', 'weight', 'last_layer', 'contract', 'missing'):
        bad, expectation = deepcopy(overlay), expected_contract
        if kind == 'stale': bad['updates'] = 2
        if kind == 'weight': bad['parameter_sha256'] = '0'*64
        if kind == 'last_layer': bad['layers'][list(bad['layers'])[-1]]['variance'] = [-1.]
        if kind == 'contract': expectation = dict(expected_contract, train_ids=['foreign'])
        if kind == 'missing': bad = None
        with pytest.raises(ValueError): pooled.apply(model, bad, 1, expectation)
        assert_nested(model.state_dict(), native)
    with pytest.raises(ValueError, match='stale'):
        save_checkpoint(tmp_path/'invalid.pt', model, optimizer, 2, dataset.tokens, identity, eval_calibration=overlay)


@pytest.mark.parametrize('packed', [False, True])
def test_pooled_calibration_travels_with_normal_frozen_snapshot(resources, tmp_path, monkeypatch, packed):
    from direct_s2st.recovery import freeze_files
    from direct_s2st.colab import publish, latest, restore
    model, dataset = resources
    optimizer = torch.optim.Adam(model.parameters())
    optimization_step(model, optimizer, dataset[0])
    with pooled.temporary_eval(model): overlay = pooled.fit(model, dataset, 1)
    identity = dict(training=dict(tt2_recipe=POOLED, postnet_calibration=pooled.contract(dataset)))
    work = tmp_path/'work'
    checkpoint = work/'checkpoints/checkpoint_last.pt'
    save_checkpoint(checkpoint, model, optimizer, 1, dataset.tokens, identity,
        eval_calibration=overlay, rank_states=[capture_rank_state(model)])
    from direct_s2st.io import atomic_write_json
    atomic_write_json(work/'training-monitor.json', dict(fixture=True, best_update=1))
    frozen = tmp_path/'frozen'
    hashes = freeze_files(work, checkpoint, frozen)
    monkeypatch.setenv('S2ST_DRIVE_SAFE', '1' if packed else '0')
    backup = tmp_path/'backup'
    publish(frozen, backup, dict(fixture='local, no Drive access'), 1)
    snapshot = latest(backup, dict(fixture='local, no Drive access'))
    assert snapshot[1]['files'] == hashes
    restore(snapshot, tmp_path/'restored')
    restored, state = load_checkpoint(tmp_path/'restored/checkpoints/checkpoint_last.pt')
    assert state['eval_calibration'] == overlay
    assert_nested(restored.state_dict(), model.state_dict())
    pooled.apply_checkpoint(restored, state)


@pytest.mark.gpu
def test_cuda_pooled_context_preserves_native_buffers_and_cuda_rng(resources):
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    model, dataset = resources
    model.cuda()
    before = capture_rank_state(model)
    with pooled.temporary_eval(model):
        overlay = pooled.fit(model, dataset, 1)
        torch.rand(11, device='cuda')
    assert_nested(capture_rank_state(model), before)
    pooled.apply(model, overlay, 1, pooled.contract(dataset))


def test_next_optimizer_update_is_bitwise_identical_with_and_without_calibration(resources):
    model, dataset = resources
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    optimization_step(model, optimizer, dataset[0])
    baseline = deepcopy(model)
    other_optimizer = torch.optim.Adam(baseline.parameters(), lr=.0003)
    other_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    rng = capture_rank_state(model)
    with pooled.temporary_eval(model):
        pooled.fit(model, dataset, 1)
        model(**dataset[1])
    calibrated_step = optimization_step(model, optimizer, dataset[2])
    after = capture_rank_state(model)
    restore_rank_state(baseline, rng)
    untouched_step = optimization_step(baseline, other_optimizer, dataset[2])
    assert calibrated_step == untouched_step
    assert_nested(model.state_dict(), baseline.state_dict())
    assert_nested(optimizer.state_dict(), other_optimizer.state_dict())
    assert_nested(after, capture_rank_state(baseline))


def test_normal_inference_applies_the_selected_checkpoints_embedded_statistics(resources, tmp_path, monkeypatch):
    from direct_s2st.translatotron2 import infer
    from direct_s2st.hashing import sha256_file
    from direct_s2st.io import atomic_write_json
    model, dataset = resources
    with torch.no_grad():
        model.phone_projection.bias.fill_(0.)
        model.phone_projection.bias[model.EOS] = 20.
    optimizer = torch.optim.Adam(model.parameters())
    optimization_step(model, optimizer, dataset[0])
    with pooled.temporary_eval(model): overlay = pooled.fit(model, dataset, 1)
    data, run = tmp_path/'data', tmp_path/'run'
    data.mkdir()
    atomic_write_json(data/'mel-spec.json', SPEC)
    identity = {'mel-spec.json': sha256_file(data/'mel-spec.json'),
        'training': dict(tt2_recipe=POOLED, postnet_calibration=pooled.contract(dataset))}
    checkpoint = run/'checkpoints/checkpoint_last.pt'
    save_checkpoint(checkpoint, model, optimizer, 1, dataset.tokens, identity, eval_calibration=overlay)
    native = deepcopy(model.state_dict())
    with pooled.temporary_eval(model):
        pooled.apply(model, overlay, 1, pooled.contract(dataset))
        result = model.generate(dataset[0]['source'], dataset[0]['source_lengths'])
        expected = result['post_mel'][0, :int(result['frame_lengths'][0])].cpu().numpy()
    wav = tmp_path/'fixture.wav'
    wav.write_bytes(b'synthetic: frontend replaced by fixture')
    common = tmp_path/'common'
    common.mkdir()
    (common/'test.jsonl').write_text('synthetic manifest, replaced by fixture')
    monkeypatch.setattr(infer, 'read_common_manifest', lambda path: iter([dict(pair_id='fixture', split='test',
        ja_audio=str(wav), en_audio=str(wav), en_text='fixture')]))
    monkeypatch.setattr(infer, 'read_table', lambda path: [dict(id='fixture')])
    monkeypatch.setattr(infer, 'source_features', lambda path, spec: dataset[0]['source'][0])
    predictions = run/'predictions/mels.jsonl'
    monkeypatch.setattr(sys, 'argv', ['infer', '--data-root', str(data), '--run-root', str(run),
        '--common-root', str(common), '--predictions', str(predictions), '--checkpoint-selection', 'last'])
    infer.main()
    row = json.loads(predictions.read_text(encoding='utf-8'))
    assert row['status'] == 'success', row
    np.testing.assert_array_equal(np.load(row['mel_path'], allow_pickle=False), expected)
    original, state = load_checkpoint(checkpoint)
    assert_nested(original.state_dict(), native)
    assert state['eval_calibration'] == overlay


def test_explicit_new_recipe_config_and_notebook_keep_all_other_recipes(resources, tmp_path):
    from direct_s2st.colab import make_config
    from direct_s2st.recipes import option
    data, environment = tmp_path/'data', tmp_path/'environment.json'
    root = data/'translatotron2/fairseq'
    root.mkdir(parents=True)
    (root/'mel-spec.json').write_text(json.dumps(SPEC))
    (root/'data-lock.json').write_text('{"fixture":true}')
    environment.write_text('{"fixture":true}')
    cfg = make_config(ROOT, data, environment, 'tt2', 'fisher', performance='gpu80',
        reproduction_mode='paper_practical', tt2_recipe=POOLED)
    assert option(cfg['command'], '--tt2-recipe') == POOLED
    assert cfg['research_metadata']['effective_batch_size'] == 1024
    assert float(option(cfg['command'], '--learning-rate')) == .0003
    for mode in ('smoke', 'paper_exact'):
        with pytest.raises(ValueError, match='new run'): validate_recipe(POOLED, mode)
    with pytest.raises(ValueError, match='FP32'):
        make_config(ROOT, data, environment, 'tt2', 'fisher', performance='gpu80',
            reproduction_mode='paper_practical', tt2_recipe=POOLED, precision='bf16')
    notebook = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    source = ''.join(next(c['source'] for c in notebook['cells']
        if c['cell_type'] == 'code' and 'TT2_RECIPE =' in ''.join(c['source'])))
    namespace = {}
    exec(source.split('def ensure_workspace():')[0], namespace)
    names = {}
    s2ut = namespace['training_run_name']('s2ut')
    for recipe in (LEGACY, STABILIZED, POOLED):
        namespace['TT2_RECIPE'] = recipe
        names[recipe] = namespace['training_run_name']('tt2')
        assert namespace['training_run_name']('s2ut') == s2ut
    assert len(set(names.values())) == 3
    assert namespace['training_mode']('tt2') == 'paper_practical'


def test_normal_cell5_pooled_launch_resume_is_recipe_isolated(tmp_path, monkeypatch):
    # Exercise the existing normal-cell recovery harness with the new recipe.
    import test_tt2_stabilized_recipe as old
    monkeypatch.setattr(old, 'STABILIZED', POOLED)
    old.test_cell5_keeps_legacy_and_stabilized_runs_separate(tmp_path)
    fresh = tmp_path/'normal'
    fresh.mkdir()
    old.test_normal_cell5_stabilized_launch_and_resume_preserve_config(fresh, monkeypatch)


def test_normal_cell5_does_not_automatically_adopt_a_seconds_v1_snapshot(tmp_path):
    sys.path.insert(0, str(ROOT/'scripts/colab'))
    import training_session as session
    from test_cell5_recovery import snapshot
    cfg, path, marker = snapshot(tmp_path, 'tt2-seconds-v1')
    cfg['command'] = ['--tt2-recipe', STABILIZED]
    path.write_text(json.dumps(cfg))
    before = path.read_bytes(), marker.read_bytes()
    assert session.choose_run(tmp_path, 'tt2', 'tt2-pooled-v2', tt2_recipe=POOLED) == ('tt2-pooled-v2', None)
    with pytest.raises(ValueError, match='recipe differs'):
        session.choose_run(tmp_path, 'tt2', 'tt2-pooled-v2', 'tt2-seconds-v1', tt2_recipe=POOLED)
    assert before == (path.read_bytes(), marker.read_bytes())


@pytest.mark.parametrize('env', [('WORLD_SIZE', '2'), ('S2ST_TRAIN_PRECISION', 'bf16')])
def test_unsupported_pooled_runtime_rejected_before_reading_data(tmp_path, monkeypatch, env):
    from direct_s2st.translatotron2 import train
    monkeypatch.setenv(*env)
    monkeypatch.setattr(train, 'PreparedDataset', lambda *a: pytest.fail('runtime rejected before data I/O'))
    monkeypatch.setattr(sys, 'argv', ['tt2', '--data-root', str(tmp_path/'data'), '--run-root', str(tmp_path/'run'),
        '--model-size', 'fisher', '--reproduction-mode', 'paper_practical', '--tt2-recipe', POOLED,
        '--batch-size', '16', '--max-updates', '1'])
    with pytest.raises(SystemExit): train._main()
    assert not (tmp_path/'run').exists()


def test_native_cli_pooled_saves_best_and_last_with_native_resume_state(resources, tmp_path, monkeypatch):
    from direct_s2st.translatotron2 import train
    import subprocess
    monkeypatch.setattr(train, 'PreparedDataset', lambda root, split: Dataset(split))
    monkeypatch.setattr(train, 'fingerprint', lambda root: dict(fixture='synthetic'))
    monkeypatch.setattr(ModelConfig, 'fisher', classmethod(lambda cls: cls.smoke()))
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'b'*40)
    for key in ('S2ST_TT2_PRACTICAL_WARMUP', 'S2ST_BATCH_PROBE'):
        monkeypatch.delenv(key, raising=False)
    for key in ('S2ST_TRAIN_OPTIMIZE', 'S2ST_STREAM_INPUTS', 'S2ST_TRAIN_ADAPTIVE_BATCH'):
        monkeypatch.setenv(key, '0')
    monkeypatch.setenv('S2ST_TRAIN_PRECISION', 'default')
    monkeypatch.setenv('S2ST_TT2_MONITOR_POLICY', json.dumps(dict(version=3, interval=1, patience=5, min_relative_improvement=.01)))
    args = ['tt2', '--data-root', str(tmp_path/'data'), '--run-root', str(tmp_path/'run'),
        '--model-size', 'fisher', '--reproduction-mode', 'paper_practical', '--tt2-recipe', POOLED,
        '--batch-size', '16', '--update-freq', '64', '--max-updates', '1']
    monkeypatch.setattr(sys, 'argv', args)
    train.main()
    last = tmp_path/'run/checkpoints/checkpoint_last.pt'
    best = last.with_name('checkpoint_best.pt')
    model, state = load_checkpoint(last)
    best_before = best.read_bytes()
    assert state['updates'] == state['eval_calibration']['updates'] == 1
    assert state['data_fingerprint']['training']['tt2_recipe'] == POOLED
    assert state['monitor_state']['best_update'] == 1
    pooled.apply_checkpoint(model, state)
    monkeypatch.setattr(sys, 'argv', args[:-1] + ['2', '--restore-file', str(last)])
    train.main()
    model, state = load_checkpoint(last)
    assert state['updates'] == state['eval_calibration']['updates'] == 2
    assert state['optimizer']['param_groups'][0]['lr'] == .0003*2/400
    assert len(state['monitor_state']['history']) == 2
    pooled.apply_checkpoint(model, state)
    best_model, best_state = load_checkpoint(best)
    pooled.apply_checkpoint(best_model, best_state)
    if state['monitor_state']['best_update'] == 1: assert best.read_bytes() == best_before
    assert 'schedule_transition' not in state
    # Actual uninterrupted CLI execution has identical model/Adam/RNG/history.
    fresh_args = args[:-1] + ['2']
    fresh_args[fresh_args.index('--run-root')+1] = str(tmp_path/'uninterrupted')
    monkeypatch.setattr(sys, 'argv', fresh_args)
    train.main()
    uninterrupted_model, uninterrupted = load_checkpoint(tmp_path/'uninterrupted/checkpoints/checkpoint_last.pt')
    # model above was explicitly calibrated for inference; compare native states.
    native_model, native_state = load_checkpoint(last)
    assert_nested(native_model.state_dict(), uninterrupted_model.state_dict())
    for key in ('optimizer', 'rank_states', 'eval_calibration', 'data_fingerprint'):
        assert_nested(native_state[key], uninterrupted[key])
    a, b = deepcopy(native_state['monitor_state']), deepcopy(uninterrupted['monitor_state'])
    # The serialized ZIP's temporary basename is not a numerical identity.
    a.pop('best_sha256'); b.pop('best_sha256')
    assert_nested(a, b)


def test_final_calibration_does_not_add_off_interval_monitor_evaluations(resources, tmp_path, monkeypatch):
    from direct_s2st.translatotron2 import train
    import subprocess
    monkeypatch.setattr(train, 'PreparedDataset', lambda root, split: Dataset(split))
    monkeypatch.setattr(train, 'fingerprint', lambda root: dict(fixture='synthetic'))
    monkeypatch.setattr(ModelConfig, 'fisher', classmethod(lambda cls: cls.smoke()))
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'b'*40)
    for key in ('S2ST_TT2_PRACTICAL_WARMUP', 'S2ST_BATCH_PROBE'):
        monkeypatch.delenv(key, raising=False)
    for key in ('S2ST_TRAIN_OPTIMIZE', 'S2ST_STREAM_INPUTS', 'S2ST_TRAIN_ADAPTIVE_BATCH'):
        monkeypatch.setenv(key, '0')
    monkeypatch.setenv('S2ST_TRAIN_PRECISION', 'default')
    monkeypatch.setenv('S2ST_TT2_MONITOR_POLICY', json.dumps(dict(version=3, interval=10, patience=5, min_relative_improvement=.01)))
    monkeypatch.setattr(sys, 'argv', ['tt2', '--data-root', str(tmp_path/'data'), '--run-root', str(tmp_path/'run'),
        '--model-size', 'fisher', '--reproduction-mode', 'paper_practical', '--tt2-recipe', POOLED,
        '--batch-size', '16', '--update-freq', '64', '--max-updates', '1'])
    train.main()
    model, state = load_checkpoint(tmp_path/'run/checkpoints/checkpoint_last.pt')
    assert state['eval_calibration']['updates'] == state['updates'] == 1
    assert state['monitor_state']['best_update'] is None
    assert not (tmp_path/'run/validation').exists()
    assert len(state['monitor_state']['history']) == 1
    pooled.apply_checkpoint(model, state)
