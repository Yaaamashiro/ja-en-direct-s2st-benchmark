import ast
import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from direct_s2st.config import load_config
from direct_s2st.recipes import option, tt2_batch, tt2_source_spec, validate_s2ut_paper

ROOT = Path(__file__).resolve().parents[2]


def test_effective_batch_is_explicit_and_never_silently_changed():
    assert tt2_batch('paper_exact', 8) == 128
    assert tt2_batch('paper_exact', 4, world_size=2) == 128
    assert tt2_batch('paper_practical', 8) == 1
    assert tt2_batch('paper_practical', 8, 16) == 16
    with pytest.raises(ValueError, match='divide'):
        tt2_batch('paper_exact', 3)
    with pytest.raises(ValueError, match='1024'):
        tt2_batch('paper_exact', 8, 1)


def test_fisher_model_shape_and_loss_contract():
    from dataclasses import asdict
    from direct_s2st.translatotron2.model import ModelConfig
    values = asdict(ModelConfig.fisher())
    expected = dict(encoder_dim=144, encoder_layers=16, heads=4, convolution_kernel=32,
                    attention_dim=512, attention_output_dim=256, attention_heads=4,
                    linguistic_dim=256, linguistic_layers=4, embedding_dim=96, zoneout=.1,
                    duration_dim=64, acoustic_dim=1024, acoustic_layers=2,
                    prenet_dim=128, postnet_dim=512, phone_weight=10., duration_weight=1., label_smoothing=.1)
    assert {k:values[k] for k in expected} == expected


def test_colab_fisher_recipe_preserves_existing_target_features(tmp_path):
    from direct_s2st.colab import make_config
    target = load_config(ROOT/'configs/translatotron2/prepare.yaml')['mel']
    target.update(log_transform='natural_log_clamp_eps', normalization='none')
    data = tmp_path/'data'
    prepared = data/'translatotron2/fairseq'
    prepared.mkdir(parents=True)
    (prepared/'mel-spec.json').write_text(json.dumps(target))
    (prepared/'data-lock.json').write_text('{"fixture":true}')
    env = tmp_path/'environment.json'
    env.write_text('{"repository":"fixture-revision"}')
    exact = make_config(ROOT, data, env, model_size='fisher', performance='gpu80')
    command, metadata = exact['command'], exact['research_metadata']
    assert option(command, '--learning-rate') == '0.0042'
    assert option(command, '--warmup-updates') == '10000'
    assert float(option(command, '--l2-regularization')) == 1e-6
    assert option(command, '--model-size') == 'fisher'
    assert option(command, '--update-freq') == '128'
    assert metadata['effective_batch_size'] == 1024
    assert metadata['target_feature_config'] == target
    assert metadata['source_feature_config'] == tt2_source_spec(ROOT)
    assert metadata['source_feature_config']['f_min'] == 125
    assert metadata['source_feature_config']['f_max'] == 7600
    assert metadata['recommended_total_updates'] is None
    practical = make_config(ROOT, data, env, model_size='fisher', performance='gpu80', reproduction_mode='paper_practical')
    assert practical['research_metadata']['effective_batch_size'] == 8
    assert practical != exact
    assert json.loads((prepared/'mel-spec.json').read_text()) == target
    with pytest.raises(ValueError, match='batch statistics'):
        make_config(ROOT, data, env, model_size='fisher', optimize=True, adaptive_batch=True)


def test_pinned_s2ut_fisher_registry_and_official_recipe():
    # Evaluate actual pinned architecture functions, without a mock of dimensions
    # or importing old fairseq/Hydra into Python 3.13.
    path = ROOT/'third_party/fairseq/fairseq/models/speech_to_speech/s2s_transformer.py'
    nodes = [n for n in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in (
                 'base_s2st_transformer_encoder_architecture', 's2ut_architecture_base', 's2ut_architecture_fisher')]
    for node in nodes:
        node.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    args = Namespace()
    namespace['s2ut_architecture_fisher'](args)
    assert (args.encoder_layers, args.encoder_embed_dim, args.encoder_attention_heads) == (12, 256, 4)
    assert (args.decoder_layers, args.decoder_embed_dim, args.decoder_attention_heads) == (6, 256, 8)
    command = load_config(ROOT/'configs/s2ut/train.yaml')['training']['command']
    validate_s2ut_paper(command)
    changed = command[:]
    changed[changed.index('--update-freq')+1] = '1'
    with pytest.raises(ValueError):
        validate_s2ut_paper(changed)
    validate_s2ut_paper(changed, exact=False)
    changed[changed.index('--lr')+1] = '0.001'
    with pytest.raises(ValueError):
        validate_s2ut_paper(changed, exact=False)
    for cap in ('--batch-size', '--max-sentences'):
        with pytest.raises(ValueError, match='utterance batch cap'):
            validate_s2ut_paper(command + [cap, '32'])


def test_s2ut_colab_metadata_and_recipe_resume_guard(tmp_path, monkeypatch):
    import subprocess
    from direct_s2st.colab import make_config
    from direct_s2st.s2ut import preflight
    from direct_s2st.s2ut.fairseq_train import prepare_research_metadata
    unit = load_config(ROOT/'configs/s2ut/prepare.yaml')
    identity = dict(hubert_revision=unit['hubert']['revision'], kmeans_sha256=unit['kmeans']['sha256'],
                    hubert_layer=6, kmeans_clusters=100)
    data = tmp_path/'data'
    prepared = data/'s2ut/fairseq'
    prepared.mkdir(parents=True)
    (prepared/'data-lock.json').write_text(json.dumps(dict(unit_configuration=identity, target='reduced_units')))
    (data/'s2ut/units').mkdir()
    (data/'s2ut/units/unit-lock.json').write_text(json.dumps(identity))
    env = tmp_path/'environment.json'
    env.write_text('{"repository":"fixture"}')
    monkeypatch.setattr(preflight, 'validate_training', lambda *a: {'model_dimensions': {'encoder_embed_dim':256}})
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'fixture-revision\n')
    config = make_config(ROOT, data, env, 's2ut', 'fisher', performance='gpu80')
    assert option(config['command'], '--arch') == 's2ut_transformer_fisher'
    assert option(config['command'], '--update-freq') == '4'
    assert config['research_metadata']['unit_artifacts'] == identity
    command = ['wrapper'] + [v.format(run_root=str(tmp_path/'run'), updates='2') for v in config['command'][3:]]
    clean, (run, meta) = prepare_research_metadata(command)
    assert '--reproduction-mode' not in clean and '--vocoder-mode' not in clean
    assert meta['hubert_revision'] == identity['hubert_revision']
    assert meta['precision'] == 'fp16' and meta['max_updates'] == 2
    checkpoint = run/'checkpoints/checkpoint_last.pt'
    restored = command + ['--restore-file', str(checkpoint)]
    restored[restored.index('--max-update')+1] = '3'
    _, (_, resume_meta) = prepare_research_metadata(restored)
    assert resume_meta['training_fingerprint'] == meta['training_fingerprint']
    assert resume_meta['max_updates'] == 3
    changed = restored[:]
    changed[changed.index('--reproduction-mode')+1] = 'paper_practical'
    with pytest.raises(ValueError, match='identity mismatch'):
        prepare_research_metadata(changed)
    with pytest.raises(ValueError):
        make_config(ROOT, data, env, 's2ut', 'fisher', max_tokens=2000)
    calibrated = make_config(ROOT, data, env, 's2ut', 'fisher', performance='gpu80',
                             optimize=True, calibrate_batch=True)
    from direct_s2st.batch_calibration import candidates, selected_config
    assert candidates(calibrated) == [0]
    assert selected_config(calibrated, 0)['command'] == calibrated['command']
    for key, value in [('S2ST_TRAIN_FIXED_MICROBATCH', '32'), ('S2ST_TRAIN_ADAPTIVE_BATCH', '1'),
                       ('S2ST_TRAIN_PRECISION', 'bf16')]:
        with monkeypatch.context() as patch:
            patch.setenv(key, value)
            with pytest.raises(ValueError, match='forbids additional batch'):
                prepare_research_metadata(command)
    with monkeypatch.context() as patch:
        patch.setenv('S2ST_TRAIN_FIXED_MICROBATCH', '32')
        practical = command[:]
        practical[practical.index('--reproduction-mode')+1] = 'paper_practical'
        practical[practical.index('--save-dir')+1] = str(tmp_path/'practical/checkpoints')
        _, (_, practical_meta) = prepare_research_metadata(practical)
        assert practical_meta['physical_batch_cap'] == 32


def test_fisher_preflight_rejects_auxiliary_loss_changes(tmp_path, monkeypatch):
    import sys
    import types
    import yaml
    from direct_s2st.s2ut import preflight
    from direct_s2st.s2ut.multitask import load_settings
    path = ROOT/'third_party/fairseq/fairseq/models/speech_to_speech/s2s_transformer.py'
    nodes = [n for n in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in (
                 'base_s2st_transformer_encoder_architecture', 's2ut_architecture_base', 's2ut_architecture_fisher')]
    for node in nodes:
        node.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    fake = types.ModuleType('fairseq.models')
    fake.ARCH_CONFIG_REGISTRY = {'s2ut_transformer_fisher':namespace['s2ut_architecture_fisher']}
    monkeypatch.setitem(sys.modules, 'fairseq.models', fake)
    monkeypatch.setattr(preflight, 'validate_prepared', lambda *a, **kw: {})
    monkeypatch.setattr(preflight, 'read_tsv', lambda path, columns: {'a': {'tgt_n_frames':'10', 'tgt_text':'a b'}})
    cfg = {key:dict(value, data=str(tmp_path)) for key, value in load_settings().items()}
    config_path = tmp_path/'config_multitask.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    command = load_config(ROOT/'configs/s2ut/train.yaml')['training']['command']
    evidence = preflight.validate_training(tmp_path, command)
    assert evidence['model_dimensions']['decoder_attention_heads'] == 8
    cfg['source_letter']['loss_weight'] = 1.
    config_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='auxiliary task/loss'):
        preflight.validate_training(tmp_path, command)


def test_revision_migration_preserves_legacy_data_and_requires_overwrite(tmp_path, monkeypatch):
    import importlib.util
    import subprocess
    spec = importlib.util.spec_from_file_location('paper_migration', ROOT/'scripts/colab/use_paper_recipe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path/'corpus'))
    revision, old = 'a'*40, 'b'*40
    monkeypatch.setattr(subprocess, 'check_output', lambda cmd, **kw: revision if 'rev-parse' in cmd else '')
    persistent = tmp_path/'experiment'
    persistent.mkdir()
    (persistent/'repository-revision.txt').write_text(old+'\n')
    (persistent/'legacy.json').write_text('{"fixture":true}')
    data = persistent/'data'
    data.mkdir()
    (data/'features.zip').write_bytes(b'preserved')
    assert module.migrate(persistent, revision)['status'] == 'PLAN_ONLY'
    assert (persistent/'repository-revision.txt').read_text().strip() == old
    assert module.migrate(persistent, revision, overwrite=True)['status'] == 'APPLIED'
    assert (persistent/'repository-revision.before-paper-aaaaaaaaaaaa.txt').read_text().strip() == old
    assert (data/'features.zip').read_bytes() == b'preserved'
    assert module.migrate(persistent, revision)['status'] == 'ALREADY_CURRENT'
    (persistent/'new.json').write_text('{"research_metadata":{"reproduction_mode":"paper_exact"}}')
    (persistent/'repository-revision.txt').write_text(old+'\n')
    with pytest.raises(ValueError, match='already exists'):
        module.migrate(persistent, revision, overwrite=True)
    preserved = (persistent/'new.json').read_bytes()
    assert module.migrate(persistent, revision, overwrite=True,
                          separate_recipe_v2_runs=True)['status'] == 'APPLIED'
    assert (persistent/'new.json').read_bytes() == preserved
    assert (data/'features.zip').read_bytes() == b'preserved'
    (persistent/'repository-revision.txt').write_text(old+'\n')
    (persistent/'s2ut-recipe-v2.json').write_bytes(preserved)
    with pytest.raises(ValueError, match='recipe-v2 configuration already exists'):
        module.migrate(persistent, revision, overwrite=True, separate_recipe_v2_runs=True)


def test_streaming_accumulation_matches_list_and_resumes(tmp_path):
    from direct_s2st.translatotron2.batching import StreamingBatches
    from direct_s2st.translatotron2.engine import optimization_step, save_checkpoint, load_checkpoint
    from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
    torch.set_num_threads(1)
    torch.manual_seed(8)
    samples = [dict(source=torch.randn(1, 24, 80), source_lengths=torch.tensor([24]),
                   phones=torch.tensor([[3, 4, 2]]), phone_lengths=torch.tensor([3]),
                   target=torch.randn(1, 12, 80), target_lengths=torch.tensor([12])) for _ in range(2)]
    class Dataset(SimpleNamespace):
        def __len__(self):
            return len(self.rows)
    dataset = Dataset(rows=[dict(id=str(i), tgt_n_frames='12') for i in range(2)], labels={'0':'a b', '1':'a b'})
    model = Translatotron2(ModelConfig.smoke(), 6)
    other = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt2 = torch.optim.Adam(other.parameters(), lr=1e-4)
    rng = torch.get_rng_state()
    expected = optimization_step(model, optimizer, samples)
    torch.set_rng_state(rng)
    actual = optimization_step(other, opt2, StreamingBatches(iter(samples), dataset, 1, 1, 2))
    assert actual == expected
    for key, value in model.state_dict().items():
        assert torch.equal(value, other.state_dict()[key])
    path = tmp_path/'stream.pt'
    save_checkpoint(path, other, opt2, 1, list('abcdef'), {'mode':'paper_exact'})
    expected = optimization_step(other, opt2, StreamingBatches(iter(samples), dataset, 2, 1, 2))
    restored, state = load_checkpoint(path, restore_rng=True, expected_fingerprint={'mode':'paper_exact'})
    restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-4)
    restored_opt.load_state_dict(state['optimizer'])
    assert optimization_step(restored, restored_opt, StreamingBatches(iter(samples), dataset, 2, 1, 2)) == expected
    with pytest.raises(ValueError, match='fingerprint'):
        load_checkpoint(path, expected_fingerprint={'mode':'paper_practical'})


def test_paper_vocoders_need_no_locally_trained_vocoder(tmp_path):
    from direct_s2st.colab_compare import build_configs, stage_direct_artifacts
    modes = dict(tt2_vocoder_mode='griffin_lim', s2ut_vocoder_mode='paper')
    runs = {}
    for system in ('s2ut', 'translatotron2'):
        source = tmp_path/system
        (source/'checkpoints').mkdir(parents=True)
        (source/'checkpoints/checkpoint_last.pt').write_bytes(b'fixture')
        (source/'gradient-audit-rank-0.json').write_text('{}')
        runs[system] = source
    stage_direct_artifacts(tmp_path/'runs', 'paper', runs, {}, **modes)
    cfg = build_configs(ROOT, tmp_path/'runs', 'paper', **modes)
    assert cfg['mel']['infer']['command'][2] == 'direct_s2st.vocoders.griffin_lim'
    assert '--checkpoint' not in cfg['mel']['infer']['command']
    assert '--official-fisher' in cfg['unit']['infer']['command']
    alternate = build_configs(ROOT, tmp_path/'runs', 'hifigan')
    assert '--checkpoint' in alternate['mel']['infer']['command']
    assert '--official-fisher' not in alternate['unit']['infer']['command']


def test_reference_vocoder_fetch_is_hash_locked(tmp_path, monkeypatch):
    from direct_s2st.vocoders import reference
    calls = []
    monkeypatch.setattr(reference, 'download_artifact', lambda url, path, **kw: calls.append((url, path, kw)))
    reference.ensure_reference(tmp_path/'checkpoint', tmp_path/'config')
    assert calls[0][2]['sha256'] == reference.CHECKPOINT_SHA256
    assert calls[1][2]['sha256'] == reference.CONFIG_SHA256
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='CORPUS_ROOT'):
        reference.ensure_reference(tmp_path/'unsafe', tmp_path/'unsafe2')


def test_griffin_lim_real_cpu_reconstruction_and_resume(tmp_path, monkeypatch):
    import librosa
    from direct_s2st.vocoders import griffin_lim as gl
    from direct_s2st.io import read_jsonl
    spec = load_config(ROOT/'configs/translatotron2/prepare.yaml')['mel']
    spec.update(log_transform='natural_log_clamp_eps', normalization='none')
    wave = np.sin(2*np.pi*440*np.arange(3200)/16000).astype(np.float32)*.1
    magnitude = np.abs(librosa.stft(wave, n_fft=512, win_length=400, hop_length=160, pad_mode='reflect'))
    filters = librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000)
    feature = np.log(np.maximum(filters @ magnitude, 1e-5)).T
    feature_path = tmp_path/'mel.npy'
    np.save(feature_path, feature)
    spec_path = tmp_path/'mel-spec.json'
    spec_path.write_text(json.dumps(spec))
    inputs = tmp_path/'input.jsonl'
    inputs.write_text(json.dumps(dict(pair_id='a', system_id='translatotron2', status='success', mel_path=str(feature_path)))+'\n')
    output = tmp_path/'output'
    result = gl.vocode(inputs, output, spec_path, iterations=2)
    assert result['failures'] == 0 and result['checkpoint_sha256'] is None
    row = list(read_jsonl(output/'predictions.jsonl'))[0]
    assert row['output_duration'] == pytest.approx(.2)
    assert Path(row['output_audio']).is_file()
    monkeypatch.setattr(gl, 'reconstruct', lambda *a, **kw: pytest.fail('valid completed WAV must be reused'))
    assert gl.vocode(inputs, output, spec_path, iterations=2, resume=True) == result
    with pytest.raises(ValueError, match='identity'):
        gl.vocode(inputs, output, spec_path, iterations=3, resume=True)
