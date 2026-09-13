"""Independent-implementation mechanics, not evidence of paper metric parity."""
from dataclasses import replace
from pathlib import Path
import json
import pytest

torch = pytest.importorskip('torch')
from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
from direct_s2st.translatotron2.conformer import MaskedBatchNorm
from direct_s2st.translatotron2.engine import optimization_step, save_checkpoint, load_checkpoint


@pytest.fixture
def sample():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    return dict(source=torch.randn(1, 24, 80), source_lengths=torch.tensor([24]),
        phones=torch.tensor([[3, 4, 2]]), phone_lengths=torch.tensor([3]),
        target=torch.randn(1, 12, 80), target_lengths=torch.tensor([12]))


def test_masked_normalization_excludes_padding():
    norm = MaskedBatchNorm(2)
    x = torch.tensor([[[1., 3., 1e6], [2., 6., 1e6]]], requires_grad=True)
    y = norm(x, torch.tensor([[True, True, False]]))
    torch.testing.assert_close(norm.running_mean, torch.tensor([.2, .4]))
    torch.testing.assert_close(y[:, :, :2], torch.tensor([[[-1., 1.], [-1., 1.]]]), atol=1e-5, rtol=1e-5)
    assert y[:, :, 2].count_nonzero() == 0
    y.square().sum().backward()
    assert x.grad[:, :, 2].count_nonzero() == 0


def test_kernel32_relative_attention_and_distinct_context_width(sample):
    cfg = replace(ModelConfig.smoke(), convolution_kernel=32, attention_dim=32, attention_output_dim=12)
    model = Translatotron2(cfg, 6)
    output = model(**sample)
    output['loss'].backward()
    assert model.encoder.layers[0].depthwise.kernel_size == (32,)
    assert model.encoder.layers[0].attention.relative.weight.grad.abs().sum() > 0
    assert model.context_projection.weight.grad.abs().sum() > 0
    assert output['post_mel'].shape == sample['target'].shape


def test_encoder_padding_width_does_not_affect_valid_positions(sample):
    model = Translatotron2(ModelConfig.smoke(), 6).eval()
    source = sample['source'][:, :17]
    short, lengths = model.encode(source, torch.tensor([17]))
    padded, _ = model.encode(torch.nn.functional.pad(source, (0, 0, 0, 7), value=12345.), torch.tensor([17]))
    torch.testing.assert_close(short, padded[:, :int(lengths[0])], atol=1e-6, rtol=1e-5)


def test_beam_synthesizes_selected_states_without_reference(sample):
    model = Translatotron2(ModelConfig.smoke(), 6).eval()
    with torch.no_grad():
        model.phone_projection.weight.zero_()
        model.phone_projection.bias.zero_()
        model.phone_projection.bias[2] = 10
        model.phone_projection.bias[3] = 5
    generated = model.generate(sample['source'], sample['source_lengths'], beam_size=3, max_phones=4)
    assert generated['phones'].tolist() == [[3, 2]]
    assert generated['phone_lengths'].tolist() == [1]
    assert torch.isfinite(generated['post_mel']).all()
    with pytest.raises(ValueError):
        model.generate(sample['source'], sample['source_lengths'], beam_size=0)


def test_accumulated_optimizer_and_resume(sample, tmp_path):
    model = Translatotron2(ModelConfig.smoke(), 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batches = [sample, {key: value.clone() for key, value in sample.items()}]
    first = optimization_step(model, optimizer, batches)
    assert first['gradient_norm'] > 0
    checkpoint = tmp_path / 'accumulated.pt'
    save_checkpoint(checkpoint, model, optimizer, 1, list('abcdef'), {'fixture': 'accumulated'})
    expected = optimization_step(model, optimizer, batches)
    resumed, state = load_checkpoint(checkpoint, restore_rng=True)
    restored_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(state['optimizer'])
    assert optimization_step(resumed, restored_optimizer, batches) == expected


def test_dev_validation_preserves_rng_and_does_not_update_weights(sample):
    from direct_s2st.translatotron2.validation import validate
    model = Translatotron2(ModelConfig.smoke(), 6)
    state = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = validate(model, [sample], 'cpu')
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert torch.equal(rng, torch.get_rng_state()) and model.training
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, state[key])


def test_reference_presets_and_source_target_specs():
    from direct_s2st.config import load_config
    from direct_s2st.vocoders.inference import validate_config
    root = Path(__file__).resolve().parents[2]
    fisher, covost = ModelConfig.fisher(), ModelConfig.covost2()
    assert fisher.context_dim == 256 and fisher.attention_dim == 512
    assert covost.context_dim == 512 and covost.attention_heads == 8 and covost.linguistic_layers == 6
    cfg = load_config(root / 'configs/translatotron2/prepare-paper.yaml')
    assert cfg['source_mel']['n_mels'] == 80 and cfg['mel']['n_mels'] == 128
    vocoder = json.loads((root / 'configs/vocoder/mel-paper-generator.json').read_text())
    validate_config('mel', vocoder, sample_rate=24000, mel=vocoder['mel'])
    for key, value in cfg['mel'].items():
        assert vocoder['mel'][key] == value


def test_v1_checkpoint_loads_legacy_architecture(sample, tmp_path):
    cfg = replace(ModelConfig.smoke(), architecture_revision=1)
    model = Translatotron2(cfg, 6)
    opt = torch.optim.Adam(model.parameters())
    optimization_step(model, opt, sample)
    checkpoint = tmp_path / 'legacy.pt'
    save_checkpoint(checkpoint, model, opt, 1, list('abcdef'), {})
    state = torch.load(checkpoint, weights_only=True)
    state['format'] = 'direct-s2st-tt2-v1'
    for name in ('architecture_revision', 'attention_output_dim', 'attention_heads', 'specaugment_time_masks'):
        del state['model_config'][name]
    torch.save(state, checkpoint)  # Deliberate legacy-format unit fixture, not E2E.
    restored, _ = load_checkpoint(checkpoint)
    assert restored.config.architecture_revision == 1
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
