"""Synthetic tensors test mechanics, not speech quality or real-data E2E."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchaudio')
from direct_s2st.translatotron2.model import ModelConfig, Translatotron2, gaussian_upsample
from direct_s2st.translatotron2.engine import optimization_step, save_checkpoint, load_checkpoint


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = Translatotron2(ModelConfig.smoke(), 7)
    batch = dict(source=torch.randn(2, 24, 80), source_lengths=torch.tensor([24, 17]),
                 phones=torch.tensor([[3, 4, 5, 2], [4, 3, 2, 0]]), phone_lengths=torch.tensor([4, 3]),
                 target=torch.randn(2, 12, 80), target_lengths=torch.tensor([12, 9]))
    return model, batch


def test_all_losses_and_gradients_and_real_optimizer(setup):
    model, batch = setup
    before = model.embedding.weight.detach().clone()
    losses = optimization_step(model, torch.optim.Adam(model.parameters()), batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.equal(before, model.embedding.weight)
    for prefix in ('encoder', 'linguistic', 'shared_attention', 'duration', 'range_predictor', 'acoustic'):
        assert sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith(prefix)) > 0


def test_mel_loss_reaches_linguistic_attention_and_durations(setup):
    model, batch = setup
    model(**batch)['mel_loss'].backward()
    for prefix in ('linguistic', 'shared_attention', 'duration', 'range_predictor'):
        assert sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith(prefix)) > 0


def test_checkpoint_resume_is_exact(setup, tmp_path):
    model, batch = setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimization_step(model, optimizer, batch)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, model, optimizer, 1, list('abcdefg'), {'fixture': 'synthetic'})
    expected = optimization_step(model, optimizer, batch)
    restored, state = load_checkpoint(path, expected_fingerprint={'fixture': 'synthetic'}, restore_rng=True)
    opt = torch.optim.Adam(restored.parameters(), lr=1e-4)
    opt.load_state_dict(state['optimizer'])
    actual = optimization_step(restored, opt, batch)
    assert actual == expected
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='fingerprint'):
        load_checkpoint(path, expected_fingerprint={'fixture': 'changed'})


def test_gaussian_duration_gradient_and_padding():
    states = torch.randn(2, 3, 5, requires_grad=True)
    durations = torch.tensor([[1., 2., 3.], [2., 1., 0.]], requires_grad=True)
    output, weights = gaussian_upsample(states, durations, torch.ones(2, 3),
                                        torch.tensor([3, 2]), torch.tensor([8, 6]))
    assert torch.count_nonzero(weights[1, :, 2]) == 0
    assert torch.count_nonzero(output[1, 6:]) == 0
    output.square().sum().backward()
    assert durations.grad.abs().sum() > 0


def test_reference_free_generation_and_fail_closed_limits(setup):
    model, batch = setup
    model.eval()
    # Force deterministic EOS for this interface test, not a trained speech result.
    with torch.no_grad():
        model.phone_projection.weight.zero_()
        model.phone_projection.bias.zero_()
        model.phone_projection.bias[2] = 10
    output = model.generate(batch['source'], batch['source_lengths'], max_phones=4)
    assert output['phone_lengths'].tolist() == [1, 1]
    assert torch.isfinite(output['post_mel']).all()
    with torch.no_grad():
        model.phone_projection.bias[3] = 20
    with pytest.raises(ValueError, match='without EOS'):
        model.generate(batch['source'], batch['source_lengths'], max_phones=2)


def test_target_padding_does_not_change_losses(setup):
    model, batch = setup
    model.eval()
    first = model(**batch)
    batch['target'][1, 9:] = 1e6
    batch['phones'][1, 3:] = 6
    second = model(**batch)
    for key in ('loss', 'mel_loss', 'duration_loss', 'phone_loss'):
        torch.testing.assert_close(first[key], second[key])


def test_phoneme_state_changes_acoustic_output(setup):
    model, batch = setup
    model.eval()
    before = model(**batch)['post_mel']
    batch['phones'][0, 0] = 6
    after = model(**batch)['post_mel']
    assert not torch.allclose(before[0], after[0])


@pytest.mark.parametrize('workers', [0, 2])
def test_training_cli_updates_and_resumes(setup, tmp_path, monkeypatch, workers):
    from direct_s2st.translatotron2 import train
    model, batch = setup

    class SyntheticDataset:
        # This explicitly tests the trainer, not the unexecuted real audio frontend.
        spec = {'n_mels': 80}
        tokens = list('abcdefg')
        def __init__(self, root, split):
            assert split == 'train'
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return batch

    monkeypatch.setattr(train, 'PreparedDataset', SyntheticDataset)
    monkeypatch.setattr(train, 'fingerprint', lambda root: {'fixture': 'synthetic'})
    arguments = ['tt2-train', '--num-workers', str(workers), '--batch-size', '2',
                 '--data-root', str(tmp_path / 'data'), '--run-root', str(tmp_path / 'run'),
                 '--max-updates', '2']
    monkeypatch.setattr('sys.argv', arguments)
    train.main()
    checkpoint = tmp_path / 'run/checkpoints/checkpoint_last.pt'
    _, state = load_checkpoint(checkpoint)
    assert state['updates'] == 2
    from direct_s2st.io import ExistingOutputError
    with pytest.raises(ExistingOutputError):
        train.main()
    monkeypatch.setattr('sys.argv', arguments[:-1] + ['3', '--restore-file', str(checkpoint)])
    train.main()
    _, state = load_checkpoint(checkpoint)
    assert state['updates'] == 3
    assert (tmp_path / 'run/losses/00000003.json').is_file()
    # A fresh uninterrupted run must match the worker-prefetched resumed run.
    continuous = arguments[:]
    continuous[continuous.index('--run-root')+1] = str(tmp_path/'continuous')
    continuous[continuous.index('--max-updates')+1] = '3'
    continuous[continuous.index('--num-workers')+1] = '0'
    monkeypatch.setattr('sys.argv', continuous)
    train.main()
    _, expected = load_checkpoint(tmp_path/'continuous/checkpoints/checkpoint_last.pt')
    for name, value in state['model'].items():
        assert torch.equal(value, expected['model'][name]), name
