import threading
import pytest
from direct_s2st.prefetch import ordered_samples
from direct_s2st.colab import make_config


def test_parallel_prefetch_is_ordered_and_bounded():
    barrier = threading.Barrier(2)
    def read(index):
        if index < 2:
            barrier.wait(timeout=5)
        return index
    with ordered_samples(read, range(10), workers=2, prefetch=1) as samples:
        assert list(samples) == list(range(10))


def test_prefetch_surfaces_errors():
    def read(index):
        if index == 1:
            raise ValueError('bad sample')
        return index
    with pytest.raises(ValueError, match='bad sample'):
        with ordered_samples(read, range(8), workers=2) as samples:
            list(samples)


@pytest.mark.parametrize('kind,batch', [('tt2', 8), ('unit', 16), ('mel', 16)])
def test_gpu80_recipe_options(tmp_path, kind, batch):
    root = tmp_path / ('translatotron2/fairseq' if kind == 'tt2' else 'common')
    root.mkdir(parents=True)
    (root/'data-lock.json').write_text('{}')
    cfg = make_config(tmp_path, tmp_path, tmp_path/'environment.json', kind, performance='gpu80')
    command = cfg['command']
    for key, value in [('--batch-size', batch), ('--num-workers', 4), ('--save-interval-updates', 50)]:
        assert command[command.index(key)+1] == str(value)


def test_vocoder_segments_batch_without_padding():
    torch = pytest.importorskip('torch')
    from direct_s2st.vocoders.train import segment_batch
    torch.manual_seed(7)
    samples = [(torch.ones(1, 24), [1]*6), (torch.ones(1, 40)*2, [2]*10)]
    conditioning, real = segment_batch(samples, spec={'n_fft': 8}, hop=4,
                                      segment_frames=8, mel=None, device='cpu')
    assert conditioning.shape == (2, 6)
    assert real.shape == (2, 1, 24)
    assert (real[0] == 1).all() and (real[1] == 2).all()


def test_mel_segments_batch_and_rng_resume():
    torch = pytest.importorskip('torch')
    from direct_s2st.vocoders.train import segment_batch
    samples = [(torch.arange(40).float()[None], None), (torch.arange(32).float()[None], None)]
    def mel(x):
        return x.reshape(1, -1, 4).mean(-1)[:, None]
    kwargs = dict(spec={'n_fft': 8}, hop=4, segment_frames=6, mel=mel, device='cpu')
    state = torch.get_rng_state()
    first = segment_batch(samples, **kwargs)
    torch.set_rng_state(state)
    second = segment_batch(samples, **kwargs)
    assert first[0].shape == (2, 1, 6)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
