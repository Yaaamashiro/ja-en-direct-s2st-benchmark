import importlib.util
import json
from pathlib import Path
import subprocess
import pytest
from direct_s2st.journal import Journal
from direct_s2st.io import ExistingOutputError, read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_journal_requires_identical_resume_and_preserves_progress(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    journal = Journal(path, {'checkpoint': 'one'})
    journal.record({'pair_id': 'a', 'status': 'success'})
    journal.record({'pair_id': 'b', 'status': 'failed'})
    with pytest.raises(ExistingOutputError):
        Journal(path, {'checkpoint': 'one'})
    with pytest.raises(ValueError, match='identity'):
        Journal(path, {'checkpoint': 'two'}, resume=True)
    resumed = Journal(path, {'checkpoint': 'one'}, resume=True)
    resumed.record({'pair_id': 'b', 'status': 'success'})
    assert len(list(read_jsonl(path))) == 2
    assert all(r['status'] == 'success' for r in resumed.rows.values())


def test_all_system_plan_and_fitted_vocoder_paths(tmp_path):
    suite = load_module('suite_driver', ROOT / 'scripts/smoke/suite.py')
    configs, stages = suite.build_suite('trial', tmp_path / '.env', updates=2)
    assert len(stages) == 23
    assert stages[-2]['stage'] == 'acceptance'
    assert stages[-1]['stage'] == 'comparison'
    assert 'unit-fit' in {s['stage'] for s in stages}
    for kind in ('unit', 'mel'):
        command = configs[kind]['infer']['command']
        assert command[command.index('--checkpoint')+1] == '{run_root}/vocoder-'+kind+'/generator.pt'
    assert configs['comparison']['run_ids'] == ['trial-s2ut', 'trial-translatotron2', 'trial-cascade']
    with pytest.raises(ValueError):
        suite.build_suite('trial', tmp_path / '.env', limit=101)


def test_paper_suite_frontend_and_vocoder_are_consistent(tmp_path):
    suite = load_module('suite_paper', ROOT / 'scripts/smoke/suite.py')
    configs, stages = suite.build_suite('paper', tmp_path / '.env', tt2_recipe='fisher')
    assert len(stages) == 23
    assert configs['tt2-paper-prepare']['mel']['n_mels'] == 128
    assert configs['tt2-paper-prepare']['source_mel']['n_mels'] == 80
    train = configs['mel']['train']['command']
    assert train[train.index('--config')+1].endswith('mel-paper-generator.json')
    infer = configs['mel']['infer']['command']
    assert infer[infer.index('--sample-rate')+1] == '24000'
    prepare = next(s for s in stages if s['stage'] == 'translatotron2-prepare')
    assert any('tt2-paper-prepare.yaml' in item for item in prepare['command'])


def test_suite_resumes_failed_stage_without_repeating_pass(tmp_path):
    suite = load_module('suite_resume', ROOT / 'scripts/smoke/suite.py')
    stages = [dict(stage='a', status='NOT_RUN', command=['fixture', 'a']),
              dict(stage='b', status='NOT_RUN', command=['fixture', 'b'])]
    calls = []
    def fail_second(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, int(command[1] == 'b'))
    assert suite.execute({}, stages, tmp_path, runner=fail_second) == 1
    calls.clear()
    assert suite.execute({}, stages, tmp_path, resume=True,
                         runner=lambda command, **kw: (calls.append(command) or subprocess.CompletedProcess(command, 0))) == 0
    assert calls == [['fixture', 'b', '--resume']]


def test_both_generator_configs_have_exact_output_stride():
    import math
    from direct_s2st.vocoders.inference import validate_config
    for kind in ('unit', 'mel'):
        cfg = json.loads((ROOT / f'configs/vocoder/{kind}-generator.json').read_text())
        validate_config(kind, cfg, sample_rate=16000, mel=cfg['mel'])
        assert math.prod(cfg['upsample_rates']) == (320 if kind == 'unit' else 160)
        assert all((k - stride) % 2 == 0 for k, stride in zip(cfg['upsample_kernel_sizes'], cfg['upsample_rates']))


def test_vocoder_duration_targets_are_real_run_lengths():
    torch = pytest.importorskip('torch')
    from direct_s2st.vocoders.train import run_lengths
    tokens, lengths = run_lengths([2, 2, 2, 3, 4, 4])
    assert tokens.tolist() == [2, 3, 4]
    assert lengths.tolist() == [3, 1, 2]
    with pytest.raises(ValueError):
        run_lengths([1.5, 2])


@pytest.mark.parametrize('batch_size', [1, 2])
def test_real_pinned_generator_and_official_discriminator_optimizer(batch_size):
    torch = pytest.importorskip('torch')
    pytest.importorskip('torchaudio')
    from direct_s2st.vocoders.train import gan_step
    from direct_s2st.vocoders.discriminators import DiscriminatorP
    source = ROOT / 'third_party/fairseq/fairseq/models/text_to_speech/hifigan.py'
    if not source.is_file():
        pytest.skip('initialize pinned fairseq submodule')
    upstream = load_module('pinned_hifigan_for_cpu_test', source)
    torch.set_num_threads(1)
    torch.manual_seed(41)
    generator = upstream.Generator(dict(model_in_dim=4, upsample_initial_channel=16,
        upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4], resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]]))
    class OnePeriod(torch.nn.Module):
        # Use the exact upstream period discriminator, restricted to one period
        # to bound CPU test memory. Production uses all MPD/MSD discriminators.
        def __init__(self):
            super().__init__()
            self.discriminator = DiscriminatorP(2)
        def forward(self, real, fake):
            r, rf = self.discriminator(real)
            f, ff = self.discriminator(fake)
            return [r], [f], [rf], [ff]
    discriminators = torch.nn.ModuleList([OnePeriod()])
    optim_g = torch.optim.AdamW(generator.parameters(), lr=1e-4)
    optim_d = torch.optim.AdamW(discriminators.parameters(), lr=1e-4)
    # Synthetic tensors explicitly test optimization mechanics, not corpus E2E.
    conditioning, real = torch.randn(batch_size, 4, 64), torch.randn(batch_size, 1, 256).tanh()
    before = generator.conv_pre.weight_v.detach().clone()
    def spectral_fixture(x):
        return torch.stft(x, 64, hop_length=16, window=torch.hann_window(64),
                          return_complex=True).abs().clamp_min(1e-5).log()
    losses = gan_step(generator, discriminators, optim_g, optim_d, conditioning, real, spectral_fixture)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert optim_g.state and optim_d.state
    assert not torch.equal(before, generator.conv_pre.weight_v)


def test_tt2_padded_batch_and_schedule():
    torch = pytest.importorskip('torch')
    from direct_s2st.translatotron2.batching import collate, learning_rate
    def row(frames):
        return dict(source=torch.ones(1, frames, 80), source_lengths=torch.tensor([frames]),
                    target=torch.ones(1, frames+1, 80), target_lengths=torch.tensor([frames+1]),
                    phones=torch.tensor([[3, 2]]), phone_lengths=torch.tensor([2]))
    batch = collate([row(4), row(7)])
    assert batch['source'].shape == (2, 7, 80)
    assert batch['source'][0, 4:].count_nonzero() == 0
    assert learning_rate(1., 2, 4) == 0.5
    assert learning_rate(1., 16, 4) == 0.5


def test_s2ut_gradient_audit_rejects_missing_branches():
    torch = pytest.importorskip('torch')
    from direct_s2st.s2ut.fairseq_train import TrainingAudit, GROUPS
    model = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in GROUPS})
    audit = TrainingAudit()
    audit.attach(model)
    with pytest.raises(ValueError, match='incomplete'):
        audit.result()
    loss = sum(layer(torch.ones(1, 2)).square().sum() for layer in model.values())
    audit.record(loss, 1., {'multitask': {key: {'loss': 1.} for key in
                                        ('source_letter', 'target_letter', 'decoder_target_ctc')}})
    loss.backward()
    assert audit.result()['status'] == 'PASS'
    assert all(value > 0 for value in audit.result()['max_abs_gradient_by_group'].values())


def test_acceptance_rejects_silent_wav_without_claiming_e2e(tmp_path):
    import numpy as np
    import soundfile as sf
    from direct_s2st.io import atomic_write_json, atomic_write_jsonl
    from direct_s2st.hashing import sha256_file
    from direct_s2st.evaluation.acceptance import verify_suite
    common, run = tmp_path / 'common', tmp_path / 'cascade-fixture'
    wav = tmp_path / 'silent.wav'
    sf.write(wav, np.zeros(1600), 16000)
    atomic_write_jsonl(common / 'test.jsonl', [dict(pair_id='a', split='test', ja_audio=str(wav), en_audio=str(wav), en_text='test')])
    prediction = dict(pair_id='a', system_id='cascade', run_id=run.name, split='test',
        source_audio=str(wav), reference_audio=str(wav), reference_text='test', output_audio=str(wav),
        output_duration=0.1, inference_seconds=1., real_time_factor=10., status='success', error=None)
    predictions = run / 'predictions/predictions.jsonl'
    atomic_write_jsonl(predictions, [prediction])
    atomic_write_jsonl(run / 'metrics/per_sample.jsonl', [dict(pair_id='a', evaluation_status='success', hypothesis_raw='test')])
    atomic_write_json(run / 'metrics/metrics.json', {'samples': 1})
    atomic_write_json(run / 'metrics/per_sample.lock.json',
        dict(predictions=sha256_file(predictions), audio={str(wav): sha256_file(wav)}))
    with pytest.raises(ValueError, match='silent'):
        verify_suite(common, [run], tmp_path / 'report')
    assert not (tmp_path / 'report/e2e-acceptance.json').exists()
    prediction['reference_text'] = 'changed'
    atomic_write_jsonl(predictions, [prediction], overwrite=True)
    with pytest.raises(ValueError, match='stale'):
        verify_suite(common, [run], tmp_path / 'report')


def test_evaluation_resume_rejects_changed_audio(tmp_path):
    import numpy as np
    import soundfile as sf
    from direct_s2st.io import atomic_write_jsonl
    from direct_s2st.evaluation.run import evaluate_predictions
    wav = tmp_path / 'audio.wav'
    sf.write(wav, np.ones(1600) * 0.1, 16000)
    predictions = tmp_path / 'predictions.jsonl'
    atomic_write_jsonl(predictions, [dict(pair_id='a', system_id='cascade', run_id='test',
        source_audio=str(wav), reference_audio=str(wav), reference_text='test', output_audio=str(wav),
        output_duration=0.1, inference_seconds=1., real_time_factor=10., status='success', error=None)])
    evaluate_predictions(predictions, tmp_path / 'metrics', transcribe=lambda _: 'test')
    sf.write(wav, np.ones(1600) * 0.2, 16000)
    with pytest.raises(ValueError, match='identity'):
        evaluate_predictions(predictions, tmp_path / 'metrics', transcribe=lambda _: 'test', resume=True)
