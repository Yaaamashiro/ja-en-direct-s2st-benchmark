"""Train fixed-fairseq HiFi-GAN generators with official MPD/MSD losses."""
import argparse
from ..progress import operation, track
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from ..io import ExistingOutputError, atomic_write_json
from ..s2ut.reduce_units import validate_units
from .discriminators import (MultiPeriodDiscriminator, MultiScaleDiscriminator,
                             discriminator_loss, generator_loss, feature_loss)
from .inference import validate_config


def run_lengths(units):
    values = torch.as_tensor(validate_units(list(units), clusters=100), dtype=torch.long)
    if values.ndim != 1 or values.numel() == 0 or (values < 0).any() or (values >= 100).any():
        raise ValueError('expected nonempty KM100 original units')
    return torch.unique_consecutive(values, return_counts=True)


def duration_objective(generator, units):
    reduced, counts = run_lengths(units)
    reduced = reduced.to(next(generator.parameters()).device)
    prediction = generator.dur_predictor(generator.dict(reduced[None]))
    return F.mse_loss(prediction, counts.to(prediction).log1p()[None])


def finite_step(loss, optimizer, parameters):
    parameters = list(parameters)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite vocoder loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError('nonfinite vocoder gradient')
    if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in parameters):
        raise ValueError('missing vocoder gradients')
    torch.nn.utils.clip_grad_norm_(parameters, 1000., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError('nonfinite vocoder parameters')


def gan_step(generator, discriminators, optim_g, optim_d, conditioning, real, mel,
             *, units=None, duration_weight=1.0):
    generator.train()
    discriminators.train()
    from ..train_runtime import precision_context
    device = real.device
    with precision_context(device):
        fake = generator(code=conditioning, dur_prediction=False) if units is not None else generator(conditioning)
    if fake.shape != real.shape:
        raise ValueError(f'generator/wave alignment mismatch: {fake.shape} vs {real.shape}')
    optim_d.zero_grad(set_to_none=True)
    d_loss = fake.new_zeros(())
    with precision_context(device):
        for discriminator in discriminators:
            real_scores, fake_scores, _, _ = discriminator(real, fake.detach())
            d_loss = d_loss + discriminator_loss([s.float() for s in real_scores], [s.float() for s in fake_scores])[0]
    finite_step(d_loss, optim_d, discriminators.parameters())
    optim_g.zero_grad(set_to_none=True)
    for parameter in discriminators.parameters():
        parameter.requires_grad_(False)
    try:
        adversarial, feature = fake.new_zeros(()), fake.new_zeros(())
        with precision_context(device):
            for discriminator in discriminators:
                _, generated, features_real, features_fake = discriminator(real, fake)
                adversarial = adversarial + generator_loss([s.float() for s in generated])[0]
                feature = feature + feature_loss([[v.float() for v in group] for group in features_real],
                                                [[v.float() for v in group] for group in features_fake])
        # Keep STFT/log objectives in FP32 even when convolutions use BF16.
        mel_loss = F.l1_loss(mel(fake.squeeze(1).float()), mel(real.squeeze(1).float()))
        if units is None:
            duration = fake.new_zeros(())
        elif isinstance(units[0], (list, tuple)):
            # No padded phones are introduced into the duration predictor.
            duration = torch.stack([duration_objective(generator, sequence) for sequence in units]).mean()
        else:
            duration = duration_objective(generator, units)
        total = adversarial + feature + 45 * mel_loss + duration_weight * duration
        finite_step(total, optim_g, generator.parameters())
    finally:
        for parameter in discriminators.parameters():
            parameter.requires_grad_(True)
    return {k: float(v.detach()) for k, v in dict(generator=total, discriminator=d_loss,
              adversarial=adversarial, feature=feature, mel=mel_loss, duration=duration).items()}


class LogMel(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        from fairseq.data.audio.audio_utils import TTSSpectrogram, TTSMelScale
        self.spectrum = TTSSpectrogram(spec['n_fft'], spec['win_length'], spec['hop_length'])
        self.mel = TTSMelScale(spec['n_mels'], spec['sample_rate'], spec['f_min'],
                               spec['f_max'], spec['n_fft'] // 2 + 1)
        self.eps = spec['eps']

    def forward(self, waveform):
        return self.mel(self.spectrum(waveform)).clamp_min(self.eps).log()


def load_wave(row, sample_rate, kind):
    import soundfile as sf
    from ..train_runtime import cached_file
    with cached_file(row['en_audio']) as local:
        waveform, rate = sf.read(local or row['en_audio'], dtype='float32', always_2d=True)
    if waveform.shape[1] != 1:
        raise ValueError('fitting requires mono audio')
    wave = torch.from_numpy(waveform[:, 0])[None]
    if rate != sample_rate:
        if kind == 'unit':
            raise ValueError('unit fitting requires original 16 kHz audio')
        import torchaudio.functional as audio_functional
        wave = audio_functional.resample(wave, rate, sample_rate)
    if not torch.isfinite(wave).all() or not torch.count_nonzero(wave):
        raise ValueError('invalid or silent training audio')
    return wave


def segment_batch(samples, *, spec, hop, segment_frames, mel, device):
    """Crop equal-length real segments without padding synthetic audio."""
    available = []
    for wave, units in samples:
        length = len(units) if units is not None else wave.size(1) // hop
        if units is not None and not 0 <= wave.size(1) - length * hop < 2 * hop:
            raise ValueError('original unit/wave alignment exceeds HuBERT tail tolerance')
        available.append(length)
    frames = min(segment_frames, *available)
    if frames * hop <= spec['n_fft'] // 2:
        raise ValueError('training segment too short for reflect-padded mel')
    real, conditioning = [], []
    for (wave, units), length in zip(samples, available):
        wave = wave.to(device, non_blocking=True)
        offset = int(torch.randint(0, length - frames + 1, (1,)))
        real.append(wave[:, offset * hop:(offset + frames) * hop, None].transpose(1, 2))
        conditioning.append(torch.tensor([units[offset:offset + frames]], device=device) if units is not None
                            else mel(wave)[:, :, offset:offset + frames].detach())
    return torch.cat(conditioning), torch.cat(real)


def save_state(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.pt.tmp')
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@operation('vocoders/train: main')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['unit', 'mel'], required=True)
    parser.add_argument('--common-root', type=Path, required=True)
    parser.add_argument('--units-root', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--split', choices=['train'], default='train')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-updates', type=int, default=2)
    parser.add_argument('--segment-frames', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--save-interval-updates', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.0002)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verification-result', type=Path)
    parser.add_argument('--verified-inputs', type=Path)
    args = parser.parse_args()
    if min(args.max_updates, args.segment_frames, args.batch_size, args.prefetch_factor, args.save_interval_updates) < 1 or args.num_workers < 0 or args.learning_rate <= 0:
        parser.error('updates, segment frames and learning rate must be positive')
    if args.resume and args.overwrite:
        parser.error('resume and overwrite are mutually exclusive')
    if os.environ.get('CORPUS_ROOT') and args.output_root.resolve().is_relative_to(Path(os.environ['CORPUS_ROOT']).resolve()):
        raise ValueError('vocoder output must not be in CORPUS_ROOT')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    spec = config['mel']
    validate_config(args.kind, config, sample_rate=spec['sample_rate'], mel=spec)
    if spec['normalize_volume'] or spec['normalization'] != 'none' or spec['log_transform'] != 'natural_log_clamp_eps':
        raise ValueError('fitting requires unnormalized natural-log mel')
    from .verification import verify_inputs, save_receipt, load_receipt
    if args.verify_only:
        if args.verification_result is None or args.verified_inputs is not None:
            parser.error('verify-only requires a verification-result and no verified-inputs')
        save_receipt(args.verification_result, verify_inputs(args.common_root, args.kind, args.units_root))
        return
    checkpoint = args.output_root / 'generator.pt'
    if args.output_root.exists() and any(args.output_root.iterdir()) and not (args.resume or args.overwrite):
        raise ExistingOutputError(str(args.output_root))
    if args.resume and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    verified = (load_receipt(args.verified_inputs, args.common_root, args.kind, args.units_root)
                if args.verified_inputs else verify_inputs(args.common_root, args.kind, args.units_root))
    rows = verified['rows']
    identity = {'common': verified['common'],
                'config': config, 'learning_rate': args.learning_rate,
                'segment_frames': args.segment_frames, 'kind': args.kind,
                'seed': args.seed, 'audio': verified['audio'], 'units': verified['units']}
    if args.batch_size != 1:
        identity['batch_size'] = args.batch_size
    if os.environ.get('S2ST_TRAIN_PRECISION', 'default') != 'default':
        identity['precision'] = os.environ['S2ST_TRAIN_PRECISION']
    all_units = verified['sequences']
    if args.kind == 'unit':
        if spec['sample_rate'] != 16000:
            raise ValueError('HuBERT units require 16 kHz audio')
        identity['unit_lock'] = json.loads((args.units_root / 'unit-lock.json').read_text(encoding='utf-8'))
        if identity['unit_lock']['hubert_layer'] != 6 or identity['unit_lock']['kmeans_clusters'] != 100:
            raise ValueError('unit fitting requires HuBERT layer6 KM100')
        from fairseq.models.text_to_speech.codehifigan import CodeGenerator
        factory = CodeGenerator
        if math.prod(config['upsample_rates']) != 320:
            raise ValueError('HuBERT 16 kHz units require 320-sample upsampling')
    else:
        from fairseq.models.text_to_speech.hifigan import Generator
        factory = Generator
    torch.manual_seed(args.seed)
    generator = factory(config).to(args.device)
    discriminators = torch.nn.ModuleList([MultiPeriodDiscriminator(), MultiScaleDiscriminator()]).to(args.device)
    mel = LogMel(spec).to(args.device)
    optim_g = torch.optim.AdamW(generator.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(discriminators.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
    start = 0
    if args.resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('format') != 'direct-s2st-vocoder-v1' or state['identity'] != identity or state['updates'] < 1:
            raise ValueError('vocoder resume identity mismatch')
        generator.load_state_dict(state['generator'], strict=True)
        discriminators.load_state_dict(state['discriminators'], strict=True)
        optim_g.load_state_dict(state['optimizer_g'])
        optim_d.load_state_dict(state['optimizer_d'])
        torch.set_rng_state(state['rng'])
        if state['cuda_rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start = state['updates']
    atomic_write_json(args.output_root / 'config.json', config, resume=args.resume, overwrite=args.overwrite)
    from ..prefetch import ordered_samples
    from ..train_runtime import Timings, enabled, stopping, checkpoint_saved
    timing = Timings()
    from ..s2ut.unit_storage import records, stamp
    unit_records = records(args.units_root) if args.kind == 'unit' else {}
    def load(index):
        row = rows[index % len(rows)]
        from ..preparation import file_stamp
        if file_stamp(row['en_audio']) != row['_verified_audio_stamp']:
            raise ValueError(f"{row['pair_id']}: audio changed after verification")
        if args.kind == 'unit':
            path = args.units_root / 'train/original' / (row['pair_id'] + '.units')
            record = unit_records.get(row['pair_id'])
            current = stamp(record, 'original') if record else file_stamp(path)
            if current != row['_verified_unit_stamp']:
                raise ValueError(f"{row['pair_id']}: units changed after verification")
        wave = load_wave(row, spec['sample_rate'], args.kind)
        if file_stamp(row['en_audio']) != row['_verified_audio_stamp']:
            raise ValueError(f"{row['pair_id']}: audio changed while loading")
        if enabled() and args.device.startswith('cuda'):
            wave = wave.pin_memory()
        return wave, all_units.get(row['pair_id'])
    indices = range(start * args.batch_size, args.max_updates * args.batch_size)
    with ordered_samples(load, indices, args.num_workers, args.prefetch_factor) as loaded:
        for update in track(range(start + 1, args.max_updates + 1), 'vocoder: train updates'):
            with timing.measure('data_wait'):
                samples = [next(loaded) for _ in range(args.batch_size)]
            with timing.measure('transfer_and_conditioning', args.device if enabled() else None):
                conditioning, real = segment_batch(samples, spec=spec,
                    hop=math.prod(config['upsample_rates']), segment_frames=args.segment_frames,
                    mel=mel, device=args.device)
            units = [sample[1] for sample in samples] if args.kind == 'unit' else None
            with timing.measure('optimization', args.device if enabled() else None):
                losses = gan_step(generator, discriminators, optim_g, optim_d, conditioning, real, mel, units=units)
            stop = stopping()
            if update % args.save_interval_updates == 0 or update == args.max_updates or stop:
                save_state(checkpoint, dict(format='direct-s2st-vocoder-v1', generator=generator.state_dict(),
                           discriminators=discriminators.state_dict(), optimizer_g=optim_g.state_dict(),
                           optimizer_d=optim_d.state_dict(), updates=update, identity=identity,
                           rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
                with timing.measure('local_freeze'):
                    checkpoint_saved(args.output_root, checkpoint, update,
                                     force=stop or update == args.max_updates)
            atomic_write_json(args.output_root / 'losses' / f'{update:08d}.json', losses,
                              overwrite=args.overwrite or args.resume)
            print(json.dumps({'update': update, **losses}), flush=True)
            if enabled():
                timing.report(update, real.numel() / spec['sample_rate'], units='audio_seconds')
            if stop:
                break



if __name__ == '__main__':
    main()
