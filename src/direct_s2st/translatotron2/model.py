"""Experimental TT2 core; see docs/TRANSLATOTRON2.md for reproduction limits.

No text is fed to the acoustic encoder. The linguistic LSTM drives a shared
attention; both its hidden state and acoustic context condition the synthesizer.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.models import Conformer
from .conformer import RelativeConformer, ConvSubsample2d, MaskedBatchNorm


@dataclass
class ModelConfig:
    architecture_revision: int = 2
    input_dim: int = 80
    mel_dim: int = 80
    encoder_dim: int = 144
    encoder_layers: int = 16
    heads: int = 4
    attention_dim: int = 512
    attention_output_dim: int | None = None
    attention_heads: int = 4
    specaugment_masks: int = 2
    specaugment_time_masks: int = 10
    convolution_kernel: int = 32
    linguistic_dim: int = 256
    linguistic_layers: int = 4
    embedding_dim: int = 96
    duration_dim: int = 64
    acoustic_dim: int = 1024
    acoustic_layers: int = 2
    prenet_dim: int = 128
    postnet_dim: int = 512
    dropout: float = 0.1
    zoneout: float = 0.1
    phone_weight: float = 10.0
    duration_weight: float = 1.0
    label_smoothing: float = 0.1

    @property
    def context_dim(self):
        return self.attention_output_dim or self.attention_dim

    @classmethod
    def fisher(cls):
        return cls(attention_output_dim=256)

    @classmethod
    def covost2(cls):
        return cls(attention_output_dim=512, attention_heads=8, linguistic_dim=512,
                   linguistic_layers=6, embedding_dim=256, duration_dim=128, dropout=0.2)

    @classmethod
    def conversational(cls):
        return cls(attention_output_dim=512, attention_heads=8, linguistic_dim=512,
                   linguistic_layers=4, embedding_dim=256, duration_dim=128, dropout=0.2)

    @classmethod
    def smoke(cls):
        return cls(encoder_dim=16, encoder_layers=1, linguistic_dim=16,
                   linguistic_layers=2, embedding_dim=8, duration_dim=8,
                   acoustic_dim=16, acoustic_layers=2, prenet_dim=8,
                   postnet_dim=16, convolution_kernel=7, dropout=0.0,
                   attention_dim=16, specaugment_masks=0)


def sequence_mask(lengths, size):
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


class ZoneoutStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, probability):
        super().__init__()
        self.cells = nn.ModuleList(nn.LSTMCell(input_dim if i == 0 else hidden_dim,
                                              hidden_dim) for i in range(layers))
        self.hidden_dim, self.probability = hidden_dim, probability

    def forward(self, x, state=None):
        if state is None:
            state = [(x.new_zeros(x.size(0), self.hidden_dim),
                      x.new_zeros(x.size(0), self.hidden_dim)) for _ in self.cells]
        result = []
        for cell, (old_h, old_c) in zip(self.cells, state):
            h, c = cell(x, (old_h, old_c))
            if self.training:
                h = torch.where(torch.rand_like(h) < self.probability, old_h, h)
                c = torch.where(torch.rand_like(c) < self.probability, old_c, c)
            else:
                h = self.probability * old_h + (1 - self.probability) * h
                c = self.probability * old_c + (1 - self.probability) * c
            result.append((h, c))
            x = h
        return x, result


class PackedPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                           bidirectional=True, batch_first=True)
        self.projection = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                  batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.size(1))
        return (F.softplus(self.projection(y).squeeze(-1)) + 1e-4) * sequence_mask(lengths, x.size(1))


def gaussian_upsample(states, durations, ranges, phone_lengths, frame_lengths):
    """Differentiable normalized Gaussian interpolation (no hard repeats).

    Training rescales predicted durations to known utterance length; the separate
    duration loss constrains the *unscaled* sum. No aligned phone labels needed.
    """
    scaled = durations * (frame_lengths / durations.sum(1).clamp_min(1e-5))[:, None]
    centers = scaled.cumsum(1) - scaled / 2
    positions = torch.arange(int(frame_lengths.max()), device=states.device) + 0.5
    sigma = ranges.clamp_min(1e-3)
    logits = -0.5 * ((positions[None, :, None] - centers[:, None]) / sigma[:, None]).square()
    logits = logits - sigma[:, None].log()
    logits = logits.masked_fill(~sequence_mask(phone_lengths, states.size(1))[:, None], -torch.inf)
    weights = logits.softmax(-1)
    output = torch.bmm(weights, states)
    return output * sequence_mask(frame_lengths, output.size(1))[:, :, None], weights


class Translatotron2(nn.Module):
    PAD, BOS, EOS = 0, 1, 2

    def __init__(self, config: ModelConfig, vocabulary_size: int):
        super().__init__()
        if vocabulary_size < 4:
            raise ValueError("phoneme vocabulary must contain real phones and three special tokens")
        self.config = c = config
        if c.architecture_revision == 1:
            self.subsample = nn.ModuleList([
                nn.Conv1d(c.input_dim, c.encoder_dim, 3, stride=2, padding=1),
                nn.Conv1d(c.encoder_dim, c.encoder_dim, 3, stride=2, padding=1)])
            self.encoder = Conformer(c.encoder_dim, c.heads, c.encoder_dim * 4,
                                     c.encoder_layers, c.convolution_kernel,
                                     dropout=c.dropout, use_group_norm=True)
        elif c.architecture_revision == 2:
            self.subsample = ConvSubsample2d(c.input_dim, c.encoder_dim)
            self.encoder = RelativeConformer(c.encoder_dim, c.heads, c.encoder_layers,
                                             c.convolution_kernel, c.dropout)
        else:
            raise ValueError('unsupported TT2 architecture revision')
        self.embedding = nn.Embedding(vocabulary_size, c.embedding_dim, padding_idx=self.PAD)
        self.linguistic = ZoneoutStack(c.embedding_dim + c.context_dim, c.linguistic_dim,
                                       c.linguistic_layers, c.zoneout)
        self.query = nn.Linear(c.linguistic_dim, c.attention_dim)
        self.shared_attention = nn.MultiheadAttention(c.attention_dim, c.attention_heads,
            kdim=c.encoder_dim, vdim=c.encoder_dim, dropout=c.dropout, batch_first=True)
        self.context_projection = nn.Identity() if c.context_dim == c.attention_dim else nn.Linear(c.attention_dim, c.context_dim)
        conditioning_dim = c.linguistic_dim + c.context_dim
        self.phone_projection = nn.Linear(conditioning_dim, vocabulary_size)
        self.duration = PackedPredictor(conditioning_dim, c.duration_dim)
        self.range_predictor = PackedPredictor(conditioning_dim + 1, c.duration_dim)
        self.prenet = nn.Sequential(nn.Linear(c.mel_dim, c.prenet_dim), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(c.prenet_dim, c.prenet_dim), nn.ReLU(), nn.Dropout(0.5))
        self.acoustic = ZoneoutStack(conditioning_dim + c.prenet_dim, c.acoustic_dim,
                                     c.acoustic_layers, c.zoneout)
        self.mel_projection = nn.Linear(c.acoustic_dim, c.mel_dim)
        layers = []
        for i in range(5):
            layers.append(nn.Conv1d(c.mel_dim if i == 0 else c.postnet_dim,
                                    c.mel_dim if i == 4 else c.postnet_dim, 5, padding=2))
            if c.architecture_revision == 2:
                layers.append(MaskedBatchNorm(c.mel_dim if i == 4 else c.postnet_dim))
            if i != 4:
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(c.dropout))
        self.postnet = nn.Sequential(*layers)

    def encode(self, source, lengths):
        if source.ndim != 3 or source.size(2) != self.config.input_dim:
            raise ValueError("source must be [batch, frames, input_dim]")
        if (lengths < 1).any() or (lengths > source.size(1)).any() or not torch.isfinite(source).all():
            raise ValueError("invalid source features/lengths")
        x = source.masked_fill(~sequence_mask(lengths, source.size(1))[:, :, None], 0)
        if self.training and self.config.specaugment_masks:
            x = x.clone()
            for batch_index, length in enumerate(lengths.tolist()):
                for axis, size, maximum, count in ((0, length, max(1, length//20), self.config.specaugment_time_masks),
                                                    (1, x.size(2), max(1, int(x.size(2)*0.33)), self.config.specaugment_masks)):
                    for _ in range(count):
                        width = int(torch.randint(maximum + 1, (1,), device=x.device))
                        start = int(torch.randint(size - width + 1, (1,), device=x.device))
                        if axis == 0:
                            x[batch_index, start:start+width] = 0
                        else:
                            x[batch_index, :length, start:start+width] = 0
        if isinstance(self.subsample, ConvSubsample2d):
            x, lengths = self.subsample(x, lengths)
        else:
            for layer in self.subsample:
                x = F.silu(layer(x.transpose(1, 2))).transpose(1, 2)
                lengths = (lengths + 1) // 2
                x = x.masked_fill(~sequence_mask(lengths, x.size(1))[:, :, None], 0)
        x, lengths = self.encoder(x, lengths)
        return x, lengths

    def linguistic_step(self, token, context, state, memory, memory_lengths):
        hidden, state = self.linguistic(torch.cat([self.embedding(token), context], -1), state)
        context, attention = self.shared_attention(self.query(hidden)[:, None], memory, memory,
            key_padding_mask=~sequence_mask(memory_lengths, memory.size(1)))
        context = self.context_projection(context[:, 0])
        conditioning = torch.cat([hidden, context], -1)
        return self.phone_projection(conditioning), conditioning, context, state, attention

    def synthesize(self, conditioning, phone_lengths, frame_lengths=None, target=None, max_frames=3000):
        durations = self.duration(conditioning, phone_lengths)
        ranges = self.range_predictor(torch.cat([conditioning, durations[:, :, None]], -1), phone_lengths)
        if not torch.isfinite(durations).all() or not torch.isfinite(ranges).all():
            raise ValueError('nonfinite duration/range prediction')
        if frame_lengths is None:
            frame_lengths = durations.sum(1).round().long().clamp_min(1)
            if (frame_lengths > max_frames).any():
                raise ValueError("predicted duration exceeds max_frames; not silently truncating")
        expanded, weights = gaussian_upsample(conditioning, durations, ranges, phone_lengths, frame_lengths)
        previous = conditioning.new_zeros(conditioning.size(0), self.config.mel_dim)
        state, frames = None, []
        for t in range(expanded.size(1)):
            hidden, state = self.acoustic(torch.cat([expanded[:, t], self.prenet(previous)], -1), state)
            frame = self.mel_projection(hidden)
            frames.append(frame)
            previous = target[:, t] if target is not None else frame
        mask = sequence_mask(frame_lengths, expanded.size(1))[:, :, None]
        mel = torch.stack(frames, 1) * mask
        # Mask between convolutions to prevent padded positions leaking back in.
        residual = mel.transpose(1, 2)
        for layer in self.postnet:
            residual = (layer(residual, mask.squeeze(-1)) if isinstance(layer, MaskedBatchNorm) else layer(residual)) * mask.transpose(1, 2)
        post_mel = (mel + residual.transpose(1, 2)) * mask
        return dict(mel=mel, post_mel=post_mel, durations=durations,
                    upsampling=weights, frame_lengths=frame_lengths)

    def forward(self, source, source_lengths, phones, phone_lengths, target, target_lengths):
        if (phone_lengths < 2).any() or (phone_lengths > phones.size(1)).any():
            raise ValueError("phones require at least one phone followed by EOS")
        if not torch.all(phones.gather(1, (phone_lengths - 1)[:, None]) == self.EOS):
            raise ValueError("phone sequence must terminate in EOS")
        if (target_lengths < 1).any() or (target_lengths > target.size(1)).any() or not torch.isfinite(target).all():
            raise ValueError("invalid target mel/lengths")
        memory, lengths = self.encode(source, source_lengths)
        context = memory.new_zeros(memory.size(0), self.config.context_dim)
        state, logits, conditioning = None, [], []
        token = phones.new_full((phones.size(0),), self.BOS)
        for t in range(int(phone_lengths.max())):
            scores, hidden, context, state, _ = self.linguistic_step(token, context, state, memory, lengths)
            logits.append(scores)
            conditioning.append(hidden)
            token = phones[:, t]
        logits = torch.stack(logits, 1)
        # EOS predicts termination, not an acoustic duration.
        conditioning = torch.stack(conditioning, 1)[:, :-1]
        result = self.synthesize(conditioning, phone_lengths - 1, target_lengths, target)
        labels = phones[:, :logits.size(1)].masked_fill(~sequence_mask(phone_lengths, logits.size(1)), self.PAD)
        phone_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=self.PAD,
                                     label_smoothing=self.config.label_smoothing)
        duration_loss = (result['durations'].sum(1) - target_lengths).square().mean()
        mask = sequence_mask(target_lengths, result['mel'].size(1))[:, :, None]
        reference = target[:, :mask.size(1)].masked_fill(~mask, 0)
        mel_loss = sum(((result[key] - reference).abs() * mask).sum() /
                       (mask.sum() * self.config.mel_dim) for key in ('mel', 'post_mel'))
        result.update(phone_logits=logits, phone_loss=phone_loss, duration_loss=duration_loss,
                      mel_loss=mel_loss, loss=mel_loss + self.config.phone_weight * phone_loss +
                      self.config.duration_weight * duration_loss)
        return result

    @torch.no_grad()
    def generate(self, source, source_lengths, max_phones=400, max_frames=3000, beam_size=1, length_penalty=0.0):
        if self.training:
            raise ValueError("call eval() before reference-free generation")
        if min(max_phones, max_frames, beam_size) < 1 or length_penalty < 0:
            raise ValueError('invalid generation limits')
        if beam_size > 1:
            from .decoding import beam_generate
            return beam_generate(self, source, source_lengths, beam_size, max_phones, max_frames, length_penalty)
        memory, lengths = self.encode(source, source_lengths)
        context = memory.new_zeros(memory.size(0), self.config.context_dim)
        token = torch.full((source.size(0),), self.BOS, device=source.device, dtype=torch.long)
        done = torch.zeros_like(token, dtype=torch.bool)
        phone_lengths = torch.zeros_like(token)
        state, states, tokens = None, [], []
        for step in range(max_phones + 1):
            scores, hidden, context, state, _ = self.linguistic_step(token, context, state, memory, lengths)
            scores[:, [self.PAD, self.BOS]] = -torch.inf
            if step == 0:
                scores[:, self.EOS] = -torch.inf
            token = scores.argmax(-1)
            done = done | token.eq(self.EOS)
            phone_lengths += ~done
            states.append(hidden)
            tokens.append(token)
            if done.all():
                break
        if not done.all():
            raise ValueError("phoneme decoding reached max_phones without EOS")
        result = self.synthesize(torch.stack(states, 1)[:, :-1], phone_lengths, max_frames=max_frames)
        result.update(phones=torch.stack(tokens, 1), phone_lengths=phone_lengths)
        return result
