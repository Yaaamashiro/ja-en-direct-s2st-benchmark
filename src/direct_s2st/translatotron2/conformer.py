"""Relative-position Conformer with length-aware convolution/normalization.

Independent implementation of Conformer/Transformer-XL component equations;
see docs/REFERENCE_PARITY.md. No modifications to installed fairseq/torchaudio.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def mask_for(lengths, width):
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]


class MaskedBatchNorm(nn.Module):
    def __init__(self, channels, momentum=0.1, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.momentum, self.eps = momentum, eps

    def forward(self, x, valid):
        mask = valid[:, None].to(x)
        if self.training:
            count = mask.sum().clamp_min(1)
            mean = (x * mask).sum((0, 2)) / count
            variance = ((x - mean[None, :, None]).square() * mask).sum((0, 2)) / count
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_((variance * count / (count - 1).clamp_min(1)).detach(), self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            mean, variance = self.running_mean, self.running_var
        output = (x - mean[None, :, None]) * torch.rsqrt(variance[None, :, None] + self.eps)
        return (output * self.weight[None, :, None] + self.bias[None, :, None]) * mask


class RelativeAttention(nn.Module):
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        if dimension % heads or dimension % 2:
            raise ValueError('relative attention needs even dimension divisible by heads')
        self.dimension, self.heads = dimension, heads
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.relative = nn.Linear(dimension, dimension, bias=False)
        self.content_bias = nn.Parameter(torch.zeros(heads, dimension // heads))
        self.position_bias = nn.Parameter(torch.zeros(heads, dimension // heads))
        self.output = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        batch, width, dimension = x.shape
        q, k, v = self.qkv(x).reshape(batch, width, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        positions = torch.arange(1-width, width, device=x.device, dtype=x.dtype)
        frequencies = torch.exp(torch.arange(0, dimension, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.) / dimension))
        phase = positions[:, None] * frequencies[None]
        encoding = torch.cat([phase.sin(), phase.cos()], -1)
        relative = self.relative(encoding).view(2*width-1, self.heads, -1).permute(1, 0, 2)
        content_scores = torch.matmul(q + self.content_bias[None, :, None], k.transpose(-1, -2))
        relative_scores = torch.einsum('bhtd,hrd->bhtr', q + self.position_bias[None, :, None], relative)
        index = torch.arange(width, device=x.device)
        offsets = index[:, None] - index[None, :] + width - 1
        position_scores = relative_scores.gather(-1, offsets[None, None].expand(batch, self.heads, -1, -1))
        scores = (content_scores + position_scores) / math.sqrt(dimension // self.heads)
        scores = scores.masked_fill(~valid[:, None, None], -torch.inf)
        weights = self.dropout(scores.softmax(-1))
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, width, dimension)
        return self.output(output) * valid[:, :, None]


class ConformerBlock(nn.Module):
    def __init__(self, dimension, heads, kernel, dropout):
        super().__init__()
        self.kernel = kernel
        def feedforward():
            return nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension*4),
                nn.SiLU(), nn.Dropout(dropout), nn.Linear(dimension*4, dimension), nn.Dropout(dropout))
        self.ff1, self.ff2 = feedforward(), feedforward()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = RelativeAttention(dimension, heads, dropout)
        self.conv_norm = nn.LayerNorm(dimension)
        self.pointwise_in = nn.Conv1d(dimension, dimension*2, 1)
        self.depthwise = nn.Conv1d(dimension, dimension, kernel, groups=dimension)
        self.batch_norm = MaskedBatchNorm(dimension)
        self.pointwise_out = nn.Conv1d(dimension, dimension, 1)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(dimension)

    def forward(self, x, valid):
        mask = valid[:, :, None]
        x = (x + 0.5*self.ff1(x)) * mask
        x = (x + self.dropout(self.attention(self.attention_norm(x), valid))) * mask
        convolution = self.conv_norm(x).transpose(1, 2)
        convolution = F.glu(self.pointwise_in(convolution), dim=1) * valid[:, None]
        left = (self.kernel - 1) // 2
        convolution = self.depthwise(F.pad(convolution, (left, self.kernel - 1 - left)))
        convolution = self.pointwise_out(F.silu(self.batch_norm(convolution, valid))).transpose(1, 2)
        x = (x + self.dropout(convolution)) * mask
        return self.final_norm(x + 0.5*self.ff2(x)) * mask


class RelativeConformer(nn.Module):
    def __init__(self, dimension, heads, layers, kernel, dropout):
        super().__init__()
        self.layers = nn.ModuleList(ConformerBlock(dimension, heads, kernel, dropout) for _ in range(layers))

    def forward(self, x, lengths):
        valid = mask_for(lengths, x.size(1))
        for layer in self.layers:
            x = layer(x, valid)
        return x, lengths


class ConvSubsample2d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(1, output_dim, 3, stride=2, padding=1),
                                     nn.Conv2d(output_dim, output_dim, 3, stride=2, padding=1)])
        self.projection = nn.Linear(output_dim*((input_dim+3)//4), output_dim)

    def forward(self, x, lengths):
        x = x[:, None]
        for layer in self.layers:
            x = F.relu(layer(x))
            lengths = (lengths + 1)//2
            x = x * mask_for(lengths, x.size(2))[:, None, :, None]
        x = self.projection(x.transpose(1, 2).flatten(2))
        return x * mask_for(lengths, x.size(1))[:, :, None], lengths
