"""Train-only pooled postnet statistics for an explicit, fresh practical recipe.

The optimizer's native buffers are NEVER replaced in a saved training state.
Evaluation statistics are embedded separately in the checkpoint and validated
before inference. No dev targets, train-mode forwards or gradient steps fit them.
"""
from contextlib import contextmanager
import hashlib

import torch

from ..progress import operation, track
from .batching import collate
from .conformer import MaskedBatchNorm
from .engine import capture_rank_state, restore_rank_state

FORMAT = 'tt2-pooled-postnet-eval-v1'
MAX_SAMPLES = 128
BATCH_SIZE = 4


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Moments:
    """FP64 parallel Welford; pool frames, not batch variances or padded zeros."""
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = None

    def add(self, x, valid):
        if x.ndim != 3 or valid.shape != (x.size(0), x.size(2)) or valid.dtype != torch.bool:
            raise ValueError('expected [B,C,T] and boolean [B,T] mask')
        values = x.detach().transpose(1, 2)[valid].double()
        if not values.numel():
            return
        if not torch.isfinite(values).all():
            raise ValueError('nonfinite valid calibration input')
        n, mean = values.size(0), values.mean(0)
        m2 = (values - mean).square().sum(0)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, mean, m2
            return
        if self.mean.shape != mean.shape:
            raise ValueError('calibration channel mismatch')
        delta, total = mean - self.mean, self.count + n
        self.m2 += m2 + delta.square() * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def finish(self):
        if self.count < 2:
            raise ValueError('at least two valid frames required')
        return self.mean, (self.m2 / (self.count - 1)).clamp_min(0)


def layers(model):
    selected = {name: module for name, module in model.named_modules()
                if name.startswith('postnet.') and isinstance(module, MaskedBatchNorm)}
    if len(selected) != 5:
        raise ValueError('expected five native postnet normalization layers')
    return selected


@contextmanager
def temporary_eval(model):
    """Restore all buffers/RNG/mixed module flags, including on exceptions.

    Parameters and existing gradients are read-only; no expensive parameter or
    Adam clone is needed. Tests compare both, and the next real optimizer step.
    """
    state = capture_rank_state(model)
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        restore_rank_state(model, state)
        for module, training in flags:
            module.training = training


def selection(dataset):
    if getattr(dataset, 'split', None) != 'train':
        raise ValueError('postnet calibration accepts train only')
    ids = [row['id'] for row in dataset.rows]
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('empty/duplicate calibration IDs')
    # Stable under manifest ordering; unrelated to optimizer batch size and RNG.
    indices = sorted(range(len(ids)), key=lambda i: (hashlib.sha256(ids[i].encode()).hexdigest(), ids[i]))[:MAX_SAMPLES]
    return indices, [ids[i] for i in indices]


def contract(dataset):
    _, ids = selection(dataset)
    return dict(format=FORMAT, method='pooled_teacher', selection='sha256_train_id_order_v1',
                max_samples=MAX_SAMPLES, batch_size=BATCH_SIZE, train_ids=ids,
                normalization='layerwise_masked_pooled_unbiased_variance_eval_inputs')


@operation('tt2: train-only pooled postnet calibration')
@torch.no_grad()
def fit(model, dataset, update):
    if type(update) is not int or update < 1:
        raise ValueError('calibration requires an optimizer update')
    if any(module.training for module in model.modules()) or torch.is_grad_enabled():
        raise ValueError('fit in temporary_eval context')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('pooled recipe requires FP32 parameters')
    selected = layers(model)
    indices, _ = selection(dataset)
    device = next(model.parameters()).device
    cached = []
    for start in track(range(0, len(indices), BATCH_SIZE), 'tt2: calibration train batches'):
        batch = collate([dataset[i] for i in indices[start:start+BATCH_SIZE]])
        output = model(**{key: value.to(device) for key, value in batch.items()})
        # Raw acoustic output is independent of postnet buffers. Retain its
        # padded batch geometry to match the original full-model fit exactly.
        cached.append((output['mel'].detach().cpu(), batch['target_lengths']))
        del output, batch
    result = {}
    for name, module in track(selected.items(), 'tt2: pooled postnet layers'):
        moments = Moments()
        for mel, lengths in cached:
            mask = torch.arange(mel.size(1), device=device)[None] < lengths.to(device)[:, None]
            residual = mel.to(device).transpose(1, 2)
            for layer in model.postnet:
                if layer is module:
                    moments.add(residual, mask)
                    break  # No need to run later layers for this fit.
                residual = (layer(residual, mask) if isinstance(layer, MaskedBatchNorm)
                            else layer(residual)) * mask[:, None]
        mean, variance = moments.finish()
        module.running_mean.copy_(mean.to(module.running_mean))
        module.running_var.copy_(variance.to(module.running_var))
        result[name] = dict(mean=module.running_mean.tolist(), variance=module.running_var.tolist(),
                            valid_frames=moments.count)
    return dict(contract(dataset), updates=update, parameter_sha256=parameter_digest(model), layers=result)


def apply(model, overlay, update, expected=None):
    """Fail before first mutation for missing/stale/foreign/malformed metadata."""
    if (not isinstance(overlay, dict) or overlay.get('format') != FORMAT or
            overlay.get('updates') != update or type(update) is not int or update < 1 or
            overlay.get('method') != 'pooled_teacher' or
            overlay.get('parameter_sha256') != parameter_digest(model)):
        raise ValueError('missing/stale/mismatched postnet calibration; use a calibrated best checkpoint')
    if expected is not None and any(overlay.get(key) != value for key, value in expected.items()):
        raise ValueError('postnet calibration train contract mismatch')
    ids = overlay.get('train_ids', [])
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('invalid calibration train IDs')
    selected = layers(model)
    if set(overlay.get('layers', {})) != set(selected):
        raise ValueError('calibration layer set mismatch')
    converted = []
    for name, module in selected.items():
        value = overlay['layers'][name]
        mean = torch.tensor(value['mean'], dtype=module.running_mean.dtype, device=module.running_mean.device)
        variance = torch.tensor(value['variance'], dtype=module.running_var.dtype, device=module.running_var.device)
        if (mean.shape != module.running_mean.shape or variance.shape != module.running_var.shape or
                not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or
                (variance < 0).any() or type(value['valid_frames']) is not int or value['valid_frames'] < 2):
            raise ValueError('invalid pooled normalization buffers')
        converted.append((module, mean, variance))
    with torch.no_grad():
        for module, mean, variance in converted:
            module.running_mean.copy_(mean)
            module.running_var.copy_(variance)


def apply_checkpoint(model, state):
    from .recipe import POOLED
    training = state.get('data_fingerprint', {}).get('training', {})
    if training.get('tt2_recipe') == POOLED:
        expected = training.get('postnet_calibration')
        if not isinstance(expected, dict) or expected.get('format') != FORMAT:
            raise ValueError('missing checkpoint calibration contract')
        apply(model, state.get('eval_calibration'), state['updates'], expected)
