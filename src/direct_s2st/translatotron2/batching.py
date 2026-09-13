"""Padded mini-batches and deterministic warmup/inverse-square-root schedule."""
import math
import torch


def collate(samples):
    if not samples:
        raise ValueError('empty mini-batch')
    batch = {}
    for field, length_field in [('source', 'source_lengths'), ('phones', 'phone_lengths'), ('target', 'target_lengths')]:
        values = []
        for sample in samples:
            values.extend(sample[field][index, :int(length)] for index, length in enumerate(sample[length_field]))
        batch[field] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
        batch[length_field] = torch.tensor([len(value) for value in values])
    return batch


def learning_rate(base, update, warmup):
    if base <= 0 or update < 1 or warmup < 0:
        raise ValueError('invalid learning-rate schedule')
    return base if warmup == 0 else base * min(update / warmup, math.sqrt(warmup / update))
