"""Padded mini-batches and deterministic warmup/inverse-square-root schedule."""
import math
import torch


class StreamingBatches:
    """One update without retaining 1024 utterances/features in RAM.

    Counts come from checked TSV/phone labels; loaded batches are checked again
    before backward. A consumed stream is never silently replayed on OOM.
    """
    def __init__(self, samples, dataset, update, batch_size, update_freq, rank=0, world_size=1, timing=None):
        self.samples, self.batch_size, self.frequency = samples, batch_size, update_freq
        first = (update - 1) * update_freq * batch_size * world_size
        indices = [(first + micro * batch_size * world_size + rank * batch_size + i) % len(dataset)
                   for micro in range(update_freq) for i in range(batch_size)]
        self.expected = [(int(dataset.rows[i]['tgt_n_frames']),
                          len(dataset.labels[dataset.rows[i]['id']].split()) + 1) for i in indices]
        self.counts = [sum(v[0] for v in self.expected), sum(v[1] for v in self.expected), len(indices)]
        self.source_frames = 0
        self.timing = timing

    def __len__(self):
        return self.frequency

    def __iter__(self):
        from ..progress import track
        from ..train_runtime import enabled, pin_batch
        from contextlib import nullcontext
        for micro in track(range(self.frequency), 'tt2: accumulate microbatches'):
            with (self.timing.measure('stream_data_wait_and_collate') if self.timing else nullcontext()):
                batch = collate([next(self.samples) for _ in range(self.batch_size)])
                if enabled() and torch.cuda.is_available():
                    batch = pin_batch(batch)
            expected = self.expected[micro * self.batch_size:(micro + 1) * self.batch_size]
            if batch['target_lengths'].tolist() != [v[0] for v in expected] or batch['phone_lengths'].tolist() != [v[1] for v in expected]:
                raise ValueError('loaded accumulation batch differs from immutable manifest counts')
            self.source_frames += int(batch['source_lengths'].sum())
            yield batch


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
