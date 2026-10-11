"""Scoped fairseq hook mechanics, without launching real fairseq/GPU training."""
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('cap', [0, 2])
def test_probe_counts_whole_accumulated_update_not_physical_cap(monkeypatch, cap):
    from direct_s2st.s2ut import performance
    from direct_s2st import batch_probe
    measured, reports, chunks = [], [], []
    class Observer:
        def begin(self): pass
        def finish(self, samples): measured.append(samples)
    class Timing:
        def __init__(self): self.seconds = {}
        def measure(self, *args):
            from contextlib import nullcontext
            return nullcontext()
        def report(self, update, amount, **kwargs): reports.append(kwargs)
    class Trainer:
        data_parallel_world_size, cuda = 1, False
        def __init__(self):
            self.updates, self.model, self.optimizer = 0, torch.nn.Linear(1, 1), object()
        def get_num_updates(self): return self.updates
        def train_step(self, samples, raise_oom=False):
            chunks.extend(int(s['nsentences']) for s in samples)
            self.updates += 1
            return {'fixture': True}
        def save_checkpoint(self, *args): pass
        def load_checkpoint(self, *args): pass
        def get_train_iterator(self, *args): pass
    class Task:
        def train_step(self, *args, **kwargs): pass
    audio = SimpleNamespace(get_features_or_waveform=lambda *a: None)
    checkpoints = SimpleNamespace(save_checkpoint=lambda *a: None)
    for name, module in {
        'fairseq': SimpleNamespace(checkpoint_utils=checkpoints),
        'fairseq.trainer': SimpleNamespace(Trainer=Trainer),
        'fairseq.tasks.speech_to_speech': SimpleNamespace(SpeechToSpeechTask=Task),
        'fairseq.data.audio': SimpleNamespace(speech_to_text_dataset=audio),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv('S2ST_TRAIN_OPTIMIZE', '1')
    monkeypatch.setenv('S2ST_TRAIN_ADAPTIVE_BATCH', '0')
    monkeypatch.setenv('S2ST_TRAIN_FIXED_MICROBATCH', str(cap))
    monkeypatch.setenv('S2ST_TRAIN_PRECISION', 'default')
    monkeypatch.delenv('S2ST_TRAIN_DEADLINE', raising=False)
    monkeypatch.setattr(batch_probe, 'Probe', Observer)
    monkeypatch.setattr(performance, 'Timings', Timing)
    samples = [dict(nsentences=n, ntokens=3*n, target_lengths=torch.tensor([3]*n)) for n in (5, 3)]
    original = Trainer.train_step
    with performance.runtime_hooks(None):
        assert Trainer().train_step(samples) == {'fixture': True}
    assert Trainer.train_step is original
    assert measured == [8]
    assert chunks == ([5, 3] if cap == 0 else [2, 2, 1, 2, 1])
    assert reports[0]['microbatch'] == (5 if cap == 0 else 2)
