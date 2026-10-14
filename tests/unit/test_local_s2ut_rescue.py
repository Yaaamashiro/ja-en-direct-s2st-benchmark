import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
from rescue_local_s2ut import rescue
from direct_s2st.colab import publish, latest
from direct_s2st.s2ut.checkpoint import validate_checkpoint
from direct_s2st.s2ut.monitor import S2UTMonitor


def fixture(tmp_path, monkeypatch):
    from direct_s2st.s2ut import fairseq_train
    monkeypatch.setattr(fairseq_train, 'prepare_research_metadata',
                        lambda command, **kw: (command, (tmp_path/'work', {'training_fingerprint':'synthetic-reviewed'})))
    work = tmp_path/'work'
    (work/'checkpoints').mkdir(parents=True)
    lock = tmp_path/'input.json'
    lock.write_text('immutable input')
    config = dict(kind='s2ut', identity_files=[str(lock)], checkpoint='checkpoints/checkpoint_last.pt',
        command=['{python}', '-m', 'direct_s2st.s2ut.fairseq_train', str(tmp_path/'data'),
                 '--save-dir', '{run_root}/checkpoints', '--max-update', '{updates}'])
    identity = dict(config=config, work=str(work), inputs={str(lock):hashlib.sha256(lock.read_bytes()).hexdigest()})
    def save(update, fingerprint='synthetic-reviewed'):
        # Explicit synthetic fairseq-shaped state; never a production model.
        monitor = S2UTMonitor(dict(version=2, interval=10, patience=5, min_relative_improvement=.01), 400)
        values = {key:1. for key in monitor.metrics}
        for count in range(1, update+1): monitor.observe(count, values, None, .001)
        state = dict(model={name+'.weight':torch.ones(1) for name in
                     ('encoder', 'source_letter_decoder', 'target_letter_decoder', 'decoder_target_ctc_decoder')},
            optimizer_history=[dict(num_updates=update)], last_optimizer_state={'state':{0:{'exp_avg':torch.zeros(1)}}},
            extra_state=dict(s2st_recipe_fingerprint=fingerprint, s2st_monitor=monitor.state()))
        torch.save(state, work/config['checkpoint'])
        (work/'training-monitor.json').write_text(json.dumps(monitor.state()))
        return state
    save(1)
    publish(work, tmp_path/'drive', identity, 1)
    return work, config, identity, save


def test_newer_local_checkpoint_is_verified_and_published_before_drive_restore(tmp_path, monkeypatch):
    work, cfg, identity, save = fixture(tmp_path, monkeypatch)
    save(9)
    before = (work/cfg['checkpoint']).read_bytes()
    assert rescue(cfg, work, tmp_path/'drive', 1000) == 9
    assert latest(tmp_path/'drive', identity)[1]['updates'] == 9
    assert len(list((tmp_path/'drive').glob('*/snapshot.json'))) == 2
    assert (work/cfg['checkpoint']).read_bytes() == before
    assert rescue(cfg, work, tmp_path/'drive', 1000) is None


@pytest.mark.parametrize('invalid', ['recipe', 'report', 'optimizer', 'inputs', 'loss_scale', 'empty_optimizer'])
def test_local_rescue_refuses_mismatched_state_without_changing_durable_point(tmp_path, monkeypatch, invalid):
    work, cfg, identity, save = fixture(tmp_path, monkeypatch)
    state = save(9, fingerprint='wrong-recipe' if invalid == 'recipe' else 'synthetic-reviewed')
    if invalid == 'report': (work/'training-monitor.json').write_text('{}')
    if invalid == 'optimizer':
        state['last_optimizer_state']['state'][0]['exp_avg'] = torch.tensor(float('nan'))
        torch.save(state, work/cfg['checkpoint'])
    if invalid in ('loss_scale', 'empty_optimizer'):
        if invalid == 'loss_scale': state['last_optimizer_state']['loss_scale'] = -1.
        else: state['last_optimizer_state']['state'] = {}
        torch.save(state, work/cfg['checkpoint'])
    if invalid == 'inputs': Path(cfg['identity_files'][0]).write_text('changed data')
    with pytest.raises(ValueError): rescue(cfg, work, tmp_path/'drive', 1000)
    assert latest(tmp_path/'drive', identity)[1]['updates'] == 1


def test_checkpoint_model_monitor_and_optimizer_integrity(tmp_path, monkeypatch):
    work, cfg, _, save = fixture(tmp_path, monkeypatch)
    state = save(9)
    state['extra_state']['s2st_monitor']['bad_evaluations'] = 1
    torch.save(state, work/cfg['checkpoint'])
    with pytest.raises(ValueError, match='history/counters'):
        validate_checkpoint(work/cfg['checkpoint'])
