import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
from tt2_schedule_overlay import MONITOR, functions, patched_monitor, prepare


def baseline(revision):
    return subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(),
        '-C', str(ROOT), 'show', revision+':src/'+MONITOR])


@pytest.mark.parametrize('revision', ['d70516a221199262a0fbd26637c6c5a8b86f73df', '6e311c45b1fe95f8646e41172e9dda878f53e570',
                                    '41c6ca22a7c45bd19374999bf10945d3e7ce3804',
                                    '334c14bb9681efd3d6d73b8ab4419c4f5301d7ae', '761a87f', 'f9cbe6d'])
def test_reviewed_overlay_changes_only_five_functions_and_supports_v1(revision, tmp_path):
    original = baseline(revision)
    current = (ROOT/'src'/MONITOR).read_bytes()
    changed = patched_monitor(original, current)
    def without_functions(source):
        lines = source.decode().splitlines(keepends=True)
        for a, b in sorted(functions(source.decode()).values(), reverse=True):
            lines[a:b] = []
        return '\n'.join(line for line in ''.join(lines).splitlines() if line.strip())
    assert without_functions(original) == without_functions(changed)
    ast.parse(changed)
    # Execute the actual patched legacy module, not a mocked scheduler.
    module = {'__name__':'direct_s2st.translatotron2.schedule_fixture', '__package__':'direct_s2st.translatotron2'}
    exec(compile(changed, 'reviewed-overlay', 'exec'), module)
    old = dict(version=1, base_warmup=10000, warmup=1000, adopted_at=85, mode='paper_practical')
    assert module['resume_schedule'](10000, 133, {'schedule_transition':old}, 500)[0] == 500
    from test_tt2_monitor import POLICY, losses, bind_best
    monitor = module['Monitor'](POLICY, 1000)
    for update in range(1, 134):
        if monitor.observe(update, losses(1), losses(1), .001):
            bind_best(monitor, tmp_path)
    assert monitor.state()['warmup'] == 1000
    saved = monitor.state()
    restored = module['Monitor'].restore(None, saved, 133, 500, tmp_path)
    assert restored.history == saved['history']
    restored.observe(134, losses(1), losses(1), .001)
    assert module['Monitor'].restore(None, restored.state(), 134, 500, tmp_path).state() == restored.state()
    policy = dict(version=2, interval=1, patience=5, min_relative_improvement=.01)
    migrated = module['Monitor'].restore(policy, restored.state(), 134, 500, tmp_path)
    assert migrated.history == restored.history
    assert migrated.state()['policy_history'][0]['adopted_at'] == 134
    assert module['Monitor'].restore(None, migrated.state(), 134, 500, tmp_path).state() == migrated.state()
    v3 = module['Monitor'].restore(dict(policy, version=3), migrated.state(), 134, 500, tmp_path)
    assert v3.history == migrated.history and v3.bad_evaluations == 0
    assert v3.state()['stopping_metrics'] == ['loss', 'mel_loss', 'phone_loss', 'duration_loss']
    assert module['Monitor'].restore(None, v3.state(), 134, 500, tmp_path).state() == v3.state()
    assert module['absolute_update_limit']('tt2', 10000, v3.policy) == 1000
    assert module['absolute_update_limit']('s2ut', 10000, v3.policy) == 3000


def test_overlay_is_resumable_and_never_overwrites_changed_copies(tmp_path):
    execution = tmp_path/'execution'
    target = execution/'src'/MONITOR
    target.parent.mkdir(parents=True)
    target.write_bytes(baseline('d70516a221199262a0fbd26637c6c5a8b86f73df'))
    model = execution/'src/direct_s2st/translatotron2/model.py'
    model.write_text('unchanged model')
    cfg = execution/'configs/translatotron2/train-fisher.yaml'
    cfg.parent.mkdir(parents=True)
    cfg.write_text('unchanged config')
    overlay = prepare(execution, ROOT, tmp_path/'persistent', 'fixture')
    assert (overlay/'src/direct_s2st/translatotron2/model.py').read_bytes() == model.read_bytes()
    assert (overlay/'configs/translatotron2/train-fisher.yaml').read_bytes() == cfg.read_bytes()
    assert prepare(execution, ROOT, tmp_path/'persistent', 'fixture') == overlay
    (overlay/'src/direct_s2st/translatotron2/model.py').write_text('changed')
    with pytest.raises(FileExistsError):
        prepare(execution, ROOT, tmp_path/'persistent', 'fixture')
    assert model.read_text() == 'unchanged model'
    with pytest.raises(ValueError, match='reviewed monitor'):
        patched_monitor(target.read_bytes()+b'\n# unexpected edit\n', (ROOT/'src'/MONITOR).read_bytes())
