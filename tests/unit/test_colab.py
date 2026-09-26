import json
from pathlib import Path
import subprocess
import pytest
from direct_s2st.colab import latest, publish, run_session, safe_relative


def fixture(tmp_path):
    lock = tmp_path / 'data-lock.json'
    lock.write_text('{}')
    cfg = dict(kind='tt2', checkpoint='checkpoints/checkpoint_last.pt',
        identity_files=[str(lock)], command=['fixture', '{run_root}', '{updates}'],
        resume_args=['--restore-file', '{checkpoint}'])
    calls = []
    def runner(args, **kwargs):
        calls.append(args)
        path = Path(args[1]) / cfg['checkpoint']
        path.parent.mkdir(parents=True, exist_ok=True)
        # Explicit synthetic metadata fixture, never a production checkpoint.
        path.write_text(args[2])
        return subprocess.CompletedProcess(args, 0)
    return cfg, calls, runner


def test_chunk_resume_and_snapshot_integrity(tmp_path):
    cfg, calls, runner = fixture(tmp_path)
    kwargs = dict(work=tmp_path/'work', backup=tmp_path/'drive', runner=runner,
                  inspect_checkpoint=lambda p, k: int(p.read_text()))
    assert run_session(cfg, **kwargs)['durable_updates'] == 2
    assert len(calls) == 2 and '--restore-file' in calls[1]
    assert run_session(cfg, total=3, resume=True, **kwargs)['durable_updates'] == 3
    assert list(tmp_path.glob('work.interrupted-*'))
    with pytest.raises(FileExistsError):
        run_session(cfg, **kwargs)
    Path(cfg['identity_files'][0]).write_text('changed')
    with pytest.raises(ValueError, match='identity'):
        run_session(cfg, resume=True, **kwargs)


def test_corrupt_and_incomplete_snapshot_falls_back(tmp_path):
    work = tmp_path/'work'
    work.mkdir()
    (work/'state').write_text('one')
    first = publish(work, tmp_path/'drive', {'a': 1}, 1)
    second = publish(work, tmp_path/'drive', {'a': 1}, 2)
    (second/'files/state').write_text('broken')
    (tmp_path/'drive/999999999999-incomplete').mkdir()
    assert latest(tmp_path/'drive', {'a': 1})[0] == first


def test_timeout_keeps_last_committed_chunk(tmp_path):
    cfg, calls, runner = fixture(tmp_path)
    def timeout(args, **kwargs):
        if calls:
            raise subprocess.TimeoutExpired(args, 1)
        return runner(args, **kwargs)
    result = run_session(cfg, work=tmp_path/'work', backup=tmp_path/'drive', runner=timeout,
                         inspect_checkpoint=lambda p, k: int(p.read_text()))
    assert result == dict(status='PAUSED', durable_updates=1)


def test_no_checkpoint_no_publish(tmp_path):
    cfg, _, _ = fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_session(cfg, work=tmp_path/'work', backup=tmp_path/'drive',
            runner=lambda *a, **k: subprocess.CompletedProcess(a, 0),
            inspect_checkpoint=lambda p, k: int(p.read_text()))
    assert not (tmp_path/'drive').exists()


@pytest.mark.parametrize('value', ['../escape', '/absolute', 'C:/drive', 'a\\b'])
def test_snapshot_path_rejection(value):
    with pytest.raises(ValueError):
        safe_relative(value)


def test_scope_and_full_guards(tmp_path, monkeypatch):
    cfg, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='confirm-training'):
        run_session(cfg, work=tmp_path/'work', backup=tmp_path/'drive', total=11)
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='CORPUS_ROOT'):
        run_session(cfg, work=tmp_path/'work', backup=tmp_path/'drive')


def test_real_tt2_checkpoint_survives_snapshot_resume(tmp_path):
    torch = pytest.importorskip('torch')
    from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
    from direct_s2st.translatotron2.engine import optimization_step, save_checkpoint, load_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(9)
    batch = dict(source=torch.randn(1, 24, 80), source_lengths=torch.tensor([24]),
        phones=torch.tensor([[3, 4, 2]]), phone_lengths=torch.tensor([3]),
        target=torch.randn(1, 12, 80), target_lengths=torch.tensor([12]))
    cfg, _, _ = fixture(tmp_path)
    def runner(args, **kwargs):
        checkpoint = Path(args[1]) / cfg['checkpoint']
        if '--restore-file' in args:
            model, state = load_checkpoint(checkpoint, restore_rng=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            optimizer.load_state_dict(state['optimizer'])
        else:
            model = Translatotron2(ModelConfig.smoke(), 6)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimization_step(model, optimizer, batch)
        save_checkpoint(checkpoint, model, optimizer, int(args[2]), list('abcdef'), {}, overwrite=True)
        return subprocess.CompletedProcess(args, 0)
    kwargs = dict(work=tmp_path/'work', backup=tmp_path/'drive', runner=runner)
    run_session(cfg, total=1, **kwargs)
    model, state = load_checkpoint(tmp_path/'work'/cfg['checkpoint'], restore_rng=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(state['optimizer'])
    optimization_step(model, optimizer, batch)
    run_session(cfg, total=2, resume=True, **kwargs)
    restored, _ = load_checkpoint(tmp_path/'work'/cfg['checkpoint'])
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name


def test_notebook_cells_are_valid_python():
    import ast
    notebook = Path(__file__).resolve().parents[2] / 'notebooks/colab_training.ipynb'
    state = json.loads(notebook.read_text(encoding='utf-8'))
    for cell in state['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs'] == []
