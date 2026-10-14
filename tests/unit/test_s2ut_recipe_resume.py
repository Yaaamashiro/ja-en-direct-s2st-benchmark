"""Real recipe construction, including the literal argv[0] of legacy -m runs."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from direct_s2st.colab import make_config, publish, latest
from direct_s2st.config import load_config
from direct_s2st.journal import digest
from direct_s2st.s2ut import fairseq_train, preflight
from direct_s2st.s2ut.monitor import S2UTMonitor
from direct_s2st.s2ut.recipe_identity import bind_recipe, MODULE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
from preflight_s2ut_resume import verify, selected, runtime_environment
from rescue_local_s2ut import rescue


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    for key in ('S2ST_INPUT_MAP', 'S2ST_DRIVE_SAFE', 'S2ST_TRAIN_FIXED_MICROBATCH',
                'S2ST_TRAIN_ADAPTIVE_BATCH', 'S2ST_TRAIN_PRECISION'):
        monkeypatch.delenv(key, raising=False)
    source = subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(),
                                     '-C', str(ROOT), 'show', '2486164:src/direct_s2st/s2ut/fairseq_train.py'])
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_research_metadata')
    node.decorator_list = []
    namespace = dict(__file__=str(ROOT/'src/direct_s2st/s2ut/fairseq_train.py'),
                     __package__='direct_s2st.s2ut', Path=Path, json=json, os=os,
                     atomic_write_json=fairseq_train.atomic_write_json, track=fairseq_train.track,
                     sys=sys)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-pinned-recipe', 'exec'), namespace)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'fixture-revision\n')
    checks = []
    def dimensions(*a, **kw):
        checks.append(kw.get('check_audio', True))
        return {'model_dimensions': {'encoder_embed_dim':256}}
    monkeypatch.setattr(preflight, 'validate_training', dimensions)
    unit = load_config(ROOT/'configs/s2ut/prepare.yaml')
    units = dict(hubert_revision=unit['hubert']['revision'], kmeans_sha256=unit['kmeans']['sha256'],
                 hubert_layer=unit['hubert_layer'], kmeans_clusters=unit['kmeans_clusters'])
    data = tmp_path/'data'
    root = data/'s2ut/fairseq'
    root.mkdir(parents=True)
    (root/'data-lock.json').write_text(json.dumps(dict(unit_configuration=units, target='reduced_units')))
    lock = data/'s2ut/units/unit-lock.json'
    lock.parent.mkdir()
    lock.write_text(json.dumps(units))
    environment = tmp_path/'environment.json'
    environment.write_text('{"repository":"fixture"}')
    cfg = make_config(ROOT, data, environment, 's2ut', 'fisher', performance='gpu80',
                      reproduction_mode='paper_practical', s2ut_warmup_updates=400, optimize=True)
    work = tmp_path/'work'
    command = ['/content/s2st-execution/'+'a'*40+'/.fp16-overlays/'+'b'*64+'/src/direct_s2st/s2ut/fairseq_train.py']
    command += [v.format(run_root=str(work), updates='1000') for v in cfg['command'][3:]]
    _, (_, metadata) = namespace['prepare_research_metadata'](command)
    return cfg, work, command, metadata, checks


@pytest.mark.parametrize('entry', [MODULE, 'wrapper', '/content/reviewed-copy/src/direct_s2st/s2ut/fairseq_train.py'])
def test_actual_legacy_fingerprint_and_lock_bytes_survive_relocated_resume(recipe, entry):
    cfg, work, command, metadata, checks = recipe
    marker = work/'s2ut-recipe-lock.json'
    before = marker.read_bytes()
    relocated = [entry]+command[1:]+['--restore-file', str(work/cfg['checkpoint'])]
    relocated[relocated.index('--max-update')+1] = '2000'
    _, (_, current) = fairseq_train.prepare_research_metadata(relocated)
    assert current['training_fingerprint'] == metadata['training_fingerprint']
    assert current['max_updates'] == 2000
    assert marker.read_bytes() == before
    assert checks[-1] is True  # Actual training still checks the audio.


@pytest.mark.parametrize('change', ['warmup', 'batch', 'data', 'physical_cap', 'launcher', 'fingerprint', 'extra_field'])
def test_recipe_guard_only_exempts_argv_zero_and_preserves_original_on_failure(recipe, change):
    cfg, work, command, _, _ = recipe
    marker = work/'s2ut-recipe-lock.json'
    saved = json.loads(marker.read_text(encoding='utf-8'))
    candidate = deepcopy(saved)
    candidate['command'][0] = MODULE
    if change in ('warmup', 'batch'):
        flag = '--warmup-updates' if change == 'warmup' else '--max-tokens'
        candidate['command'][candidate['command'].index(flag)+1] = '9999'
    elif change == 'data': candidate['data']['data-lock.json'] = 'changed'
    elif change == 'physical_cap': candidate['recipe']['physical_batch_cap'] = 32
    elif change == 'launcher': candidate['command'][0] = '/unreviewed/other_trainer.py'
    elif change == 'fingerprint':
        saved['recipe']['training_fingerprint'] = '0'*64
        marker.write_text(json.dumps(saved), encoding='utf-8')
    elif change == 'extra_field': candidate['unexpected'] = True
    before = marker.read_bytes()
    with pytest.raises(ValueError): bind_recipe(marker, candidate, restoring=True)
    assert marker.read_bytes() == before


def test_new_recipe_uses_module_identity_and_missing_resume_lock_never_restarts(recipe):
    _, work, _, _, _ = recipe
    saved = json.loads((work/'s2ut-recipe-lock.json').read_text(encoding='utf-8'))
    marker = work/'new/s2ut-recipe-lock.json'
    with pytest.raises(ValueError, match='missing'):
        bind_recipe(marker, saved, restoring=True)
    assert not marker.exists()
    fingerprint = bind_recipe(marker, saved)
    identity = json.loads(marker.read_text(encoding='utf-8'))
    assert identity['command'][0] == MODULE
    assert identity['recipe']['training_fingerprint'] == fingerprint
    del identity['recipe']['training_fingerprint']
    from direct_s2st.s2ut.recipe_identity import canonical, differences
    assert digest(identity) == fingerprint, list(differences(canonical(saved), identity))
    assert fingerprint != saved['recipe']['training_fingerprint']


def saved_checkpoint(cfg, work, fingerprint, update):
    path = work/cfg['checkpoint']
    path.parent.mkdir(parents=True, exist_ok=True)
    monitor = S2UTMonitor(dict(version=2, interval=10, patience=5, min_relative_improvement=.01), 400)
    for count in range(1, update+1):
        monitor.observe(count, {k:1. for k in monitor.metrics}, None, .001)
    state = dict(model={name+'.weight':torch.ones(1) for name in
        ('encoder', 'source_letter_decoder', 'target_letter_decoder', 'decoder_target_ctc_decoder')},
        optimizer_history=[dict(num_updates=update)], last_optimizer_state={'state':{0:{'exp_avg':torch.zeros(1)}}},
        extra_state=dict(s2st_recipe_fingerprint=fingerprint, s2st_monitor=monitor.state()))
    torch.save(state, path)
    (work/'training-monitor.json').write_text(json.dumps(monitor.state()))
    return path


def durable(recipe, backup):
    from direct_s2st.hashing import sha256_file
    cfg, work, _, metadata, _ = recipe
    saved_checkpoint(cfg, work, metadata['training_fingerprint'], 1)
    identity = dict(config=cfg, work=str(work.resolve()), inputs={p:sha256_file(Path(p)) for p in cfg['identity_files']})
    publish(work, backup, identity, 1)
    return identity


def test_real_recipe_guard_allows_local_rescue_then_predownload_resume(recipe, tmp_path, monkeypatch):
    cfg, work, _, metadata, checks = recipe
    backup = tmp_path/'drive'
    identity = durable(recipe, backup)
    saved_checkpoint(cfg, work, metadata['training_fingerprint'], 9)
    original = (work/'s2ut-recipe-lock.json').read_bytes()
    from direct_s2st import colab, drive_staging
    monkeypatch.setattr(colab, 'stage_training', lambda *a: pytest.fail('audio download before recipe check'))
    monkeypatch.setattr(drive_staging, 'stage_audio', lambda *a, **kw: pytest.fail('audio payload download'))
    import os
    monkeypatch.setenv('S2ST_DRIVE_SAFE', '1')
    assert rescue(cfg, work, backup, 1000) == 9
    assert latest(backup, identity)[1]['updates'] == 9
    assert verify(cfg, work, backup, 1000) == {'status':'RESTORED', 'durable_updates':9}
    assert checks[-2:] == [False, False]
    assert (work/'s2ut-recipe-lock.json').read_bytes() == original
    assert os.environ['S2ST_TRAIN_FIXED_MICROBATCH'] == '0'


def test_predownload_preflight_rejects_checkpoint_fingerprint_without_optimization(recipe, tmp_path, monkeypatch):
    cfg, work, _, metadata, _ = recipe
    backup = tmp_path/'drive'
    durable(recipe, backup)
    saved_checkpoint(cfg, work, 'wrong-fingerprint', 2)
    from direct_s2st.hashing import sha256_file
    identity = dict(config=cfg, work=str(work.resolve()), inputs={p:sha256_file(Path(p)) for p in cfg['identity_files']})
    publish(work, backup, identity, 2)
    with pytest.raises(ValueError, match='checkpoint recipe fingerprint mismatch before input download'):
        verify(cfg, work, backup, 1000)
    assert latest(backup, identity)[1]['updates'] == 2


def test_saved_calibration_is_applied_to_preflight_environment_without_reselection(tmp_path):
    config = dict(kind='s2ut', command=['wrapper', '--reproduction-mode', 'paper_practical'],
                  runtime=dict(adaptive_batch=True, precision='default'), batch_calibration={'steps':3})
    (tmp_path/'batch-calibration.json').write_text(json.dumps({'selected':16}))
    actual = selected(config, tmp_path)
    assert runtime_environment(actual) == dict(S2ST_TRAIN_FIXED_MICROBATCH='16',
                                              S2ST_TRAIN_ADAPTIVE_BATCH='0', S2ST_TRAIN_PRECISION='default')
    assert config['runtime']['adaptive_batch'] is True
