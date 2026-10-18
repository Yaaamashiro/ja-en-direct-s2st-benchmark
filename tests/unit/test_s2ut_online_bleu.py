"""Paper-run routing, real BLEU scoring and bounded CPU online-evaluation contract."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from direct_s2st.s2ut import bleu_monitor as bleu
from direct_s2st.s2ut.paper_monitor import PaperMonitor
from direct_s2st.s2ut.paper_recipe import (PAPER, activate, bleu_policy, report_policy,
    validate_bleu_policy, validate_report_policy)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    import os
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    for key in ('S2ST_S2UT_MONITOR_POLICY', 'S2ST_S2UT_PAPER_MONITOR', 'S2ST_S2UT_BLEU_POLICY',
                'S2ST_INPUT_MAP', 'S2ST_DRIVE_SAFE'):
        monkeypatch.delenv(key, raising=False)


def losses(value):
    return {key: value for key in PaperMonitor.metrics}


def test_no_early_stop_beyond_old_cap_sparse_history_and_exact_resume(tmp_path):
    (tmp_path/'checkpoints').mkdir()
    monitor = PaperMonitor(report_policy(), 10000)
    for update in range(1, 4002):
        improved = monitor.observe(update, losses(10), losses(20) if monitor.due(update) else None, .0001)
        if improved:
            monitor.save_best(tmp_path, lambda p: p.write_bytes(b'full optimizer checkpoint fixture'))
    assert not monitor.stopped and monitor.update == 4001
    assert monitor.best_update == 1000 and len(monitor.history) == 40
    monitor.write_reports(tmp_path)
    before = monitor.state()
    restored = PaperMonitor.restore(report_policy(), before, 4001, 10000, tmp_path)
    assert restored.state() == before
    assert len((tmp_path/'training-monitor.csv').read_text().splitlines()) == 41
    assert 'decoder_target_ctc' in (tmp_path/'training-monitor.svg').read_text()
    bad = copy.deepcopy(before)
    del bad['history'][1]
    with pytest.raises(ValueError, match='incomplete'): PaperMonitor.restore(None, bad, 4001, 10000, tmp_path)
    with pytest.raises(ValueError, match='new run'): PaperMonitor.restore(None, None, 1670, 10000, tmp_path)
    with pytest.raises(ValueError, match='changed'): PaperMonitor.restore(None, {'version':'s2ut-dev-unit-monitor-v1'}, 1670, 10000, tmp_path)
    restored.update = 400000
    with pytest.raises(ValueError, match='schedule'): restored.observe(400001, losses(1), None, .0001)


@pytest.mark.parametrize('field,value', [('beam',1), ('seed',3), ('asr_revision','main'), ('aux_samples',501),
    ('asr_updates',[1000,1000]), ('asr_updates',[400001]), ('asr_updates',[]),
    ('asr_updates',[{}]), ('version',True), ('max_seconds',0)])
def test_policy_rejects_unreviewed_evaluation(field, value):
    policy = bleu_policy()
    policy[field] = value
    with pytest.raises(ValueError): validate_bleu_policy(policy)


def test_policies_are_run_scoped_and_legacy_is_not_adopted(monkeypatch):
    import os
    monkeypatch.setenv('S2ST_S2UT_MONITOR_POLICY', '{"version":3}')
    activate(dict(kind='s2ut', command=['--s2ut-recipe',PAPER],
                  s2ut_paper=dict(monitor=report_policy(),bleu=bleu_policy())))
    assert 'S2ST_S2UT_MONITOR_POLICY' not in os.environ
    assert json.loads(os.environ['S2ST_S2UT_PAPER_MONITOR'])['max_update'] == 400000
    activate(dict(kind='s2ut', command=[]))
    assert 'S2ST_S2UT_PAPER_MONITOR' not in os.environ
    with pytest.raises(ValueError): validate_report_policy(dict(report_policy(), max_update=3000))


def test_fixed_dev_subset_and_failed_hypotheses_remain_in_bleu_denominator():
    rows = [dict(pair_id=str(i),split='dev') for i in range(20)]
    assert bleu.selected_rows(rows,10) == bleu.selected_rows(rows[::-1],10)
    with pytest.raises(ValueError): bleu.selected_rows([dict(pair_id='test',split='test')],1)
    with pytest.raises(ValueError): bleu.selected_rows(rows+rows,10)
    refs = dict(a='this is a complete english sentence', b='another complete english sentence is here')
    perfect = {k:dict(status='success',hypothesis=v) for k,v in refs.items()}
    assert bleu.score(perfect,refs)['bleu'] == pytest.approx(100)
    perfect['b'] = dict(status='failed',hypothesis='')
    result = bleu.score(perfect,refs)
    assert result['samples'] == 2 and result['failures'] == 1 and result['bleu'] < 100
    perfect['b']['hypothesis']=refs['b']
    assert bleu.score(perfect,refs)['bleu']==result['bleu']  # Failed partial output is still empty.
    with pytest.raises(ValueError): bleu.score({'a':perfect['a']},refs)


def test_rng_module_modes_buffers_weights_and_optimizer_unchanged_on_failure():
    model = torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout())
    model.register_buffer('counter',torch.zeros(1))
    model.train(); model[1].eval()
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1,2)).sum().backward(); optimizer.step()
    weights = bleu.weight_hash(model)
    state = copy.deepcopy(optimizer.state_dict())
    modes = [m.training for m in model.modules()]
    py, np_state, rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(RuntimeError):
        with bleu.evaluation_state(model):
            assert not torch.is_grad_enabled() and not model.training
            model.counter.add_(1)
            random.random(); np.random.rand(); torch.rand(1)
            raise RuntimeError('fake ASR failure')
    assert bleu.weight_hash(model) == weights
    assert [m.training for m in model.modules()] == modes
    assert random.getstate() == py and torch.equal(rng,torch.get_rng_state())
    assert np.array_equal(np.random.get_state()[1],np_state[1])
    for key,value in state['state'].items():
        for name,tensor in value.items():
            assert torch.equal(optimizer.state_dict()['state'][key][name],tensor)


def test_auxiliary_free_decode_uses_correct_attachment_not_targets():
    class Dictionary:
        eos=lambda s:2
        bos=lambda s:0
        pad=lambda s:1
        __getitem__=lambda s,i: {3:'h',4:'i',5:'<space>'}[i]
    seen=[]
    context=torch.ones(3,1,2)
    def forward_encoder(**kwargs):
        assert set(kwargs)=={'src_tokens','src_lengths','return_all_hiddens'}
        return dict(encoder_states=[None]*8+[context],encoder_padding_mask=[])
    def decoder(tokens,encoder_out,incremental_state):
        assert encoder_out['encoder_out'][0] is context
        seen.append(tokens.tolist())
        logits=torch.full((1,1,6),-100.)
        logits[0,0,[3,4,2][len(seen)-1]]=10
        return logits,None
    model=SimpleNamespace(forward_encoder=forward_encoder,target_letter_decoder=decoder)
    task=SimpleNamespace(multitask_tasks={'target_letter':SimpleNamespace(
        args=SimpleNamespace(input_from='encoder',input_layer=8),target_dictionary=Dictionary())})
    sample={'net_input':dict(src_tokens=torch.ones(1,3,80),src_lengths=torch.tensor([3]))}
    assert bleu.decode_characters(model,task,sample,256)=='hi'
    assert seen==[[[2]],[[2,3]],[[2,3,4]]]


def online_fixture(tmp_path,monkeypatch):
    root=tmp_path/'data/s2ut/fairseq'
    root.mkdir(parents=True)
    common=tmp_path/'data/common';common.mkdir()
    rows=[dict(pair_id='a',split='dev',en_text='this is a complete english sentence'),
          dict(pair_id='b',split='dev',en_text='another complete english sentence is here')]
    (common/'dev.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'dev.tsv').write_text('id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\na\ta.wav\t3\t0 1\t2\nb\tb.wav\t3\t0 1\t2\n')
    class Dataset:
        __len__=lambda s:2
        __getitem__=lambda s,i:i
        def collater(self,items):
            return dict(id=torch.tensor(items),target='MUST NOT REACH MODEL',
                        net_input=dict(src_tokens=torch.ones(1,3,80),src_lengths=torch.tensor([3]),
                                       prev_output_tokens='GROUND TRUTH'))
    class Dictionary:
        eos=lambda s:2
        __getitem__=lambda s,i:{4:'0',5:'1'}[i]
    class Generator:
        def generate(self,models,sample):
            assert set(sample)=={'net_input'} and set(sample['net_input'])=={'src_tokens','src_lengths'}
            return [[dict(tokens=torch.tensor([4,5,2]))]]
    model=torch.nn.Linear(1,1)
    trainer=SimpleNamespace(get_model=lambda:model)
    task=SimpleNamespace(args=SimpleNamespace(data=str(root)),dataset=lambda s:Dataset(),
        build_generator=lambda *a:Generator(),target_dictionary=Dictionary())
    calls=[]
    def aux(model,task,sample,maximum):
        assert set(sample['net_input'])=={'src_tokens','src_lengths'}
        calls.append('aux');torch.rand(1)
        return 'this is a complete english sentence'
    class Vocoder:
        dict=lambda s,c:torch.ones(1,len(c[0]),2)
        dur_predictor=lambda s,c:torch.zeros(c.shape[:2])
        __call__=lambda s,**kw:torch.sin(torch.arange(1600).float()/10).reshape(1,1,-1)*.1
    def asr(path):
        assert path.is_file()
        calls.append('asr')
        if path.stem=='b': raise ValueError('deliberate sample failure')
        return 'this is a complete english sentence'
    monkeypatch.setattr(bleu,'decode_characters',aux)
    monkeypatch.setattr(bleu,'load_audio_models',lambda *a:(Vocoder(),asr))
    policy=dict(bleu_policy(),aux_samples=2,asr_samples=2,asr_updates=[1000])
    monkeypatch.setenv('S2ST_S2UT_BLEU_POLICY',json.dumps(policy))
    return trainer,task,calls


def test_online_audio_bleu_resume_reports_and_backup(tmp_path,monkeypatch):
    trainer,task,calls=online_fixture(tmp_path,monkeypatch)
    work=tmp_path/'work'
    rng=torch.get_rng_state()
    result=bleu.evaluate(trainer,task,work,1000)
    assert result['asr_samples']==2 and result['asr_failures']==1
    assert result['aux_samples']==2 and result['status']=='completed_with_failures'
    assert trainer.get_model().training and torch.equal(rng,torch.get_rng_state())
    assert calls.count('aux')==2 and calls.count('asr')==2
    bleu.evaluate(trainer,task,work,1000)
    assert calls.count('aux')==2 and calls.count('asr')==3  # Successful rows reused; failure retried.
    assert bleu.evaluate(trainer,task,work,1001) is None
    from direct_s2st.recovery import freeze_files
    checkpoint=work/'checkpoints/checkpoint_last.pt';checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'fixture')
    frozen=freeze_files(work,checkpoint,tmp_path/'frozen')
    assert 'training-bleu.svg' in frozen and 'training-bleu/000001000/identity.json' in frozen
    assert 'training-bleu/000001000/aux.lock.json' in frozen
    assert not any('/audio/' in p or 'bleu-models' in p for p in frozen)
    import shutil
    restored=tmp_path/'restored'
    shutil.copytree(tmp_path/'frozen',restored)
    bleu.evaluate(trainer,task,restored,1000)
    assert calls.count('aux')==2 and calls.count('asr')==4
    with torch.no_grad(): trainer.get_model().weight.add_(1)
    with pytest.raises(FileExistsError): bleu.evaluate(trainer,task,work,1000)


def test_time_budget_does_not_drop_unprocessed_examples(tmp_path,monkeypatch):
    trainer,task,calls=online_fixture(tmp_path,monkeypatch)
    ticks=iter([0])
    monkeypatch.setattr(bleu,'time',SimpleNamespace(monotonic=lambda:next(ticks,2000)))
    result=bleu.evaluate(trainer,task,tmp_path/'work',1000)
    assert result['aux_samples']==result['asr_samples']==2
    assert result['aux_failures']==result['asr_failures']==2
    assert result['aux_bleu']==result['asr_bleu']==0 and calls==[]


def test_new_run_recipe_command_and_legacy_routing(tmp_path,monkeypatch):
    import os
    from direct_s2st.colab import make_config
    from direct_s2st.s2ut import preflight
    from direct_s2st.recipes import option
    from direct_s2st.config import load_config
    cfg=load_config(ROOT/'configs/s2ut/prepare.yaml')
    identity=dict(hubert_model=cfg['hubert']['model'],hubert_revision=cfg['hubert']['revision'],kmeans_sha256=cfg['kmeans']['sha256'],
                  hubert_layer=6,kmeans_clusters=100)
    data=tmp_path/'data';prepared=data/'s2ut/fairseq';prepared.mkdir(parents=True)
    (prepared/'data-lock.json').write_text(json.dumps(dict(unit_configuration=identity,target='reduced_units')))
    lock=data/'s2ut/units/unit-lock.json';lock.parent.mkdir();lock.write_text(json.dumps(identity))
    env=tmp_path/'environment.json';env.write_text('{}')
    monkeypatch.setattr(preflight,'validate_training',lambda *a,**kw:{'model_dimensions':{}})
    config=make_config(ROOT,data,env,'s2ut','fisher',performance='gpu80',s2ut_recipe=PAPER,
                       s2ut_vocoder_mode='paper',optimize=True)
    command=config['command']
    for flag,value in [('--warmup-updates','10000'),('--lr','0.0005'),('--max-tokens','20000'),('--update-freq','4')]:
        assert option(command,flag)==value
    assert '--fp16' in command and option(command,'--s2ut-recipe')==PAPER
    assert str(data/'common/dev.jsonl') in config['identity_files']
    assert config['s2ut_paper']['monitor']['mode']=='report_only'
    import subprocess
    from direct_s2st.s2ut.fairseq_train import prepare_research_metadata
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'fixture-revision\n')
    requested=['wrapper']+[part.format(run_root=str(tmp_path/'run'),updates='500000') for part in command[3:]]
    actual,(_,metadata)=prepare_research_metadata(requested,check_audio=False)
    assert option(actual,'--max-update')=='400000' and '--s2ut-recipe' not in actual
    assert metadata['s2ut_recipe']==PAPER and metadata['independent_early_stopping'] is False
    assert metadata['online_dev_evaluation']['monitor']['interval']==1000
    assert metadata['paper_training_target_requested'] and not metadata['paper_training_length_reached']
    with pytest.raises(ValueError,match='exact Fisher'):
        make_config(ROOT,data,env,'s2ut','fisher',s2ut_recipe=PAPER,s2ut_vocoder_mode='paper',s2ut_warmup_updates=400)
    sys.path.insert(0,str(ROOT/'scripts/colab'))
    from training_session import choose_run
    persistent=tmp_path/'persistent';persistent.mkdir()
    legacy=dict(kind='s2ut',command=[])
    (persistent/'s2ut-old.json').write_text(json.dumps(legacy))
    marker=persistent/'checkpoints/s2ut-old/000001/snapshot.json';marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(dict(updates=1670,identity=dict(config=legacy))))
    assert choose_run(persistent,'s2ut','s2ut-paper',s2ut_recipe=PAPER)==('s2ut-paper',None)
    assert choose_run(persistent,'s2ut','s2ut-paper')[0]=='s2ut-old'
    with pytest.raises(ValueError,match='differs'): choose_run(persistent,'s2ut','s2ut-paper','s2ut-old',s2ut_recipe=PAPER)


def test_notebook_new_default_is_separate_safe_run_and_compiles():
    notebook=json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']),'notebook','exec')
    source=''.join(notebook['cells'][1]['source'])
    scope={};exec(source,scope)
    assert scope['S2UT_RECIPE']==PAPER and scope['S2UT_TOTAL_UPDATES']==400000
    assert scope['S2UT_WARMUP_UPDATES']==10000 and not scope['RUN_TRAINING']
    assert scope['training_run_name']('s2ut').endswith('-paper-fisher-v1')
    assert scope['training_mode']('s2ut')=='paper_exact'
    scope['S2UT_RECIPE']='legacy_short';scope['S2UT_WARMUP_UPDATES']=400
    assert scope['training_run_name']('s2ut').endswith('-recipe-v3-warmup400')


def test_dependency_preflight_checks_imports_and_reports_missing(monkeypatch):
    import importlib
    import importlib.metadata
    from direct_s2st.s2ut.paper_recipe import preflight_dependencies
    checked=[]
    def load(name):
        checked.append(name)
        if name=='editdistance': raise ImportError('missing')
    monkeypatch.setattr(importlib,'import_module',load)
    with pytest.raises(RuntimeError,match='editdistance'): preflight_dependencies()
    assert {'sacrebleu','transformers','sentencepiece','torch','soundfile'} <= set(checked)
    monkeypatch.setattr(importlib,'import_module',lambda n:SimpleNamespace(pipeline=object()))
    monkeypatch.setattr(importlib.metadata,'version',lambda n:'fixed-version')
    assert preflight_dependencies()['transformers']=='fixed-version'
    def native_failure(name):
        if name.startswith('fairseq.'): raise RuntimeError('incompatible registry')
        return SimpleNamespace(pipeline=object())
    monkeypatch.setattr(importlib,'import_module',native_failure)
    with pytest.raises(RuntimeError,match='Python 3.10'): preflight_dependencies()


def test_completed_paper_checkpoint_never_constructs_fairseq_or_steps(tmp_path,monkeypatch,capsys):
    from direct_s2st.s2ut import fairseq_train,checkpoint
    # Boundary-control test, not a native editdistance integration check.
    monkeypatch.setitem(sys.modules,'editdistance',SimpleNamespace(eval=lambda *a:1))
    monkeypatch.setenv('S2ST_S2UT_PAPER_MONITOR',json.dumps(report_policy()))
    command=['wrapper','--restore-file',str(tmp_path/'last.pt'),'--max-update','400000']
    monkeypatch.setattr(fairseq_train,'prepare_research_metadata',lambda argv:(command,None))
    monkeypatch.setattr(torch,'load',lambda *a,**kw:dict(optimizer_history=[dict(num_updates=400000)],
        extra_state=dict(s2st_monitor=dict(policy=report_policy(),stopped=False))))
    checked=[]
    monkeypatch.setattr(checkpoint,'validate_checkpoint',lambda p:checked.append(p))
    fairseq_train.main()
    assert checked==[tmp_path/'last.pt']
    assert 'optimizer_steps=0' in capsys.readouterr().err


def test_cuda_failures_are_fatal_not_scored_as_failed_examples():
    assert bleu.fatal_device_error(torch.cuda.OutOfMemoryError('OOM'))
    assert bleu.fatal_device_error(RuntimeError('CUDA device-side assert triggered'))
    assert not bleu.fatal_device_error(ValueError('decoding did not terminate'))
