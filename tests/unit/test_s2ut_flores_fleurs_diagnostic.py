import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

from direct_s2st.s2ut import flores_fleurs_diagnostic as diag
from direct_s2st.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from direct_s2st.benchmark_storage import read
from direct_s2st.drive_staging import raw_sha
from direct_s2st.journal import digest
from direct_s2st.cascade.flores_fleurs_data import sha_text
from direct_s2st.vocoders.reference import UNIT_CONFIGURATION

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/colab'))
import s2ut_flores_fleurs_diagnostic as launcher


def wav(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,np.sin(np.arange(1600)/10)*.1,16000,subtype='PCM_16')


def source_fixture(tmp_path, n=3):
    persistent = tmp_path/'drive/experiment'
    samples = [dict(sample_id=f'flores:devtest:{i}',flores_split='devtest',flores_id=str(i),
        fleurs_split='test',fleurs_id=str(i+500),ja_text=f'日本語{i}。',
        en_reference=f'the very old book is on the table {i}',
        locator=dict(split='test',row_group=0,offset=i)) for i in range(n)]
    for row in samples:
        row.update(ja_text_sha256=sha_text(row['ja_text']),en_text_sha256=sha_text(row['en_reference']))
    audit = dict(selected_samples=samples,selected=n,counts=dict(matched=500),
                 datasets=[dict(repository='google/fleurs',revision='a'*40),
                           dict(repository='openlanguagedata/flores_plus',revision='b'*40)])
    environment = dict(python='3.10.18', packages={k:'v' for k in ('torch','torchaudio','transformers',
        'tokenizers','safetensors','numpy','librosa','soundfile','soxr','sacrebleu','accelerate','huggingface-hub')})
    plan = dict(purpose=diag.CASCADE_PURPOSE,selected=samples,dataset=audit,
                context=dict(evaluation_asr=diag.ASR),environment=environment,
                implementation={k:raw_sha(ROOT/'src/direct_s2st'/k) for k in
                    ('cascade/asr.py','cascade/whisper.py','evaluation/transcribe.py','evaluation/bleu.py')})
    root = persistent/'diagnostics/flores-fleurs-cascade'/digest(plan)[:24]
    atomic_write_json(root/'identity.json',plan)
    atomic_write_json(root/'dataset-audit.json',audit)
    atomic_write_json(root/'execution-audit/evaluation_asr.json',
        dict(adapter='EvaluationASR English transcribe',dtype='torch.float16',generation_config={}))
    rows=[]
    for sample in samples:
        path = root/'natural'/(sample['flores_id']+'.wav'); wav(path)
        audio = diag.audio_audit(path)
        request = dict(run=digest(plan),sample_id=sample['sample_id'],condition='natural',stage='natural_audio',
            inputs=dict(locator=sample['locator'],dataset=audit['datasets'],text_sha=sample['ja_text_sha256']))
        body = dict(request=request,status='success',audio=[audio])
        atomic_write_json(root/'stage-results'/(digest([sample['sample_id'],'natural','natural_audio'])+'.json'),
                          dict(body=body,sha256=digest(body)))
        rows.append(dict(sample,natural_input_audio_status='success',natural_input_audio_path=str(path),
                         natural_audio_path=str(path),natural_input_audio_sha256=raw_sha(path)))
    atomic_write_jsonl(root/'per_sample.jsonl',rows)
    atomic_write_json(root/'metrics.json',dict(purpose=diag.CASCADE_PURPOSE,identity_sha256=digest(plan),
        metrics=dict(samples=n,natural_final_ASR_BLEU=21.153432814483),per_sample=rows))
    return persistent,root


def experiment(tmp_path, n=3):
    persistent,source = source_fixture(tmp_path,n)
    prepared = persistent/'data/s2ut/fairseq'
    prepared.mkdir(parents=True)
    (prepared/'config.yaml').write_text(yaml.safe_dump(dict(input_channels=1,input_feat_per_channel=80,
        transforms={'*':['utterance_cmvn'],'_train':['utterance_cmvn','specaugment']})),encoding='utf-8')
    (prepared/'dict.txt').write_text(''.join(f'{i} 1\n' for i in range(100)),encoding='utf-8')
    tasks={}
    for task in ('source_letter','target_letter','decoder_target_ctc'):
        d=prepared/task;d.mkdir();(d/'dict.txt').write_text('x 1\n',encoding='utf-8')
        (d/'test.tsv').write_text('original private labels',encoding='utf-8')
        tasks[task]=dict(data=d.as_posix(),dict=(d/'dict.txt').as_posix(),decoder_type='ctc')
    (prepared/'config_multitask.yaml').write_text(yaml.safe_dump(tasks),encoding='utf-8')
    atomic_write_json(prepared/'data-lock.json',dict(unit_configuration=UNIT_CONFIGURATION))
    (prepared/'test.tsv').write_text('original training test',encoding='utf-8')
    (prepared/'train.tsv').write_text('original train',encoding='utf-8')
    common=persistent/'data/common';atomic_write_jsonl(common/'test.jsonl',[dict(original='test')])
    work=tmp_path/'local/models/s2ut-run'
    checkpoint=work/'checkpoints/checkpoint_best.pt';checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'trusted checkpoint fixture')
    run=dict(name='s2ut-run',work=str(work),updates=400000,data_root=str(prepared),
             environment_sha256='e'*64)
    metadata={str(p):raw_sha(p) for p in prepared.rglob('*') if p.is_file()}
    context=diag.source_context(source)
    plan=diag.prepare_plan(persistent,tmp_path/'external',context,run,metadata,checkpoint,390000,'c'*40)
    return plan


def snapshot(path):
    return {str(p):p.read_bytes() for p in Path(path).rglob('*') if p.is_file()}


def test_exact_saved_sample_reference_audio_identity_and_no_training_changes(tmp_path):
    plan=experiment(tmp_path)
    source=Path(plan['source']['root']);training=Path(plan['run']['data_root']).parents[1]
    original={**snapshot(source),**snapshot(training)}
    prepared=diag.prepare_external(plan)
    assert {r['sample_id'] for r in plan['source']['samples']}=={r['sample_id'] for r in read(source/'identity.json')['selected']}
    assert all(sha_text(r['en_reference'])==r['en_text_sha256'] for r in plan['source']['samples'])
    from direct_s2st.s2ut.multitask import read_tsv
    tsv=read_tsv(prepared/'test.tsv',('id','src_audio','src_n_frames','tgt_audio','tgt_n_frames'))
    for row in plan['source']['samples']:
        r=tsv[row['pair_id']]
        assert r['tgt_audio']=='0' and r['tgt_n_frames']=='1' and r['src_n_frames']=='10'
        assert raw_sha(r['src_audio'])==row['source_sha256']
        assert row['en_reference'] not in (prepared/'test.tsv').read_text(encoding='utf-8')
    assert (prepared/'config.yaml').read_bytes()==(Path(plan['run']['data_root'])/'config.yaml').read_bytes()
    for task,cfg in diag.read_yaml(prepared/'config_multitask.yaml').items():
        assert Path(cfg['data']).is_relative_to(prepared)
        assert raw_sha(cfg['dict'])==plan['metadata'][str(Path(plan['run']['data_root'])/task/'dict.txt')]
        assert all(r['tgt_text']=='<unk>' for r in read_tsv(Path(cfg['data'])/'test.tsv',('id','tgt_text')).values())
    diag.prepare_external(plan)
    assert original=={**snapshot(source),**snapshot(training)}


@pytest.mark.parametrize('mode',['missing','duplicate','text','reference','audio_sha','receipt'])
def test_reject_corrupt_or_misaligned_8c_artifacts(tmp_path,mode):
    _,root=source_fixture(tmp_path)
    rows=list(read_jsonl(root/'per_sample.jsonl'))
    if mode=='missing':rows.pop()
    elif mode=='duplicate':rows[1]=copy.deepcopy(rows[0])
    elif mode=='text':rows[0]['ja_text']='changed'
    elif mode=='reference':rows[0]['en_reference']='changed'
    elif mode=='audio_sha':Path(rows[0]['natural_audio_path']).write_bytes(b'changed WAV')
    else:
        p=next((root/'stage-results').glob('*.json'));doc=read(p);doc['sha256']='0'*64
        atomic_write_json(p,doc,overwrite=True)
    atomic_write_jsonl(root/'per_sample.jsonl',rows,overwrite=True)
    with pytest.raises((ValueError,RuntimeError,OSError)):
        diag.source_context(root)


def test_reference_and_source_text_sha_are_checked_even_if_reports_agree(tmp_path):
    _,root=source_fixture(tmp_path)
    plan=read(root/'identity.json');plan['selected'][0]['en_text_sha256']='0'*64
    plan['dataset']['selected_samples'][0]['en_text_sha256']='0'*64
    new=root.parent/digest(plan)[:24];root.rename(new)
    # Fail closed even though an edited identity has a matching directory.
    atomic_write_json(new/'identity.json',plan,overwrite=True)
    with pytest.raises(ValueError):diag.source_context(new)


def test_checkpoint_change_new_identity_and_modified_bound_data_rejected(tmp_path):
    plan=experiment(tmp_path)
    Path(plan['checkpoint']).write_bytes(b'changed model')
    other=diag.prepare_plan(Path(plan['source']['root']).parents[2],Path(plan['local']).parent,
        plan['source'],plan['run'],plan['metadata'],plan['checkpoint'],390000,'c'*40)
    assert other['identity']!=plan['identity'] and other['output']!=plan['output']
    with pytest.raises(ValueError,match='checkpoint SHA'):diag.verify_plan(plan)
    p=Path(plan['run']['data_root'])/'config.yaml';p.write_text('modified',encoding='utf-8')
    with pytest.raises(ValueError,match='metadata changed'):diag.verify_plan(other)


def test_per_sample_order_irrelevant_canonical_id_join(tmp_path):
    _,root=source_fixture(tmp_path)
    before=diag.source_context(root)
    atomic_write_jsonl(root/'per_sample.jsonl',reversed(list(read_jsonl(root/'per_sample.jsonl'))),overwrite=True)
    after=diag.source_context(root)
    assert after['samples']==before['samples'] and after['cascade_natural_bleu']==before['cascade_natural_bleu']


def test_source_changes_after_plan_are_rejected(tmp_path):
    plan=experiment(tmp_path)
    p=Path(plan['source']['root'])/'metrics.json'
    p.write_text(p.read_text(encoding='utf-8')+' ',encoding='utf-8')
    with pytest.raises(ValueError,match='source artifact changed'):diag.prepare_external(plan)


def test_selector_prefers_saved_300_and_explicit_never_escapes(tmp_path):
    persistent,root=source_fixture(tmp_path,n=300)
    other=root.parent/'another';atomic_write_json(other/'metrics.json',dict(purpose=diag.CASCADE_PURPOSE,metrics={'samples':10}))
    assert diag.choose_cascade(persistent)==root
    assert diag.choose_cascade(persistent,root.name)==root
    with pytest.raises(RuntimeError):diag.choose_cascade(persistent,'../escape')
    with pytest.raises(ValueError):diag.choose_cascade(tmp_path/'unknown')


def runner_fixture(calls,stop_at=None):
    def run(command,**kwargs):
        shard=int(command[command.index('--distributed-rank')+1])
        count=int(command[command.index('--distributed-world-size')+1])
        calls.append(shard)
        if shard==stop_at:raise KeyboardInterrupt('fixture interrupt')
        assert command[2]=='direct_s2st.s2ut.external_generate'
        assert command[command.index('--prefix-size')+1]=='0'
        prepared=Path(command[3])
        from direct_s2st.s2ut.multitask import read_tsv
        ids=list(read_tsv(prepared/'test.tsv',('id','src_audio','src_n_frames','tgt_audio','tgt_n_frames')))
        destination=Path(command[command.index('--results-path')+1]);destination.mkdir(parents=True,exist_ok=True)
        (destination/'generate-test.txt').write_text(''.join(f'S-{i}\t0\nD-{i}\t-1\t1 2 3\n'
            for i in range(len(ids)) if i%count==shard),encoding='utf-8')
    return run


def test_units_resume_after_runtime_loss_reuses_done_shards_and_no_reference_leak(tmp_path):
    plan=experiment(tmp_path,n=35);diag.prepare_external(plan)
    calls=[]
    with pytest.raises(KeyboardInterrupt):diag.units(plan,runner=runner_fixture(calls,1))
    assert calls==[0,1]
    local=Path(plan['local']);shutil.rmtree(local/'generation')
    result=diag.units(plan,runner=runner_fixture(calls))
    assert calls==[0,1,1] and len(result)==35
    diag.units(plan,runner=lambda *a,**kw:pytest.fail('no regeneration'))
    assert len(list(read_jsonl(Path(plan['output'])/'units.jsonl')))==35
    assert all('reference_text' in r for r in result)  # attached AFTER runner


def test_source_only_adapter_removes_targets_aux_and_prefixes_without_mutating_sample():
    from direct_s2st.s2ut.external_generate import source_only
    observed=[]
    task=SimpleNamespace(inference_step=lambda g,m,s,**kw:observed.append(s))
    tasks=SimpleNamespace(setup_task=lambda:task)
    original=tasks.setup_task
    sample=dict(net_input=dict(src_tokens='source',src_lengths=10,prev_output_tokens='DUMMY',tgt_speaker=None),
                target='DUMMY',target_lengths=1,multitask={'dummy':'labels'})
    with source_only(tasks):
        t=tasks.setup_task();t.inference_step(None,None,sample)
        assert observed[0]['target'] is None and 'multitask' not in observed[0]
        assert 'prev_output_tokens' not in observed[0]['net_input']
        assert observed[0]['net_input']['src_tokens']=='source'
        with pytest.raises(ValueError,match='prefix'):t.inference_step(None,None,sample,prefix_tokens='reference')
    assert tasks.setup_task is original and sample['target']=='DUMMY'


@pytest.mark.parametrize('content,expected',[
    ('D-0\t0\t1 2\n',('success','failed')),
    ('D-0\t0\t999\nD-1\t0\t1\n',('failed','success')),
    ('D-0\t0\t\n',('failed','failed'))])
def test_sample_generation_failure_remains_in_denominator(tmp_path,content,expected):
    p=tmp_path/'gen.txt';p.write_text(content,encoding='utf-8')
    rows=diag.parse_units(p,['a','b'])
    assert tuple(rows[k]['status'] for k in ['a','b'])==expected


@pytest.mark.parametrize('content',['D-0\t0\t1\nD-0\t0\t1\n','D-2\t0\t1\n','D-0\tbroken\n'])
def test_invalid_numeric_alignment_fatal(tmp_path,content):
    p=tmp_path/'gen.txt';p.write_text(content,encoding='utf-8')
    with pytest.raises(ValueError):diag.parse_units(p,['a','b'])


def install_vocoder(monkeypatch,calls):
    from direct_s2st.vocoders.reference import CHECKPOINT_SHA256,CONFIG_SHA256
    def ensure(checkpoint,config):
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'fixture')
        atomic_write_json(config,dict(sampling_rate=16000),overwrite=True)
    def vocode(kind,inputs,output,*args,**kwargs):
        assert kind=='unit' and kwargs['duration_prediction'] and kwargs['fatal_cuda']
        calls.append('vocoder');rows=[]
        for i,r in enumerate(read_jsonl(inputs)):
            path=output/'audio'/f'{i:08d}.wav'
            if r['status']=='success':wav(path)
            rows.append(dict(r,output_audio=str(path) if r['status']=='success' else None,
                output_sha256=raw_sha(path) if r['status']=='success' else None,
                output_duration=.1 if r['status']=='success' else None,real_time_factor=.1))
        atomic_write_jsonl(output/'predictions.jsonl',rows,overwrite=True)
        atomic_write_json(output/'predictions.lock.json',dict(backend='fairseq_unit_hifigan',
            checkpoint_sha256=CHECKPOINT_SHA256,config_sha256=CONFIG_SHA256,input_sha256=raw_sha(inputs),
            sample_rate=16000,duration_prediction=True,mel=None),overwrite=True)
    monkeypatch.setattr('direct_s2st.vocoders.reference.ensure_reference',ensure)
    monkeypatch.setattr('direct_s2st.vocoders.inference.vocode',vocode)


def test_real_zip_journal_pipeline_resume_failures_and_originals_untouched(tmp_path,monkeypatch):
    from direct_s2st.s2ut.benchmark_worker import audio
    plan=experiment(tmp_path);source=Path(plan['source']['root']);training=Path(plan['run']['data_root'])
    before={**snapshot(source),**snapshot(training)}
    diag.prepare_external(plan);diag.units(plan,runner=runner_fixture([]))
    local,output=Path(plan['local']),Path(plan['output'])
    units=list(read_jsonl(local/'units/predictions.jsonl'))
    units[1].update(status='failed',error='fixture generation failure',units=[],units_sha256=digest([]))
    atomic_write_jsonl(local/'units/predictions.jsonl',units,overwrite=True)
    atomic_write_jsonl(output/'units.jsonl',units,overwrite=True)
    calls=[];install_vocoder(monkeypatch,calls);audio(plan)
    def load(**kw):
        assert kw==dict(plan['asr'],device=0)
        def transcribe(path):
            calls.append('asr')
            return 'the very old book is on the table'
        return transcribe
    result=diag.evaluate(plan,asr_factory=load)
    assert result['metrics']['sample_count']==3 and result['metrics']['generation_failures']==1
    assert result['metrics']['vocoder_failures']==0 and result['metrics']['evaluation_asr_failures']==0
    from direct_s2st.evaluation.bleu import corpus_bleu,normalize_english
    expected=corpus_bleu(['the very old book is on the table','','the very old book is on the table'],
                       [normalize_english(r['en_reference']) for r in plan['source']['samples']])
    assert result['metrics']['bleu']==expected
    assert len(list(read_jsonl(output/'per_sample.jsonl')))==3
    saved_units=(output/'units.jsonl').read_bytes()
    edited=list(read_jsonl(output/'units.jsonl'));edited[0]['units']=[99]
    atomic_write_jsonl(output/'units.jsonl',edited,overwrite=True)
    with pytest.raises(ValueError,match='Unit manifest differs'):diag.restore_audio(plan)
    (output/'units.jsonl').write_bytes(saved_units)
    # The VM disappears: completed Unit shards, PCM16 WAVs and ASR journal restore.
    shutil.rmtree(local/'audio');shutil.rmtree(local/'scores')
    diag.evaluate(plan,asr_factory=lambda **kw:pytest.fail('ASR must restore'))
    assert calls==['vocoder','asr','asr']
    assert before=={**snapshot(source),**snapshot(training)}


def test_asr_load_or_dependency_errors_are_fatal_not_sample_failures(tmp_path,monkeypatch):
    from direct_s2st.s2ut.benchmark_worker import audio
    plan=experiment(tmp_path);diag.prepare_external(plan);diag.units(plan,runner=runner_fixture([]))
    install_vocoder(monkeypatch,[]);audio(plan)
    def broken(**kwargs):raise ImportError('missing ASR dependency')
    with pytest.raises(ImportError,match='dependency'):diag.evaluate(plan,asr_factory=broken)
    assert not (Path(plan['output'])/'metrics.json').exists()


def test_asr_runtime_same_versions_required_but_not_same_gpu(tmp_path):
    plan=experiment(tmp_path)
    env=copy.deepcopy(plan['source']['environment']);env['gpu']='another GPU'
    diag.verify_asr_runtime(plan,env)
    env['packages']['numpy']='different'
    with pytest.raises(ValueError,match='numpy'):diag.verify_asr_runtime(plan,env)


def test_cli_help_and_inspect_preflight_need_no_site_packages(tmp_path):
    script=ROOT/'scripts/colab/s2ut_flores_fleurs_diagnostic.py'
    result=subprocess.run([sys.executable,'-S','-B',str(script),'--help'],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0 and '--cascade-id' in result.stdout and '--limit' not in result.stdout


def test_notebook_all_old_cells_identical_and_one_8e_after_8d():
    current=read(ROOT/'notebooks/colab_training.ipynb')
    prior=json.loads(subprocess.check_output(['git','-c','safe.directory='+str(ROOT),'-C',str(ROOT),
        'show','1a2c552e30834ac7284aa913177f68b166dba678:notebooks/colab_training.ipynb'],encoding='utf-8'))
    added=[c for c in current['cells'] if ''.join(c['source']).startswith('#@title 8E.')]
    assert len(added)==1 and [c for c in current['cells'] if c not in added]==prior['cells']
    i=current['cells'].index(added[0]);assert ''.join(current['cells'][i-1]['source']).startswith('#@title 8D.')
    assert ''.join(current['cells'][i+1]['source']).startswith('#@title 9.')
    code=''.join(added[0]['source']);ast.parse(code,feature_version=(3,10))
    assert 'LIMIT' not in code and 'TRAIN_TARGET' not in code and 'RUN_BENCHMARK' not in code
    for cell in current['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<cell>','exec')


def test_output_guard_rejects_persistent_local_before_writes(tmp_path):
    plan=experiment(tmp_path);persistent=Path(plan['source']['root']).parents[2]
    before=snapshot(persistent)
    with pytest.raises(ValueError,match='outside persistent'):
        diag.prepare_plan(persistent,persistent/'forbidden',plan['source'],plan['run'],plan['metadata'],
            plan['checkpoint'],390000,'c'*40)
    assert snapshot(persistent)==before


def test_normal_launcher_separates_environments_and_completion_resume_is_model_free(tmp_path,monkeypatch):
    import inference_session
    import flores_fleurs_runtime
    from direct_s2st.s2ut.benchmark_worker import audio
    plan=experiment(tmp_path)
    persistent=Path(plan['source']['root']).parents[2]
    local=Path(plan['local']).parent
    repo=tmp_path/'repo';repo.mkdir()
    run=plan['run'];calls=[]
    monkeypatch.setattr(launcher,'training_context',lambda *a:(run,plan['metadata'],Path(plan['checkpoint']),390000))
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'c'*40+'\n')
    monkeypatch.setattr(inference_session,'environment',lambda *a:'fairseq-python')
    def ensure(*a,**kw):
        calls.append('isolated-8c-runtime')
        return 'asr-python',{},plan['source']['environment']
    monkeypatch.setattr(flores_fleurs_runtime,'ensure',ensure)
    install_vocoder(monkeypatch,calls)
    def invoke(*command,env=None):
        if '-c' in command:
            assert command[0]=='fairseq-python'
            return
        stage=command[command.index('--worker-stage')+1]
        p=read(Path(command[command.index('--plan')+1]))
        calls.append(stage)
        assert 'S2ST_INPUT_MAP' not in env
        if stage=='evaluate':
            assert command[0]=='asr-python' and 'fairseq' not in env['PYTHONPATH']
            diag.evaluate(p,asr_factory=lambda **kw:lambda path:'the very old book is on the table')
        else:
            assert command[0]=='fairseq-python'
            assert '/content/s2st-runtime/fairseq' in env['PYTHONPATH']
            {'prepare':diag.prepare_external,'units':lambda p:diag.units(p,runner=runner_fixture([])),
             'audio':audio}[stage](p)
    monkeypatch.setattr(launcher,'invoke',invoke)
    first=launcher.launch(repo,persistent,local,resume=True)
    assert first['metrics']['sample_count']==3
    assert calls==['prepare','units','audio','vocoder','isolated-8c-runtime','evaluate']
    monkeypatch.setattr(inference_session,'environment',lambda *a:pytest.fail('no rebuild'))
    second=launcher.launch(repo,persistent,local,resume=True)
    assert first==second and calls==['prepare','units','audio','vocoder','isolated-8c-runtime','evaluate']
    with pytest.raises(ValueError,match='enable resume'):
        launcher.launch(repo,persistent,local,resume=False)


def test_cell1_then_8e_without_cell2_or_existing_runtime(tmp_path,monkeypatch):
    nb=read(ROOT/'notebooks/colab_training.ipynb')
    config=next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('#@title 1.'))
    code=next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('#@title 8E.'))
    code=code.replace("'/content/ja-en-direct-s2st-benchmark'",repr(str(ROOT)))
    code=code.replace("'/content/drive/MyDrive'",repr(str(tmp_path)))
    persistent,_=source_fixture(tmp_path)
    namespace={};exec(config,namespace)
    namespace.update(OUTPUT_BASE=str(persistent.parent.parent),EXPERIMENT=persistent.name,MODE=persistent.parent.name)
    assert not any(k in namespace for k in ('REPO','PERSISTENT','DATA'))
    def git(command,**kw):
        assert command[0]=='git' and not any(v in command for v in ('fetch','checkout','reset'))
        return SimpleNamespace(returncode=0,stdout='' if 'status' in command else 'c'*40,stderr='')
    monkeypatch.setattr(subprocess,'run',git)
    import runpy
    old=runpy.run_path
    captured=[]
    def run_path(path,*a,**kw):
        if str(path).endswith('cascade_diagnostic_runtime.py'):
            def run_json(args,env,**kwargs):
                captured.append((args,env))
                assert '--resume' in args and args[args.index('--persistent')+1]==str(persistent)
                assert '--limit' not in args
                return dict(display='test 8E',metrics={})
            return dict(run_json=run_json)
        return old(path,*a,**kw)
    monkeypatch.setattr(runpy,'run_path',run_path)
    exec(code,namespace)
    assert len(captured)==1 and namespace['flores_fleurs_s2ut']['display']=='test 8E'


def test_saved_dev_best_selection_and_unit_permutation_verification(tmp_path,monkeypatch):
    import inference_session
    plan=experiment(tmp_path)
    run=copy.deepcopy(plan['run'])
    run['config']=dict(command=['python','-m','trainer',run['data_root']])
    run['identity']=dict(inputs=plan['metadata'])
    atomic_write_json(Path(run['work'])/'training-monitor.json',dict(best_update=390000))
    monkeypatch.setattr(inference_session,'saved_run',lambda p,k,r,l:run)
    selected=[]
    def best(work,mode):
        selected.append((work,mode));return Path(plan['checkpoint'])
    monkeypatch.setattr('direct_s2st.translatotron2.monitor.select_checkpoint',best)
    result=launcher.training_context(tmp_path,'existing-run',tmp_path/'local')
    assert selected==[(run['work'],'best')] and result[-1]==390000
    assert result[1]==plan['metadata']
    path=Path(run['data_root'])/'data-lock.json'
    lock=read(path);lock['unit_configuration']['kmeans_sha256']='0'*64
    atomic_write_json(path,lock,overwrite=True)
    run['identity']['inputs'][str(path)]=raw_sha(path)
    with pytest.raises(ValueError,match='official Unit vocoder'):
        launcher.training_context(tmp_path,'existing-run',tmp_path/'local')


def test_incompatible_8c_implementation_stops_before_audio_and_models(tmp_path):
    _,root=source_fixture(tmp_path)
    plan=read(root/'identity.json');plan['implementation']['evaluation/bleu.py']='0'*64
    new=root.parent/digest(plan)[:24];root.rename(new)
    atomic_write_json(new/'identity.json',plan,overwrite=True)
    report=read(new/'metrics.json');report['identity_sha256']=digest(plan)
    atomic_write_json(new/'metrics.json',report,overwrite=True)
    with pytest.raises(ValueError,match='normalization implementation'):
        diag.source_context(new,verify_audio=False)


def test_empty_asr_is_counted_as_failure_and_retryable_journal_can_be_empty(tmp_path,monkeypatch):
    from direct_s2st.s2ut.benchmark_worker import audio
    plan=experiment(tmp_path);diag.prepare_external(plan);diag.units(plan,runner=runner_fixture([]))
    install_vocoder(monkeypatch,[]);audio(plan)
    local=Path(plan['local']);(local/'scores').mkdir()
    # A process can die after creating the empty JSONL but before its first record.
    # evaluate_predictions creates its identity lock before reading this journal.
    identity=dict(plan=plan['identity'],asr=plan['asr'],normalization=diag.NORMALIZATION,
                  bleu_denominator=plan['bleu_denominator'])
    diag.restore_audio(plan)
    from direct_s2st.journal import Journal
    predictions=list(read_jsonl(local/'predictions.jsonl'))
    locked=dict(predictions=raw_sha(local/'predictions.jsonl'),evaluation=identity,
        speaker_enabled=False,blaser_enabled=False,
        audio={r[field]:raw_sha(r[field]) for r in predictions for field in ('output_audio','source_audio')},
        bleu_denominator='all_samples_failed_as_empty')
    journal=Journal(local/'scores/per_sample.jsonl',locked,resume=True)
    journal.path.write_bytes(b'')
    result=diag.evaluate(plan,asr_factory=lambda **kw:lambda path:'')
    assert result['metrics']['sample_count']==3 and result['metrics']['evaluation_asr_failures']==3
    assert result['metrics']['bleu']==0


def test_inspect_only_has_no_runtime_setup_or_output_mutations(tmp_path,monkeypatch):
    persistent,root=source_fixture(tmp_path)
    before=snapshot(persistent)
    monkeypatch.setattr(launcher,'ensure_host_yaml',lambda *a:pytest.fail('no install'))
    result=launcher.launch(tmp_path/'missing-repo',persistent,tmp_path/'local',execute=False)
    assert result['samples']==3 and snapshot(persistent)==before
    assert not (tmp_path/'local').exists()


def test_changed_local_path_or_prepared_asset_never_overwrites_training(tmp_path):
    plan=experiment(tmp_path);training=Path(plan['run']['data_root']);before=snapshot(training)
    changed=copy.deepcopy(plan);changed['local']=str(training)
    with pytest.raises(ValueError,match='paths changed'):diag.prepare_external(changed)
    assert snapshot(training)==before
    prepared=diag.prepare_external(plan)
    (prepared/'config.yaml').write_text('damaged local data',encoding='utf-8')
    with pytest.raises(ValueError,match='asset retained'):diag.prepare_external(plan)
    assert (prepared/'config.yaml').read_text(encoding='utf-8')=='damaged local data'
    assert snapshot(training)==before


@pytest.mark.parametrize('lose_scores',[False,True])
def test_interrupted_asr_reuses_newer_local_or_durable_tail(tmp_path,monkeypatch,lose_scores):
    from direct_s2st.s2ut.benchmark_worker import audio
    plan=experiment(tmp_path,n=20);diag.prepare_external(plan);diag.units(plan,runner=runner_fixture([]))
    install_vocoder(monkeypatch,[]);audio(plan)
    calls=[]
    def load(**kw):
        def transcribe(path):
            calls.append(path)
            if len(calls)==18:raise KeyboardInterrupt('interrupted after durable 16, local 17')
            return 'the very old book is on the table'
        return transcribe
    with pytest.raises(KeyboardInterrupt):diag.evaluate(plan,asr_factory=load)
    local,output=Path(plan['local']),Path(plan['output'])
    assert len(list(read_jsonl(local/'scores/per_sample.jsonl')))==17
    assert (output/'progress/asr.json').is_file()
    if lose_scores:shutil.rmtree(local/'scores')
    resumed=[]
    def resume(**kw):
        return lambda path:(resumed.append(path) or 'the very old book is on the table')
    result=diag.evaluate(plan,asr_factory=resume)
    assert len(resumed)==(4 if lose_scores else 3)
    assert result['metrics']['sample_count']==20 and result['metrics']['evaluation_asr_failures']==0


@pytest.mark.parametrize('change',['reference','hypothesis','lock'])
def test_saved_and_local_asr_conflicts_fail_before_model_loading(tmp_path,monkeypatch,change):
    from direct_s2st.s2ut.benchmark_worker import audio
    plan=experiment(tmp_path);diag.prepare_external(plan);diag.units(plan,runner=runner_fixture([]))
    install_vocoder(monkeypatch,[]);audio(plan)
    diag.evaluate(plan,asr_factory=lambda **kw:lambda p:'the very old book is on the table')
    path=Path(plan['local'])/'scores/per_sample.jsonl'
    rows=list(read_jsonl(path))
    if change=='lock':
        lock=read(path.with_suffix('.lock.json'));lock['evaluation']['asr']['revision']='f'*40
        atomic_write_json(path.with_suffix('.lock.json'),lock,overwrite=True)
    else:
        rows[0]['reference_raw' if change=='reference' else 'hypothesis_raw']='changed text'
        atomic_write_jsonl(path,rows,overwrite=True)
    before=path.read_bytes()
    with pytest.raises(ValueError):
        diag.evaluate(plan,asr_factory=lambda **kw:pytest.fail('no model load for corrupt progress'))
    assert path.read_bytes()==before


@pytest.mark.parametrize('change',['prepared','source'])
def test_changed_external_input_rejected_before_generation(tmp_path,change):
    plan=experiment(tmp_path);diag.prepare_external(plan)
    local=Path(plan['local'])
    path=local/'prepared/test.tsv' if change=='prepared' else local/'inputs'/(plan['ids'][0]+'.wav')
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError):
        diag.units(plan,runner=lambda *a,**kw:pytest.fail('no model run for modified external input'))


@pytest.mark.parametrize('corrupt',[False,True])
def test_locally_finished_unit_shard_survives_upload_failure_and_is_validated(tmp_path,monkeypatch,corrupt):
    plan=experiment(tmp_path);diag.prepare_external(plan)
    calls=[]
    def interrupted(*a,**kw):raise OSError('publication interrupted after generation')
    with monkeypatch.context() as patch:
        patch.setattr(diag,'publish_tree',interrupted)
        with pytest.raises(OSError,match='publication interrupted'):
            diag.units(plan,runner=runner_fixture(calls))
    assert calls==[0]
    saved=Path(plan['local'])/'generation/part-00000/records.json'
    if corrupt:
        doc=read(saved);next(iter(doc['records'].values()))['units']=[99]
        atomic_write_json(saved,doc,overwrite=True)
        before=saved.read_bytes()
        with pytest.raises(ValueError,match='shard identity'):
            diag.units(plan,runner=lambda *a,**kw:pytest.fail('no model load'))
        assert saved.read_bytes()==before
    else:
        result=diag.units(plan,runner=lambda *a,**kw:pytest.fail('successful Unit generation must be reused'))
        assert len(result)==3 and (Path(plan['output'])/'generation/part-00000.json').is_file()


def test_completed_audio_stage_with_failure_is_frozen_before_asr_resume(tmp_path,monkeypatch):
    plan=experiment(tmp_path);diag.prepare_external(plan);diag.units(plan,runner=runner_fixture([]))
    local,output=Path(plan['local']),Path(plan['output'])
    rows=list(read_jsonl(local/'units/predictions.jsonl'))
    rows[1].update(status='failed',error='fixture generation failure',units=[],units_sha256=digest([]))
    atomic_write_jsonl(local/'units/predictions.jsonl',rows,overwrite=True)
    atomic_write_jsonl(output/'units.jsonl',rows,overwrite=True)
    calls=[];install_vocoder(monkeypatch,calls)
    diag.audio(plan);before=(output/'predictions.jsonl').read_bytes()
    seen=[]
    def interrupted(**kw):
        def transcribe(path):
            seen.append(path)
            if len(seen)==2:raise KeyboardInterrupt('ASR interrupted after its first success')
            return 'the very old book is on the table'
        return transcribe
    with pytest.raises(KeyboardInterrupt):diag.evaluate(plan,asr_factory=interrupted)
    # A completed failed waveform row remains a failed hypothesis, without
    # rerunning its vocoder and changing the already-bound ASR input manifest.
    diag.audio(plan)
    assert calls==['vocoder'] and (output/'predictions.jsonl').read_bytes()==before
    resumed=[]
    result=diag.evaluate(plan,asr_factory=lambda **kw:lambda p:(resumed.append(p) or 'the very old book is on the table'))
    assert len(resumed)==1 and result['metrics']['generation_failures']==1
