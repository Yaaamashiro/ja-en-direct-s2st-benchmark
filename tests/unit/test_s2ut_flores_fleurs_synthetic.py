import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from direct_s2st.s2ut import flores_fleurs_synthetic as syn
from direct_s2st.s2ut import flores_fleurs_diagnostic as core
from direct_s2st.benchmark_storage import read
from direct_s2st.drive_staging import raw_sha
from direct_s2st.io import atomic_write_json,atomic_write_jsonl,read_jsonl
from direct_s2st.journal import digest
from direct_s2st.evaluation.bleu import corpus_bleu,normalize_english
from test_s2ut_flores_fleurs_diagnostic import legacy_plan,runner_fixture,install_vocoder,snapshot,ROOT

sys.path.insert(0,str(ROOT/'scripts/colab'))
import s2ut_flores_fleurs_synthetic as launcher
import s2ut_flores_fleurs_diagnostic as natural_launcher
import cascade_diagnostic_runtime as transport


def saved_natural(tmp_path,monkeypatch,n=3):
    released,plan=legacy_plan(tmp_path,n=n,synthetic=True)
    monkeypatch.setattr(syn,'SAMPLES',n)
    monkeypatch.setattr(syn,'MODEL',dict(name=plan['run']['name'],best_update=plan['best_update'],
        updates=plan['run']['updates'],checkpoint_sha256=plan['checkpoint_sha256']))
    released.prepare_external(plan);released.units(plan,runner=runner_fixture([]))
    install_vocoder(monkeypatch,[]);released.audio(plan)
    released.evaluate(plan,asr_factory=lambda **kw:lambda p:'the very old book is on the table')
    root=Path(plan['output'])
    atomic_write_json(root/'evaluation-runtime.json',plan['source']['environment'])
    atomic_write_json(root/'evaluation-asr-audit.json',plan['source']['asr_audit'])
    atomic_write_json(root/'completion.json',dict(identity=plan['identity'],
        files={name:raw_sha(root/name) for name in syn.RESULT_FILES}))
    return syn.natural_context(plan['persistent_base'],root.name)


def synthetic_plan(tmp_path,monkeypatch,n=3):
    natural=saved_natural(tmp_path,monkeypatch,n)
    context=syn.source_context(natural)
    old=natural['plan']
    plan=syn.prepare_plan(old['persistent_base'],tmp_path/'synthetic-work',natural,context,
        old['run'],old['metadata'],old['checkpoint'],'d'*40)
    return natural,plan


def prepared_audio(tmp_path,monkeypatch,n=3):
    natural,plan=synthetic_plan(tmp_path,monkeypatch,n)
    core.prepare_external(plan);core.units(plan,runner=runner_fixture([]));core.audio(plan)
    return natural,plan


def test_identical_ids_texts_references_model_and_exact_saved_synthetic_audio(tmp_path,monkeypatch):
    natural,plan=synthetic_plan(tmp_path,monkeypatch)
    before=natural['plan'];ctx=plan['source']
    assert plan['ids']==before['ids']
    original=core.indexed(before['source']['samples']);synthetic=core.indexed(ctx['samples'])
    assert original.keys()==synthetic.keys()
    for key,row in synthetic.items():
        assert all(row[k]==original[key][k] for k in core.SAMPLE_FIELDS)
        assert raw_sha(row['source_audio'])==row['source_sha256']
        assert row['source_sha256']!=original[key]['source_sha256']
        assert row['source_duration']!=original[key]['source_duration']
    for key in ('checkpoint_sha256','best_update','beam','vocoder','asr','normalization','bleu_denominator'):
        assert plan[key]==before[key]
    assert plan['run']['updates']==before['run']['updates']
    assert Path(plan['output']).parent.name=='flores-fleurs-s2ut-synthetic'
    assert plan['identity']!=before['identity']
    syn.verify_plan(plan)


@pytest.mark.parametrize('change',['missing','duplicate','reference','source_text','natural_wave','source_sha'])
def test_reject_changed_id_reference_or_natural_wave_before_preparation(tmp_path,monkeypatch,change):
    natural,plan=synthetic_plan(tmp_path,monkeypatch)
    ctx=copy.deepcopy(plan['source']);rows=ctx['samples']
    if change=='missing':rows.pop()
    elif change=='duplicate':rows[1]=copy.deepcopy(rows[0])
    elif change=='reference':rows[0]['en_reference']='a different reference'
    elif change=='source_text':rows[0]['ja_text']='別の文章'
    elif change=='natural_wave':rows[0]['source_audio']=natural['plan']['source']['samples'][0]['source_audio']
    else:rows[0]['source_sha256']=natural['plan']['source']['samples'][0]['source_sha256']
    with pytest.raises((ValueError,RuntimeError)):
        syn.check_source_difference(ctx,natural['plan']['source'])


@pytest.mark.parametrize('change',['wav_bytes','group_hash','group_request','missing_item','duplicate_item','8c_reference'])
def test_saved_8c_corruption_fails_without_tts_fallback(tmp_path,monkeypatch,change):
    natural=saved_natural(tmp_path,monkeypatch)
    import direct_s2st.cascade.flores_fleurs_diagnostic as cascade
    monkeypatch.setattr(cascade,'synthetic_stage',lambda *a,**kw:pytest.fail('source TTS must never run'))
    root=Path(natural['plan']['source']['root'])
    if change=='wav_bytes':
        row=next(read_jsonl(root/'per_sample.jsonl'));Path(row['synthetic_input_audio_path']).write_bytes(b'wrong audio')
    elif change=='8c_reference':
        rows=list(read_jsonl(root/'per_sample.jsonl'));rows[0]['en_reference']='wrong'
        atomic_write_jsonl(root/'per_sample.jsonl',rows,overwrite=True)
    else:
        path=next(p for p in (root/'stage-results').glob('*.json') if read(p)['body']['request']['condition']=='synthetic')
        document=read(path)
        if change=='group_hash':document['sha256']='0'*64
        else:
            if change=='group_request':document['body']['request']['condition']='natural'
            elif change=='missing_item':document['body']['items'].pop()
            else:document['body']['items'].append(copy.deepcopy(document['body']['items'][0]))
            document['sha256']=digest(document['body'])
        atomic_write_json(path,document,overwrite=True)
    with pytest.raises((ValueError,RuntimeError,OSError)):
        syn.source_context(natural)


@pytest.mark.parametrize('field',['checkpoint_sha256','best_update','beam','vocoder','asr','normalization'])
def test_non_source_scientific_changes_are_rejected(tmp_path,monkeypatch,field):
    natural=saved_natural(tmp_path,monkeypatch)
    plan=copy.deepcopy(natural['plan'])
    plan[field]=({} if isinstance(plan[field],dict) else plan[field]+1 if isinstance(plan[field],int) else 'different')
    with pytest.raises(ValueError):syn.check_model(plan)


def test_source_metadata_aligns_by_id_even_with_reordered_8c_rows(tmp_path,monkeypatch):
    natural=saved_natural(tmp_path,monkeypatch)
    # The saved canonical report order can differ from the Natural source order.
    old=copy.deepcopy(natural)
    old['plan']['source']['samples'].reverse()
    a=syn.source_context(natural);b=syn.source_context(old)
    assert a['samples']==b['samples']


def test_synthetic_sources_and_references_never_enter_generation_targets(tmp_path,monkeypatch):
    natural,plan=synthetic_plan(tmp_path,monkeypatch)
    before={**snapshot(natural['root']),**snapshot(plan['source']['root']),**snapshot(plan['run']['data_root'])}
    prepared=core.prepare_external(plan)
    text=(prepared/'test.tsv').read_text(encoding='utf-8')
    from direct_s2st.s2ut.multitask import read_tsv
    rows=read_tsv(prepared/'test.tsv',('id','src_audio','src_n_frames','tgt_audio','tgt_n_frames'))
    for item in plan['source']['samples']:
        row=rows[item['pair_id']]
        assert row['tgt_audio']=='0' and row['tgt_n_frames']=='1'
        assert raw_sha(row['src_audio'])==item['source_sha256']
        assert item['en_reference'] not in text and item['ja_text'] not in text
    for task,cfg in core.read_yaml(prepared/'config_multitask.yaml').items():
        assert all(r['tgt_text']=='<unk>' for r in read_tsv(Path(cfg['data'])/'test.tsv',('id','tgt_text')).values())
    assert before=={**snapshot(natural['root']),**snapshot(plan['source']['root']),**snapshot(plan['run']['data_root'])}


def test_real_pipeline_report_comparisons_transport_and_no_original_artifact_writes(tmp_path,monkeypatch):
    natural,plan=prepared_audio(tmp_path,monkeypatch)
    protected={**snapshot(natural['root']),**snapshot(plan['source']['root'])}
    result=core.evaluate(plan,asr_factory=lambda **kw:lambda p:'the very old book is on the table')
    metrics=result['metrics'];natural_bleu=natural['metrics']['bleu']
    assert metrics['condition']=='synthetic' and metrics['sample_count']==3
    assert metrics['s2ut_natural_bleu']==natural_bleu
    assert metrics['s2ut_synthetic_minus_natural']==metrics['bleu']-natural_bleu
    assert metrics['cascade_synthetic_minus_natural']==20.9756-21.153432814483
    assert metrics['s2ut_synthetic_over_natural']==metrics['bleu']/natural_bleu
    assert metrics['natural_8e_identity']==natural['plan']['identity']
    assert not metrics['original_comparison']['included']  # no invented original scores
    assert 'Original KFTT/JESC test |' not in result['display']
    assert 'does not by itself prove a sole cause' in result['display']
    rows=list(read_jsonl(Path(plan['output'])/'per_sample.jsonl'))
    assert all(r['source_audio_sha256']==r['source_sha256'] and r['generated_audio_sha256']==r['output_sha256'] for r in rows)
    assert protected=={**snapshot(natural['root']),**snapshot(plan['source']['root'])}
    code='import json,sys; print("[progress] 8F completed",file=sys.stderr); print(json.dumps('+repr(result)+',ensure_ascii=False))'
    assert transport.run_json([sys.executable,'-B','-c',code],transport.cpu_environment(),name='8F',
                              result_kind='external_s2ut_synthetic')==result
    with pytest.raises(RuntimeError,match='external S2UT'):
        transport.run_json([sys.executable,'-B','-c',code],transport.cpu_environment(),name='8E',result_kind='external_s2ut')
    zero=copy.deepcopy(plan);zero['natural_s2ut_bleu']=0
    other=syn.save_report(zero,copy.deepcopy(metrics),list(read_jsonl(Path(plan['local'])/'scores/per_sample.jsonl')))
    assert other['metrics']['s2ut_synthetic_over_natural'] is None and 'undefined' in other['display']


@pytest.mark.parametrize('failure',['generation','vocoder','asr'])
def test_failures_remain_in_all_sample_bleu_denominator(tmp_path,monkeypatch,failure):
    natural,plan=synthetic_plan(tmp_path,monkeypatch)
    core.prepare_external(plan);core.units(plan,runner=runner_fixture([]))
    if failure=='generation':
        for path in (Path(plan['local'])/'units/predictions.jsonl',Path(plan['output'])/'units.jsonl'):
            rows=list(read_jsonl(path));rows[1].update(status='failed',units=[],units_sha256=digest([]),error='fixture generation failure')
            atomic_write_jsonl(path,rows,overwrite=True)
    if failure=='vocoder':
        import direct_s2st.vocoders.inference as vocoder
        original=vocoder.vocode
        def fail_one(*a,**kw):
            result=original(*a,**kw);path=Path(a[2])/'predictions.jsonl'
            rows=list(read_jsonl(path));rows[1].update(status='failed',output_audio=None,output_sha256=None,error='fixture vocoder failure')
            atomic_write_jsonl(path,rows,overwrite=True);return result
        monkeypatch.setattr(vocoder,'vocode',fail_one)
    core.audio(plan)
    calls=[]
    def load(**kw):
        def transcribe(path):
            calls.append(path)
            if failure=='asr' and len(calls)==2:raise ValueError('fixture ASR failure')
            return 'the very old book is on the table'
        return transcribe
    result=core.evaluate(plan,asr_factory=load)
    metric=result['metrics'];assert metric['sample_count']==3
    assert metric[{'generation':'generation_failures','vocoder':'vocoder_failures','asr':'evaluation_asr_failures'}[failure]]==1
    expected=corpus_bleu(['the very old book is on the table','','the very old book is on the table'],
                        [normalize_english(r['en_reference']) for r in plan['source']['samples']])
    assert metric['bleu']==expected


def test_resume_restores_unit_wave_and_asr_journals_without_completed_work(tmp_path,monkeypatch):
    natural,plan=synthetic_plan(tmp_path,monkeypatch,n=35)
    core.prepare_external(plan);calls=[]
    with pytest.raises(KeyboardInterrupt):core.units(plan,runner=runner_fixture(calls,1))
    assert calls==[0,1]
    shutil.rmtree(Path(plan['local'])/'generation')
    core.units(plan,runner=runner_fixture(calls));assert calls==[0,1,1]
    vocoder_calls=[];install_vocoder(monkeypatch,vocoder_calls);core.audio(plan)
    done=(Path(plan['output'])/'predictions.jsonl').read_bytes()
    seen=[]
    def partial(**kw):
        def transcribe(path):
            seen.append(path)
            if len(seen)==18:raise KeyboardInterrupt('fixture interrupted after 16 durable, 17 local')
            return 'the very old book is on the table'
        return transcribe
    with pytest.raises(KeyboardInterrupt):core.evaluate(plan,asr_factory=partial)
    shutil.rmtree(plan['local'])
    core.prepare_external(plan)
    core.units(plan,runner=lambda *a,**kw:pytest.fail('no repeated Unit extraction'))
    core.audio(plan)
    resumed=[]
    result=core.evaluate(plan,asr_factory=lambda **kw:lambda p:(resumed.append(p) or 'the very old book is on the table'))
    assert len(resumed)==19 and result['metrics']['sample_count']==35
    assert vocoder_calls==['vocoder','vocoder']  # two groups once, no synthesis on resume
    assert (Path(plan['output'])/'predictions.jsonl').read_bytes()==done


def test_normal_launcher_completed_resume_has_no_model_or_stage_calls(tmp_path,monkeypatch):
    import inference_session
    import flores_fleurs_runtime
    natural=saved_natural(tmp_path,monkeypatch);old=natural['plan']
    monkeypatch.setattr(natural_launcher,'training_context',lambda *a:(old['run'],old['metadata'],Path(old['checkpoint']),old['best_update']))
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'d'*40+'\n')
    monkeypatch.setattr(inference_session,'environment',lambda *a:'fairseq-python')
    monkeypatch.setattr(flores_fleurs_runtime,'ensure',lambda *a,**kw:('asr-python',{},natural['runtime']))
    stages=[]
    def invoke(*command,env=None):
        if '-c' in command:return
        stage=command[command.index('--worker-stage')+1];stages.append(stage)
        plan=read(Path(command[command.index('--plan')+1]));syn.verify_plan(plan)
        if stage=='evaluate':core.evaluate(plan,asr_factory=lambda **kw:lambda p:'the very old book is on the table')
        else:{'prepare':core.prepare_external,'units':lambda p:core.units(p,runner=runner_fixture([])),'audio':core.audio}[stage](plan)
    monkeypatch.setattr(natural_launcher,'invoke',invoke)
    repo=tmp_path/'repo';repo.mkdir()
    first=launcher.launch(repo,old['persistent_base'],tmp_path/'synthetic-work')
    assert stages==['prepare','units','audio','evaluate']
    before=snapshot(first['output_root'])
    monkeypatch.setattr(inference_session,'environment',lambda *a:pytest.fail('no runtime for completed result'))
    monkeypatch.setattr(natural_launcher,'training_context',lambda *a:pytest.fail('no model checkpoint restoration'))
    monkeypatch.setattr(syn,'source_context',lambda *a,**kw:pytest.fail('no source WAV/group rescan for completed result'))
    assert launcher.launch(repo,old['persistent_base'],tmp_path/'synthetic-work')==first
    assert before==snapshot(first['output_root']) and stages==['prepare','units','audio','evaluate']


def test_cli_help_is_stdlib_only_and_old_notebook_cells_are_identical():
    helper=ROOT/'scripts/colab/s2ut_flores_fleurs_synthetic.py'
    result=subprocess.run([sys.executable,'-S','-B',str(helper),'--help'],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0 and '--natural-id' in result.stdout and '--limit' not in result.stdout
    notebook=read(ROOT/'notebooks/colab_training.ipynb')
    original=json.loads(subprocess.check_output(['git','-c','safe.directory='+str(ROOT),'-C',str(ROOT),
        'show','901ec4b15c4fab4ffbfee1a6efee35626bc00940:notebooks/colab_training.ipynb'],encoding='utf-8'))
    added=[c for c in notebook['cells'] if ''.join(c['source']).startswith('#@title 8F.')]
    assert len(added)==1 and [c for c in notebook['cells'] if c not in added]==original['cells']
    i=notebook['cells'].index(added[0]);assert ''.join(notebook['cells'][i-1]['source']).startswith('#@title 8E.')
    assert ''.join(notebook['cells'][i+1]['source']).startswith('#@title 9.')
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']),feature_version=(3,10))


def test_inspect_only_does_not_import_training_environment_or_install_dependencies(tmp_path,monkeypatch):
    natural=saved_natural(tmp_path,monkeypatch)
    import builtins
    original=builtins.__import__
    def checked_import(name,*a,**kw):
        if name=='inference_session':pytest.fail('inspect-only must not import training environment helpers')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',checked_import)
    monkeypatch.setattr(natural_launcher,'ensure_host_yaml',lambda *a:pytest.fail('no host dependency installation'))
    before=snapshot(natural['plan']['persistent_base'])
    result=launcher.launch(tmp_path/'missing-repo',natural['plan']['persistent_base'],tmp_path/'unused',execute=False)
    assert result['samples']==3 and before==snapshot(natural['plan']['persistent_base'])
    assert not (tmp_path/'unused').exists()


def formal_original_fixture(persistent,plan):
    common=Path(persistent)/'original-common';manifest=common/'test.jsonl'
    refs=[dict(pair_id='original-'+str(i),en_text='the very old book is on the table '+str(i)) for i in range(3)]
    atomic_write_jsonl(manifest,refs)
    paths={}
    for system in ('cascade','s2ut'):
        body=dict(version=1,split='test',system_id=system,common_root=str(common),common_sha256=raw_sha(manifest),
            asr=plan['asr'],bleu_denominator=plan['bleu_denominator'],run=dict(name='formal-'+system),
            checkpoint_sha256=plan['checkpoint_sha256'],beam=plan['beam'],vocoder='official_hubert100_ljspeech')
        identity=digest(body);root=Path(persistent)/'benchmarks'/system/identity[:24]
        saved=dict(body,identity=identity,local='not used',output=str(root))
        atomic_write_json(root/'plan.json',saved)
        atomic_write_json(root.parent/'latest.json',dict(plan=str(root/'plan.json'),sha256=raw_sha(root/'plan.json')))
        predictions=[dict(pair_id=r['pair_id'],reference_text=r['en_text']) for r in refs]
        atomic_write_jsonl(root/'predictions.jsonl',predictions)
        evaluation=dict(plan=identity,predictions=raw_sha(root/'predictions.jsonl'),asr=plan['asr'],
                        normalization=plan['normalization'],bleu_denominator=plan['bleu_denominator'])
        rows=[dict(pair_id=r['pair_id'],reference_raw=r['en_text'],generation_status='success',evaluation_status='success',
                   hypothesis_raw=r['en_text'] if system=='cascade' else 'the very old book is on the table') for r in refs]
        score_path=root/'results/per_sample.jsonl';atomic_write_jsonl(score_path,rows)
        bleu=corpus_bleu([normalize_english(r['hypothesis_raw']) for r in rows],[normalize_english(r['en_text']) for r in refs])
        atomic_write_json(root/'results/metrics.json',dict(bleu=bleu,evaluation_identity=evaluation,system_id=system,
            run_id='formal-'+system,samples=3,bleu_denominator=plan['bleu_denominator']))
        b=dict(identity=evaluation,files={'per_sample.jsonl':dict(sha256=raw_sha(score_path))})
        atomic_write_json(root/'progress'/('asr-'+digest(evaluation)[:24]+'.json'),dict(body=b,sha256=digest(b)))
        paths[system]=root
    return paths


def test_original_test_row_comes_from_verified_artifacts_and_rejects_changed_score(tmp_path,monkeypatch):
    natural,plan=prepared_audio(tmp_path,monkeypatch)
    roots=formal_original_fixture(plan['persistent_base'],plan)
    protected={**snapshot(natural['root']),**snapshot(plan['source']['root']),**snapshot(Path(plan['persistent_base'])/'benchmarks')}
    result=core.evaluate(plan,asr_factory=lambda **kw:lambda p:'the very old book is on the table')
    original=result['metrics']['original_comparison']
    assert original['included'] and original['samples']==3 and 'Original KFTT/JESC test |' in result['display']
    assert original['scores']=={s:read(p/'results/metrics.json')['bleu'] for s,p in roots.items()}
    assert protected=={**snapshot(natural['root']),**snapshot(plan['source']['root']),**snapshot(Path(plan['persistent_base'])/'benchmarks')}
    path=roots['s2ut']/'results/metrics.json';m=read(path);m['bleu']=1
    atomic_write_json(path,m,overwrite=True)
    assert not syn.original_comparison(plan['persistent_base'],plan)['included']


@pytest.mark.parametrize('execute,cold,saved_code,corrupt',[(True,False,False,False),(False,False,False,False),
    (True,True,False,False),(True,False,True,False),(True,True,True,False),(True,False,True,True)])
def test_normal_cell1_to_8f_uses_real_result_receiver(tmp_path,monkeypatch,capsys,execute,cold,saved_code,corrupt):
    notebook=read(ROOT/'notebooks/colab_training.ipynb')
    config=next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('#@title 1.'))
    code=next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('#@title 8F.'))
    primary=tmp_path/'repo';cache=tmp_path/'code';revision='d'*40;older='e'*40
    code=code.replace("'/content/ja-en-direct-s2st-benchmark'",repr(str(primary)))
    code=code.replace("'/content/drive/MyDrive'",repr(str(tmp_path)))
    code=code.replace("'/content/s2st-flores-fleurs-s2ut-synthetic-code'",repr(str(cache)))
    if not execute:code=code.replace('FLEURS_SYNTHETIC_RUN = True','FLEURS_SYNTHETIC_RUN = False')
    persistent=tmp_path/'drive/experiment';(persistent/'diagnostics/flores-fleurs-s2ut').mkdir(parents=True)
    report=(dict(output_root='/saved/synthetic',display='Synthetic比較結果',metrics=dict(purpose=syn.PURPOSE,condition='synthetic',
        sample_count=300,bleu=4.0,s2ut_natural_bleu=.5,s2ut_synthetic_bleu=4.0,s2ut_synthetic_minus_natural=3.5,
        cascade_natural_bleu=21.15,cascade_synthetic_bleu=20.97,cascade_synthetic_minus_natural=20.97-21.15)) if execute else
        dict(cascade_root='/saved/8c',samples=300,display='入力確認のみ'))
    def checkout(root):
        script=root/'scripts/colab/s2ut_flores_fleurs_synthetic.py';script.parent.mkdir(parents=True,exist_ok=True)
        script.write_text('# S2UT_FLORES_FLEURS_SYNTHETIC_VERSION = 1\nimport json,sys\nprint("[progress] saved 8F report",file=sys.stderr)\n'
            'print(json.dumps('+repr(report)+',ensure_ascii=False))\n',encoding='utf-8')
        module=root/'src/direct_s2st/s2ut/flores_fleurs_synthetic.py';module.parent.mkdir(parents=True);module.touch()
        (root/'scripts/colab/cascade_diagnostic_runtime.py').write_bytes((ROOT/'scripts/colab/cascade_diagnostic_runtime.py').read_bytes())
    expected_revision=older if saved_code else revision
    if not cold:
        checkout(primary);checkout(cache/revision)
        if saved_code:checkout(cache/older)
    if saved_code:
        body=dict(version=1,purpose=syn.PURPOSE,condition='synthetic',execution_revision=older,
                  natural_evaluation_identity='a'*64)
        identity=digest(body);output=persistent/'diagnostics/flores-fleurs-s2ut-synthetic'/identity[:24]
        saved=dict(body,identity=identity,output=str(output),local='unused in bootstrap')
        if corrupt:saved['identity']='0'*64
        atomic_write_json(output/'plan.json',saved)
    calls=[];fetched={}
    def git(command,**kw):
        assert command[0]=='git';args=command[3:];calls.append(args[0])
        if args[0]=='init':(Path(command[2])/'.git').mkdir()
        if args[0]=='fetch':fetched[command[2]]=revision if args[-1]=='main' else args[-1]
        if args[0]=='worktree':checkout(Path(args[-2]))
        result=''
        if args[0]=='rev-parse':
            result=fetched[command[2]] if args[1].startswith('FETCH_HEAD') else (
                Path(command[2]).name if Path(command[2]).name in (revision,older) else revision)
        return type('Result',(),dict(returncode=0,stdout=result,stderr=''))()
    monkeypatch.setattr(subprocess,'run',git)
    namespace={};exec(config,namespace)
    namespace.update(OUTPUT_BASE=str(persistent.parent.parent),EXPERIMENT=persistent.name,MODE=persistent.parent.name)
    if corrupt:
        with pytest.raises(AssertionError,match='identity'):exec(code,namespace)
        assert 'flores_fleurs_s2ut_synthetic' not in namespace
        return
    exec(code,namespace)
    assert namespace['flores_fleurs_s2ut_synthetic']==report
    assert namespace['_fs_execution_revision']==expected_revision
    assert namespace['_fs_execution_code']==cache/expected_revision
    assert namespace['_fs_args'][2]==str(cache/expected_revision/'scripts/colab/s2ut_flores_fleurs_synthetic.py')
    if saved_code:assert namespace['_fs_natural_id']=='a'*24
    assert report['display'] in capsys.readouterr().out and ('fetch' in calls)==cold
