import ast
import copy
from dataclasses import dataclass
import json
import io
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from direct_s2st.cascade import flores_fleurs_data as data
from direct_s2st.cascade import flores_fleurs_diagnostic as diag
from direct_s2st.cascade import source_corpus_tts as source
from direct_s2st.evaluation.bleu import corpus_bleu,normalize_english
from direct_s2st.io import atomic_write_json
from direct_s2st.journal import digest

ROOT=Path(__file__).resolve().parents[2]


def datasets():
    ja=[dict(split='dev' if i%2 else 'devtest',id=str(i),text=f'日本語の文{i}。') for i in range(310)]
    en=[dict(split=r['split'],id=r['id'],text='this is the English reference '+r['id']) for r in ja]
    fleurs=[dict(split='test' if i%3 else 'validation',id=str(1000+i),path=f'recording-{i}.wav',
                 raw_transcription=r['text'],gender=i%2,locator=dict(split='test',row_group=0,offset=i)) for i,r in enumerate(ja)]
    return fleurs,ja,en


def test_text_join_and_english_reference_never_depend_on_row_order():
    f,j,e=datasets()
    first,audit=data.join(f,j,e)
    for rows in (f,j,e):
        random.Random(25).shuffle(rows)
    second,again=data.join(f,j,e)
    assert first==second and audit==again
    assert len(first)==300 and audit['counts']['matched']==310
    for row in first:
        assert row['en_reference'].endswith(row['flores_id'])
        assert row['fleurs_id']!=row['flores_id']


def test_only_unique_safe_normalization_matches_no_fuzzy():
    f,j,e=datasets()
    j[0]['text']='Ａ　日本語 です。'; f[0]['raw_transcription']='A\t日本語\u2003です。'
    f[1]['raw_transcription']+='違い'
    j.append(dict(j[2]))  # Ambiguous even when duplicate text/ref is identical.
    f.append(dict(f[3],id='duplicate-recording',path='other.wav'))
    rows,audit=data.join(f,j,e,0)
    assert audit['counts']==dict(matched=308,unmatched=1,ambiguous=1,duplicate_recordings=1,input_recordings=311)
    assert any(r['ja_text']=='A 日本語 です。' for r in rows)
    assert all(r['flores_id'] not in ('1','2') for r in rows)


def test_duplicate_english_ids_are_ambiguous_not_arbitrary_first():
    f,j,e=datasets(); e.append(dict(e[2],text='conflicting reference'))
    rows,audit=data.join(f,j,e,0)
    assert audit['counts']['ambiguous']==1 and all(r['flores_id']!='2' for r in rows)


def test_recording_selection_and_limit_are_deterministic():
    f,j,e=datasets();f.extend([dict(f[0],id='other-id',path='different.wav'),dict(f[0],id='third-id',path='third.wav')])
    first,a=data.join(f,j,e,0)
    f.reverse();j.reverse();e.reverse()
    second,b=data.join(f,j,e,0)
    assert first==second and a==b and a['counts']['duplicate_recordings']==2
    limited,_=data.join(f,j,e,2)
    assert limited==first[:2]


def test_selected_natural_wav_is_preserved_without_requantization(tmp_path):
    encoded=io.BytesIO()
    sf.write(encoded,np.sin(np.arange(1600)/10)*.1,16000,format='WAV',subtype='FLOAT')
    blob=encoded.getvalue()
    reader=data.OfficialData(tmp_path/'cache')
    reader.audio_tables['test',0]=[dict(id=12,path='official.wav',raw_transcription='日本語。',audio={'bytes':blob})]
    sample=dict(locator={'split':'test','row_group':0,'offset':0},fleurs_id='12',
                fleurs_path='official.wav',raw_transcription='日本語。')
    output=tmp_path/'selected.wav'
    audit=reader.audio(sample,output)
    assert output.read_bytes()==blob and audit['original_subtype']=='FLOAT'
    assert audit['original_audio_sha256']==diag.checksum(output)
    sample['fleurs_id']='different-recording'
    with pytest.raises(RuntimeError,match='no longer matches'):reader.audio(sample,tmp_path/'wrong.wav')


@pytest.mark.parametrize('limit',[-1,True,1.5])
def test_invalid_limits_fail(limit):
    with pytest.raises(ValueError,match='LIMIT'):
        data.join(*datasets(),limit)


def test_less_than_300_stops_even_when_requesting_two():
    f,j,e=datasets()
    with pytest.raises(ValueError,match='Fewer than 300'):
        data.join(f[:299],j,e,2)


@pytest.mark.parametrize('dataset,split',[(0,'train'),(1,'train'),(2,'train')])
def test_training_splits_are_rejected(dataset,split):
    rows=list(datasets());rows[dataset][0]['split']=split
    with pytest.raises(ValueError,match='forbidden'):
        data.join(*rows)


def wav(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,.1*np.sin(np.arange(1600)/10),16000)


def test_stage_resume_reuses_successful_wav_and_checks_text_config_sha(tmp_path):
    cache=diag.StageCache(tmp_path,'identity')
    request=cache.request('sample','gold','target_tts',dict(text_sha='a',model='sha'))
    calls=[]
    def call(path): calls.append(path);wav(path);return dict(text='')
    first=cache.run(request,call,audio_folder='output-audio/gold')
    assert cache.run(request,lambda _:pytest.fail('must not regenerate'),audio_folder='output-audio/gold')==first
    sf.write(calls[0],np.ones(1600)*.2,16000)
    second=cache.run(request,call,audio_folder='output-audio/gold')
    assert len(calls)==2 and second['audio'][0]['path']!=first['audio'][0]['path']
    changed=copy.deepcopy(request);changed['inputs']['text_sha']='b'
    cache.run(changed,call,audio_folder='output-audio/gold')
    changed['inputs']['model']='new-config';cache.run(changed,call,audio_folder='output-audio/gold')
    assert len(calls)==4


def test_failure_is_retryable_and_atomic_attempts_preserved(tmp_path):
    cache=diag.StageCache(tmp_path,'identity');request=cache.request('sample','natural','mt',{})
    def fail(_): raise ValueError('sample failed')
    bad=cache.run(request,fail)
    good=cache.run(request,lambda _:dict(text='translation'))
    assert bad['status']=='failed' and good['status']=='success'
    assert len(list((tmp_path/'checkpoints').rglob('*.json')))==2
    assert cache.get(request)==good


def test_changed_checkpoint_body_is_rejected(tmp_path):
    cache=diag.StageCache(tmp_path,'i');r=cache.request('id','gold','mt',{})
    cache.run(r,lambda _:dict(text='good'))
    p=next((tmp_path/'checkpoints').rglob('*.json'))
    d=json.loads(p.read_text());d['body']['text']='edited';atomic_write_json(p,d,overwrite=True)
    with pytest.raises(RuntimeError,match='changed'):
        cache.get(r)


@pytest.mark.parametrize('wave',[np.zeros(50),np.full(50,np.nan),np.ones((50,2))])
def test_invalid_audio_is_not_a_success(tmp_path,wave):
    path=tmp_path/'invalid.wav';sf.write(path,wave,16000,subtype='FLOAT')
    with pytest.raises(ValueError,match='invalid'):
        diag.audio_audit(path)


@pytest.mark.parametrize('directory',['data','checkpoints','benchmarks','evaluations','runs'])
def test_output_cannot_enter_protected_directories(tmp_path,directory):
    persistent=tmp_path/directory/'nested'
    with pytest.raises(RuntimeError,match='overlaps'):
        diag.safe_output(persistent,[tmp_path/directory])
    with pytest.raises(RuntimeError,match='inside'):
        diag.inside(tmp_path/'diagnostic',tmp_path/'official/results.json')


def test_no_guessed_model_revision():
    for config in ({'model':'whisper'}, {'model':'whisper','revision':'main'}, {'model':'whisper','revision':'abc123'}):
        with pytest.raises(RuntimeError,match='no guesses'):
            diag.pinned(config,'ASR')
    assert diag.pinned(dict(model='whisper',revision='a'*40),'ASR')['revision']=='a'*40


def receipt(tmp_path):
    evidence=tmp_path/'original-generation-receipt.json'
    evidence.write_text('actual saved generation metadata fixture',encoding='utf-8')
    doc=dict(schema='source-corpus-tts-v1',repository=source.REPOSITORY,code_revision='a'*40,
        config_relative_path='configs/colab-pro.yaml',config_sha256='c'*64,language='Japanese',
        effective_tts=dict(model_id='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',revision='b'*40,device='cuda:0',
            dtype='bfloat16',speaker='Ono_Anna',instruct='',max_new_tokens=2048,do_sample=True,
            subtalker_do_sample=True,remove_invalid_values=False,renormalize_logits=True,output_sample_rate=16000,
            max_content_retries=1,batch_size=130),attention='sdpa',
        packages={'qwen-tts':'0.1.1','transformers':'4.57.3','torch':'2.7.1','numpy':'2.2.6','soxr':'1.1.0'},
        evidence=[dict(path=str(evidence),sha256=diag.checksum(evidence))])
    path=tmp_path/'provenance.json';atomic_write_json(path,doc)
    return path,doc


def test_repository_yaml_alone_is_not_historical_proof(tmp_path):
    with pytest.raises(RuntimeError,match=source.ERROR):source.proof(None)
    p=tmp_path/'current.yaml';p.write_text('tts: {speaker: Ono_Anna}')
    with pytest.raises(RuntimeError,match=source.ERROR):source.proof(p)


@pytest.mark.parametrize('field',['instruct','revision','batch_size','do_sample','dtype'])
def test_incomplete_source_generation_provenance_fails(tmp_path,field):
    p,d=receipt(tmp_path);del d['effective_tts'][field];atomic_write_json(p,d,overwrite=True)
    with pytest.raises(RuntimeError,match=source.ERROR):source.proof(p)


def test_proof_binds_actual_evidence_hash(tmp_path):
    p,d=receipt(tmp_path);assert source.proof(p)['receipt']==d
    Path(d['evidence'][0]['path']).write_text('changed')
    with pytest.raises(RuntimeError,match='SHA mismatch'):source.proof(p)


def test_source_preflight_cli_needs_no_model_or_scoring_packages(tmp_path):
    # A fresh stdlib-only interpreter must emit the real cause, not ImportError.
    result=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/colab/flores_fleurs_diagnostic.py'),
        '--preflight-source'],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode and source.ERROR in result.stderr and 'ModuleNotFoundError' not in result.stderr
    assert not list(tmp_path.iterdir())


def fixture(tmp_path):
    samples,audit=data.join(*datasets(),2)
    audit['datasets']=[dict(repository=data.FLEURS,revision=data.FLEURS_REVISION),dict(repository=data.FLORES,revision=data.FLORES_REVISION)]
    saved=tmp_path/'official';saved.mkdir()
    evidence=saved/'metrics.json';evidence.write_text('official immutable result')
    ctx=dict(evaluation_root=str(saved),pipeline=dict(asr={'model':'source','revision':'1'*40},
        mt={'model':'facebook/nllb-200-distilled-600M','revision':'2'*40,'source_language':'jpn_Jpan','target_language':'eng_Latn'},
        tts={'model':'qwen','revision':'3'*40,'speaker':'speaker','language':'English'}),
        evaluation_asr={'model_id':'english','revision':'4'*40},inputs_sha256={str(evidence):diag.checksum(evidence)})
    p,_=receipt(tmp_path);proven=source.proof(p)
    root=diag.prepare(tmp_path/'persistent',ctx,samples,audit,proven,{'revision':'a'*40},{'gpu':'fake'})
    return root,samples,evidence


class FakeData:
    def audio(self,sample,destination):wav(destination)


@dataclass
class RunConfig:
    output_dir: Path


@dataclass
class Config:
    run: RunConfig


class FakeSource:
    def __init__(self,plan,destination):self.config=Config(RunConfig(destination))
    def group(self,samples):
        rows=[]
        for s in samples:
            path=self.config.run.output_dir/(digest(s['sample_id'])[:16]+'.wav');wav(path)
            rows.append(dict(generation_status='ok',wav_16k=str(path),tts_seconds=.1))
        return rows


class FakeAdapter:
    calls=[]
    fail_stage=None
    def __init__(self,plan,stage):self.stage=stage
    def load(self):pass
    def __call__(self,source,destination):
        self.calls.append(self.stage)
        if self.fail_stage==self.stage:raise ValueError('intentional stage failure')
        if destination:wav(destination)
        return dict(text='the very old book is on the table' if self.stage in ('mt','evaluation_asr') else '日本語')


def test_complete_three_branches_resume_and_official_results_are_unchanged(tmp_path):
    root,samples,evidence=fixture(tmp_path)
    before=evidence.read_bytes(),evidence.stat().st_mtime_ns
    FakeAdapter.calls=[]
    for stage in diag.STAGES:
        diag.worker(root,stage,data=FakeData(),adapter_factory=FakeAdapter,source_factory=FakeSource)
    report=diag.report(root)
    calls=len(FakeAdapter.calls)
    assert calls==22  # two samples * (2 source ASR + 3 MT + 3 TTS + 3 EN ASR)
    for stage in diag.STAGES:
        diag.worker(root,stage,data=FakeData(),adapter_factory=FakeAdapter,
                    source_factory=lambda *a:pytest.fail('source successful WAV must be reused'))
    assert len(FakeAdapter.calls)==calls
    assert diag.report(root)==report
    assert (evidence.read_bytes(),evidence.stat().st_mtime_ns)==before
    assert report['metrics']['samples']==2
    assert all(report['metrics'][c+'_failures']==0 for c in diag.CONDITIONS)
    for row in report['per_sample']:
        assert row['en_reference']==next(s['en_reference'] for s in samples if s['sample_id']==row['sample_id'])
        assert all(row[c+'_mt_en_text'] for c in diag.CONDITIONS)
    for filename in ('metrics.json','report.md','per_sample.jsonl','per_sample.csv','dataset-audit.json','model-audit.json'):
        assert (root/filename).is_file()


def test_all_failed_hypotheses_remain_in_corpus_bleu_denominator_and_cer():
    samples=[];values=[]
    for i in range(2):
        samples.append(dict(sample_id=str(i),ja_text='日本語。',en_reference='the very old book is on the table'))
        row={}
        for c in diag.CONDITIONS:
            row[c,'source_asr']=dict(status='success',text='日本語。')
            row[c,'input_audio']=dict(status='success',text='')
            row[c,'mt']=dict(status='success' if i==0 else 'failed',text='the very old book is on the table' if i==0 else '')
            row[c,'target_tts']=dict(status='success' if i==0 else 'failed',text='')
            row[c,'evaluation_asr']=dict(status='success' if i==0 else 'failed',text='the very old book is on the table' if i==0 else '')
        values.append(row)
    m,rows=diag.summarize(samples,values)
    expected=corpus_bleu(['the very old book is on the table',''],['the very old book is on the table']*2)
    for c in diag.CONDITIONS:
        assert m[c+'_MT_BLEU']==expected and m[c+'_final_ASR_BLEU']==expected and m[c+'_failures']==1
    assert expected != sum(r['gold_mt_sentence_bleu'] for r in rows)/2


def test_new_cell_order_and_all_existing_notebook_cells_are_identical():
    b=json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    previous=json.loads(subprocess.check_output(['git','-c','safe.directory='+str(ROOT),'-C',str(ROOT),
        'show','36451d53ad92d1d172fa4c4334dfae3a7966d805:notebooks/colab_training.ipynb'],encoding='utf-8'))
    new=[c for c in b['cells'] if ''.join(c['source']).startswith('#@title 8C.')]
    assert len(new)==1
    assert [c for c in b['cells'] if c not in new and not ''.join(c['source']).startswith('#@title 8D.')]==previous['cells']
    i=b['cells'].index(new[0]);assert ''.join(b['cells'][i-1]['source']).startswith('#@title 8B.')
    assert ''.join(b['cells'][i+1]['source']).startswith(('#@title 8D.', '#@title 9.'))
    code=''.join(new[0]['source']);ast.parse(code,feature_version=(3,10))
    assert code.index('_ff_preflight =')<code.index("_ff_runtime['ensure']")
    assert 'FLORES_FLEURS_LIMIT = 300' in code
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in
        ('run','cli','ensure_runtime','ensure_workspace') for n in ast.walk(ast.parse(code)))


def test_failed_stages_retry_without_repeating_successful_upstream_models(tmp_path):
    root,samples,_=fixture(tmp_path)
    FakeAdapter.calls=[];FakeAdapter.fail_stage='target_tts'
    try:
        for stage in diag.STAGES:
            diag.worker(root,stage,data=FakeData(),adapter_factory=FakeAdapter,source_factory=FakeSource)
        report=diag.report(root)
        assert all(report['metrics'][c+'_failures']==2 for c in diag.CONDITIONS)
    finally:
        FakeAdapter.fail_stage=None
    calls=len(FakeAdapter.calls)
    for stage in diag.STAGES:
        diag.worker(root,stage,data=FakeData(),adapter_factory=FakeAdapter,source_factory=lambda *a:pytest.fail('source reused'))
    assert FakeAdapter.calls[calls:]==['target_tts']*6+['evaluation_asr']*6
    assert all(diag.report(root,overwrite=True)['metrics'][c+'_failures']==0 for c in diag.CONDITIONS)


def test_changed_runtime_or_dataset_creates_new_identity(tmp_path):
    root,_,_=fixture(tmp_path);plan=diag.read(root/'identity.json')
    context=plan['context']
    args=(tmp_path/'persistent',context,plan['selected'],plan['dataset'],plan['source_proof'],plan['source_code'])
    second=diag.prepare(*args,{'gpu':'different runtime'})
    assert root!=second
    changed=copy.deepcopy(plan['dataset']);changed['datasets'][0]['revision']='f'*40
    third=diag.prepare(args[0],context,args[2],changed,args[4],args[5],plan['environment'])
    assert third not in (root,second)


def test_source_cer_keeps_punctuation_and_failed_asr_counts_as_empty():
    samples=[dict(sample_id='1',ja_text='Ａ　あ。',en_reference='some reference with enough words')]
    values={('natural','source_asr'):dict(status='failed',text='do not use this'),
            ('synthetic','source_asr'):dict(status='success',text='Aあ')}
    metrics,rows=diag.summarize(samples,[values])
    assert metrics['natural_source_CER']==1 and metrics['synthetic_source_CER']==pytest.approx(1/3)
    assert rows[0]['natural_asr_ja_text']==''


def test_cuda_and_loading_errors_are_fatal_not_silently_emptied(tmp_path):
    cache=diag.StageCache(tmp_path,'i');r=cache.request('sample','gold','mt',{})
    def cuda(_):raise RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError,match='CUDA'):cache.run(r,cuda)
    assert not list(tmp_path.rglob('*.json'))


def test_official_gated_flores_error_never_falls_back_to_mirror(tmp_path,monkeypatch):
    import huggingface_hub
    calls=[]
    def denied(**kwargs):calls.append(kwargs);raise PermissionError('gated')
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',denied)
    with pytest.raises(RuntimeError,match='Accept the dataset terms and authenticate'):
        data.OfficialData(tmp_path).load()
    assert len(calls)==1 and calls[0]['repo_id']==data.FLORES and calls[0]['revision']==data.FLORES_REVISION
    assert not list(tmp_path.iterdir())


def test_synthetic_group_failure_retries_only_failed_members_and_keeps_successful_wavs(tmp_path):
    root,samples,_=fixture(tmp_path)
    calls=[]
    class FailingSource(FakeSource):
        def group(self,samples):
            calls.append([s['sample_id'] for s in samples])
            rows=super().group(samples)
            rows[0]=dict(generation_status='failed',generation_error='bad source',tts_seconds=.1)
            return rows
    diag.worker(root,'synthetic_audio',source_factory=FailingSource)
    waves={str(p):p.read_bytes() for p in (root/'synthetic-source').rglob('*.wav')}
    diag.worker(root,'synthetic_audio',source_factory=FailingSource)
    assert len(calls)==2 and len(calls[0])==2 and calls[1]==calls[0][:1]
    assert all(Path(p).read_bytes()==b for p,b in waves.items())
    assert len(list((root/'synthetic-source').rglob('*.wav')))==3


def test_synthetic_batch_exception_does_not_forget_already_successful_members(tmp_path):
    root,_,_=fixture(tmp_path)
    calls=[]
    class Source(FakeSource):
        def group(self,samples):
            calls.append([s['sample_id'] for s in samples])
            if len(calls)==2:raise ValueError('temporary generation failure')
            rows=super().group(samples)
            if len(calls)==1:rows[0]=dict(generation_status='failed',generation_error='bad source')
            return rows
    for _ in range(3):diag.worker(root,'synthetic_audio',source_factory=Source)
    assert len(calls[0])==2 and calls[1]==calls[2]==calls[0][:1]


def test_worker_identity_tampering_is_rejected_before_model_load(tmp_path):
    root,_,_=fixture(tmp_path);plan=diag.read(root/'identity.json');plan['version']=100
    atomic_write_json(root/'identity.json',plan,overwrite=True)
    with pytest.raises(RuntimeError,match='identity/path mismatch'):
        diag.worker(root,'mt',adapter_factory=lambda *a:pytest.fail('no load'))


def test_all_new_python_files_parse_for_colab_python310():
    for name in ('src/direct_s2st/cascade/flores_fleurs_data.py','src/direct_s2st/cascade/flores_fleurs_diagnostic.py',
                 'src/direct_s2st/cascade/source_corpus_tts.py','scripts/colab/flores_fleurs_diagnostic.py',
                 'scripts/colab/flores_fleurs_runtime.py'):
        ast.parse((ROOT/name).read_text(encoding='utf-8'),feature_version=(3,10))


def test_source_bridge_uses_original_corpus_helpers_not_english_tts_adapter():
    text=(ROOT/'src/direct_s2st/cascade/source_corpus_tts.py').read_text(encoding='utf-8')
    assert 's2st_corpus.runner' in text and 'self.runner._generate_missing(' in text
    assert 'generate_custom_voice(' not in text and 'from .tts import QwenTTS' not in text
    for k in ('S2ST_TTS_BATCH_SIZE','S2ST_ASR_BATCH_SIZE'):
        assert k in text


def test_original_source_checkout_is_read_only_and_sha_checked(tmp_path,monkeypatch):
    p,d=receipt(tmp_path)
    code=tmp_path/'original-corpus';cfg=code/'configs/colab-pro.yaml';cfg.parent.mkdir(parents=True)
    cfg.write_text('original tracked YAML',encoding='utf-8')
    py=code/'src/s2st_corpus/runner.py';py.parent.mkdir(parents=True);py.write_text('original source',encoding='utf-8')
    d['config_sha256']=diag.checksum(cfg);atomic_write_json(p,d,overwrite=True)
    proven=source.proof(p);calls=[]
    def git(root,*args):calls.append(args);return d['code_revision'] if 'rev-parse' in args else ''
    monkeypatch.setattr(source,'git',git)
    before=cfg.read_bytes(),py.read_bytes()
    audit=source.verify_code(code,proven)
    assert audit['config_sha256']==d['config_sha256'] and 'src/s2st_corpus/runner.py' in audit['source_files']
    assert before==(cfg.read_bytes(),py.read_bytes()) and not any('checkout' in args or 'reset' in args for args in calls)
    cfg.write_text('modified YAML')
    with pytest.raises(RuntimeError,match='config SHA'):source.verify_code(code,proven)


def test_saved_pipeline_settings_and_eval_whisper_are_loaded_without_defaults(tmp_path,monkeypatch):
    ctx=dict(inputs_sha256={'official.json':'a'*64})
    pipeline=dict(asr={'model':'JA-ASR','revision':'a'*40},
        mt={'model':'facebook/nllb-200-distilled-600M','revision':'b'*40,'source_language':'jpn_Jpan','target_language':'eng_Latn'},
        tts={'model':'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice','revision':'c'*40,'speaker':'Ono_Anna','language':'English',
             'generation':dict(policy='official_sampling_eos_v1',seed=17,max_audio_seconds=120,max_wall_seconds=600)})
    evaluation=dict(model_id='EN-ASR',revision='d'*40)
    atomic_write_json(tmp_path/'plan.json',dict(pipeline=pipeline,asr=evaluation))
    atomic_write_json(tmp_path/'results/metrics.json',dict(evaluation_identity={'asr':evaluation}))
    monkeypatch.setattr(diag,'saved_context',lambda *a:ctx)
    actual=diag.context(tmp_path/'manifest',tmp_path)
    assert actual['pipeline']==pipeline and actual['evaluation_asr']==evaluation
    del evaluation['revision'];atomic_write_json(tmp_path/'plan.json',dict(pipeline=pipeline,asr=evaluation),overwrite=True)
    atomic_write_json(tmp_path/'results/metrics.json',dict(evaluation_identity={'asr':evaluation}),overwrite=True)
    with pytest.raises(RuntimeError,match='no guesses'):diag.context(tmp_path/'manifest',tmp_path)


def test_missing_target_policy_never_infers_sampling_from_current_defaults(tmp_path,monkeypatch):
    pipeline=dict(asr={'model':'JA','revision':'a'*40},
        mt={'model':'facebook/nllb-200-distilled-600M','revision':'b'*40,'source_language':'jpn_Jpan','target_language':'eng_Latn'},
        tts={'model':'qwen','revision':'c'*40,'speaker':'Ono_Anna','language':'English'})
    evaluation={'model_id':'EN','revision':'d'*40}
    atomic_write_json(tmp_path/'plan.json',dict(pipeline=pipeline,asr=evaluation))
    atomic_write_json(tmp_path/'results/metrics.json',dict(evaluation_identity={'asr':evaluation}))
    monkeypatch.setattr(diag,'saved_context',lambda *a:dict(inputs_sha256={}))
    with pytest.raises(RuntimeError,match='cannot guess historical decoding defaults'):diag.context('manifest',tmp_path)


def test_real_saved_generic_context_uses_evaluation_identity_asr_not_a_nonexistent_top_level_key(tmp_path,monkeypatch):
    from test_cascade_corpus_diagnostic import fixture as formal_fixture,snapshot
    common,output,*_=formal_fixture(tmp_path,monkeypatch,benchmark=False)
    pipeline=dict(asr={'model':'JA-ASR','revision':'a'*40},
        mt={'model':'facebook/nllb-200-distilled-600M','revision':'b'*40,'source_language':'jpn_Jpan','target_language':'eng_Latn'},
        tts={'model':'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice','revision':'c'*40,'speaker':'Ono_Anna','language':'English',
             'generation':dict(policy='official_sampling_eos_v1',seed=17,max_audio_seconds=120,max_wall_seconds=600)})
    lock=output/'predictions/predictions.lock.json';saved=diag.read(lock);saved['models']=pipeline
    atomic_write_json(lock,saved,overwrite=True)
    before=snapshot(tmp_path)
    actual=diag.context(common,output)
    assert actual['evaluation_asr']=={'model':'pinned-evaluation-asr','revision':'a'*40}
    assert actual['pipeline']==pipeline and snapshot(tmp_path)==before
