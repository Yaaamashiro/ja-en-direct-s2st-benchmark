"""8C external three-condition Cascade diagnostic; never a formal benchmark."""
import contextlib
import csv
import dataclasses
import io
import json
from pathlib import Path
import re
import sys
import time
import uuid

from .corpus_diagnostic import checksum, read, normalize_japanese_cer, edit_distance
from .flores_fleurs_data import sha_text
from .mt_diagnostic import fatal, saved_context, validate_model
from ..evaluation.bleu import corpus_bleu, sentence_bleu, normalize_english
from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_text, atomic_write_bytes
from ..journal import digest
from ..progress import operation, track

VERSION = 1
PURPOSE = 'external_flores_fleurs_diagnostic_not_formal_benchmark'
CONDITIONS = ('gold','natural','synthetic')
STAGES = ('natural_audio','synthetic_audio','source_asr','mt','target_tts','evaluation_asr')


def pinned(config, label, id_field='model'):
    if (not isinstance(config,dict) or not isinstance(config.get(id_field),str)
            or not re.fullmatch('[0-9a-f]{40}',str(config.get('revision','')))):
        raise RuntimeError('8C requires saved immutable '+label+' model and revision; no guesses/defaults')
    return config


@operation('8C: bind saved Cascade context without changing official results')
def context(manifest, evaluation_root):
    ctx = saved_context(manifest,evaluation_root)
    root = Path(evaluation_root)
    dedicated = (root/'plan.json').is_file()
    if dedicated:
        plan = read(root/'plan.json')
        pipeline = plan['pipeline']
    else:
        pipeline = read(root/'predictions/predictions.lock.json')['models']
    metrics=read(root/('results/metrics.json' if dedicated else 'metrics/metrics.json'))
    evaluation=metrics.get('evaluation_identity',{}).get('asr')
    if dedicated and evaluation!=plan.get('asr'):
        raise RuntimeError('8C saved evaluation Whisper differs between the plan and scored evaluation identity')
    validate_model(pipeline.get('mt'))
    source = pinned(pipeline.get('asr'),'source Whisper')
    target = pinned(pipeline.get('tts'),'target Qwen TTS')
    evaluation = pinned(evaluation,'evaluation Whisper','model_id' if evaluation and 'model_id' in evaluation else 'model')
    if set(source)!={'model','revision'} or target.get('language')!='English':
        raise RuntimeError('8C unsupported saved source ASR / target TTS settings; review before reproduction')
    if set(target)-{'model','revision','speaker','language','generation'} or not target.get('speaker'):
        raise RuntimeError('8C unsupported/incomplete saved target TTS configuration')
    from .qwen_generation import generation_policy
    if (not isinstance(target.get('generation'),dict)
            or set(target['generation'])!={'policy','seed','max_audio_seconds','max_wall_seconds'}):
        raise RuntimeError('8C saved target TTS generation settings are incomplete; cannot guess historical decoding defaults')
    generation_policy(target.get('generation'))
    if set(evaluation)-{'model','model_id','revision'}:
        raise RuntimeError('8C unsupported saved English evaluation ASR arguments')
    return dict(pipeline=pipeline,evaluation_asr=evaluation, inputs_sha256=ctx['inputs_sha256'],
                evaluation_root=str(root), all_score_matches_saved=True)


def safe_output(persistent, protected=()):
    persistent = Path(persistent).absolute()
    root = persistent/'diagnostics/flores-fleurs-cascade'
    defaults = [persistent/x for x in ('data','checkpoints','benchmarks','evaluations','runs')]
    for forbidden in [*defaults,*map(Path,protected)]:
        if root.resolve().is_relative_to(forbidden.resolve()):
            raise RuntimeError('8C diagnostic output overlaps corpus/evaluation/checkpoint directory')
    if any(p.is_symlink() for p in (root,*root.parents)):
        raise RuntimeError('8C diagnostic output may not traverse symlinks')
    return root


def inside(root, path):
    root,path = Path(root).absolute(),Path(path).absolute()
    if not path.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in (path,*path.parents)):
        raise RuntimeError('8C path must remain inside the diagnostic root without symlinks')
    return path


def audio_audit(path):
    import numpy as np
    import soundfile as sf
    wave, rate = sf.read(path, dtype='float32',always_2d=False)
    if (wave.ndim!=1 or rate<=0 or not len(wave) or not np.isfinite(wave).all()
            or not np.any(wave) or len(wave)/rate>600):
        raise ValueError('invalid/empty/non-finite/silent/non-mono or excessive WAV')
    return dict(path=str(Path(path).absolute()),sha256=checksum(path),sample_rate=int(rate),mono=True,
                finite=True,non_silent=True,duration=len(wave)/rate)


class StageCache:
    """Atomic immutable attempts. Failed/corrupt outputs are never reused.

    Request includes text/input-WAV SHA, model/config and full run identity.
    Audio lives at unique attempt paths so retry never overwrites old WAVs.
    """
    def __init__(self, root, identity):
        self.root,self.identity=Path(root),identity
        self._verified,self._lookups,self._source_values={},{},{}

    def valid_wav(self, wav):
        p=inside(self.root,Path(wav['path']))
        try:
            stat=p.stat()
            signature=(stat.st_size,stat.st_mtime_ns,wav['sha256'])
            if self._verified.get(str(p))!=signature:
                if checksum(p)!=wav['sha256']:
                    return False
                self._verified[str(p)]=signature
            return True
        except OSError:
            return False

    def request(self, sample_id, condition, stage, inputs):
        return dict(run=self.identity,sample_id=sample_id,condition=condition,stage=stage,inputs=inputs)

    def get(self, request):
        key=digest(request)
        if key in self._lookups:
            body=self._lookups[key]
            if body is not None and all(self.valid_wav(w) for w in body.get('audio',[])):
                return body
            self._lookups.pop(key)
        folder = inside(self.root,self.root/'checkpoints'/digest(request)[:24])
        for path in sorted(folder.glob('*.json')):
            value=read(path)
            body=value.get('body',{})
            if value.get('sha256')!=digest(body) or body.get('request')!=request:
                raise RuntimeError('8C checkpoint content/identity changed: '+str(path))
            if body.get('status')!='success':
                continue
            try:
                for wav in body.get('audio',[]):
                    if not self.valid_wav(wav):
                        break
                else:
                    self._lookups[key]=body
                    return body
            except OSError:
                pass
        return None

    def run(self, request, call, *, audio_folder=None):
        prior=self.get(request)
        if prior is not None:
            return prior
        token=uuid.uuid4().hex[:16]
        destination=inside(self.root,self.root/audio_folder/(token+'.wav')) if audio_folder else None
        if destination:
            destination.parent.mkdir(parents=True,exist_ok=True)
        body=dict(request=request,status='failed',text='',error=None,audio=[])
        start=time.perf_counter()
        try:
            result=call(destination)
            if not isinstance(result,dict):
                raise RuntimeError('8C adapter returned an invalid result')
            body.update(result)
            if destination:
                body['audio']=[audio_audit(destination)]
            body['status']=result.get('status','success')
        except Exception as error:
            if fatal(error) or not isinstance(error,ValueError):
                raise
            body.update(status='failed',text='',error=f'{type(error).__name__}: {error}',audio=[])
        body['seconds']=time.perf_counter()-start
        path=inside(self.root,self.root/'checkpoints'/digest(request)[:24]/(token+'.json'))
        atomic_write_json(path,dict(body=body,sha256=digest(body)),resume=True)
        self._lookups.pop(digest(request),None)
        return body


def prepare(persistent, ctx, samples, dataset_audit, proven, code_audit, environment, protected=()):
    if not samples or len({r['sample_id'] for r in samples})!=len(samples):
        raise ValueError('8C requires distinct selected canonical sentences')
    if dataset_audit['counts']['matched']<300:
        raise ValueError('8C requires at least 300 matched unique sentences before limit selection')
    for row in samples:
        if (sha_text(row['ja_text'])!=row['ja_text_sha256'] or sha_text(row['en_reference'])!=row['en_text_sha256']):
            raise ValueError('8C selected text audit mismatch')
    code = Path(__file__).parent
    implementations={str(p.relative_to(code.parent)):checksum(p) for p in [
        Path(__file__),code/'flores_fleurs_data.py',code/'source_corpus_tts.py',code/'asr.py',code/'whisper.py',
        code/'mt.py',code/'mt_diagnostic.py',code/'tts.py',code/'qwen_generation.py',code.parent/'evaluation/transcribe.py',
        code.parent/'evaluation/bleu.py']}
    repo=code.parents[2]
    implementations.update({str(p.relative_to(repo)):checksum(p) for p in (
        repo/'scripts/colab/flores_fleurs_diagnostic.py',repo/'scripts/colab/flores_fleurs_runtime.py',
        repo/'requirements/diagnostic-flores.txt')})
    identity=dict(version=VERSION,purpose=PURPOSE,context=ctx,selected=samples,dataset=dataset_audit,
                  source_proof=proven,source_code=code_audit,environment=environment,implementation=implementations,
                  conditions=CONDITIONS,source_attempt=0,source_qc_selection=False,failure_policy='all_selected_failed_as_empty')
    identity=json.loads(json.dumps(identity))
    base=safe_output(persistent,[ctx['evaluation_root'],*protected])
    root=inside(base,base/digest(identity)[:24])
    atomic_write_json(root/'identity.json',identity,resume=True)
    atomic_write_json(root/'dataset-audit.json',dataset_audit,resume=True)
    atomic_write_json(root/'model-audit.json',dict(pipeline=ctx['pipeline'],evaluation_asr=ctx['evaluation_asr'],
                                                 source_corpus=proven,source_code=code_audit),resume=True)
    return root


def audio_result(value):
    if value is None or value['status']!='success' or len(value.get('audio',[]))!=1:
        raise ValueError('upstream audio stage failed')
    return value['audio'][0]


class Adapters:
    """Only one GPU model is loaded in each worker process."""
    def __init__(self,plan,stage):
        self.plan,self.stage,self.model=plan,stage,None

    def load(self):
        if self.model is not None:
            return self.model
        ctx=self.plan['context']
        with contextlib.redirect_stdout(sys.stderr):
            if self.stage=='source_asr':
                from .asr import WhisperASR
                cfg=ctx['pipeline']['asr']
                self.model=WhisperASR(model_id=cfg['model'],revision=cfg['revision'],device=0)
            elif self.stage=='mt':
                from .mt_diagnostic import NllbProbe
                self.model=NllbProbe(ctx['pipeline']['mt'])
            elif self.stage=='target_tts':
                from .tts import QwenTTS
                cfg=ctx['pipeline']['tts']
                self.model=QwenTTS(model_id=cfg['model'],revision=cfg['revision'],speaker=cfg['speaker'],
                                   language=cfg['language'],generation=cfg.get('generation'))
            elif self.stage=='evaluation_asr':
                from ..evaluation.transcribe import EvaluationASR
                cfg=ctx['evaluation_asr']
                self.model=EvaluationASR(model_id=cfg.get('model_id') or cfg['model'],revision=cfg['revision'],device=0)
            else:
                raise RuntimeError('unsupported 8C model stage')
        return self.model

    def audit(self):
        if hasattr(self.model,'audit'):
            return self.model.audit
        if self.stage=='target_tts':
            return dict(runtime=self.model.runtime,policy=self.model.generation,snapshot=self.model.snapshot)
        pipeline=self.model.pipeline if self.stage=='source_asr' else self.model.model.pipeline
        return dict(dtype=str(next(pipeline.model.parameters()).dtype),
                    generation_config=pipeline.model.generation_config.to_dict(),
                    adapter='WhisperASR Japanese transcribe' if self.stage=='source_asr' else 'EvaluationASR English transcribe')

    def __call__(self,source,destination):
        model=self.load()
        with contextlib.redirect_stdout(sys.stderr):
            if self.stage=='target_tts':
                model(source,destination)
                return dict(text='',generation=getattr(model,'last_generation',None))
            if self.stage=='mt':
                result=model(source)
                if not result.get('translation','').strip():
                    raise ValueError('empty NLLB output')
                return dict(text=result['translation'],generation=result)
            result=model(Path(source))
            if not result.strip():
                raise ValueError('empty ASR output')
            return dict(text=result)


def stage_requests(sample, stage, values, plan):
    """All branches depend on the SAME gold JA and English reference identity."""
    for condition in CONDITIONS:
        if stage=='natural_audio':
            if condition=='natural':
                yield condition,dict(locator=sample['locator'],dataset=plan['dataset']['datasets'],
                                     text_sha=sample['ja_text_sha256']),None
            continue
        if stage=='source_asr':
            if condition=='gold':
                continue
            upstream=values.get((condition,'input_audio'))
            wav=audio_result(upstream) if upstream and upstream['status']=='success' else None
            yield condition,dict(audio=wav,model=plan['context']['pipeline']['asr']),wav['path'] if wav else None
        elif stage=='mt':
            previous=values.get((condition,'source_asr'))
            source=sample['ja_text'] if condition=='gold' else previous['text'] if previous and previous['status']=='success' else ''
            yield condition,dict(text_sha=sha_text(source),model=plan['context']['pipeline']['mt']),source
        elif stage=='target_tts':
            previous=values.get((condition,'mt'))
            source=previous['text'] if previous and previous['status']=='success' else ''
            yield condition,dict(text_sha=sha_text(source),model=plan['context']['pipeline']['tts']),source
        elif stage=='evaluation_asr':
            previous=values.get((condition,'target_tts'))
            wav=audio_result(previous) if previous and previous['status']=='success' else None
            yield condition,dict(audio=wav,model=plan['context']['evaluation_asr']),wav['path'] if wav else None


def source_groups(samples,batch_size):
    samples=sorted(samples,key=lambda r:(len(r['ja_text']),r['sample_id']))
    return [samples[i:i+batch_size] for i in range(0,len(samples),batch_size)]


@operation('8C: source corpus fixed-batch synthesis/reuse')
def synthetic_stage(root,plan,cache,source_factory):
    model=None
    n=plan['source_proof']['receipt']['effective_tts']['batch_size']
    for group in track(source_groups(plan['selected'],n),'8C: Japanese source TTS groups'):
        request=cache.request(digest([s['sample_id'] for s in group]),'synthetic','input_audio_group',
                              dict(samples=group,source=plan['source_proof'],batch_size=n))
        previous=cache.get(request)
        if previous:
            save_stage_result(root,request,previous)
            continue
        # Keep individually verified successes even if another batch member
        # failed. The original corpus helper generates only pending inputs;
        # its configured batch ceiling and seed routine are unchanged.
        kept={}
        pointer=root/'stage-results'/(digest([request['sample_id'],'synthetic','input_audio_group'])+'.json')
        if pointer.is_file():
            document=read(pointer)
            if document['sha256']!=digest(document['body']) or document['body']['request']!=request:
                raise RuntimeError('8C source group progress identity mismatch')
            kept={item['sample_id']:item for item in document['body'].get('items',[])
                  if item['status']=='success' and all(cache.valid_wav(w) for w in item.get('audio',[]))}
        pending=[s for s in group if s['sample_id'] not in kept]
        destination=inside(root,root/'synthetic-source'/uuid.uuid4().hex[:16])
        if pending and model is None:
            model=source_factory(plan,destination)
            if hasattr(model,'audit'):
                atomic_write_json(root/'execution-audit/synthetic-source.json',model.audit,resume=True)
        elif pending:
            model.config=dataclasses.replace(model.config,
                run=dataclasses.replace(model.config.run,output_dir=destination))
        def synth(_):
            try:
                outputs=model.group(pending) if pending else []
            except ValueError as error:
                outputs=[dict(generation_status='failed',generation_error=f'{type(error).__name__}: {error}') for _ in pending]
            if len(outputs)!=len(pending):
                raise RuntimeError('source corpus batch size mismatch')
            updated=dict(kept)
            for sample,result in zip(pending,outputs):
                if result.get('generation_status')!='ok':
                    updated[sample['sample_id']]=dict(sample_id=sample['sample_id'],status='failed',text='',audio=[],error=result.get('generation_error'))
                else:
                    try:
                        wav=audio_audit(inside(root,Path(result['wav_16k'])))
                        if wav['sample_rate']!=16000:
                            raise ValueError('corpus Japanese TTS did not produce recorded 16kHz PCM16')
                        updated[sample['sample_id']]=dict(sample_id=sample['sample_id'],status='success',text='',audio=[wav],
                                          error=None,provenance=result,seconds=result['tts_seconds'])
                    except ValueError as error:
                        updated[sample['sample_id']]=dict(sample_id=sample['sample_id'],status='failed',text='',audio=[],
                                                        error=f'{type(error).__name__}: {error}')
            items=[updated[s['sample_id']] for s in group]
            return dict(text='',items=items,audio=[w for item in items for w in item['audio']],
                        status='success' if all(r['status']=='success' for r in items) else 'failed')
        body=cache.run(request,synth)
        save_stage_result(root,request,body)


def save_stage_result(root,request,body):
    # Small owned index files may be atomically updated after a new retry;
    # immutable attempts and all output WAVs are preserved.
    path=inside(root,root/'stage-results'/digest([request['sample_id'],request['condition'],request['stage']]))
    atomic_write_json(path.with_suffix('.json'),dict(body=body,sha256=digest(body)),overwrite=True)


def values_for(sample,root,plan,cache):
    values={}
    groups=source_groups(plan['selected'],plan['source_proof']['receipt']['effective_tts']['batch_size'])
    group=next(g for g in groups if any(s['sample_id']==sample['sample_id'] for s in g))
    request=cache.request(digest([s['sample_id'] for s in group]),'synthetic','input_audio_group',
                          dict(samples=group,source=plan['source_proof'],batch_size=plan['source_proof']['receipt']['effective_tts']['batch_size']))
    pointer=root/'stage-results'/ (digest([request['sample_id'],'synthetic','input_audio_group'])+'.json')
    group_key=digest(request)
    if group_key not in cache._source_values and pointer.is_file():
        document=read(pointer)
        body=document['body']
        if document['sha256']!=digest(body) or body['request']!=request:
            raise RuntimeError('8C source stage index changed')
        items=body.get('items') or [dict(sample_id=s['sample_id'],status='failed',text='',audio=[],
                                       error=body.get('error') or 'source synthesis failed',seconds=body.get('seconds')) for s in group]
        cache._source_values[group_key]={item['sample_id']:item for item in items}
    item=cache._source_values.get(group_key,{}).get(sample['sample_id'])
    if item:
        if not all(cache.valid_wav(w) for w in item.get('audio',[])):
            item=dict(status='failed',text='',audio=[],error='source WAV checksum mismatch')
        values['synthetic','input_audio']=item
    for stage in ('natural_audio','source_asr','mt','target_tts','evaluation_asr'):
        for condition,inputs,source in stage_requests(sample,stage,values,plan):
            request=cache.request(sample['sample_id'],condition,stage,inputs)
            result=cache.get(request)
            if result is None:
                pointer=root/'stage-results'/(digest([sample['sample_id'],condition,stage])+'.json')
                if pointer.is_file():
                    doc=read(pointer)
                    if doc['sha256']!=digest(doc['body']):
                        raise RuntimeError('8C failed stage index checksum mismatch')
                    if doc['body']['request']==request and doc['body']['status']=='failed':
                        result=doc['body']
            if result:
                values[condition,'input_audio' if stage=='natural_audio' else stage]=result
    return values


def owned_plan(root):
    root=Path(root)
    plan=read(root/'identity.json')
    if (root.name!=digest(plan)[:24] or plan.get('purpose')!=PURPOSE
            or root.parent.name!='flores-fleurs-cascade' or root.parent.parent.name!='diagnostics'):
        raise RuntimeError('8C worker identity/path mismatch')
    inside(root,root)
    return plan


@operation('8C: sequential stage worker')
def worker(root,stage,*,adapter_factory=Adapters,data=None,source_factory=None):
    root=Path(root)
    plan=owned_plan(root)
    cache=StageCache(root,digest(plan))
    if stage=='synthetic_audio':
        return synthetic_stage(root,plan,cache,source_factory)
    if stage not in STAGES:
        raise ValueError('unknown diagnostic stage')
    adapter=adapter_factory(plan,stage) if stage not in ('natural_audio',) else None
    for sample in track(plan['selected'],'8C: '+stage,key=lambda r:r['sample_id']):
        values=values_for(sample,root,plan,cache)
        for condition,inputs,source in stage_requests(sample,stage,values,plan):
            request=cache.request(sample['sample_id'],condition,stage,inputs)
            if cache.get(request):
                continue
            if stage=='natural_audio':
                def call(destination):
                    metadata=data.audio(sample,destination)
                    return dict(text='',**(metadata or {}))
                folder='natural'
            else:
                # Load errors/dependency failures are fatal before per-sample handling.
                if source:
                    adapter.load()
                    if callable(getattr(adapter,'audit',None)):
                        atomic_write_json(root/'execution-audit'/(stage+'.json'),adapter.audit(),resume=True)
                def call(destination,source=source):
                    if not source:
                        raise ValueError('upstream stage failed/empty; retained as empty hypothesis')
                    return adapter(source,destination)
                folder='output-audio/'+condition if stage=='target_tts' else None
            body=cache.run(request,call,audio_folder=folder)
            save_stage_result(root,request,body)


def summarize(samples,all_values):
    references=[normalize_english(s['en_reference']) for s in samples]
    metrics, per_sample = dict(samples=len(samples)),[]
    for sample,values in zip(samples,all_values):
        row={k:v for k,v in sample.items() if k not in ('locator','raw_transcription')}
        row['gold_ja_text']=sample['ja_text']
        for condition in CONDITIONS:
            stages={stage:values.get((condition,stage)) for stage in ('input_audio','source_asr','mt','target_tts','evaluation_asr')}
            for stage,value in stages.items():
                if value:
                    row[condition+'_'+stage+'_status']=value['status']
                    row[condition+'_'+stage+'_error']=value.get('error')
                    row[condition+'_'+stage+'_seconds']=value.get('seconds')
                    for wav in value.get('audio',[]):
                        for name,field in wav.items():
                            row[condition+'_'+stage+'_'+name]=field
            for stage,field in (('source_asr','asr_ja_text'),('mt','mt_en_text'),('evaluation_asr','output_asr_en_text')):
                value=stages[stage]
                row[condition+'_'+field]=value['text'] if value and value['status']=='success' else ''
            row[condition+'_status']='success' if all(stages[k] and stages[k]['status']=='success' for k in
                (('mt','target_tts','evaluation_asr') if condition=='gold' else tuple(stages))) else 'failed'
            row[condition+'_error']='; '.join(str(value['error']) for value in stages.values() if value and value.get('error')) or None
            for stage,field in (('mt','mt_en_text'),('final','output_asr_en_text')):
                row[condition+'_'+stage+'_sentence_bleu']=sentence_bleu(normalize_english(row[condition+'_'+field]),normalize_english(sample['en_reference']))
            if condition!='gold':
                ref=normalize_japanese_cer(sample['ja_text'])
                hyp=normalize_japanese_cer(row[condition+'_asr_ja_text'])
                row[condition+'_cer_edits']=edit_distance(ref,hyp)
                row[condition+'_cer_chars']=len(ref)
                row[condition+'_cer']=row[condition+'_cer_edits']/len(ref)
        row['natural_audio_path']=row.get('natural_input_audio_path')
        row['synthetic_audio_path']=row.get('synthetic_input_audio_path')
        row['synthetic_source_audio']=row['synthetic_audio_path']
        for condition in CONDITIONS:
            row[condition+'_output_audio']=row.get(condition+'_target_tts_path')
        per_sample.append(row)
    for condition in CONDITIONS:
        for stage,field in (('MT_BLEU','mt_en_text'),('final_ASR_BLEU','output_asr_en_text')):
            metrics[condition+'_'+stage]=corpus_bleu([normalize_english(r[condition+'_'+field]) for r in per_sample],references)
        metrics[condition+'_failures']=sum(r[condition+'_status']!='success' for r in per_sample)
        if condition!='gold':
            metrics[condition+'_source_CER']=sum(r[condition+'_cer_edits'] for r in per_sample)/sum(r[condition+'_cer_chars'] for r in per_sample)
        metrics[condition+'_target_tts_penalty']=metrics[condition+'_MT_BLEU']-metrics[condition+'_final_ASR_BLEU']
    for condition in ('natural','synthetic'):
        metrics[condition+'_source_asr_penalty']=metrics['gold_MT_BLEU']-metrics[condition+'_MT_BLEU']
    metrics['synthetic_vs_natural_MT_gap']=metrics['synthetic_MT_BLEU']-metrics['natural_MT_BLEU']
    return metrics,per_sample


@operation('8C: save separate external diagnostic report')
def report(root,*,overwrite=False):
    root=Path(root)
    plan=owned_plan(root)
    cache=StageCache(root,digest(plan))
    for path,sha in {**plan['context']['inputs_sha256'],**plan['source_proof']['inputs_sha256']}.items():
        if checksum(path)!=sha:
            raise RuntimeError('8C original evidence changed during diagnosis: '+path)
    metrics,per_sample=summarize(plan['selected'],[values_for(s,root,plan,cache) for s in plan['selected']])
    display=['=== 8C: FLORES/FLEURS external Cascade sanity check ===',
        'Diagnostic only; official KFTT/JESC results unchanged. Corpus BLEU; failures included as empty.',
        'Synthetic source: original generation settings, attempt 0, without corpus CER-based acceptance/retry selection.',
        'Matched unique sentences: '+str(plan['dataset']['counts']['matched']),
        'Selected: '+str(len(per_sample)), 'Matching counts: '+json.dumps(plan['dataset']['counts']),
        '', 'Condition           Source CER       MT BLEU      Final ASR-BLEU     Failures']
    for c in CONDITIONS:
        cer=f"{metrics[c+'_source_CER']:.4f}" if c!='gold' else '-'
        display.append(f"{c:16} {cer:>10} {metrics[c+'_MT_BLEU']:13.4f} {metrics[c+'_final_ASR_BLEU']:18.4f} {metrics[c+'_failures']:5}/{len(per_sample)}")
    display+=['']+[key+f': {value:+.4f} BLEU' for key,value in metrics.items() if 'penalty' in key or key.endswith('_gap')]
    display+=['','These differences are diagnostic clues, not proof of a sole cause.', 'Saved: '+str(root)]
    result=dict(purpose=PURPOSE,identity_sha256=digest(plan),output_root=str(root),metrics=metrics,
                all_score_matches_saved=True,display='\n'.join(display),per_sample=per_sample)
    # The normal cell explicitly requests replacement of ONLY these owned views.
    atomic_write_json(root/'metrics.json',result,overwrite=overwrite,resume=True)
    atomic_write_text(root/'report.md',result['display']+'\n',overwrite=overwrite,resume=True)
    atomic_write_jsonl(root/'per_sample.jsonl',per_sample,overwrite=overwrite,resume=True)
    stream=io.StringIO(newline='')
    fields=sorted({k for r in per_sample for k in r})
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows(per_sample)
    atomic_write_bytes(root/'per_sample.csv',stream.getvalue().encode('utf-8'),overwrite=overwrite,resume=True)
    return result
