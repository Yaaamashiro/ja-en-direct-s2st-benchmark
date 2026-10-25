"""8F: frozen 8C Synthetic WAVs, bound to an already completed Natural 8E.

Only data validation, plan ownership and reporting are condition-specific.
Preparation, source-only fairseq generation, vocoding and ASR journals use 8E.
"""
import copy
import csv
import hashlib
import io
import math
from pathlib import Path
import re
import wave

from . import flores_fleurs_diagnostic as core
from ..benchmark_storage import guarded, read
from ..cascade.flores_fleurs_diagnostic import source_groups
from ..drive_staging import raw_sha
from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_text, read_jsonl
from ..journal import digest
from ..progress import operation, track

PURPOSE = 'external_flores_fleurs_s2ut_synthetic_not_formal_benchmark'
VERSION = 1
SAMPLES = 300
MODEL = dict(name='s2ut-training-experiment-001-paper-fisher-v1', best_update=91000, updates=400000,
             checkpoint_sha256='d2ee19bec9ce5f12c19f6b7ac64cf68516ea7010b373aa878a6d9e1f34fc0c7b')
NATURAL_CODE = {
    '4e18c8f872561c57a6cb32f115164f5c331db972':'30c67e811a2aa7a52c76c430b949dfc83d9a6a0bc97213ed066bcd1cf4e719c1',
    '32cc967f01808d5abc82df2d709b5c8262a9e961':'74830023d326238d9e55621942fe119586dd2b1ee8adb8aeb45cc18b480962c6',
}
RESULT_FILES = ('metrics.json','per_sample.jsonl','per_sample.csv','report.md','audio-index.json',
                'units.jsonl','predictions.jsonl','external-input-audit.json','evaluation-runtime.json')
INFERENCE_SCOPES = ('source_context','prepare_external','verify_external','path_values','parse_units','shard_records',
    'units','audio','restore_audio','evaluate','restore_asr_progress','SAMPLE_FIELDS','ASR','NORMALIZATION',
    'verify_asr_runtime','canonical_packages','verify_saved_asr_dependencies')
INFERENCE_SHA256 = 'a4dafadcdd215aa41b136c882a2a9559431efb95520baf9c675ab58d4a0cebc4'


def number(value, label):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('missing/nonfinite '+label)
    return value


def checked_completion(root, *, expected_identity=None):
    root=Path(root)
    receipt=read(root/'completion.json')
    if (expected_identity is not None and receipt.get('identity')!=expected_identity
            or not re.fullmatch('[0-9a-f]{64}',str(receipt.get('identity','')))
            or not isinstance(receipt.get('files'),dict) or not set(RESULT_FILES)<=set(receipt['files'])):
        raise ValueError('completed external evaluation receipt is incomplete or has another identity')
    for name,sha in track(receipt['files'].items(),'8F: verify completed report files'):
        from ..benchmark_storage import safe_name
        path=core.inside(root,root.joinpath(*safe_name(name).parts))
        if raw_sha(path)!=sha:
            raise ValueError('completed external evaluation checksum differs: '+name)
    return receipt


def check_model(plan):
    if (plan['run']['name']!=MODEL['name'] or plan['run']['updates']!=MODEL['updates']
            or plan['best_update']!=MODEL['best_update'] or plan['checkpoint_sha256']!=MODEL['checkpoint_sha256']):
        raise ValueError('8F requires the specified completed 8E S2UT run/checkpoint; no model substitution')
    from ..vocoders.reference import BASE, CHECKPOINT_SHA256, CONFIG_SHA256
    expected=dict(url=BASE,checkpoint='g_00500000',checkpoint_sha256=CHECKPOINT_SHA256,
                  config_sha256=CONFIG_SHA256,duration_prediction=True)
    if (plan['beam']!=10 or plan['vocoder']!=expected or plan['asr']!=core.ASR
            or plan['normalization']!=core.NORMALIZATION or plan['bleu_denominator']!='all_samples_failed_as_empty'):
        raise ValueError('8F Natural checkpoint decoding/vocoder/ASR/scoring settings differ')


def code_release(files, revision):
    from .flores_fleurs_resume import relative_files
    values={name:sha for name,(_,sha) in relative_files(files).items()}
    if revision not in NATURAL_CODE or digest(values)!=NATURAL_CODE[revision]:
        raise ValueError('8F Natural inference implementation is not a reviewed 8E release')


def validate_natural_documents(plan, metrics, rows, runtime, asr_audit):
    """Pure metadata check, also usable on real saved JSON without model/WAVs."""
    body={k:v for k,v in plan.items() if k not in ('identity','local','output')}
    if digest(body)!=plan['identity'] or plan['purpose']!=core.PURPOSE or plan['version']!=core.VERSION:
        raise ValueError('8F Natural plan identity/purpose/version mismatch')
    check_model(plan);code_release(plan['execution_files'],plan['execution_revision'])
    from .flores_fleurs_resume import relative_files,scope_sha
    if scope_sha(Path(core.__file__),INFERENCE_SCOPES)!=INFERENCE_SHA256:
        raise ValueError('8F source-only inference/ASR implementation differs from Natural 8E')
    excluded={'src/direct_s2st/s2ut/flores_fleurs_diagnostic.py','src/direct_s2st/s2ut/flores_fleurs_resume.py',
              'scripts/colab/s2ut_flores_fleurs_diagnostic.py'}
    current=relative_files(core.implementation_files())
    for relative,(_,sha) in relative_files(plan['execution_files']).items():
        if relative not in excluded:
            path=current[relative][0];content=path.read_bytes()
            if sha not in {hashlib.sha256(content).hexdigest(),hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()}:
                raise ValueError('8F model/vocoder/ASR/dependency implementation differs: '+relative)
    ctx=plan['source'];selected=core.indexed(ctx['samples'])
    scored=core.indexed(rows)
    if (len(selected)!=SAMPLES or selected.keys()!=scored.keys() or len(set(plan['ids']))!=SAMPLES
            or set(plan['ids'])!={r['pair_id'] for r in selected.values()}):
        raise ValueError('8F requires the identical complete Natural 8E 300-sample ID set')
    if (metrics.get('purpose')!=core.PURPOSE or metrics.get('condition')!='natural'
            or metrics.get('sample_count')!=SAMPLES or metrics.get('checkpoint_sha256')!=plan['checkpoint_sha256']
            or metrics.get('best_update')!=plan['best_update'] or metrics.get('training_updates')!=plan['run']['updates']
            or metrics.get('beam')!=plan['beam'] or metrics.get('vocoder')!=plan['vocoder']
            or metrics.get('evaluation_asr')!=plan['asr'] or metrics.get('normalization')!=plan['normalization']
            or metrics.get('cascade_8c_identity')!=ctx['identity']):
        raise ValueError('8F Natural report differs from its checkpoint/condition/source identity')
    number(metrics.get('bleu'),'Natural S2UT BLEU')
    for key,item in selected.items():
        row=scored[key]
        if any(row.get(k)!=item.get(k) for k in core.SAMPLE_FIELDS+('pair_id','source_sha256','source_audio')):
            raise ValueError('8F Natural per-sample ID/reference/audio metadata differs: '+key)
    if asr_audit!=ctx['asr_audit']:
        raise ValueError('8F Natural actual ASR dtype/decoding configuration differs from 8C')
    core.verify_asr_runtime(plan,runtime)
    core.verify_saved_asr_dependencies(ctx)


@operation('8F: select and verify completed Natural 8E without recalculation')
def natural_context(persistent, requested=''):
    base=Path(persistent).absolute()/'diagnostics/flores-fleurs-s2ut'
    if requested:
        root=Path(requested) if Path(requested).is_absolute() else base/requested
        core.inside(base,root)
        candidates=[root]
    else:
        candidates=[p.parent for p in base.glob('*/completion.json')]
    matches=[]
    for root in track(sorted(candidates),'8F: inspect completed Natural evaluations'):
        plan=read(root/'plan.json')
        if not requested and plan.get('run',{}).get('name')!=MODEL['name']:
            continue
        checked_completion(root,expected_identity=plan['identity'])
        if (root.parent!=base or root.name!=plan['identity'][:24] or Path(plan['output'])!=root
                or Path(plan['persistent_base'])!=Path(persistent).absolute()):
            raise ValueError('8F Natural evaluation owned path differs')
        metrics=read(root/'metrics.json');runtime=read(root/'evaluation-runtime.json')
        audit=read(root/'evaluation-asr-audit.json');rows=list(read_jsonl(root/'per_sample.jsonl'))
        validate_natural_documents(plan,metrics,rows,runtime,audit)
        files={str(root/name):raw_sha(root/name) for name in
               ('plan.json','completion.json',*RESULT_FILES,'evaluation-asr-audit.json')}
        if 'execution_overlay' in metrics:
            overlay=metrics['execution_overlay']
            code_release(overlay['execution_files'],overlay['execution_revision'])
            if (overlay.get('base_identity')!=plan['identity'] or overlay.get('scientific_settings_unchanged') is not True
                    or overlay.get('base_execution_files_sha256')!=digest(plan['execution_files'])
                    or overlay.get('base_revision')!=plan['execution_revision']
                    or metrics.get('code_revision')!=overlay['execution_revision']):
                raise ValueError('8F Natural reviewed execution overlay differs')
        elif metrics.get('code_revision')!=plan['execution_revision']:
            raise ValueError('8F Natural report execution revision differs from its plan')
        matches.append(dict(root=str(root),plan=plan,metrics=metrics,files=files,runtime=runtime,asr_audit=audit))
    if not matches:
        raise ValueError('完了済みの対応する8E Natural結果がありません。同じ実験設定と保存先を確認してください')
    if len({v['plan']['source']['identity'] for v in matches})>1 and not requested:
        raise ValueError('複数の8Cに対応するNatural評価があります。8FのNATURAL_IDを指定してください')
    return max(matches,key=lambda v:Path(v['root'],'completion.json').stat().st_mtime_ns)


def verify_wav(path, saved):
    if raw_sha(path)!=saved['sha256']:
        raise ValueError('8C Synthetic source WAV SHA256 mismatch: '+str(path))
    with wave.open(str(path),'rb') as stream:
        if (stream.getnchannels()!=1 or stream.getsampwidth()!=2 or stream.getframerate()!=16000
                or stream.getcomptype()!='NONE' or stream.getnframes()/16000!=saved['duration']):
            raise ValueError('8C Synthetic WAV is not the saved mono PCM16/16kHz waveform')
        if not any(stream.readframes(stream.getnframes())):
            raise ValueError('8C Synthetic WAV is silent')


@operation('8F: bind saved Synthetic WAVs to the identical 8C and Natural 8E samples')
def source_context(natural, *, verify_audio=True):
    previous=natural['plan']['source'];root=Path(previous['root'])
    files={}
    for name in ('identity.json','dataset-audit.json','metrics.json','per_sample.jsonl','execution-audit/evaluation_asr.json'):
        path=core.inside(root,root/name);sha=raw_sha(path)
        if sha!=previous['files'].get(str(path)):
            raise ValueError('8C evidence changed since the completed Natural evaluation: '+name)
        files[str(path)]=sha
    plan=read(root/'identity.json');audit=read(root/'dataset-audit.json');report=read(root/'metrics.json')
    if (digest(plan)!=previous['identity'] or root.name!=previous['identity'][:24]
            or plan.get('purpose')!=core.CASCADE_PURPOSE or plan['dataset']!=audit
            or report.get('identity_sha256')!=previous['identity'] or report.get('purpose')!=core.CASCADE_PURPOSE):
        raise ValueError('8F saved 8C identity/audit/report mismatch')
    group_code=Path(__file__).resolve().parents[1]/'cascade/flores_fleurs_diagnostic.py'
    content=group_code.read_bytes()
    if plan.get('implementation',{}).get('cascade/flores_fleurs_diagnostic.py') not in {
            hashlib.sha256(content).hexdigest(),hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()}:
        raise ValueError('8F saved 8C Synthetic group implementation differs; do not guess source provenance')
    selected=core.indexed(plan['selected']);audited=core.indexed(audit['selected_samples'])
    rows=core.indexed(list(read_jsonl(root/'per_sample.jsonl')));embedded=core.indexed(report['per_sample'])
    old=core.indexed(previous['samples'])
    if not (selected.keys()==audited.keys()==rows.keys()==embedded.keys()==old.keys()
            and len(selected)==SAMPLES==report['metrics']['samples']==audit['selected']):
        raise ValueError('8F missing/duplicate/extra sample in 8C or Natural 8E')
    source_proof=plan['source_proof'];batch=source_proof['receipt']['effective_tts']['batch_size']
    if type(batch) is not int or batch<1:
        raise ValueError('8C Synthetic source batch provenance missing')
    synthetic={}
    for group in track(source_groups(plan['selected'],batch),'8F: verify saved Synthetic source group receipts'):
        group_id=digest([s['sample_id'] for s in group])
        pointer=root/'stage-results'/(digest([group_id,'synthetic','input_audio_group'])+'.json')
        document=read(pointer);body=document['body']
        expected=dict(run=previous['identity'],sample_id=group_id,condition='synthetic',stage='input_audio_group',
                      inputs=dict(samples=group,source=source_proof,batch_size=batch))
        if document.get('sha256')!=digest(body) or body.get('request')!=expected:
            raise ValueError('8C Synthetic group receipt SHA/request mismatch')
        items=core.indexed(body.get('items',[]))
        if items.keys()!={s['sample_id'] for s in group}:
            raise ValueError('8C Synthetic group has missing/duplicate/extra sample IDs')
        files[str(pointer)]=raw_sha(pointer)
        synthetic.update(items)
    samples=[]
    for key in track(sorted(selected),'8F: verify/reuse saved Synthetic WAV evidence'):
        item,row=selected[key],rows[key]
        if item!=audited[key] or row!=embedded[key] or any(row.get(k)!=item.get(k) or old[key].get(k)!=item.get(k)
                for k in core.SAMPLE_FIELDS):
            raise ValueError('8F Japanese text/English reference/canonical ID differs from 8C or Natural 8E')
        if (core.sha_text(item['ja_text'])!=item['ja_text_sha256']
                or core.sha_text(item['en_reference'])!=item['en_text_sha256']):
            raise ValueError('8F source/reference text SHA differs')
        saved=synthetic[key]
        if saved.get('status')!='success' or len(saved.get('audio',[]))!=1:
            raise ValueError('8C Synthetic source is missing/failed; restore its saved audio, do not regenerate/select replacements')
        wav=saved['audio'][0];path=core.inside(root/'synthetic-source',Path(wav['path']))
        if (row.get('synthetic_input_audio_status')!='success' or wav.get('sample_rate')!=16000
                or row.get('synthetic_input_audio_sha256')!=wav['sha256']
                or wav.get('mono') is not True or wav.get('finite') is not True or wav.get('non_silent') is not True
                or any(row.get(k)!=str(path) for k in ('synthetic_input_audio_path','synthetic_audio_path','synthetic_source_audio'))
                or str(path)==old[key]['source_audio'] or wav['sha256']==old[key]['source_sha256']):
            raise ValueError('8F Synthetic WAV metadata differs from 8C or incorrectly selects Natural audio')
        if verify_audio:verify_wav(path,wav)
        samples.append(dict(old[key],source_audio=str(path),source_sha256=wav['sha256'],
                            source_duration=wav['duration'],source_sample_rate=16000))
    result=copy.deepcopy(previous)
    result.update(samples=samples,files={**files,**natural['files']},condition='synthetic',
                  cascade_synthetic_bleu=number(report['metrics'].get('synthetic_final_ASR_BLEU'),'saved Cascade Synthetic BLEU'),
                  selection='exact_saved_8c_synthetic_set_bound_to_completed_natural_8e')
    if number(report['metrics'].get('natural_final_ASR_BLEU'),'saved Cascade Natural BLEU')!=previous['cascade_natural_bleu']:
        raise ValueError('8C Natural baseline differs from Natural 8E')
    return result


def implementation_files():
    package=Path(__file__).resolve().parents[1];repo=package.parents[1]
    extra=[Path(__file__),repo/'scripts/colab/s2ut_flores_fleurs_synthetic.py',
           repo/'scripts/colab/cascade_diagnostic_runtime.py',package/'cascade/flores_fleurs_diagnostic.py']
    return dict(core.implementation_files(),**{str(p):raw_sha(p) for p in extra})


def check_source_difference(ctx, previous):
    for key in ('root','identity','dataset_revisions','asr','asr_audit','environment','normalization','cascade_natural_bleu'):
        if ctx.get(key)!=previous.get(key):raise ValueError('8F changed a non-audio source condition: '+key)
    before=core.indexed(previous['samples']);after=core.indexed(ctx['samples'])
    if ctx.get('condition')!='synthetic' or before.keys()!=after.keys() or len(after)!=SAMPLES:
        raise ValueError('8F Natural/Synthetic source condition or ID set differs')
    audio_fields={'source_audio','source_sha256','source_duration'}
    for key,row in after.items():
        if {k:v for k,v in row.items() if k not in audio_fields}!={k:v for k,v in before[key].items() if k not in audio_fields}:
            raise ValueError('8F changed a non-audio sample field/reference: '+key)
        core.inside(Path(ctx['root'])/'synthetic-source',Path(row['source_audio']))
        if row['source_sha256']==before[key]['source_sha256']:
            raise ValueError('8F incorrectly selected the Natural source waveform')


def prepare_plan(persistent,local,natural,ctx,run,metadata,checkpoint,revision):
    old=natural['plan']
    check_model(old)
    check_source_difference(ctx,old['source'])
    if (run['name']!=old['run']['name'] or run['updates']!=old['run']['updates']
            or raw_sha(checkpoint)!=old['checkpoint_sha256'] or metadata!=old['metadata']
            or run.get('environment_sha256')!=old['run'].get('environment_sha256')
            or run.get('identity')!=old['run'].get('identity') or run.get('config')!=old['run'].get('config')):
        raise ValueError('8F restored model/environment/feature metadata differs from completed Natural 8E')
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('8F requires immutable code revision')
    body=dict(version=VERSION,purpose=PURPOSE,condition='synthetic',source=ctx,run=run,metadata=metadata,
        workspace_base=str(Path(local).absolute()),persistent_base=str(Path(persistent).absolute()),
        checkpoint=str(checkpoint),checkpoint_sha256=old['checkpoint_sha256'],best_update=old['best_update'],
        execution_revision=revision,execution_files=implementation_files(),
        ids=old['ids'],natural_evaluation_identity=old['identity'],natural_evaluation_root=natural['root'],
        natural_s2ut_bleu=natural['metrics']['bleu'],natural_asr_runtime=natural['runtime'],
        **{k:copy.deepcopy(old[k]) for k in ('beam','vocoder','asr','normalization','bleu_denominator')})
    identity=digest(body);owned=Path(local).absolute()/identity[:24]
    output=Path(persistent).absolute()/'diagnostics/flores-fleurs-s2ut-synthetic'/identity[:24]
    if owned.resolve().is_relative_to(Path(persistent).resolve()):
        raise ValueError('8F local workspace must remain outside persistent experiments')
    guarded(owned);guarded(output)
    plan=dict(body,identity=identity,local=str(owned),output=str(output))
    atomic_write_json(output/'plan.json',plan,resume=True)
    return plan


def verify_frozen_plan(plan):
    body={k:v for k,v in plan.items() if k not in ('identity','local','output')}
    root=Path(plan['output'])
    if (digest(body)!=plan['identity'] or plan.get('purpose')!=PURPOSE or plan.get('condition')!='synthetic'
            or plan.get('version')!=VERSION or root!=Path(plan['persistent_base'])/'diagnostics/flores-fleurs-s2ut-synthetic'/plan['identity'][:24]
            or Path(plan['local'])!=Path(plan['workspace_base'])/plan['identity'][:24]
            or Path(plan['local']).resolve().is_relative_to(Path(plan['persistent_base']).resolve())):
        raise ValueError('8F Synthetic plan identity/condition/owned path differs')
    core.inside(root,root);check_model(plan)
    if plan['execution_files']!=implementation_files():
        raise ValueError('8F execution implementation differs')
    old=read(Path(plan['natural_evaluation_root'])/'plan.json')
    if old['identity']!=plan['natural_evaluation_identity'] or old['ids']!=plan['ids']:
        raise ValueError('8F Natural/Synthetic sample IDs differ')
    check_source_difference(plan['source'],old['source'])
    if (plan['metadata']!=old['metadata'] or plan['run'].get('identity')!=old['run'].get('identity')
            or plan['run'].get('config')!=old['run'].get('config')
            or plan['run'].get('environment_sha256')!=old['run'].get('environment_sha256')):
        raise ValueError('8F model environment/feature metadata differs from Natural 8E')
    for key in ('checkpoint_sha256','best_update','beam','vocoder','asr','normalization','bleu_denominator'):
        if old[key]!=plan[key]:raise ValueError('8F changed a non-source condition: '+key)


def verify_plan(plan):
    verify_frozen_plan(plan)
    core.verify_source(plan['source'])
    for path,sha in track(plan['metadata'].items(),'8F: verify bound feature metadata'):
        if raw_sha(path)!=sha:raise ValueError('8F model metadata changed: '+path)
    if raw_sha(plan['checkpoint'])!=plan['checkpoint_sha256']:
        raise ValueError('8F checkpoint SHA differs')


@operation('8F: read completed comparison without source WAVs or checkpoint restoration')
def completed_result(persistent,natural,revision):
    base=Path(persistent)/'diagnostics/flores-fleurs-s2ut-synthetic'
    files=implementation_files()
    matches=[]
    for marker in base.glob('*/completion.json'):
        plan=read(marker.parent/'plan.json')
        if (plan.get('natural_evaluation_identity')==natural['plan']['identity']
                and plan.get('execution_revision')==revision and plan.get('execution_files')==files):
            matches.append((marker.stat().st_mtime_ns,marker.parent,plan))
    if not matches:return None
    _,root,plan=max(matches,key=lambda v:v[0])
    verify_frozen_plan(plan)
    if plan['natural_s2ut_bleu']!=natural['metrics']['bleu']:
        raise ValueError('8F completed comparison has another Natural baseline')
    evidence=dict(natural['files'])
    for name in ('identity.json','dataset-audit.json','metrics.json','per_sample.jsonl','execution-audit/evaluation_asr.json'):
        path=str(Path(natural['plan']['source']['root'])/name)
        evidence[path]=natural['plan']['source']['files'][path]
    for path,sha in track(evidence.items(),'8F: verify frozen comparison evidence'):
        if plan['source']['files'].get(path)!=sha or raw_sha(path)!=sha:
            raise ValueError('8F completed comparison evidence changed: '+path)
    checked_completion(root,expected_identity=plan['identity'])
    audit=read(root/'external-input-audit.json')
    if (audit.get('plan')!=plan['identity'] or audit.get('references_in_generation_input') is not False
            or audit.get('inputs')!={r['pair_id']:r['source_sha256'] for r in plan['source']['samples']}):
        raise ValueError('8F completed source-input audit differs')
    metrics=read(root/'metrics.json')
    if (metrics.get('purpose')!=PURPOSE or metrics.get('condition')!='synthetic'
            or metrics.get('natural_8e_identity')!=natural['plan']['identity']
            or metrics.get('checkpoint_sha256')!=plan['checkpoint_sha256']):
        raise ValueError('8F completed metrics identity differs')
    return dict(output_root=str(root),metrics=metrics,display=(root/'report.md').read_text(encoding='utf-8'))


def original_comparison(persistent,plan):
    """Optional original test row: verify both formal journals and recompute text BLEU only."""
    from ..evaluation.bleu import corpus_bleu,normalize_english
    try:
        values={};files={};common=None
        for system in ('cascade','s2ut'):
            base=Path(persistent)/'benchmarks'/system
            pointer=read(base/'latest.json');path=core.inside(base,Path(pointer['plan']))
            if raw_sha(path)!=pointer['sha256']:raise ValueError('original latest/plan SHA differs')
            saved=read(path);body={k:v for k,v in saved.items() if k not in ('identity','local','output')}
            if (digest(body)!=saved['identity'] or saved['split']!='test' or saved.get('system_id','s2ut')!=system
                    or saved['asr']!=plan['asr'] or saved['bleu_denominator']!=plan['bleu_denominator']):
                raise ValueError('original test identity/ASR/failure policy differs')
            if system=='s2ut' and (saved['checkpoint_sha256']!=plan['checkpoint_sha256']
                    or saved.get('beam')!=plan['beam'] or saved.get('vocoder')!='official_hubert100_ljspeech'):
                raise ValueError('original S2UT checkpoint/beam/official vocoder differs')
            root=path.parent;metric_path=root/'results/metrics.json';metrics=read(metric_path)
            score_path=root/'results/per_sample.jsonl';scores=core.indexed(list(read_jsonl(score_path)),'pair_id')
            manifest=Path(saved['common_root'])/'test.jsonl'
            if raw_sha(manifest)!=saved['common_sha256']:raise ValueError('original common manifest differs')
            refs=core.indexed(list(read_jsonl(manifest)),'pair_id')
            mapping={k:r['en_text'] for k,r in refs.items()}
            if common is not None and common!=mapping:raise ValueError('original test ID/reference sets differ')
            common=mapping
            identity=metrics['evaluation_identity'];receipt_path=root/'progress'/('asr-'+digest(identity)[:24]+'.json')
            receipt=read(receipt_path);b=receipt['body']
            if (receipt['sha256']!=digest(b) or b['identity']!=identity or identity['plan']!=saved['identity']
                    or identity['asr']!=plan['asr'] or identity['predictions']!=raw_sha(root/'predictions.jsonl')
                    or b['files']['per_sample.jsonl']['sha256']!=raw_sha(score_path)
                    or scores.keys()!=refs.keys() or metrics.get('bleu_denominator')!=plan['bleu_denominator']
                    or identity.get('normalization')!=plan['normalization'] or metrics.get('samples')!=len(refs)
                    or metrics.get('system_id')!=system or metrics.get('run_id')!=saved['run']['name']):
                raise ValueError('original formal score journal differs')
            keys=sorted(refs)
            if any(scores[k]['reference_raw']!=refs[k]['en_text'] for k in keys):
                raise ValueError('original formal English references differ')
            bleu=corpus_bleu([normalize_english(scores[k]['hypothesis_raw']) if scores[k]['evaluation_status']=='success' else '' for k in keys],
                             [normalize_english(refs[k]['en_text']) for k in keys])
            if not math.isclose(bleu,number(metrics.get('bleu'),'original BLEU'),abs_tol=1e-8,rel_tol=0):
                raise ValueError('original saved BLEU does not match its complete journal')
            values[system]=metrics['bleu']
            files.update({str(p):raw_sha(p) for p in (base/'latest.json',path,metric_path,score_path,manifest,receipt_path,root/'predictions.jsonl')})
        return dict(included=True,scores=values,files=files,samples=len(common))
    except (OSError,ValueError,KeyError,TypeError) as error:
        return dict(included=False,reason=f'{type(error).__name__}: {error}')


@operation('8F: save source-condition comparison without rewriting Natural or Cascade results')
def save_report(plan,metrics,scores):
    expected=Path(plan['persistent_base'])/'diagnostics/flores-fleurs-s2ut-synthetic'/plan['identity'][:24]
    if Path(plan['output'])!=expected:
        raise ValueError('8F reports must remain in their own Synthetic namespace')
    core.inside(expected,expected)
    rows=core.report_rows(plan,scores);output=Path(plan['output'])
    for row in rows:
        row.update(source_audio_sha256=row['source_sha256'],generated_audio_sha256=row['output_sha256'])
    natural=plan['natural_s2ut_bleu'];synthetic=metrics['bleu']
    cascade_natural=plan['source']['cascade_natural_bleu'];cascade_synthetic=plan['source']['cascade_synthetic_bleu']
    delta=synthetic-natural;cascade_delta=cascade_synthetic-cascade_natural
    ratio=synthetic/natural if natural!=0 else None
    original=original_comparison(plan['persistent_base'],plan)
    metrics.update(purpose=PURPOSE,condition='synthetic',sample_count=len(rows),samples=len(rows),
        s2ut_natural_bleu=natural,s2ut_synthetic_bleu=synthetic,s2ut_synthetic_minus_natural=delta,
        synthetic_minus_natural_s2ut=delta,s2ut_synthetic_over_natural=ratio,
        cascade_natural_bleu=cascade_natural,cascade_synthetic_bleu=cascade_synthetic,
        cascade_synthetic_minus_natural=cascade_delta,generation_failures=sum(r['generation_status']!='success' for r in rows),
        vocoder_failures=sum(r['generation_status']=='success' and r['vocoder_status']!='success' for r in rows),
        evaluation_asr_failures=metrics['asr_failures'],checkpoint_sha256=plan['checkpoint_sha256'],
        best_update=plan['best_update'],training_updates=plan['run']['updates'],beam=plan['beam'],vocoder=plan['vocoder'],
        evaluation_asr=plan['asr'],evaluation_asr_audit=plan['source']['asr_audit'],normalization=plan['normalization'],
        natural_8e_identity=plan['natural_evaluation_identity'],natural_8e_root=plan['natural_evaluation_root'],
        source_artifact_identity=plan['source']['identity'],cascade_8c_identity=plan['source']['identity'],
        cascade_8c_root=plan['source']['root'],code_revision=plan['execution_revision'],
        source_audio_sha256={r['sample_id']:r['source_sha256'] for r in plan['source']['samples']},
        sample_set_sha256=digest([{k:r[k] for k in core.SAMPLE_FIELDS} for r in plan['source']['samples']]),
        original_comparison=original)
    lines=['# FLEURS/FLORES S2UT source-condition comparison','',f'Samples: {len(rows)}','',
           '| Evaluation | Cascade ASR-BLEU | S2UT ASR-BLEU |','|---|---:|---:|']
    if original['included']:
        lines.append(f"| Original KFTT/JESC test | {original['scores']['cascade']:.4f} | {original['scores']['s2ut']:.4f} |")
    lines += [f'| FLEURS/FLORES Natural | {cascade_natural:.4f} | {natural:.4f} |',
              f'| FLEURS/FLORES Synthetic | {cascade_synthetic:.4f} | {synthetic:.4f} |','',
              f'S2UT Synthetic - Natural: {delta:+.4f} BLEU',f'Cascade Synthetic - Natural: {cascade_delta:+.4f} BLEU',
              'S2UT Synthetic / Natural: '+(f'{ratio:.4f}' if ratio is not None else 'undefined (Natural BLEU is zero)'),'',
              'S2UT checkpoint: `'+plan['checkpoint_sha256']+'`',f"Dev-best update: {plan['best_update']}; training reached: {plan['run']['updates']}; beam: {plan['beam']}",'',
              f"Failures: generation={metrics['generation_failures']}; vocoder={metrics['vocoder_failures']}; ASR={metrics['evaluation_asr_failures']}",
              'All samples remain in the BLEU denominator; failures are empty hypotheses.','',
              'Natural and Synthetic use identical sample IDs, references, checkpoint, decoding configuration, vocoder and evaluation ASR. Only the Japanese source audio condition differs.',
              '8C Synthetic WAVs are reused byte-for-byte; no source TTS generation or new sample selection.','',
              'This difference is a diagnostic clue and does not by itself prove a sole cause.',
              '8C identity: `'+plan['source']['identity']+'`; Natural 8E identity: `'+plan['natural_evaluation_identity']+'`.',
              'Code: `'+plan['execution_revision']+'`. WAV ZIPs: `audio/`; index: `audio-index.json`.']
    if not original['included']:lines+=['','Original test row omitted: '+original['reason']]
    text='\n'.join(lines)+'\n'
    atomic_write_json(output/'metrics.json',metrics,overwrite=True)
    atomic_write_jsonl(output/'per_sample.jsonl',rows,overwrite=True)
    stream=io.StringIO(newline='')
    fields=['sample_id','flores_id','fleurs_id','ja_text','en_reference','output_asr_en_text','sentence_bleu','status','error',
            'source_audio_sha256','generated_audio_sha256','units_sha256','audio_archive','audio_member','audio_archive_sha256']
    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    atomic_write_text(output/'per_sample.csv',stream.getvalue(),overwrite=True)
    atomic_write_text(output/'report.md',text,overwrite=True)
    return dict(output_root=str(output),metrics=metrics,display=text)
