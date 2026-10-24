"""Synthetic JA listening comparison; Cascade MT is a pseudo-reference, NOT truth."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import uuid

from .benchmark_storage import guarded, read
from .drive_staging import copy_verified, raw_sha
from .io import atomic_write_json
from .journal import digest
from .progress import operation

PURPOSE = 'synthetic_ja_cascade_pseudo_reference_not_benchmark'
PHASES = ('source', 'cascade-asr', 'cascade-mt', 'candidate-model', 'cascade-audio', 'candidate-audio', 'evaluate')


def validate_request(text, model):
    if model not in ('tt2', 's2ut', 's2t_tts'):
        raise ValueError('comparison candidate must be tt2, s2ut or s2t_tts')
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500 or any(ord(c) < 32 for c in text):
        raise ValueError('enter one nonempty Japanese sentence, at most 500 characters, without control characters')
    return text.strip()


def read_stage(plan, phase):
    path = Path(plan['output'])/(phase+'.json')
    if not path.is_file():
        return None
    document = read(path)
    body = document['body']
    if document['sha256'] != digest(body) or body['plan'] != plan['identity'] or body['phase'] != phase:
        raise ValueError('interactive stage identity/checksum changed; original retained')
    for name, sha in body['files'].items():
        path = Path(plan['output'])/name
        if not path.resolve().is_relative_to(Path(plan['output']).resolve()) or raw_sha(path) != sha:
            raise ValueError('interactive saved artifact changed; no silent regeneration')
    return body


def artifact(plan, phase, key):
    body = read_stage(plan, phase)
    if not body or body['status'] != 'success':
        raise ValueError('upstream '+phase+' failed; no model invoked')
    name = body['artifacts'][key]
    source, target = Path(plan['output'])/name, Path(plan['local'])/name
    if not target.is_file() or raw_sha(target) != body['files'][name]:
        copy_verified(source, target, body['files'][name])
    return target


def data(plan, phase):
    body = read_stage(plan, phase)
    if not body or body['status'] != 'success':
        raise ValueError('upstream '+phase+' failed; no model invoked')
    return body['data']


def save_stage(plan, phase, key, values, files, error=None):
    from .cascade.benchmark_worker import checked_wav
    for source in files.values():
        if Path(source).suffix == '.wav':
            checked_wav(Path(source))  # Check PCM readback too, not only pre-conversion floats.
    token = uuid.uuid4().hex
    output = guarded(plan['output'])
    checksums, names = {}, {}
    for label, source in files.items():
        name = f'artifacts/{phase}/{token}/{Path(source).name}'
        sha = raw_sha(source)
        copy_verified(source, output/name, sha)
        names[label], checksums[name] = name, sha
    body = dict(plan=plan['identity'], phase=phase, inputs=key, data=values, artifacts=names, files=checksums,
                status='failed' if error else 'success', error=error)
    document = dict(body=body, sha256=digest(body))
    atomic_write_json(output/'history'/phase/(token+'.json'), document)
    atomic_write_json(output/(phase+'.json'), document, overwrite=True)
    return body


def dependencies(phase):
    return {'source': (), 'cascade-asr': ('source',), 'cascade-mt': ('cascade-asr',),
            'candidate-model': ('source',), 'cascade-audio': ('cascade-mt',),
            'candidate-audio': ('candidate-model',),
            'evaluate': ('source', 'cascade-mt', 'cascade-audio', 'candidate-audio')}[phase]


def release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def voice(config):
    from .cascade.tts import QwenTTS
    return QwenTTS(model_id=config['model'], revision=config['revision'], speaker=config['speaker'],
                   language=config['language'], generation=config['generation'])


def timed(load, infer):
    from .inference_performance import Measurement
    with Measurement() as timing:
        model = load()
        timing.loaded()
        values = infer(model)
    del model
    release()
    return values, timing.result


@operation('interactive: pinned S2UT source-only inference')
def s2ut(plan, source):
    """Use the pinned evaluation frontend and beam search without any targets."""
    import torch
    import soundfile as sf
    from fairseq import checkpoint_utils, utils
    from fairseq.dataclass.configs import GenerationConfig
    from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
    from omegaconf import OmegaConf, open_dict
    run = plan['run']
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [run['checkpoint']], arg_overrides=dict(data=run['data_root'], config_yaml='config.yaml',
                                               multitask_config_yaml='config_multitask.yaml'), strict=True)
    if len(models) != 1 or not task.args.target_is_code or task.args.target_code_size != 100 or task.args.n_frames_per_step != 1:
        raise ValueError('interactive S2UT requires the existing KM100 reduced-unit model')
    generation = OmegaConf.structured(GenerationConfig(beam=10, max_len_a=1.0))
    with open_dict(cfg):
        cfg.generation = generation
    model = models[0].float().cuda().eval()  # Same FP32 inference as existing generation_command (no --fp16).
    model.prepare_for_inference_(cfg)
    generator = task.build_generator(models, generation)
    def generate(_):
        dataset = SpeechToTextDataset('test', False, task.data_cfg, [str(source)], [max(1, round(sf.info(source).duration*100))], ids=['interactive'])
        sample = dataset.collater([dataset[0]])
        if sample.get('target') is not None or sample['net_input'].get('prev_output_tokens') is not None:
            raise ValueError('interactive source-only dataset unexpectedly contains a target')
        sample = utils.move_to_cuda(sample)
        if sample['net_input']['src_lengths'][0] > task.max_positions()[0]:
            raise ValueError('interactive source exceeds the saved S2UT source-position limit; no truncation')
        with torch.inference_mode():
            best = task.inference_step(generator, models, sample)[0][0]['tokens']
        if not len(best) or int(best[-1]) != task.target_dictionary.eos():
            raise ValueError('S2UT output missing final EOS')
        from .s2ut.reduce_units import validate_units
        return dict(units=validate_units([int(task.target_dictionary[int(t)]) for t in best[:-1]], clusters=100),
                    beam=10, max_len_a=1.0, precision='float32', reference_used=False)
    return generate, (models, cfg, task, generator)


def tt2_load(plan):
    from .translatotron2.engine import load_checkpoint
    from .translatotron2.postnet_calibration import apply_checkpoint
    model, state = load_checkpoint(Path(plan['run']['checkpoint']), device='cuda')
    apply_checkpoint(model, state)
    model.eval()
    return model, state


def tt2_infer(plan, pair, source, directory):
    import torch
    import numpy as np
    from .translatotron2.data import source_features
    model, state = pair
    root = Path(plan['run']['data_root'])
    spec = read(root/'source-mel-spec.json') if (root/'source-mel-spec.json').is_file() else read(root/'mel-spec.json')
    spec = state['data_fingerprint'].get('training', {}).get('source_feature_config', spec)
    if state['data_fingerprint'].get('mel-spec.json') != raw_sha(root/'mel-spec.json'):
        raise ValueError('TT2 checkpoint Mel specification mismatch')
    if (root/'source-mel-spec.json').is_file() and state['data_fingerprint'].get('source-mel-spec.json') != raw_sha(root/'source-mel-spec.json'):
        raise ValueError('TT2 checkpoint source frontend mismatch')
    with torch.inference_mode():
        feature = source_features(source, spec).cuda()
        result = model.generate(feature[None], torch.tensor([len(feature)], device='cuda'),
                                max_phones=400, max_frames=3000, beam_size=1, length_penalty=0.0)
    values = result['post_mel'][0, :int(result['frame_lengths'][0])].float().cpu().numpy()
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('invalid TT2 generated Mel')
    path = directory/'predicted.npy'
    np.save(path, values, allow_pickle=False)
    return path


def unit_audio(units, path, directory):
    import math
    import torch
    from .vocoders.reference import ensure_reference
    from .vocoders.inference import validate_config, write_waveform
    from fairseq.models.text_to_speech.codehifigan import CodeGenerator
    checkpoint, config_path = directory/'g_00500000', directory/'config.json'
    ensure_reference(checkpoint, config_path)  # Verified download outside measured model loading/inference.
    cfg = read(config_path)
    validate_config('unit', cfg, sample_rate=cfg['sampling_rate'])
    def load():
        model = CodeGenerator(cfg)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['generator'], strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite official Unit vocoder')
        return model.cuda().eval()
    def infer(model):
        with torch.inference_mode():
            code = torch.tensor([units], device='cuda', dtype=torch.long)
            lengths = (model.dur_predictor(model.dict(code)).exp()-1).round().clamp_min(1)
            if not torch.isfinite(lengths).all() or lengths.sum()*math.prod(cfg['upsample_rates']) > cfg['sampling_rate']*120:
                raise ValueError('Unit vocoder duration exceeds safety limit')
            wave = model(code=code, dur_prediction=True).float().cpu().numpy()
        return write_waveform(path, wave, cfg['sampling_rate'])
    return timed(load, infer)


def perform(plan, phase, directory):
    """Phase interfaces deliberately never pass Cascade text to the candidate."""
    if phase == 'source':
        import soundfile as sf
        import numpy as np
        import scipy.signal
        config = dict(plan['cascade']['tts'], language='Japanese')
        raw = directory/'japanese-24k.wav'
        model = voice(config)
        model(plan['text'], raw)
        diagnostic = dict(model.last_generation)
        wave, rate = sf.read(raw)
        if rate != 24000 or wave.ndim != 1:
            raise ValueError('invalid Qwen Japanese source')
        normalized = scipy.signal.resample_poly(wave, 2, 3).astype(np.float32)
        path = directory/'japanese-16k.wav'
        sf.write(path, normalized, 16000, subtype='PCM_16')
        return dict(language='Japanese', tts_generation=diagnostic, duration=len(normalized)/16000,
                    resampling='scipy.signal.resample_poly 2/3; PCM16'), dict(audio=path, original_audio=raw)
    if phase == 'cascade-asr':
        from .cascade.asr import WhisperASR
        config = plan['cascade']['asr']
        source = artifact(plan, 'source', 'audio')
        text, performance = timed(lambda: WhisperASR(model_id=config['model'], revision=config['revision']), lambda model: model(source))
        if not text.strip():
            raise ValueError('empty Cascade Japanese ASR')
        return dict(text=text, performance=performance), {}
    if phase == 'cascade-mt':
        from .cascade.mt import NllbTranslator
        config = plan['cascade']['mt']
        text = data(plan, 'cascade-asr')['text']
        result, performance = timed(lambda: NllbTranslator(model_id=config['model'], revision=config['revision'],
                                  source_language=config['source_language'], target_language=config['target_language']),
                                  lambda model: model(text))
        if not result.strip():
            raise ValueError('empty Cascade English MT')
        return dict(text=result, performance=performance, reference_kind='machine_pseudo_reference_not_ground_truth'), {}
    if phase == 'candidate-model':
        source = artifact(plan, 'source', 'audio')
        kind = plan['model']
        if kind == 's2t_tts':
            from .s2t_tts.s2t import WhisperS2T
            config = plan['s2t']['s2t']
            values, performance = timed(lambda: WhisperS2T(model_id=config['model'], revision=config['revision'],
                                       language=config['language'], task=config['task']), lambda model: model(source))
            return dict(text=values, performance=performance, reference_used=False), {}
        if kind == 's2ut':
            values, performance = timed(lambda: s2ut(plan, source), lambda pair: pair[0](None))
            return dict(values, performance=performance), {}
        path, performance = timed(lambda: tt2_load(plan), lambda pair: tt2_infer(plan, pair, source, directory))
        return dict(performance=performance, reference_used=False), dict(mel=path)
    if phase in ('cascade-audio', 'candidate-audio'):
        branch = 'cascade' if phase == 'cascade-audio' else 'candidate'
        path = directory/'english.wav'
        if branch == 'cascade' or plan['model'] == 's2t_tts':
            text = data(plan, 'cascade-mt' if branch == 'cascade' else 'candidate-model')['text']
            def generate(model):
                model(text, path)
                return dict(model.last_generation)
            diagnostic, performance = timed(lambda: voice(plan['cascade']['tts']), generate)
            return dict(tts_generation=diagnostic, performance=performance, vocoder='Qwen CustomVoice'), dict(audio=path)
        if plan['model'] == 's2ut':
            units = data(plan, 'candidate-model')['units']
            _, performance = unit_audio(units, path, Path(plan['local'])/'official-unit')
            return dict(performance=performance, vocoder='official LJSpeech Unit HiFi-GAN / predicted duration'), dict(audio=path)
        from .vocoders.griffin_lim import inverse_basis, reconstruct, validate_spec
        from .vocoders.inference import write_waveform
        import numpy as np
        spec = read(Path(plan['run']['data_root'])/'mel-spec.json')
        validate_spec(spec)
        feature = np.load(artifact(plan, 'candidate-model', 'mel'), allow_pickle=False)
        _, performance = timed(lambda: inverse_basis(spec),
                               lambda basis: write_waveform(path, reconstruct(feature, spec, basis, iterations=32, seed=1), spec['sample_rate']))
        return dict(performance=performance, vocoder='Griffin-Lim', iterations=32, seed=1), dict(audio=path)
    return evaluate(plan), {}


@operation('interactive: English output ASR, excluded from inference timing')
def evaluate(plan):
    from .evaluation.transcribe import EvaluationASR
    from .evaluation.bleu import normalize_english, sentence_bleu
    reference = data(plan, 'cascade-mt')['text']
    saved_evaluation = read_stage(plan, 'evaluate')
    saved_evaluation = saved_evaluation['data'] if saved_evaluation and saved_evaluation['status'] == 'success' else {}
    results, model = {}, None
    for branch in ('cascade', 'candidate'):
        audio_stage = read_stage(plan, branch+'-audio')
        if not audio_stage or audio_stage['status'] != 'success':
            results[branch] = dict(status='failed', error=audio_stage.get('error') if audio_stage else 'no audio', sentence_bleu=None)
            continue
        try:
            saved = saved_evaluation.get('branches', {}).get(branch, {})
            current_sha = audio_stage['files'][audio_stage['artifacts']['audio']]
            if (saved.get('status') == 'success' and saved.get('audio_sha256') == current_sha
                    and saved_evaluation.get('reference') == reference):
                results[branch] = saved
                continue
            if model is None:
                cfg = plan['evaluation']['asr']
                model = EvaluationASR(model_id=cfg['model'], revision=cfg['revision'])
            hypothesis = model(artifact(plan, branch+'-audio', 'audio'))
            results[branch] = dict(status='success', asr_text=hypothesis,
                                   audio_sha256=current_sha,
                                   sentence_bleu=sentence_bleu(normalize_english(hypothesis), normalize_english(reference)))
        except Exception as error:
            results[branch] = dict(status='failed', error=f'{type(error).__name__}: {error}', sentence_bleu=None)
    return dict(reference=reference, reference_source='cascade_mt_pseudo_reference', human_reference=False, branches=results)


@operation('interactive: one resumable comparison phase')
def worker(plan, phase):
    if plan.get('purpose') != PURPOSE or phase not in PHASES:
        raise ValueError('not an interactive comparison plan')
    guarded(plan['output'])
    guarded(plan['local'])
    if raw_sha(Path(plan['environment_path'])) != plan['environment_sha256']:
        raise ValueError('interactive dependency environment changed')
    upstream = {p: read_stage(plan, p) for p in dependencies(phase)}
    key = digest(upstream)
    previous = read_stage(plan, phase)
    evaluation_complete = (phase != 'evaluate' or (previous and all(
        value.get('status') == 'success' for value in previous.get('data', {}).get('branches', {}).values())
        and len(previous.get('data', {}).get('branches', {})) == 2))
    if previous and previous['status'] == 'success' and previous['inputs'] == key and evaluation_complete:
        print(f'[interactive] phase={phase} reused=True original_measurements=True', file=sys.stderr, flush=True)
        return previous
    directory = guarded(Path(plan['local'])/'attempts'/phase/uuid.uuid4().hex)
    try:
        values, files = perform(plan, phase, directory)
        return save_stage(plan, phase, key, values, files)
    except Exception as error:
        import traceback
        traceback.print_exc()
        return save_stage(plan, phase, key, {}, {}, f'{type(error).__name__}: {error}')


def report(plan, reused=()):
    guarded(plan['output'])
    from .inference_performance import branch_performance
    import soundfile as sf
    stages = {p: read_stage(plan, p) for p in PHASES}
    source = stages['source']
    source_seconds = source['data']['duration'] if source and source['status'] == 'success' else None
    result = dict(purpose=PURPOSE, model=plan['model'], text=plan['text'], plan=plan['identity'],
                  human_reference=False, reference_source='cascade_mt_pseudo_reference',
                  note='BLEU is machine-reference agreement, not translation accuracy. '
                       'Single first-call measurement, no warmup/repeats; cached stages show their original timings.',
                  reused_stages=list(reused), audio={}, branches={}, stages=stages)
    for branch in ('cascade', 'candidate'):
        stage = stages[branch+'-audio']
        names = ('cascade-asr', 'cascade-mt', 'cascade-audio') if branch == 'cascade' else ('candidate-model', 'candidate-audio')
        if stage and stage['status'] == 'success':
            name = stage['artifacts']['audio']
            path = Path(plan['output'])/name
            result['audio'][branch] = dict(path=str(path), sha256=stage['files'][name])
            parts = [stages[p]['data'].get('performance') if stages[p] else None for p in names]
            performance = branch_performance(parts, source_seconds, sf.info(path).duration)
            result['branches'][branch] = dict(status='success', performance=performance)
        else:
            result['branches'][branch] = dict(status='failed', performance=None, error=stage.get('error') if stage else 'not generated')
    if source and source['status'] == 'success':
        name = source['artifacts']['audio']
        result['audio']['source'] = dict(path=str(Path(plan['output'])/name), sha256=source['files'][name])
    if stages['evaluate'] and stages['evaluate']['status'] == 'success':
        result['evaluation'] = stages['evaluate']['data']
    performances = [row.get('performance') for row in result['branches'].values()]
    sessions = {s for p in performances if p for s in p['measurement_sessions']}
    hardware = {tuple(h) for p in performances if p for h in p['hardware']}
    result['performance_same_session'] = all(performances) and len(sessions) == 1 and len(hardware) <= 1
    result['run'] = {k: plan['run'][k] for k in ('name', 'updates', 'checkpoint_sha256')} if plan.get('run') else None
    atomic_write_json(Path(plan['output'])/'comparison.json', result, overwrite=True)
    from .io import atomic_write_text
    lines = ['# Synthetic Japanese listening comparison (NOT a benchmark)', '',
             'Reference: Cascade MT pseudo-reference, NOT human ground truth.', '',
             result['note'], '', '| branch | inference seconds | worker acquisition/load seconds | peak process RAM GiB | RAM % | peak Torch VRAM GiB | VRAM % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for branch, row in result['branches'].items():
        perf = row.get('performance')
        if not perf:
            lines.append('| '+branch+' | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |')
        else:
            def value(key, scale=1):
                return 'unavailable' if perf.get(key) is None else f'{perf[key]/scale:.3f}'
            lines.append('| '+branch+' | '+' | '.join((value('inference_seconds'), value('model_load_seconds'),
                         value('process_rss_peak_bytes', 1024**3), value('process_rss_percent_system_ram'),
                         value('torch_allocated_peak_bytes', 1024**3),
                         value('torch_allocated_percent_total_vram')))+' |')
    lines += ['', 'Both branches measured in one session: '+str(result['performance_same_session']),
              'Partial resumes may combine original measurements from different sessions/hardware. '
              'Remeasure for a same-session comparison; preserved old timings are never relabeled as new.',
              'RAM is the worker process RSS (50 ms sampling), not all Colab RAM. VRAM is PyTorch allocator peak, '
              'not nvidia-smi total memory or GPU compute utilization. Sequential stages use max, not sum, of memory peaks.',
              'Inference is first-call, without warmup or repetitions. Acquisition/load may include HF downloads. '
              'Drive training-snapshot restoration is outside the worker and excluded; acquisition times are not a fair cold-download comparison. '
              'Common source synthesis and evaluation ASR are excluded from both branch timings.']
    atomic_write_text(Path(plan['output'])/'report.md', '\n'.join(lines)+'\n', overwrite=True)
    return result


def main():
    from .inference_logging import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=(*PHASES, 'report'), required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if args.phase == 'report':
        report(plan)
        return
    for path, sha in plan.get('metadata', {}).items():
        if plan.get('run') and path == plan['run']['checkpoint'] and args.phase != 'candidate-model':
            continue
        if raw_sha(Path(path)) != sha:
            raise ValueError('interactive inference metadata/checkpoint changed')
    worker(plan, args.phase)


if __name__ == '__main__':
    main()
