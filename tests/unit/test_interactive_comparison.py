import ast
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import soundfile as sf
import torch

from direct_s2st import interactive_comparison as worker
from direct_s2st.inference_performance import Measurement, branch_performance
from direct_s2st.io import atomic_write_json
from direct_s2st.drive_staging import raw_sha
from direct_s2st.journal import digest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
import interactive_session as session


def wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.sin(np.arange(1600)/10)*.1, 16000)
    return path


def fixture(tmp_path):
    environment = tmp_path/'environment.json'
    atomic_write_json(environment, {'packages': 'fixed test environment'})
    plan = dict(purpose=worker.PURPOSE, identity='a'*64, text='今日は晴れです。', model='s2ut',
                local=str(tmp_path/'l'), output=str(tmp_path/'d'), environment_path=str(environment),
                environment_sha256=raw_sha(environment), evaluation={'asr': {'model':'model', 'revision':'b'*40}}, run=None)
    return plan


def performance(seconds=1, ram=10, vram=None):
    return dict(success=True, inference_seconds=seconds, model_load_seconds=.5,
                process_rss_peak_bytes=ram, process_rss_percent_system_ram=10,
                torch_allocated_peak_bytes=vram, torch_reserved_peak_bytes=vram,
                torch_allocated_percent_total_vram=None, gpu_name=None)


def fake_phases(monkeypatch, *, fail_candidate=False):
    calls, source_paths = [], []
    def perform(plan, phase, directory):
        calls.append(phase)
        if phase == 'source':
            return dict(duration=.1), dict(audio=wav(directory/'source.wav'))
        if phase == 'cascade-asr':
            source_paths.append(worker.artifact(plan, 'source', 'audio'))
            return dict(text='認識した日本語', performance=performance()), {}
        if phase == 'cascade-mt':
            return dict(text='the book is new', performance=performance()), {}
        if phase == 'candidate-model':
            source_paths.append(worker.artifact(plan, 'source', 'audio'))
            if fail_candidate:
                raise ValueError('phoneme decoding reached max_phones without EOS')
            return dict(units=[1, 2, 3], reference_used=False, performance=performance()), {}
        if phase.endswith('-audio'):
            worker.data(plan, 'candidate-model' if phase == 'candidate-audio' else 'cascade-mt')
            return dict(performance=performance()), dict(audio=wav(directory/'english.wav'))
        return worker.evaluate(plan), {}
    monkeypatch.setattr(worker, 'perform', perform)
    asr_calls = []
    def asr(**kwargs):
        def recognize(path):
            asr_calls.append(path)
            return 'the book is new'
        return recognize
    monkeypatch.setattr('direct_s2st.evaluation.transcribe.EvaluationASR', asr)
    return calls, source_paths, asr_calls


def test_two_branches_share_source_and_use_pseudo_reference_only_after_generation(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    calls, sources, asr_calls = fake_phases(monkeypatch)
    for phase in worker.PHASES:
        worker.worker(plan, phase)
    result = worker.report(plan)
    assert sources[0] == sources[1] and raw_sha(sources[0]) == raw_sha(sources[1])
    assert len(asr_calls) == 2
    assert result['human_reference'] is False
    assert result['evaluation']['reference_source'] == 'cascade_mt_pseudo_reference'
    assert result['evaluation']['branches']['candidate']['sentence_bleu'] == 100
    assert result['branches']['cascade']['performance']['inference_seconds'] == 3
    assert result['branches']['candidate']['performance']['inference_seconds'] == 2
    assert result['branches']['candidate']['performance']['input_rtf'] == 20
    assert result['branches']['candidate']['performance']['torch_allocated_peak_bytes'] is None
    assert set(result['audio']) == {'source', 'cascade', 'candidate'}
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(plan['output']).rglob('*') if p.is_file()}
    # VM loss: all successful stages restore their small durable artifacts; no recomputation.
    Path(plan['local']).rename(tmp_path/'lost-vm')
    for phase in worker.PHASES:
        worker.worker(plan, phase)
    assert calls == list(worker.PHASES)
    assert all((Path(p).read_bytes(), Path(p).stat().st_mtime_ns) == value for p, value in before.items())
    assert 'NOT human ground truth' in (Path(plan['output'])/'report.md').read_text()


def test_candidate_failure_preserves_cascade_and_never_reports_zero_latency_or_fake_bleu(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    fake_phases(monkeypatch, fail_candidate=True)
    for phase in worker.PHASES:
        worker.worker(plan, phase)
    result = worker.report(plan)
    assert result['branches']['cascade']['status'] == 'success'
    assert result['branches']['candidate']['status'] == 'failed'
    assert result['branches']['candidate']['performance'] is None
    assert result['evaluation']['branches']['candidate']['sentence_bleu'] is None
    assert 'candidate' not in result['audio']
    assert 'max_phones' in result['stages']['candidate-model']['error']


@pytest.mark.parametrize('damage', ['audio', 'identity', 'environment'])
def test_corruption_fails_before_any_model_or_regeneration(tmp_path, monkeypatch, damage):
    plan = fixture(tmp_path)
    fake_phases(monkeypatch)
    worker.worker(plan, 'source')
    if damage == 'audio':
        saved = worker.read_stage(plan, 'source')
        (Path(plan['output'])/saved['artifacts']['audio']).write_bytes(b'broken')
    elif damage == 'identity':
        (Path(plan['output'])/'source.json').write_text('{"body":{},"sha256":"bad"}')
    else:
        Path(plan['environment_path']).write_text('{}')
    monkeypatch.setattr(worker, 'perform', lambda *args: pytest.fail('must not regenerate'))
    with pytest.raises((ValueError, KeyError)):
        worker.worker(plan, 'source')


@pytest.mark.parametrize('text,model', [('', 's2ut'), ('a'*501, 'tt2'), ('bad\ntext', 'tt2'), ('hello', 'cascade'), (None, 's2ut')])
def test_bad_requests_are_rejected_without_work(text, model):
    with pytest.raises(ValueError):
        worker.validate_request(text, model)


def test_memory_is_max_of_sequential_stages_not_sum_and_unavailable_is_not_zero():
    parts = [performance(seconds=2, ram=100, vram=400), performance(seconds=3, ram=200, vram=300)]
    result = branch_performance(parts, 2, 1)
    assert result['inference_seconds'] == 5 and result['model_load_seconds'] == 1
    assert result['process_rss_peak_bytes'] == 200 and result['torch_allocated_peak_bytes'] == 400
    assert result['input_rtf'] == 2.5 and result['output_rtf'] == 5
    assert result['torch_allocated_percent_total_vram'] is None
    assert branch_performance([None], 2, 1) is None
    parts[0]['measurement_session'], parts[1]['measurement_session'] = 'old-vm', 'new-vm'
    assert branch_performance(parts, 2, 1)['mixed_measurement_sessions'] is True


def test_only_failed_output_asr_is_retried_without_regenerating_any_audio(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    calls, _, _ = fake_phases(monkeypatch)
    recognized = []
    def loader(**kwargs):
        def recognize(path):
            recognized.append(path)
            if len(recognized) == 2:
                raise ValueError('transient ASR failure')
            return 'the book is new'
        return recognize
    monkeypatch.setattr('direct_s2st.evaluation.transcribe.EvaluationASR', loader)
    for phase in worker.PHASES:
        worker.worker(plan, phase)
    first = worker.read_stage(plan, 'evaluate')['data']['branches']
    assert first['cascade']['status'] == 'success' and first['candidate']['status'] == 'failed'
    worker.worker(plan, 'evaluate')
    assert len(recognized) == 3 and worker.read_stage(plan, 'evaluate')['data']['branches']['candidate']['status'] == 'success'
    assert calls.count('source') == 1 and calls.count('cascade-audio') == 1 and calls.count('candidate-audio') == 1


def test_silent_pcm_cannot_be_published_as_success(tmp_path):
    plan = fixture(tmp_path)
    path = tmp_path/'silent.wav'
    sf.write(path, np.zeros(100), 16000)
    with pytest.raises(ValueError, match='silent'):
        worker.save_stage(plan, 'source', 'key', {}, dict(audio=path))
    assert not (Path(plan['output'])/'source.json').exists()


def test_worker_and_report_cannot_write_below_corpus(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    corpus = tmp_path/'corpus'
    corpus.mkdir()
    monkeypatch.setenv('CORPUS_ROOT', str(corpus))
    plan['output'] = str(corpus/'derived')
    monkeypatch.setattr(worker, 'perform', lambda *args: pytest.fail('must fail before model invocation'))
    for function in (lambda: worker.worker(plan, 'source'), lambda: worker.report(plan)):
        with pytest.raises(ValueError, match='outside CORPUS_ROOT'):
            function()
    assert not list(corpus.iterdir())


def test_lost_ram_access_is_unavailable_not_a_stale_baseline(monkeypatch):
    count = []
    def memory():
        count.append(1)
        if len(count) > 1:
            raise OSError('memory telemetry unavailable')
        return NS(rss=100)
    monkeypatch.setitem(sys.modules, 'psutil', NS(Process=lambda: NS(memory_info=memory), virtual_memory=lambda: NS(total=1000)))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with Measurement() as measured:
        measured.loaded()
    assert measured.result['process_rss_peak_bytes'] is None
    assert measured.result['process_rss_percent_system_ram'] is None


def test_real_cpu_measurement_reports_rss_without_claiming_vram(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with Measurement() as measurement:
        measurement.loaded()
        torch.arange(100).sum()
    assert measurement.result['success']
    assert measurement.result['process_rss_peak_bytes'] is None or measurement.result['process_rss_peak_bytes'] > 0
    assert measurement.result['inference_seconds'] >= 0
    assert measurement.result['torch_allocated_peak_bytes'] is None
    with pytest.raises(ValueError):
        with Measurement() as failed:
            failed.loaded()
            raise ValueError('failed inference')
    assert not failed.result['success'] and not failed.thread.is_alive()


def test_sampled_ram_percentage_and_cuda_peaks(monkeypatch):
    monkeypatch.setitem(sys.modules, 'psutil', NS(Process=lambda: NS(memory_info=lambda: NS(rss=100)),
                                                virtual_memory=lambda: NS(total=1000)))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    for name in ('synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: NS(total_memory=2000))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'Fake GPU')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 500)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 700)
    with Measurement() as measured:
        measured.loaded()
    assert measured.result['process_rss_percent_system_ram'] == 10
    assert measured.result['torch_allocated_percent_total_vram'] == 25
    assert measured.result['torch_allocated_peak_bytes'] == 500
    assert measured.result['torch_reserved_peak_bytes'] == 700


def test_source_is_japanese_and_resampled_once_for_both_branches(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    plan['cascade'] = dict(tts=dict(language='English', generation={'seed':17}))
    calls = []
    class Voice:
        last_generation = dict(eos_observed=True, seed=17)
        def __call__(self, text, path):
            calls.append(text)
            sf.write(path, np.sin(np.arange(2400)/10)*.1, 24000)
    def load(cfg):
        assert cfg['language'] == 'Japanese'
        return Voice()
    monkeypatch.setattr(worker, 'voice', load)
    directory = tmp_path/'attempt'
    directory.mkdir()
    values, files = worker.perform(plan, 'source', directory)
    assert calls == [plan['text']] and plan['cascade']['tts']['language'] == 'English'
    assert sf.info(files['audio']).samplerate == 16000 and sf.info(files['audio']).frames == 1600
    assert sf.info(files['original_audio']).samplerate == 24000
    assert values['duration'] == .1 and values['tts_generation']['eos_observed']


def test_tt2_inference_uses_saved_frontend_and_no_reference(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    root = tmp_path/'prepared'
    spec = {'sample_rate':16000, 'n_mels':80}
    atomic_write_json(root/'mel-spec.json', spec)
    source_spec = dict(spec, f_min=125)
    fingerprint = {'mel-spec.json': raw_sha(root/'mel-spec.json'), 'training': {'source_feature_config': source_spec}}
    plan['run'] = dict(data_root=str(root))
    source = wav(tmp_path/'ja.wav')
    def features(path, cfg):
        assert path == source and cfg == source_spec
        return torch.zeros(10, 80)
    monkeypatch.setattr('direct_s2st.translatotron2.data.source_features', features)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    original_tensor = torch.tensor
    monkeypatch.setattr(torch, 'tensor', lambda value, **kwargs: original_tensor(value))
    class Model:
        def generate(self, src, lengths, **kwargs):
            assert kwargs == dict(max_phones=400, max_frames=3000, beam_size=1, length_penalty=0.)
            return dict(post_mel=torch.ones(1, 5, 80), frame_lengths=original_tensor([5]))
    path = worker.tt2_infer(plan, (Model(), {'data_fingerprint':fingerprint}), source, tmp_path)
    assert np.load(path).shape == (5, 80)
    atomic_write_json(root/'mel-spec.json', dict(spec, sample_rate=22050), overwrite=True)
    with pytest.raises(ValueError, match='specification mismatch'):
        worker.tt2_infer(plan, (Model(), {'data_fingerprint':fingerprint}), source, tmp_path)


def test_s2ut_uses_pinned_frontend_beam10_and_no_targets_or_reference(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    @dataclass
    class GenerationConfig:
        beam: int = 10
        max_len_a: float = 1.0
    class Model:
        def float(self): return self
        def cuda(self): return self
        def eval(self): return self
        def prepare_for_inference_(self, cfg): assert cfg.generation.beam == 10
    class Dictionary:
        def eos(self): return 2
        def __getitem__(self, token): return {4:'0', 5:'1'}[token]
    sample_ids = []
    class Dataset:
        def __init__(self, split, train, cfg, audio_paths, frames, **kwargs):
            assert split == 'test' and train is False and 'tgt_texts' not in kwargs
            sample_ids.append(audio_paths)
        def __getitem__(self, index): return 'source only'
        def collater(self, items):
            return dict(net_input=dict(src_tokens=torch.zeros(1, 10, 80), src_lengths=torch.tensor([10]), prev_output_tokens=None), target=None)
    def inference(generator, models, sample):
        assert sample['target'] is None and sample['net_input']['prev_output_tokens'] is None
        return [[dict(tokens=torch.tensor([4, 5, 2]))]]
    task = NS(args=NS(target_is_code=True, target_code_size=100, n_frames_per_step=1), data_cfg=object(),
              max_positions=lambda: (6000, 1024), target_dictionary=Dictionary(), inference_step=inference,
              build_generator=lambda models, cfg: ('beam', cfg.beam))
    def load(paths, arg_overrides, strict):
        assert strict and arg_overrides['multitask_config_yaml'] == 'config_multitask.yaml'
        return [Model()], OmegaConf.create({}), task
    modules = {name: ModuleType(name) for name in ('fairseq', 'fairseq.dataclass', 'fairseq.dataclass.configs',
               'fairseq.data', 'fairseq.data.audio', 'fairseq.data.audio.speech_to_text_dataset')}
    modules['fairseq'].checkpoint_utils = NS(load_model_ensemble_and_task=load)
    modules['fairseq'].utils = NS(move_to_cuda=lambda sample: sample)
    modules['fairseq.dataclass.configs'].GenerationConfig = GenerationConfig
    modules['fairseq.data.audio.speech_to_text_dataset'].SpeechToTextDataset = Dataset
    for name, module in modules.items(): monkeypatch.setitem(sys.modules, name, module)
    path = wav(tmp_path/'source.wav')
    generate, _ = worker.s2ut(dict(run=dict(checkpoint='best.pt', data_root='fixed')), path)
    result = generate(None)
    assert result['units'] == [0, 1] and result['reference_used'] is False
    assert sample_ids == [[str(path)]]


def test_reference_dependency_is_not_an_input_to_candidate_models():
    assert worker.dependencies('candidate-model') == ('source',)
    assert worker.dependencies('candidate-audio') == ('candidate-model',)
    tree = ast.parse((ROOT/'src/direct_s2st/interactive_comparison.py').read_text(encoding='utf-8'))
    for name in ('s2ut', 'tt2_load', 'tt2_infer'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert not any(isinstance(n, ast.Constant) and n.value in ('cascade-mt', 'cascade-asr') for n in ast.walk(node))


def test_notebook_cell_is_normal_form_and_has_three_candidates():
    notebook = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    code = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('#@title 10.'))
    ast.parse(code)
    assert "INTERACTIVE_MODEL = 's2ut'" in code
    assert 's2t_tts' in code and 'tt2' in code
    assert 'INTERACTIVE_REMEASURE = False' in code
    assert '参考ASR-BLEU' in code and 'VRAM %' in code and 'interactive_session.py' in code


def test_normal_cell_executes_comparison_and_displays_two_audio_columns_and_metrics(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    fake_phases(monkeypatch)
    for phase in worker.PHASES:
        worker.worker(plan, phase)
    result = worker.report(plan)
    result['evaluation']['branches']['candidate']['asr_text'] = '<script>not HTML</script>'
    report = Path(plan['output'])/'comparison.json'
    atomic_write_json(report, result, overwrite=True)
    path = Path(plan['output'])/'plan.json'
    atomic_write_json(path, plan)
    persistent = tmp_path/'p'
    atomic_write_json(persistent/'interactive/latest.json', dict(plan=str(path), sha256=raw_sha(path), report_sha256=raw_sha(report)))
    displayed, commands = [], []
    class Audio:
        def __init__(self, filename): assert Path(filename).is_file()
        def _repr_html_(self): return '<audio controls></audio>'
    module = ModuleType('IPython.display')
    module.Audio, module.HTML, module.display = Audio, lambda html: html, displayed.append
    monkeypatch.setitem(sys.modules, 'IPython.display', module)
    notebook = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    code = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('#@title 10.'))
    namespace = dict(Path=Path, json=json, PERSISTENT=persistent, REPO=tmp_path/'repo',
                     preview_launcher=lambda: ('python', ROOT/'scripts/colab/inference_session.py'),
                     run=lambda *args: commands.append(args))
    exec(code, namespace)
    assert commands[0][commands[0].index('--model')+1] == 's2ut'
    assert '--remeasure' not in commands[0]
    assert len(displayed) == 3 and displayed[1].count('<audio') == 2
    assert '<script>' not in displayed[1] and '&lt;script&gt;' in displayed[1]
    assert 'VRAM %' in displayed[2] and 'RAM %' in displayed[2] and '取得不可' in displayed[2]
