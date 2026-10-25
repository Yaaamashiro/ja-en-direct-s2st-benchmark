import ast
import csv
import io
import json
from pathlib import Path

import pytest

from direct_s2st.cascade import mt_diagnostic as mt
from direct_s2st.cascade import corpus_diagnostic as saved
from direct_s2st.benchmark_storage import publish_tree
from direct_s2st.io import atomic_write_json
from direct_s2st.journal import digest
from test_cascade_corpus_diagnostic import fixture, snapshot, ROOT

MODEL = dict(model='facebook/nllb-200-distilled-600M', revision='f8d333a098d19b4fd9a8b18f94170487ad3f821d',
             source_language='jpn_Jpan',target_language='eng_Latn')


def context(tmp_path,monkeypatch,*,generation_failure=False):
    common,output,*_ = fixture(tmp_path/'source',monkeypatch,generation_failure=generation_failure)
    plan = saved.read(output/'plan.json')
    plan['pipeline'] = dict(mt=MODEL)
    body = {k:v for k,v in plan.items() if k not in ('identity','local','output','run')}
    plan['identity'] = digest(body)
    atomic_write_json(output/'plan.json',plan,overwrite=True)
    metrics = saved.read(output/'results/metrics.json')
    metrics['pipeline'] = plan['pipeline']
    metrics['evaluation_identity']['plan'] = plan['identity']
    atomic_write_json(output/'results/metrics.json',metrics,overwrite=True)
    atomic_write_json(output/'scores/metrics.json',metrics,overwrite=True)
    identity = metrics['evaluation_identity']
    publish_tree(output/'scores',output/'progress','asr-'+digest(identity)[:24],identity)
    return mt.saved_context(common,output), common, output


class Translator:
    audit = dict(generation_config={'max_length':200}, generate_kwargs='only forced_bos_token_id')
    def __init__(self, config):
        assert config == MODEL
        self.calls = []
    def __call__(self, text):
        self.calls.append(text)
        return dict(translation='the very old book is on the table',eos_observed=True,
                    generated_token_count=10,generation_limit_reached=False)


def test_controlled_gold_and_asr_use_only_japanese_and_preserve_original_results(tmp_path,monkeypatch):
    ctx,common,output = context(tmp_path,monkeypatch)
    before = snapshot(tmp_path/'source')
    translator = Translator(MODEL)
    report = mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=lambda cfg:translator)
    assert len(translator.calls) == 8
    expected = [text for row in ctx['samples'] for text in (row['ja_text'],row['asr_ja_text'])]
    assert translator.calls == expected
    assert not any(row['en_text'] in translator.calls for row in ctx['samples'])
    assert report['groups']['All']['samples'] == 4 and report['groups']['KFTT']['samples'] == 2
    assert report['all_score_matches_saved']
    assert snapshot(tmp_path/'source') == before
    root = Path(report['output_root'])
    assert root.is_relative_to(tmp_path/'experiment/diagnostics/cascade-gold-mt')
    assert (root/'metrics.json').is_file() and (root/'report.md').is_file()
    with (root/'per_sample.csv').open(encoding='utf-8',newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and 'gold_translation' in rows[0] and 'asr_translation' in rows[0]
    def forbidden(*args): pytest.fail('completed resume must not load any NLLB model')
    again = mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=forbidden)
    assert again == report


def test_interruption_flushes_complete_branches_and_resume_skips_them(tmp_path,monkeypatch):
    ctx,_,_ = context(tmp_path,monkeypatch)
    class Interrupted(Translator):
        def __call__(self, source):
            if len(self.calls) == 3: raise KeyboardInterrupt
            return super().__call__(source)
    interrupted = Interrupted(MODEL)
    with pytest.raises(KeyboardInterrupt):
        mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=lambda cfg:interrupted)
    resumed = Translator(MODEL)
    report = mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=lambda cfg:resumed)
    assert len(resumed.calls) == 5 and report['selected_samples'] == 4


def test_csv_with_embedded_crlf_and_unicode_resumes_byte_identically(tmp_path,monkeypatch):
    ctx,_,_=context(tmp_path,monkeypatch)
    class Lines(Translator):
        def __call__(self,source):
            return dict(translation='English\r\nwith "quotes" and 日本語',generation_limit_reached=True)
    report=mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=Lines)
    path=Path(report['output_root'])/'per_sample.csv'
    before=path.read_bytes()
    mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=lambda cfg:pytest.fail('resume must not infer'))
    assert path.read_bytes()==before
    assert report['groups']['All']['gold_generation_limit_reached']==4


def test_cuda_failure_aborts_and_preserves_previous_gold_output(tmp_path,monkeypatch):
    ctx,_,_ = context(tmp_path,monkeypatch)
    class OutOfMemory(Translator):
        def __call__(self, source):
            if self.calls: raise RuntimeError('CUDA out of memory')
            return super().__call__(source)
    with pytest.raises(RuntimeError,match='CUDA out of memory'):
        mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=OutOfMemory)
    resumed = Translator(MODEL)
    mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=lambda cfg:resumed)
    assert len(resumed.calls) == 7


def test_resume_rejects_changed_generation_audit_before_model_load(tmp_path,monkeypatch):
    ctx,_,_=context(tmp_path,monkeypatch)
    report=mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=Translator)
    audit=Path(report['output_root'])/'model-audit.json'
    document=saved.read(audit)
    document['generation_config']['max_length']=100
    atomic_write_json(audit,document,overwrite=True)
    with pytest.raises(RuntimeError,match='generation audit'):
        mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=lambda cfg:pytest.fail('must reject stale audit first'))


def test_sample_value_failure_is_empty_in_all_denominators_and_csv_handles_mixed_columns(tmp_path,monkeypatch):
    ctx,_,_ = context(tmp_path,monkeypatch)
    class Rejected(Translator):
        def __call__(self, source):
            if source == ctx['samples'][0]['ja_text']:
                raise ValueError('input too long')
            return super().__call__(source)
    report = mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=Rejected)
    assert report['groups']['All']['gold_failures'] == 1
    assert report['groups']['All']['samples'] == 4
    assert report['per_sample'][0]['gold_translation'] == ''
    assert (Path(report['output_root'])/'per_sample.csv').is_file()


@pytest.mark.parametrize('changed', ['environment','label'])
def test_changed_conditions_make_a_new_diagnostic_without_overwriting(tmp_path,monkeypatch,changed):
    ctx,_,_ = context(tmp_path,monkeypatch)
    first = mt.run(ctx,tmp_path/'experiment',{'gpu':'one'},factory=Translator)
    before = snapshot(Path(first['output_root']))
    options = dict(environment={'gpu':'two'} if changed == 'environment' else {'gpu':'one'},
                   label='another' if changed == 'label' else 'default')
    second = mt.run(ctx,tmp_path/'experiment',factory=Translator,**options)
    assert first['output_root'] != second['output_root']
    assert snapshot(Path(first['output_root'])) == before


def test_source_change_during_run_stops_final_publication(tmp_path,monkeypatch):
    ctx,common,_ = context(tmp_path,monkeypatch)
    class Changed(Translator):
        def __call__(self, source):
            result = super().__call__(source)
            if len(self.calls) == 1: common.write_text(common.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            return result
    with pytest.raises(RuntimeError,match='source metadata changed'):
        mt.run(ctx,tmp_path/'experiment',{'gpu':'fixture'},factory=Changed)
    assert not list((tmp_path/'experiment').rglob('metrics.json'))


def test_missing_model_provenance_and_model_changes_are_rejected(tmp_path,monkeypatch):
    ctx,common,output = context(tmp_path,monkeypatch)
    plan = saved.read(output/'plan.json')
    plan['pipeline']['mt']['revision'] = 'a'*40
    atomic_write_json(output/'plan.json',plan,overwrite=True)
    with pytest.raises(RuntimeError,match='not bound'):
        mt.saved_context(common,output)
    for config in ({},dict(MODEL,revision='main'),dict(MODEL,source_language='eng_Latn'),dict(MODEL,num_beams=4)):
        with pytest.raises(RuntimeError): mt.validate_model(config)


def test_diagnostic_output_cannot_overlap_corpus_or_saved_evaluation(tmp_path):
    root = tmp_path/'diagnostics/cascade-gold-mt'
    with pytest.raises(RuntimeError,match='overlaps'):
        mt.safe_output(tmp_path,root)
    with pytest.raises(RuntimeError,match='overlaps'):
        mt.safe_output(tmp_path,tmp_path/'formal',tmp_path)


def test_subsets_are_explicit_deterministic_and_not_formal_full_test(tmp_path,monkeypatch):
    ctx,common,output = context(tmp_path,monkeypatch)
    first,second = mt.saved_context(common,output,2), mt.saved_context(common,output,2)
    assert first['selected_ids'] == second['selected_ids'] and len(first['samples']) == 2
    assert first['full_test_samples'] == 4
    report = mt.run(first,tmp_path/'experiment',{'gpu':'fixture'},factory=Translator,limit=2)
    assert report['selected_samples'] == 2 and report['full_test_samples'] == 4


def test_chunk_corruption_duplicates_and_wrong_source_are_rejected(tmp_path):
    identity = dict(version=1)
    expected = {('a','gold'):digest('日本語')}
    row = dict(pair_id='a',condition='gold',input_sha256=digest('日本語'),status='success',translation='Japanese')
    journal = mt.TextChunks(tmp_path,identity,expected)
    journal.record(row)
    journal.flush()
    with pytest.raises(RuntimeError,match='duplicate'):
        journal.record(row)
    with pytest.raises(RuntimeError,match='input differs'):
        mt.TextChunks(tmp_path,identity,{('a','gold'):digest('different')})
    path = next(tmp_path.glob('*.json'))
    document = saved.read(path)
    document['body']['rows'][0]['translation'] = 'corrupt'
    path.write_text(json.dumps(document),encoding='utf-8')
    with pytest.raises(RuntimeError,match='checksum'):
        mt.TextChunks(tmp_path,identity,expected)


def test_cell_8b_is_between_8a_and_9_and_all_other_cells_are_unchanged():
    import subprocess
    git = ['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT)]
    old = json.loads(subprocess.check_output(git+['show','HEAD:notebooks/colab_training.ipynb'],encoding='utf-8'))
    new = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    skip = lambda cell: ''.join(cell['source']).startswith(('#@title 8B.', '#@title 8C.', '#@title 8D.', '#@title 8E.'))
    assert [c for c in new['cells'] if not skip(c)] == [c for c in old['cells'] if not skip(c)]
    cell = next(c for c in new['cells'] if skip(c))
    i = new['cells'].index(cell)
    assert ''.join(new['cells'][i-1]['source']).startswith('#@title 8A.')
    assert ''.join(new['cells'][i+1]['source']).startswith('#@title 8C.')
    code = ''.join(cell['source'])
    ast.parse(code,feature_version=(3,10))
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in
        ('run','cli','ensure_runtime','ensure_workspace','benchmark_launcher') for node in ast.walk(ast.parse(code)))


@pytest.fixture
def tiny_cpu_threads():
    import torch
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_real_transformers_m2m100_api_on_tiny_cpu_fixture_preserves_language_and_generate_arguments(monkeypatch,tiny_cpu_threads):
    import torch
    from types import SimpleNamespace
    from tokenizers import Tokenizer, models, processors, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, M2M100Config, M2M100ForConditionalGeneration
    from direct_s2st.cascade import mt as original
    backend = Tokenizer(models.WordLevel({'<s>':0,'<pad>':1,'</s>':2,'<unk>':3,'jpn_Jpan':4,'eng_Latn':5,'word':6,'日本語':7},unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single='jpn_Jpan $A </s>',special_tokens=[('jpn_Jpan',4),('</s>',2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',pad_token='<pad>',eos_token='</s>',
        additional_special_tokens=['jpn_Jpan','eng_Latn'],model_input_names=['input_ids','attention_mask'])
    class Batch(dict):
        def to(self,device): return self  # Deliberately keep only this tiny API fixture on CPU.
    class Tokens:
        src_lang='jpn_Jpan'
        unk_token_id=tokenizer.unk_token_id
        init_kwargs=tokenizer.init_kwargs
        convert_tokens_to_ids=tokenizer.convert_tokens_to_ids
        batch_decode=tokenizer.batch_decode
        def __call__(self,*args,**kwargs): return Batch(tokenizer(*args,**kwargs))
    torch.manual_seed(3)
    model = M2M100ForConditionalGeneration(M2M100Config(vocab_size=8,d_model=16,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
        max_position_embeddings=32,pad_token_id=1,eos_token_id=2,decoder_start_token_id=2,forced_eos_token_id=2))
    model.eval()
    model.generation_config.max_length=8
    calls=[]
    def adapter(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(torch=torch,tokenizer=Tokens(),model=model,device=torch.device('cuda'))
    monkeypatch.setattr(original,'NllbTranslator',adapter)
    diagnostic = mt.NllbProbe(MODEL)
    output = diagnostic('日本語')
    assert output['source_prefix_ids'][0] == 4 and output['output_prefix_ids'][1] == 5
    assert output['generated_token_count'] <= 8 and output['eos_observed']
    assert calls[0]['revision'] == MODEL['revision'] and calls[0]['device'] == 'cuda'
    json.dumps(diagnostic.audit)  # Real AddedToken init metadata must serialize.
    for path in (ROOT/'src/direct_s2st/cascade/mt.py',ROOT/'src/direct_s2st/cascade/mt_diagnostic.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        generate=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='generate']
        assert len(generate)==1 and [keyword.arg for keyword in generate[0].keywords] == [None,'forced_bos_token_id']
