import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from direct_s2st.cascade import source_corpus_tts as source
from direct_s2st.cascade import flores_fleurs_diagnostic as diagnostic
from direct_s2st.journal import digest

ROOT=Path(__file__).resolve().parents[2]


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'corpus-copy'
    config=dict(tts=dict(model_id='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',revision='f00cf133b78d3c2c35857faba3b1be9b98c4f971',
        device='cuda:0',dtype='auto',speaker='Ono_Anna',instruct='',max_new_tokens=2048,do_sample=True,
        subtalker_do_sample=True,remove_invalid_values=False,renormalize_logits=True,output_sample_rate=16000,max_content_retries=1),
        device=dict(attention='sdpa'))
    cell="TTS_BATCH_SIZE = 72\nASR_BATCH_SIZE = 144\nif gpu['vram_gib'] < 70:\n    TTS_BATCH_SIZE = ASR_BATCH_SIZE = 1\n"
    notebook={'cells':[{'cell_type':'code','source':cell.splitlines(keepends=True)}]}
    files={'configs/colab-pro.yaml':yaml.safe_dump(config),'notebooks/colab_pro.ipynb':json.dumps(notebook),
           'requirements/colab.txt':'qwen-tts==0.1.1\ntransformers==4.57.3\nnumpy==2.2.6\nsoxr==1.1.0\n'}
    hashes={}
    for relative,value in files.items():
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value.encode())
        hashes[relative]=hashlib.sha256(value.encode()).hexdigest()
    p=root/'src/s2st_corpus/config.py';p.parent.mkdir(parents=True);p.write_text('# fixture corpus implementation\n')
    monkeypatch.setattr(source,'CONFIRMED_FILES',hashes)
    monkeypatch.setattr(source,'git',lambda root,*args:source.CONFIRMED_REVISION if 'rev-parse' in args else '')
    runtime=dict(vram_gib=79.25,bf16_supported=True,packages={'qwen-tts':'0.1.1','transformers':'4.57.3',
        'torch':'2.7.1+cu126','numpy':'2.2.6','soxr':'1.1.0'})
    return root,runtime,config


def test_confirmation_is_explicit_and_does_not_invent_a_historical_receipt():
    proven=source.confirmed_notebook()
    assert proven['basis']=='user_confirmation' and proven['resolved'] is False
    assert proven['inputs_sha256']=={} and 'effective_tts' not in proven['receipt']
    assert proven['receipt']['confirmation']['historical_execution_revision_verified'] is False
    assert proven['receipt']['code_revision']==source.CONFIRMED_REVISION


@pytest.mark.parametrize('vram,expected',[(79.25,72),(70,72),(69.9,1),(22.5,1)])
def test_confirmed_settings_use_notebook_hardware_rule_and_record_effective_values(tmp_path,monkeypatch,vram,expected):
    root,runtime,config=fixture(tmp_path,monkeypatch);runtime['vram_gib']=vram
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=source.resolve_confirmed(source.confirmed_notebook(),root,runtime)
    assert result['resolved'] and result['basis']=='user_confirmation'
    receipt=result['receipt'];tts=receipt['effective_tts']
    assert tts['batch_size']==expected and tts['dtype']=='bfloat16'
    assert receipt['requested_tts']['dtype']=='auto' and receipt['batch_rule']['effective_batch_size']==expected
    assert {k:v for k,v in tts.items() if k not in ('dtype','batch_size')}=={k:v for k,v in config['tts'].items() if k!='dtype'}
    assert len(result['inputs_sha256'])==3 and result['receipt_sha256']==digest(receipt)
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    source.verify_code(root,result)


def test_notebook_auto_precision_is_resolved_and_audited(tmp_path,monkeypatch):
    root,runtime,_=fixture(tmp_path,monkeypatch);runtime['bf16_supported']=False
    result=source.resolve_confirmed(source.confirmed_notebook(),root,runtime)
    assert result['receipt']['effective_tts']['dtype']=='float16'


@pytest.mark.parametrize('relative',['configs/colab-pro.yaml','notebooks/colab_pro.ipynb','requirements/colab.txt'])
def test_confirmation_does_not_accept_altered_pinned_source_files(tmp_path,monkeypatch,relative):
    root,runtime,_=fixture(tmp_path,monkeypatch)
    p=root/relative;p.write_bytes(p.read_bytes()+b'\n# altered\n')
    with pytest.raises(RuntimeError,match='differs from reviewed revision'):
        source.resolve_confirmed(source.confirmed_notebook(),root,runtime)


def test_checkout_crlf_conversion_is_safe_but_actual_file_hashes_remain_audited(tmp_path,monkeypatch):
    root,runtime,_=fixture(tmp_path,monkeypatch)
    for relative in source.CONFIRMED_FILES:
        p=root/relative;p.write_bytes(p.read_bytes().replace(b'\n',b'\r\n'))
    result=source.resolve_confirmed(source.confirmed_notebook(),root,runtime)
    source.verify_code(root,result)
    assert result['receipt']['config_sha256']==source.checksum(root/'configs/colab-pro.yaml')


def test_incompatible_dependency_and_missing_runtime_data_fail_before_model_load(tmp_path,monkeypatch):
    root,runtime,_=fixture(tmp_path,monkeypatch)
    bad=copy.deepcopy(runtime);bad['packages']['qwen-tts']='0.1.2'
    with pytest.raises(RuntimeError,match='notebook dependency'):source.resolve_confirmed(source.confirmed_notebook(),root,bad)
    del runtime['vram_gib']
    with pytest.raises(RuntimeError,match='invalid corpus notebook/runtime'):source.resolve_confirmed(source.confirmed_notebook(),root,runtime)


def test_stdio_preflight_accepts_user_confirmation_without_any_python_dependencies():
    result=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/colab/flores_fleurs_diagnostic.py'),
        '--preflight-source','--confirm-corpus-notebook-defaults'],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['basis']=='user_confirmation'


def test_explicit_invalid_saved_provenance_is_not_hidden_by_user_confirmation(tmp_path):
    result=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/colab/flores_fleurs_diagnostic.py'),
        '--preflight-source','--confirm-corpus-notebook-defaults','--source-provenance',str(tmp_path/'missing.json')],
        capture_output=True,text=True,encoding='utf-8')
    assert result.returncode and source.ERROR in result.stderr


def test_unresolved_confirmation_cannot_become_a_diagnostic_identity(tmp_path):
    with pytest.raises(RuntimeError,match='must be resolved'):
        diagnostic.prepare(tmp_path,{},[],{},source.confirmed_notebook(),{},{} )
