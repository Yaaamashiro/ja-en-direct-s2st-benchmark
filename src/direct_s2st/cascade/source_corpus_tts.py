"""Proof gate and a read-only bridge to the actual corpus TTS implementation.

The corpus repository currently DOES NOT persist its full effective config.
A checked-in YAML is not proof of which overrides generated historical audio.
This module therefore refuses inference without a generation-time receipt.
"""
import contextlib
import dataclasses
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

from .corpus_diagnostic import checksum, read
from ..progress import operation,track

ERROR = 'Cannot prove exact source-corpus Japanese TTS configuration'
REPOSITORY = 'Yaaamashiro/ja-en-direct-s2st-corpus'
TTS_FIELDS = {'model_id','revision','device','dtype','speaker','instruct','max_new_tokens',
              'do_sample','subtalker_do_sample','remove_invalid_values','renormalize_logits',
              'output_sample_rate','max_content_retries','batch_size'}


@operation('8C: prove historical source-corpus Japanese TTS before GPU setup')
def proof(path):
    if not path or not Path(path).is_file():
        raise RuntimeError(ERROR+': no saved generation-time provenance. Repository YAML alone cannot prove '
            'effective instruct/sampling/dtype/batch/overrides. Set FLORES_FLEURS_SOURCE_PROVENANCE to the '
            'original saved receipt, not a reconstructed config. No dataset/model inference was started.')
    try:
        doc = read(path)
    except RuntimeError as error:
        raise RuntimeError(ERROR+': saved receipt is not valid JSON: '+str(path)) from error
    required = {'schema','repository','code_revision','config_relative_path','config_sha256','language',
                'effective_tts','attention','packages','evidence'}
    if (not required<=doc.keys() or doc['schema']!='source-corpus-tts-v1' or doc['repository']!=REPOSITORY
            or doc['language']!='Japanese' or not re.fullmatch('[0-9a-f]{40}',str(doc['code_revision']))):
        raise RuntimeError(ERROR+': incomplete generation-time provenance contract')
    cfg = doc['effective_tts']
    if (not isinstance(cfg,dict) or set(cfg)!=TTS_FIELDS or not re.fullmatch('[0-9a-f]{40}',str(cfg.get('revision')))
            or cfg.get('model_id')!='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
            or cfg.get('dtype') not in ('bfloat16','float16','float32') or cfg.get('device')!='cuda:0'
            or not isinstance(cfg.get('instruct'),str) or not isinstance(cfg.get('speaker'),str)
            or cfg.get('output_sample_rate')!=16000 or type(cfg.get('batch_size')) is not int or cfg['batch_size']<1
            or type(cfg.get('max_new_tokens')) is not int or cfg['max_new_tokens']<1
            or any(type(cfg.get(k)) is not bool for k in ('do_sample','subtalker_do_sample','remove_invalid_values','renormalize_logits'))
            or doc['attention'] not in ('sdpa','eager') or not re.fullmatch('[0-9a-f]{64}',str(doc['config_sha256']))):
        raise RuntimeError(ERROR+': incomplete/unsupported effective Japanese TTS settings; no defaults supplied')
    relative = Path(doc['config_relative_path'])
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise RuntimeError(ERROR+': unsafe source config relative path')
    packages = doc['packages']
    if not isinstance(packages,dict) or any(not packages.get(k) for k in ('qwen-tts','transformers','torch','numpy','soxr')):
        raise RuntimeError(ERROR+': saved generation dependency versions missing')
    if packages['qwen-tts']!='0.1.1' or packages['transformers']!='4.57.3':
        raise RuntimeError(ERROR+': source SDK version has not been reviewed')
    evidence = doc['evidence']
    if not isinstance(evidence,list) or not evidence:
        raise RuntimeError(ERROR+': no original provenance/config evidence')
    inputs = {str(Path(path).resolve()):checksum(path)}
    for item in track(evidence,'8C: verify original generation evidence',key=lambda item:item.get('path','-')):
        p = Path(item['path'])
        if not p.is_absolute():
            p = Path(path).parent/p
        if not p.is_file() or checksum(p)!=item.get('sha256'):
            raise RuntimeError(ERROR+': source generation evidence SHA mismatch: '+str(p))
        inputs[str(p.resolve())] = item['sha256']
    return dict(receipt=doc, inputs_sha256=inputs, receipt_sha256=checksum(path))


def git(root,*args):
    result = subprocess.run(['git','-c','safe.directory='+str(Path(root).resolve()),'-C',str(root),*args],
                            capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode:
        raise RuntimeError('8C source-code verification failed: '+(result.stderr or result.stdout))
    return result.stdout.strip()


def verify_code(root, proven):
    root = Path(root).resolve()
    doc = proven['receipt']
    if (git(root,'rev-parse','HEAD')!=doc['code_revision']
            or git(root,'status','--porcelain','--untracked-files=no','--','src','configs')):
        raise RuntimeError(ERROR+': source corpus checkout differs from recorded code; never reset the external checkout')
    config = root/doc['config_relative_path']
    if not config.resolve().is_relative_to(root) or checksum(config)!=doc['config_sha256']:
        raise RuntimeError(ERROR+': recorded corpus config SHA does not match the pinned checkout')
    files = [p for p in (root/'src/s2st_corpus').rglob('*.py')]
    if not files:
        raise RuntimeError(ERROR+': corpus Python implementation missing')
    return dict(root=str(root),repository=REPOSITORY, revision=doc['code_revision'], config_sha256=checksum(config),
                source_files={p.relative_to(root).as_posix():checksum(p) for p in sorted(files)})


class CorpusSource:
    """Use corpus runner helpers, preserving batching/RNG/PCM16/resampling.

    No corpus CLI/import/manifest/QC is run. All run paths are redirected to the
    diagnostic only. Attempt zero is generated without ASR-based rejection;
    the diagnostic measures raw source quality, never filters failures by CER.
    """
    @operation('8C: load verified original Japanese corpus TTS implementation')
    def __init__(self, proven, code, destination):
        import importlib
        import importlib.metadata as metadata
        verify_code(code,proven)
        doc = proven['receipt']
        for key,value in doc['packages'].items():
            if metadata.version(key).split('+')[0]!=value.split('+')[0]:
                raise RuntimeError('8C source dependency differs from corpus provenance: '+key)
        if 's2st_corpus.runner' in sys.modules:
            raise RuntimeError('8C source TTS must start in a fresh worker process')
        sys.path.insert(0,str(Path(code)/'src'))
        prior_bytecode=sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode=True  # Also protect read-only code when used outside our CLI.
            self.runner = importlib.import_module('s2st_corpus.runner')
            config_module = importlib.import_module('s2st_corpus.config')
        finally:
            sys.dont_write_bytecode=prior_bytecode
        # Ignore current notebook batch overrides; only receipt values apply.
        previous = {k:os.environ.pop(k,None) for k in ('S2ST_TTS_BATCH_SIZE','S2ST_ASR_BATCH_SIZE')}
        try:
            cfg = config_module.load_config(Path(code)/doc['config_relative_path'])
        finally:
            for key,value in previous.items():
                if value is not None:
                    os.environ[key]=value
        destination = Path(destination).resolve()
        self.config = dataclasses.replace(cfg,
            tts=config_module.TTSConfig(**doc['effective_tts']),
            device=dataclasses.replace(cfg.device,attention=doc['attention']),
            run=dataclasses.replace(cfg.run, output_dir=destination, input_jsonl=destination/'unused.jsonl',
                                    save_native=False,resume=False,checkpoint_every=1))
        # Corpus's own loader resolves the COMPLETE fixed snapshot locally.
        os.environ['S2ST_MODEL_ROOT']=str(Path(os.environ['HF_HOME'])/'corpus-models')
        import torch
        if not torch.cuda.is_available() or (self.config.tts.dtype=='bfloat16' and not torch.cuda.is_bf16_supported()):
            raise RuntimeError('recorded Japanese TTS precision requires a compatible CUDA GPU; no fallback')
        self.audit = dict(provenance=proven,code=verify_code(code,proven),
                          effective_tts=dataclasses.asdict(self.config.tts),attention=doc['attention'],
                          attempt=0, qc_selection=False, note='Attempt-zero external diagnostic; no CER-based rejection')

    def group(self, samples):
        records = [self.runner._new_record(dict(pair_id=s['sample_id'].replace(':','_'),
                    ja_text=s['ja_text'], en_text='unused')) for s in samples]
        targets = {(r['pair_id'],'ja') for r in records}
        # Original runner prints to stdout; diagnostic result JSON owns stdout.
        with contextlib.redirect_stdout(sys.stderr):
            self.runner._generate_missing(self.config,'external-diagnostic',records,0,targets,
                                          self.config.run.output_dir/'corpus-attempt.json')
        return [dict(r['attempts']['ja'][0],source_tts_text=r['ja_tts_text'],
                     source_tts_text_sha256=hashlib.sha256(r['ja_tts_text'].encode('utf-8')).hexdigest())
                for r in records]
