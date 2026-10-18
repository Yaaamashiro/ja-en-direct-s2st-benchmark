"""Bounded, reference-free online dev translation diagnostics; never train/test BLEU.

Aux-text BLEU and waveform/ASR-BLEU are distinct measurements. All selected IDs
remain in the denominator, including decoding/ASR failures (empty hypotheses).
Neither metric selects checkpoints or stops training.
"""
from contextlib import contextmanager
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time

from ..io import atomic_write_json, atomic_write_text, read_jsonl
from ..journal import Journal
from ..progress import operation, track
from .paper_recipe import validate_bleu_policy


def fatal_device_error(error):
    import torch
    return (isinstance(error, torch.cuda.OutOfMemoryError) or
            any(text in str(error).lower() for text in ('device-side assert','illegal memory access')))


def selected_rows(rows, count, seed=1):
    if (type(count) is not int or count < 1 or not rows or len({r['pair_id'] for r in rows}) != len(rows)
            or any(r['split'] != 'dev' for r in rows)):
        raise ValueError('online BLEU requires unique dev-only IDs')
    return sorted(rows, key=lambda r: hashlib.sha256(f'{seed}:{r["pair_id"]}'.encode()).hexdigest())[:count]


def score(records, references):
    from sacrebleu.metrics import BLEU
    from ..evaluation.bleu import normalize_english
    if not references or set(records) != set(references):
        raise ValueError('BLEU must include every selected ID, not successful samples only')
    ids = sorted(references)
    metric = BLEU(tokenize='13a', lowercase=False, effective_order=False)
    if any(records[k].get('status') not in ('success','failed') or
           not isinstance(records[k].get('hypothesis',''),str) for k in ids):
        raise ValueError('invalid BLEU hypothesis/status')
    value = metric.corpus_score([normalize_english(records[k].get('hypothesis',''))
                                if records[k]['status'] == 'success' else '' for k in ids],
                               [[normalize_english(references[k]) for k in ids]])
    return dict(bleu=float(value.score), signature=str(metric.get_signature()), samples=len(ids),
                failures=sum(records[k].get('status') != 'success' for k in ids),
                normalization='benchmark.normalize_english (lowercase/NFKC/punctuation except apostrophes)')


@contextmanager
def evaluation_state(model):
    """Keep RNG, mixed module modes and nonparameter buffers exactly as found."""
    import numpy as np
    import torch
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    py, numpy = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(), torch.inference_mode():
            model.eval()
            yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for module, training in modes:
            module.training = training
        random.setstate(py)
        np.random.set_state(numpy)


def weight_hash(model):
    import torch
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def decode_characters(model, task, sample, maximum):
    """Free-running target-letter decoder, using its actual encoder attachment."""
    import torch
    auxiliary = task.multitask_tasks['target_letter']
    if auxiliary.args.input_from != 'encoder':
        raise ValueError('target-letter BLEU requires the encoder-attached decoder')
    dictionary = auxiliary.target_dictionary
    encoder = model.forward_encoder(**sample['net_input'], return_all_hiddens=True)
    context = dict(encoder_out=[encoder['encoder_states'][auxiliary.args.input_layer]],
                   encoder_padding_mask=encoder['encoder_padding_mask'])
    decoder = model.target_letter_decoder
    tokens = torch.tensor([[dictionary.eos()]], device=sample['net_input']['src_tokens'].device)
    state, output = {}, []
    for _ in range(maximum):
        logits = decoder(tokens, encoder_out=context, incremental_state=state)[0][:,-1].float()
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite auxiliary translation logits')
        logits[:,dictionary.pad()] = -float('inf')
        logits[:,dictionary.bos()] = -float('inf')
        index = int(logits.argmax(-1).item())
        if index == dictionary.eos():
            return ''.join(' ' if dictionary[i] == '<space>' else dictionary[i] for i in output)
        output.append(index)
        tokens = torch.cat((tokens, tokens.new_tensor([[index]])), dim=1)
    raise ValueError('auxiliary decoding reached max_aux_tokens without EOS')


def decode_units(generator, model, sample, dictionary):
    result = generator.generate([model], sample)[0][0]['tokens'].tolist()
    if not result or result[-1] != dictionary.eos():
        raise ValueError('Unit decoding did not terminate with EOS')
    from .reduce_units import validate_units
    return validate_units([int(dictionary[i]) for i in result[:-1]], clusters=100)


def source_sample(dataset, index, device, dtype):
    """Keep only source tensors; collated targets must never reach decoders."""
    batch = dataset.collater([dataset[index]])
    if 'id' in batch and int(batch['id'][0]) != index:
        raise ValueError('online BLEU sample ID alignment failure')
    return dict(net_input=dict(src_tokens=batch['net_input']['src_tokens'].to(device=device, dtype=dtype),
                               src_lengths=batch['net_input']['src_lengths'].to(device=device)))


@operation('S2UT BLEU: load official Unit vocoder and fixed English ASR')
def load_audio_models(work, device, policy):
    import torch
    from ..vocoders.reference import ensure_reference
    from ..vocoders.inference import validate_config
    from ..evaluation.transcribe import EvaluationASR
    checkpoint, config = work/'bleu-models/g_00500000', work/'bleu-models/config.json'
    ensure_reference(checkpoint, config)
    cfg = json.loads(config.read_text(encoding='utf-8'))
    validate_config('unit', cfg, sample_rate=16000)
    from fairseq.models.text_to_speech.codehifigan import CodeGenerator
    vocoder = CodeGenerator(cfg)
    vocoder.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['generator'], strict=True)
    vocoder.eval().to(device)
    try:
        asr = EvaluationASR(model_id=policy['asr_model'], revision=policy['asr_revision'],
                            device=(device.index or 0) if device.type == 'cuda' else -1)
    except BaseException:
        del vocoder
        raise
    return vocoder, asr


def report(work):
    """Small immutable per-update receipts plus CSV/SVG, included in backup."""
    jobs = sorted((work/'training-bleu').glob('*/result.json'))
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in jobs]
    fields = ['update','aux_bleu','asr_bleu','aux_samples','asr_samples','aux_failures','asr_failures','status']
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key:row.get(key) for key in fields})
    atomic_write_text(work/'training-bleu.csv', table.getvalue(), overwrite=True)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="350">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="25">S2UT fixed dev: auxiliary text BLEU / speech ASR-BLEU (not test)</text>',
        '<path d="M 50 50 V 300 H 840" fill="none" stroke="#888"/>']
    maximum = max([r['update'] for r in rows] or [1])
    for key, color, offset in [('aux_bleu','#2563eb',55),('asr_bleu','#dc2626',80)]:
        points = ' '.join(f'{50+r["update"]/maximum*790:.2f},{300-r[key]*2.5:.2f}' for r in rows if r.get(key) is not None)
        parts += [f'<text x="65" y="{offset}" fill="{color}">{key}: 0..100</text>',
                  f'<polyline points="{points}" fill="none" stroke="{color}"/>']
        for row in rows:
            if row.get(key) is not None:
                parts.append(f'<circle cx="{50+row["update"]/maximum*790:.2f}" '
                             f'cy="{300-row[key]*2.5:.2f}" r="2" fill="{color}"/>')
    parts += [f'<text x="650" y="330">update {maximum}</text>', '</svg>']
    atomic_write_text(work/'training-bleu.svg', '\n'.join(parts), overwrite=True)
    atomic_write_json(work/'training-bleu.json', dict(version=1, purpose='dev_diagnostics_not_test',
                      checkpoint_selection_unchanged=True, results=rows), overwrite=True)


@operation('S2UT BLEU: bounded online dev translation')
def evaluate(trainer, task, work, update):
    policy = validate_bleu_policy(json.loads(os.environ['S2ST_S2UT_BLEU_POLICY']))
    if update % policy['aux_interval']:
        return
    import torch
    from argparse import Namespace
    from ..vocoders.inference import write_waveform
    work = Path(work)
    model = trainer.get_model()
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    data_root = Path(task.args.data)
    from .multitask import read_tsv
    table = read_tsv(data_root/'dev.tsv', ('id','src_audio','src_n_frames','tgt_audio','tgt_n_frames'))
    # Common text is read only for scoring, never passed to the generation model.
    common = list(read_jsonl(data_root.parents[1]/'common/dev.jsonl'))
    chosen = selected_rows(common, policy['aux_samples'], policy['seed'])
    if {r['pair_id'] for r in common} != set(table):
        raise ValueError('online BLEU common/prepared dev IDs differ')
    ids = [r['pair_id'] for r in chosen]
    asr_ids = set(ids[:policy['asr_samples']]) if update in policy['asr_updates'] else set()
    by_id = {key:i for i,key in enumerate(table)}
    # Fairseq dataset order is the TSV row order, not the sampled/hash order.
    dataset = task.dataset('dev')
    if len(dataset) != len(table):
        raise ValueError('online BLEU dataset/ID mapping differs')
    identity = dict(version=1, update=update, policy=policy, weights=weight_hash(model), ids=ids,
                    references={r['pair_id']:r['en_text'] for r in chosen},
                    subset='fixed_hash_sample_of_dev', generation='source_audio_only',
                    auxiliary_decoding='greedy encoder-attached target-letter decoder',
                    unit_vocoder='official LJSpeech (not Fisher-trained)',
                    asr='benchmark Whisper (not original-paper wav2vec2)')
    directory = work/'training-bleu'/f'{update:09d}'
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory/'identity.json', identity, resume=True)
    atomic_write_json(directory/'subset.json', ids, resume=True)
    aux = Journal(directory/'aux.jsonl', identity, resume=True)
    speech = Journal(directory/'asr.jsonl', identity, resume=True)
    vocoder = asr = generator = None
    started = time.monotonic()
    try:
        with evaluation_state(model):
            if any(speech.rows.get(key,{}).get('status') != 'success' for key in asr_ids):
                generator = task.build_generator([model], Namespace(beam=10, max_len_a=1, max_len_b=200))
                vocoder, asr = load_audio_models(work, device, policy)
            for pair_id in track(ids, 'S2UT BLEU: translate/score fixed dev', total=len(ids)):
                sample = None
                for journal, kind in ((aux,'aux'), (speech,'asr')):
                    if kind == 'asr' and pair_id not in asr_ids:
                        continue
                    if journal.rows.get(pair_id, {}).get('status') == 'success':
                        continue
                    record = dict(pair_id=pair_id, hypothesis='', status='failed')
                    try:
                        if time.monotonic()-started > policy['max_seconds']:
                            raise TimeoutError('online BLEU evaluation time budget reached')
                        if sample is None:
                            # Strip all targets/previous-output tokens before any model call.
                            sample = source_sample(dataset, by_id[pair_id], device, dtype)
                        if kind == 'aux':
                            hypothesis = decode_characters(model, task, sample, policy['max_aux_tokens'])
                        else:
                            units = decode_units(generator, model, sample, task.target_dictionary)
                            codes = torch.tensor([units], dtype=torch.long, device=device)
                            lengths = (vocoder.dur_predictor(vocoder.dict(codes)).exp()-1).round().clamp_min(1)
                            if not torch.isfinite(lengths).all() or lengths.sum()*320 > 16000*120:
                                raise ValueError('vocoder duration exceeds safety bound')
                            wave = vocoder(code=codes, dur_prediction=True).float().cpu().numpy()
                            path = directory/'audio'/(pair_id+'.wav')
                            write_waveform(path, wave, 16000, overwrite=True)
                            hypothesis = asr(path)
                            record['units'] = units
                        record.update(hypothesis=hypothesis, status='success', error=None)
                    except Exception as error:
                        if fatal_device_error(error):
                            raise  # Never keep optimizing in a potentially poisoned CUDA context.
                        record['error'] = f'{type(error).__name__}: {error}'
                        print(f'[training-bleu] update={update} id={pair_id} kind={kind} failed={record["error"]}',
                              file=sys.stderr, flush=True)
                    journal.record(record)
    finally:
        del generator, vocoder, asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    refs = {r['pair_id']:r['en_text'] for r in chosen}
    auxiliary = score(aux.rows, refs)
    audio = score(speech.rows, {k:refs[k] for k in asr_ids}) if asr_ids else None
    result = dict(update=update, aux_bleu=auxiliary['bleu'], aux_samples=auxiliary['samples'],
        aux_failures=auxiliary['failures'], asr_bleu=audio['bleu'] if audio else None,
        asr_samples=audio['samples'] if audio else 0, asr_failures=audio['failures'] if audio else 0,
        status=('completed_with_failures' if auxiliary['failures'] or (audio or {}).get('failures') else 'completed'),
        aux_scoring=auxiliary, asr_scoring=audio,
        seconds=time.monotonic()-started, identity=identity)
    atomic_write_json(directory/'result.json', result, overwrite=True)
    report(work)
    print('[training-bleu] '+json.dumps({k:result[k] for k in ('update','aux_bleu','asr_bleu','aux_failures','asr_failures')}),
          file=sys.stderr, flush=True)
    return result
