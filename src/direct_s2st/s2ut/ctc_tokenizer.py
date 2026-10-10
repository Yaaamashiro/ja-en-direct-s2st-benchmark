"""Train-only, immutable SentencePiece Unigram supervision (S2UT §4.2)."""
import hashlib
import io
import json
from pathlib import Path

from ..io import atomic_write_bytes, atomic_write_json
from ..progress import operation

VERSION = 'sentencepiece-unigram-1000-v1'


def canonical_text(text):
    # Match the character auxiliary labels' ASCII-whitespace policy only.
    from .multitask import tokenize
    return ''.join(' ' if token == '<space>' else token for token in tokenize(text))


def text_fingerprint(sentences):
    return hashlib.sha256(json.dumps(sentences, ensure_ascii=False).encode('utf-8')).hexdigest()


@operation('s2ut: train/reuse 1000-vocabulary Unigram tokenizer')
def build(rows, root, *, overwrite=False):
    import sentencepiece as spm
    sentences = [canonical_text(row['en_text']) for row in rows]
    if not sentences:
        raise ValueError('CTC tokenizer requires nonempty train sentences')
    fingerprint = text_fingerprint(sentences)
    identity = dict(version=VERSION, model_type='unigram', requested_vocab_size=1000,
                    vocabulary_source='train_only', train_text_sha256=fingerprint,
                    sentencepiece_version=spm.__version__, normalization='identity_ascii_whitespace')
    path, lock_path = Path(root)/'ctc.model', Path(root)/'ctc-tokenizer.json'
    if path.is_file() and lock_path.is_file() and not overwrite:
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        data = path.read_bytes()
        if any(lock.get(k) != v for k, v in identity.items()) or hashlib.sha256(data).hexdigest() != lock['model_sha256']:
            raise ValueError('CTC tokenizer identity changed; use a new prepared directory/run')
    else:
        buffer = io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=iter(sentences), model_writer=buffer,
            model_type='unigram', vocab_size=1000, hard_vocab_limit=False, character_coverage=1.0,
            normalization_rule_name='identity', remove_extra_whitespaces=False,
            bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0, num_threads=1,
            shuffle_input_sentence=False, input_sentence_size=0,
            max_sentence_length=max(4096, max(len(s.encode('utf-8')) for s in sentences)), minloglevel=2)
        data = buffer.getvalue()
        lock = dict(identity, model_sha256=hashlib.sha256(data).hexdigest())
    processor = spm.SentencePieceProcessor(model_proto=data)
    lock['actual_vocab_size'] = processor.get_piece_size()
    return processor, data, lock


def publish(root, data, lock, *, resume=False, overwrite=False):
    atomic_write_bytes(Path(root)/'ctc.model', data, resume=resume, overwrite=overwrite)
    atomic_write_json(Path(root)/'ctc-tokenizer.json', lock, resume=resume, overwrite=overwrite)


def load(root):
    import sentencepiece as spm
    root = Path(root)
    if not (root/'ctc-tokenizer.json').is_file() or not (root/'ctc.model').is_file():
        raise ValueError('legacy CTC labels: run python -m direct_s2st.s2ut.migrate_labels with --common-root and --data-root; old units/Mel are preserved')
    lock = json.loads((root/'ctc-tokenizer.json').read_text(encoding='utf-8'))
    data = (root/'ctc.model').read_bytes()
    if lock.get('version') != VERSION or lock.get('model_type') != 'unigram' or lock.get('vocabulary_source') != 'train_only' or lock.get('requested_vocab_size') != 1000:
        raise ValueError('paper S2UT requires train-only 1000-vocabulary Unigram CTC labels; migrate the old prepared data')
    if hashlib.sha256(data).hexdigest() != lock['model_sha256']:
        raise ValueError('CTC tokenizer checksum mismatch')
    processor = spm.SentencePieceProcessor(model_proto=data)
    if processor.get_piece_size() != lock['actual_vocab_size']:
        raise ValueError('CTC tokenizer vocabulary size mismatch')
    return processor, lock
