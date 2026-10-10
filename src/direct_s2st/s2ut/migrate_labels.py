"""Create a resumable Unigram-labelled sibling; never rewrite old data/units."""
import argparse
import json
import os
from pathlib import Path

from ..hashing import sha256_file
from ..io import atomic_write_json, atomic_write_text, read_jsonl
from ..progress import operation, track
from .ctc_tokenizer import VERSION
from .multitask import CHAR_VOCAB_VERSION, prepare_labels, read_tsv, validate_prepared

MIGRATED_DIRECTORY = 'fairseq-unigram-allchars-v1'


def current_labels(lock):
    linguistic = lock.get('linguistic', {})
    return (linguistic.get('ctc_version') == VERSION
            and linguistic.get('character_vocab_version') == CHAR_VOCAB_VERSION)


def paper_data_root(data, *, planned=False):
    base = Path(data)/'s2ut/fairseq'
    for name in (MIGRATED_DIRECTORY, 'fairseq-unigram-v1'):
        sibling = base.with_name(name)
        if (sibling/'paper-label-migration.json').is_file():
            base = sibling
            break
    if planned and (base/'data-lock.json').is_file():
        lock = json.loads((base/'data-lock.json').read_text(encoding='utf-8'))
        if not current_labels(lock):
            return base.with_name(MIGRATED_DIRECTORY)
    return base


@operation('s2ut: migrate CTC labels without recomputing Mel or units')
def migrate(common, source):
    common, source = Path(common), Path(source)
    lock = json.loads((source/'data-lock.json').read_text(encoding='utf-8'))
    if current_labels(lock):
        return source
    target = source.with_name(MIGRATED_DIRECTORY)
    if target == source:
        raise ValueError('migration requires the original fairseq directory')
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and target.resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('CTC migration outputs must be outside CORPUS_ROOT')
    if sha256_file(common/'dataset-lock.json') != lock['common_dataset_lock_sha256']:
        raise ValueError('common corpus identity differs from prepared S2UT data')
    names = ['data-lock.json', 'config.yaml', 'dict.txt', 'train.tsv', 'dev.tsv', 'test.tsv']
    inputs = {name: sha256_file(source/name) for name in names}
    common_inputs = {split: sha256_file(common/f'{split}.jsonl') for split in ('train', 'dev', 'test')}
    marker = target/'paper-label-migration.json'
    if marker.is_file():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved['inputs'] != inputs or saved['common_inputs'] != common_inputs:
            raise ValueError('migration inputs changed; use a new prepared directory/run')
        if any(not (target/name).resolve().is_relative_to(target.resolve()) for name in saved['outputs']):
            raise ValueError('migration artifact path must stay within the prepared directory')
        if any(not (target/name).is_file() or sha256_file(target/name) != checksum
               for name, checksum in saved['outputs'].items()):
            raise ValueError('completed CTC migration artifact checksum mismatch')
        return target
    rows = {split: list(read_jsonl(common/f'{split}.jsonl')) for split in ('train', 'dev', 'test')}
    for split in rows:
        main = read_tsv(source/f'{split}.tsv', ('id', 'src_audio', 'src_n_frames', 'tgt_audio', 'tgt_n_frames'))
        if len(rows[split]) != len(main) or {r['pair_id'] for r in rows[split]} != main.keys():
            raise ValueError('migration corpus/prepared sample IDs differ: ' + split)
    linguistic = prepare_labels(rows, target, resume=True)
    for name in track(names[1:], 's2ut: reuse prepared unit TSVs/configuration'):
        atomic_write_text(target/name, (source/name).read_text(encoding='utf-8'), resume=True)
    atomic_write_json(target/'data-lock.json', dict(lock, linguistic=linguistic), resume=True)
    validate_prepared(target, check_audio=False)
    if inputs != {name: sha256_file(source/name) for name in names}:
        raise ValueError('source prepared data changed during migration')
    if common_inputs != {split: sha256_file(common/f'{split}.jsonl') for split in rows}:
        raise ValueError('common corpus changed during migration')
    outputs = {p.relative_to(target).as_posix(): sha256_file(p) for p in target.rglob('*')
               if p.is_file() and p.name != 'paper-label-migration.json'}
    atomic_write_json(target/'paper-label-migration.json',
                      dict(version=VERSION, source=str(source.resolve()), inputs=inputs,
                           common_inputs=common_inputs, outputs=outputs), resume=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'prepared_root': str(migrate(args.common_root, args.data_root)),
                      'old_data_preserved': True, 'requires_new_s2ut_training_run': True}))


if __name__ == '__main__':
    main()
