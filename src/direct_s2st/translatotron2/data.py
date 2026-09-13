"""Read prepared benchmark data, without writing into the corpus."""
import csv
import io
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
from ..hashing import sha256_file
from ..manifests.paths import resolve_audio_path
from .prepare_fairseq import _extract_logmel_official


def read_table(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    ids = [row['id'] for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f'empty/duplicate IDs: {path}')
    return rows


def vocabulary(root):
    tokens = [line.rsplit(' ', 1)[0] for line in (Path(root) / 'target_phoneme/dict.txt').read_text(encoding='utf-8').splitlines()]
    tokens = ['<pad>', '<bos>', '<eos>'] + tokens
    if len(tokens) != len(set(tokens)) or len(tokens) < 4:
        raise ValueError('invalid phoneme dictionary')
    return tokens


def fingerprint(root):
    root = Path(root)
    seen = set()
    for split in ('train', 'dev', 'test'):
        ids = {row['id'] for row in read_table(root / f'{split}.tsv')}
        labels = {row['id'] for row in read_table(root / f'target_phoneme/{split}.tsv')}
        if ids != labels or ids & seen:
            raise ValueError('TT2 split leakage or phoneme ID mismatch')
        seen.update(ids)
    paths = ['data-lock.json', 'mel-spec.json', 'target_phoneme/dict.txt']
    if (root / 'source-mel-spec.json').is_file():
        paths.append('source-mel-spec.json')
    paths += [f'{split}.tsv' for split in ('train', 'dev', 'test')]
    paths += [f'target_phoneme/{split}.tsv' for split in ('train', 'dev', 'test')]
    paths += [p.name for p in sorted(root.glob('*.zip'))]
    return {name: sha256_file(root / name) for name in paths}


def load_mel(root, locator):
    name, offset, length = locator.rsplit(':', 2)
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or min(int(offset), int(length)) < 0:
        raise ValueError('unsafe mel ZIP locator')
    with path.open('rb') as stream:
        stream.seek(int(offset))
        values = np.load(io.BytesIO(stream.read(int(length))), allow_pickle=False)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError('invalid target mel')
    return torch.from_numpy(values.copy()).float()


def source_features(path, spec):
    # Fixed-fairseq frontend, same as target preparation; source-only CMVN.
    with tempfile.TemporaryDirectory(prefix='tt2-source-') as directory:
        output = Path(directory) / 'source.npy'
        _extract_logmel_official(Path(path), output, spec)
        values = torch.from_numpy(np.load(output, allow_pickle=False).copy()).float()
    if values.ndim != 2 or values.size(0) < 2 or not torch.isfinite(values).all():
        raise ValueError('invalid source mel')
    return (values - values.mean(0)) / values.std(0, unbiased=False).clamp_min(1e-5)


class PreparedDataset:
    def __init__(self, root, split='train'):
        self.root = Path(root)
        self.rows = read_table(self.root / f'{split}.tsv')
        labels = read_table(self.root / f'target_phoneme/{split}.tsv')
        self.labels = {row['id']: row['tgt_text'] for row in labels}
        if set(self.labels) != {row['id'] for row in self.rows}:
            raise ValueError('phoneme and acoustic IDs must match exactly')
        self.tokens = vocabulary(root)
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        self.spec = json.loads((self.root / 'mel-spec.json').read_text(encoding='utf-8'))
        source_spec_path = self.root / 'source-mel-spec.json'
        self.source_spec = json.loads(source_spec_path.read_text(encoding='utf-8')) if source_spec_path.is_file() else self.spec
        self.corpus_root = Path(os.environ['CORPUS_ROOT']) if os.environ.get('CORPUS_ROOT') else None
        for sequence in self.labels.values():
            if not sequence.split() or any(p not in self.ids or self.ids[p] < 3 for p in sequence.split()):
                raise ValueError('empty/unknown phoneme sequence')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.corpus_root is None:
            source = Path(row['src_audio'])
            if not source.is_absolute() or not source.is_file():
                raise ValueError('CORPUS_ROOT required to resolve portable source audio')
        else:
            source = resolve_audio_path(row['src_audio'], self.corpus_root)
        x = source_features(source, self.source_spec)
        y = load_mel(self.root, row['tgt_audio'])
        phones = torch.tensor([self.ids[p] for p in self.labels[row['id']].split()] + [2])
        if y.size(0) != int(row['tgt_n_frames']) or y.size(1) != self.spec['n_mels']:
            raise ValueError('mel shape does not match prepared manifest/spec')
        return dict(source=x[None], source_lengths=torch.tensor([len(x)]),
                    phones=phones[None], phone_lengths=torch.tensor([len(phones)]),
                    target=y[None], target_lengths=torch.tensor([len(y)]))
