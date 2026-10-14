"""Bounded copies of complete training state, never all historical checkpoints."""
from pathlib import Path
import shutil

from .hashing import sha256_file

ARTIFACTS = ('config.json', 'gradient-audit-rank-0.json', 'batch-calibration.json',
    'research-metadata.json', 'session-research-metadata.json', 's2ut-recipe-lock.json',
    'training-monitor.json', 'training-monitor.csv', 'training-monitor.svg',
    'checkpoints/checkpoint_best.pt')


def freeze_files(work, checkpoint, destination):
    work, checkpoint, destination = map(lambda p: Path(p).resolve(), (work, checkpoint, destination))
    relative = checkpoint.relative_to(work)
    files = [relative]+[Path(name) for name in ARTIFACTS if (work/name).is_file() and Path(name) != relative]
    hashes = {}
    for name in files:
        source, target = work/name, destination/name
        if source.is_symlink():
            raise ValueError('recovery state cannot contain symlinks')
        before = sha256_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != before or sha256_file(source) != before:
            raise ValueError('checkpoint/artifact changed while freezing recovery state')
        hashes[name.as_posix()] = before
    return hashes
