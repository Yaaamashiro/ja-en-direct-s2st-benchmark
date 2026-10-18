"""Pinned fairseq generation restricted to original dataset indices.

Only the iterator's ordered_indices is restricted, in an isolated inference
process. Numeric IDs, dictionaries, transforms, checkpoints and split files
stay unchanged. No target/reference text is passed to the generation model.
"""
import argparse
import json
import sys
from contextlib import contextmanager


@contextmanager
def selected_iterator(tasks, indices, expected_size, split):
    import numpy as np
    original = tasks.setup_task
    if (not 1 <= len(indices) <= 10 or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < expected_size for i in indices)
            or split not in ('dev', 'test')):
        raise ValueError('invalid bounded preview dataset indices')
    configured, loaded_tasks = [], []
    def setup(*args, **kwargs):
        task = original(*args, **kwargs)
        load = task.load_dataset
        loaded_tasks.append((task, load))
        def load_selected(name, *a, **kw):
            if name != split:
                raise ValueError('preview tried to load a different split')
            result = load(name, *a, **kw)
            dataset = task.dataset(name)
            if len(dataset) != expected_size:
                raise ValueError('preview dataset size differs from prepared numeric ID mapping')
            old_order = dataset.ordered_indices
            dataset.ordered_indices = lambda: np.asarray(indices, dtype=np.int64)
            configured.append((dataset, old_order))
            return result
        task.load_dataset = load_selected
        return task
    tasks.setup_task = setup
    try:
        yield
        if len(configured) != 1:
            raise ValueError('preview did not restrict exactly one inference dataset')
    finally:
        tasks.setup_task = original
        for task, load in loaded_tasks:
            task.load_dataset = load
        for dataset, order in reversed(configured):
            dataset.ordered_indices = order


def main():
    from ..inference_logging import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indices', required=True)
    parser.add_argument('--dataset-size', type=int, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    args, remaining = parser.parse_known_args()
    if remaining[:1] != ['--']:
        parser.error('expected -- followed by pinned fairseq generation arguments')
    from fairseq import tasks
    from fairseq_cli import generate
    from fairseq.data.audio import speech_to_text_dataset as audio
    from ..drive_staging import local_path
    original_audio = audio.get_features_or_waveform
    def staged_audio(path, *a, **kw):
        return original_audio(str(local_path(path)), *a, **kw)
    previous = sys.argv
    try:
        audio.get_features_or_waveform = staged_audio
        sys.argv = ['fairseq_cli.generate']+remaining[1:]
        with selected_iterator(tasks, json.loads(args.indices), args.dataset_size, args.split):
            generate.cli_main()
    finally:
        sys.argv = previous
        audio.get_features_or_waveform = original_audio


if __name__ == '__main__':
    main()
