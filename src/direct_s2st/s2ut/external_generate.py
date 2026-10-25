"""8E-only fairseq entry point: source-only decoding, original model/config."""
from contextlib import contextmanager


@contextmanager
def source_only(tasks):
    original = tasks.setup_task
    changed = []
    def setup(*args, **kwargs):
        task = original(*args, **kwargs)
        step = task.inference_step
        def infer(generator, models, sample, prefix_tokens=None, constraints=None):
            if prefix_tokens is not None or constraints is not None:
                raise ValueError('external S2UT forbids target prefixes/constraints')
            net = sample['net_input']
            if set(net)-{'src_tokens','src_lengths','prev_output_tokens','tgt_speaker'}:
                raise ValueError('unreviewed external S2UT model input')
            clean = dict(sample, net_input={k:v for k,v in net.items() if k != 'prev_output_tokens'},
                         target=None, target_lengths=None)
            clean.pop('multitask', None)
            return step(generator, models, clean, prefix_tokens=None, constraints=None)
        task.inference_step = infer
        changed.append((task, step))
        return task
    tasks.setup_task = setup
    try:
        yield
    finally:
        tasks.setup_task = original
        for task, step in changed:
            task.inference_step = step


def main():
    from ..inference_logging import configure_logging
    configure_logging()
    from fairseq import tasks
    from .staged_generate import main as generate
    with source_only(tasks):
        generate()


if __name__ == '__main__':
    main()
