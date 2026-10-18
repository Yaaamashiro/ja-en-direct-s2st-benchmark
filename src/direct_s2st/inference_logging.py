"""Keep normal inference logs, without fairseq-triggered compiler debug dumps."""
import logging


def configure_logging():
    # Pinned fairseq calls basicConfig(DEBUG) while importing all models. Install
    # a normal handler first so that call cannot enable Numba/filelock DEBUG.
    # Never force/reset existing handlers or suppress warnings/exceptions.
    logging.basicConfig(level=logging.INFO)
    root = logging.getLogger()
    if root.level < logging.INFO:
        root.setLevel(logging.INFO)
    for handler in root.handlers:
        # Parent logger levels don't filter records from explicitly DEBUG child
        # loggers. The output handler must enforce the floor too. Keep stricter
        # caller thresholds, streams, formatting and filters intact.
        if handler.level < logging.INFO:
            handler.setLevel(logging.INFO)
