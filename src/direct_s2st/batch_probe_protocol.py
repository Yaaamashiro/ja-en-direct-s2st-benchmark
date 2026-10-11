"""Shared probe protocol, including when the launcher is executed as __main__."""
import math


class ProbeComplete(BaseException):
    """Stop a disposable trainer only after its measurements were persisted."""
    progress_status = 'completed'


def validate_measurement(result, expected_steps=None):
    """Malformed/incomplete evidence is an error, never a capacity limit."""
    if not isinstance(result, dict):
        raise ValueError('invalid calibration measurement')
    if result.get('status') == 'oom':
        return
    steps = result.get('steps')
    if result.get('status') != 'ok' or not isinstance(steps, list) or not steps:
        raise ValueError('invalid calibration measurement')
    if expected_steps is not None and len(steps) != expected_steps:
        raise ValueError('incomplete calibration step count')
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError('invalid calibration measurement step')
        for field in ('seconds', 'samples', 'used', 'total'):
            value = step.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError('invalid calibration measurement: ' + field)
        if not isinstance(step['samples'], int):
            raise ValueError('invalid calibration measurement: samples must be an integer')
