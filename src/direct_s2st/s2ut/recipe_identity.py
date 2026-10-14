"""Stable recipe identity; argv[0] is a launcher location, not an experiment."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from ..io import atomic_write_json
from ..journal import digest

MODULE = 'direct_s2st.s2ut.fairseq_train'


def canonical(identity):
    value = deepcopy(identity)
    if (not isinstance(value, dict) or set(value) != {'recipe', 'command', 'data'}
            or not isinstance(value['recipe'], dict) or not isinstance(value['data'], dict)
            or not isinstance(value['command'], list) or len(value['command']) < 2
            or not all(isinstance(part, str) for part in value['command'])):
        raise ValueError('invalid S2UT recipe identity; saved files preserved')
    launcher = value['command'][0].replace('\\', '/')
    if launcher not in (MODULE, 'wrapper') and not launcher.endswith('/src/direct_s2st/s2ut/fairseq_train.py'):
        raise ValueError('unrecognized S2UT recipe launcher; saved files preserved')
    value['command'][0] = MODULE
    value['recipe'].pop('training_fingerprint', None)
    return value


def differences(saved, current, prefix=''):
    """Report keys, never dump the entire experiment/config on an error."""
    if isinstance(saved, dict) and isinstance(current, dict):
        for key in sorted(saved.keys() | current.keys()):
            field = f'{prefix}.{key}' if prefix else key
            if key not in saved or key not in current:
                yield field
            else:
                yield from differences(saved[key], current[key], field)
    elif isinstance(saved, list) and isinstance(current, list):
        if len(saved) != len(current):
            yield prefix+'.length'
        for index, (a, b) in enumerate(zip(saved, current)):
            yield from differences(a, b, f'{prefix}[{index}]')
    elif saved != current:
        yield prefix


def bind_recipe(marker, identity, *, restoring=False):
    """Reuse the *original* legacy fingerprint after exact semantic comparison.

    Old checkpoints hash the literal module path. Changing their fingerprint or
    lock would break checkpoint binding. Only argv[0] is canonicalized for the
    comparison; arguments, data hashes, optimizer/model metadata remain exact.
    """
    marker = Path(marker)
    candidate = canonical(identity)
    if marker.is_file():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        normalized = canonical(saved)
        fingerprint = saved['recipe'].get('training_fingerprint')
        original = deepcopy(saved)
        original['recipe'].pop('training_fingerprint', None)
        if fingerprint != digest(original):
            raise ValueError('S2UT saved recipe fingerprint is corrupt; saved files preserved')
        changed = list(differences(normalized, candidate))
        if changed:
            raise ValueError('S2UT recipe identity mismatch: '+', '.join(changed[:8])+
                             (' ...' if len(changed) > 8 else '')+'; saved files preserved')
        # Do not rewrite even JSON formatting in a saved immutable lock.
        if saved['command'][0] != candidate['command'][0]:
            print('[recipe-resume] launcher_path_only=true original_fingerprint_retained=true',
                  file=sys.stderr, flush=True)
        return fingerprint
    if restoring:
        raise ValueError('S2UT saved recipe lock is missing; refusing a fresh restart; saved checkpoint preserved')
    fingerprint = digest(candidate)
    candidate['recipe']['training_fingerprint'] = fingerprint
    atomic_write_json(marker, candidate, resume=True)
    return fingerprint
