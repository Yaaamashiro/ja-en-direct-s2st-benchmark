"""Real-artifact completion gate; exit codes alone never establish E2E."""
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from ..io import atomic_write_json, read_jsonl
from ..manifests.reader import read_common_manifest
from ..predictions import Prediction, SYSTEMS
from ..hashing import sha256_file


def verify_suite(common_root, run_roots, output_root, *, overwrite=False):
    common = list(read_common_manifest(Path(common_root) / 'test.jsonl'))
    expected = {r['pair_id']: r for r in common}
    if not common or len(common) != len(expected):
        raise ValueError('empty/duplicate common test IDs')
    reports, settings = [], []
    for root in map(Path, run_roots):
        rows = list(read_jsonl(root / 'predictions/predictions.jsonl'))
        evaluated = list(read_jsonl(root / 'metrics/per_sample.jsonl'))
        metrics = json.loads((root / 'metrics/metrics.json').read_text(encoding='utf-8'))
        lock = json.loads((root / 'metrics/per_sample.lock.json').read_text(encoding='utf-8'))
        if lock['predictions'] != sha256_file(root / 'predictions/predictions.jsonl'):
            raise ValueError('evaluation refers to stale predictions')
        for collection in (rows, evaluated):
            if len(collection) != len(expected) or {r['pair_id'] for r in collection} != set(expected):
                raise ValueError(f'{root.name}: IDs differ from common test')
        systems = {r['system_id'] for r in rows}
        if len(systems) != 1 or any(r['run_id'] != root.name or r.get('split') != 'test' for r in rows):
            raise ValueError('prediction run/system/split mismatch')
        for raw in rows:
            prediction = Prediction.from_dict(raw)
            if prediction.status != 'success':
                raise ValueError('generation failed: ' + prediction.pair_id)
            if prediction.reference_text != expected[prediction.pair_id]['en_text']:
                raise ValueError('reference text mismatch')
            for field, key in [('source_audio', 'ja_audio'), ('reference_audio', 'en_audio')]:
                if Path(getattr(prediction, field)).resolve() != Path(expected[prediction.pair_id][key]).resolve():
                    raise ValueError('source/reference audio mismatch')
            if lock['audio'].get(prediction.output_audio) != sha256_file(Path(prediction.output_audio)):
                raise ValueError('evaluated WAV content has changed')
            waveform, rate = sf.read(prediction.output_audio, always_2d=True)
            if waveform.shape[1] != 1 or waveform.size == 0 or not np.isfinite(waveform).all() or not np.any(waveform):
                raise ValueError('invalid/silent generated WAV')
            if prediction.output_duration is None or not math.isclose(prediction.output_duration, len(waveform)/rate, abs_tol=1/rate):
                raise ValueError('WAV duration metadata mismatch')
        if any(r.get('evaluation_status') != 'success' or not isinstance(r.get('hypothesis_raw'), str) for r in evaluated):
            raise ValueError('ASR evaluation incomplete')
        if metrics.get('samples') != len(expected) or metrics.get('asr_success_rate') != 1 or not math.isfinite(float(metrics['bleu'])):
            raise ValueError('incomplete/nonfinite metrics')
        identity = metrics.get('evaluation_identity')
        if not identity:
            raise ValueError('missing evaluation configuration identity')
        settings.append({k: v for k, v in identity.items() if k != 'run_id'})
        system = next(iter(systems))
        if system == 's2ut':
            from ..s2ut.checkpoint import validate_checkpoint
            validate_checkpoint(root / 'checkpoints/checkpoint_last.pt')
            audit = json.loads((root / 'gradient-audit-rank-0.json').read_text(encoding='utf-8'))
            if audit.get('status') != 'PASS':
                raise ValueError('S2UT gradient audit incomplete')
        elif system == 'translatotron2':
            from ..translatotron2.engine import load_checkpoint
            load_checkpoint(root / 'checkpoints/checkpoint_last.pt')
        reports.append(dict(system_id=next(iter(systems)), run_id=root.name, samples=len(rows), status='PASS'))
    if {r['system_id'] for r in reports} != set(SYSTEMS) or len(reports) != len(SYSTEMS):
        raise ValueError('exactly four baseline systems required: ' + ', '.join(SYSTEMS))
    if any(setting != settings[0] for setting in settings[1:]):
        raise ValueError('evaluation conditions differ')
    report = dict(status='PASS', scope='pipeline completeness, not quality or paper parity', runs=reports)
    atomic_write_json(Path(output_root) / 'e2e-acceptance.json', report, overwrite=overwrite)
    return report
