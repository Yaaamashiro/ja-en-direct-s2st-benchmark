"""Verify a durable S2UT resume before downloading any audio/feature shards."""
import argparse
import json
import os
from pathlib import Path

from direct_s2st.progress import operation


def selected(config, backup):
    if config.get('batch_calibration'):
        from direct_s2st.batch_calibration import selected_config
        record = json.loads((Path(backup)/'batch-calibration.json').read_text(encoding='utf-8'))
        if record.get('selected') is None:
            raise ValueError('saved S2UT batch selection is missing; saved run preserved')
        config = selected_config(config, record['selected'])
    return config


def runtime_environment(config):
    settings = config.get('runtime', {})
    return dict(S2ST_TRAIN_FIXED_MICROBATCH=str(settings.get('fixed_microbatch', 0)),
                S2ST_TRAIN_ADAPTIVE_BATCH='1' if settings.get('adaptive_batch') else '0',
                S2ST_TRAIN_PRECISION=settings.get('precision', 'default'))


@operation('cell 5: verify S2UT recipe/checkpoint before input download')
def verify(config, work, backup, total):
    if config.get('kind') != 's2ut':
        raise ValueError('S2UT resume preflight requires a saved S2UT config')
    from direct_s2st.colab import run_session, safe_relative
    # Restore-only cannot optimize or download payloads. `_staged` merely avoids
    # staging here; full normal-cell staging/byte verification still follows.
    result = run_session(config, work=work, backup=backup, total=total,
                         resume=True, restore_only=True, confirm_training=True, _staged=True)
    config = selected(config, backup)
    work = Path(work).resolve()
    checkpoint = work/safe_relative(config['checkpoint'])
    mapping = dict(python='wrapper', run_root=str(work), checkpoint=str(checkpoint), updates=str(total))
    command = ['wrapper']+[part.format(**mapping) for part in config['command'][3:]]
    command += [part.format(**mapping) for part in config['resume_args']]
    os.environ.update(runtime_environment(config))
    from direct_s2st.s2ut.fairseq_train import prepare_research_metadata
    _, research = prepare_research_metadata(command, check_audio=False)
    if research is not None:
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('extra_state', {}).get('s2st_recipe_fingerprint') != research[1]['training_fingerprint']:
            raise ValueError('S2UT checkpoint recipe fingerprint mismatch before input download; saved files preserved')
    print(f'[s2ut-resume-preflight] verified_updates={result["durable_updates"]} '
          'optimizer_steps=0 audio_downloads=0 full_payload_verification_still_required=true', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'work', 'backup'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--total', type=int, required=True)
    args = parser.parse_args()
    verify(json.loads(args.config.read_text(encoding='utf-8')), args.work, args.backup, args.total)
