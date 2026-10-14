"""Verify and publish unpublished local S2UT progress before ordinary Drive restore."""
import argparse
import json
from pathlib import Path
import tempfile

from direct_s2st.progress import operation


@operation('cell 5: verify unpublished local S2UT checkpoint')
def rescue(config, work, backup, total):
    from direct_s2st.colab import latest, publish, safe_relative
    from direct_s2st.hashing import sha256_file
    from direct_s2st.recovery import freeze_files
    from direct_s2st.s2ut.checkpoint import validate_checkpoint
    work, backup = Path(work).resolve(), Path(backup).resolve()
    if config.get('kind') != 's2ut' or work == Path(work.anchor) or work.is_relative_to(backup) or backup.is_relative_to(work):
        raise ValueError('local rescue requires a separate dedicated S2UT work directory')
    path = work/safe_relative(config['checkpoint'])
    if not path.is_file():
        return None
    from preflight_s2ut_resume import selected, runtime_environment
    config = selected(config, backup)
    inputs = {p: sha256_file(Path(p)) for p in config['identity_files']}
    identity = dict(config=config, work=str(work), inputs=inputs)
    previous = latest(backup, identity)
    if previous is None:
        raise ValueError('unpublished local S2UT state has no verified durable run identity; preserve and inspect it')
    local = validate_checkpoint(path)['num_updates']
    if local <= previous[1]['updates']:
        return None
    if local > total:
        raise ValueError('local update is beyond the requested training limit')
    checksum = sha256_file(path)
    from direct_s2st.s2ut.fairseq_train import prepare_research_metadata
    mapping = dict(python='wrapper', run_root=str(work), checkpoint=str(path), updates=str(total))
    command = ['wrapper']+[p.format(**mapping) for p in config['command'][3:]]
    command += ['--restore-file', str(path)]
    import os
    os.environ.update(runtime_environment(config))
    _, (_, metadata) = prepare_research_metadata(command, check_audio=False)
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('extra_state', {}).get('s2st_recipe_fingerprint') != metadata['training_fingerprint']:
        raise ValueError('local S2UT checkpoint recipe fingerprint mismatch')
    saved = state.get('extra_state', {}).get('s2st_monitor')
    report = work/'training-monitor.json'
    if saved is None or not report.is_file() or json.loads(report.read_text()) != saved:
        raise ValueError('local S2UT checkpoint/report alignment mismatch')
    if sha256_file(path) != checksum:
        raise ValueError('local checkpoint changed during verification; stop the existing trainer first')
    retained = Path(tempfile.mkdtemp(prefix='.pending-local-s2ut-', dir=work.parent))
    hashes = freeze_files(work, path, retained/'files')
    if hashes[config['checkpoint']] != checksum:
        raise ValueError('local checkpoint changed before recovery copy')
    publish(retained/'files', backup, identity, local)
    print(f'[local-rescue] verified_updates={local} durable_published=true original_local_preserved=true copy={retained}', flush=True)
    return local


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'work', 'backup'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--total', type=int, required=True)
    args = parser.parse_args()
    rescue(json.loads(args.config.read_text()), args.work, args.backup, args.total)
