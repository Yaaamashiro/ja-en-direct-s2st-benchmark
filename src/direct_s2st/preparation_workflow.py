"""Colab preparation steps with durable completion receipts.

Fast reuse requires immutable corpus data. Receipts compare input/output file
size and mtime, not all audio bytes. RECHECK bypasses receipts and sample caches.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .io import atomic_write_json
from .journal import digest
from .preparation import file_stamp
from .progress import operation, track


def inventory(roots):
    values = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            values[str(root)] = None
            continue
        def visible_files():
            for directory, dirs, names in os.walk(root):
                dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
                for name in sorted(names):
                    if not name.startswith('.'):
                        yield Path(directory)/name
        paths = visible_files() if root.is_dir() else [root]
        for path in track(paths, 'preparation: receipt metadata'):
            if any(part.startswith('.') for part in path.relative_to(root.parent).parts):
                continue
            if path.is_file():
                values[str(path)] = file_stamp(path)
    return values


@operation('preparation: completed-stage check')
def run_stage(marker, identity, inputs, outputs, run, *, force=False):
    marker = Path(marker)
    before = inventory(inputs)
    current = inventory(outputs)
    if marker.is_file() and not force:
        prior = json.loads(marker.read_text(encoding='utf-8'))
        if prior.get('sha256') != digest(prior['receipt']):
            raise ValueError(f'corrupt preparation receipt: {marker}')
        if prior['receipt'] == dict(identity=identity, inputs=before, outputs=current):
            print(f'[preparation-stage] {marker.stem} status=reused', file=sys.stderr, flush=True)
            return
    run()
    after = inventory(inputs)
    if after != before:
        raise ValueError('preparation inputs changed during execution; completion not recorded')
    current = inventory(outputs)
    if not current or any(value is None for value in current.values()):
        raise ValueError('preparation stage did not publish expected output')
    receipt = dict(identity=identity, inputs=after, outputs=current)
    atomic_write_json(marker, dict(receipt=receipt, sha256=digest(receipt)), overwrite=True)
    print(f'[preparation-stage] {marker.stem} status=completed', file=sys.stderr, flush=True)


@operation('preparation: CPU audio ZIP publishing')
def prepare_audio_packs(data):
    """No HuBERT/GPU or all-audio local materialization; resume per ZIP."""
    from .drive_staging import stage_audio, ensure_local
    import tempfile
    data = Path(data)
    if not (data / 'common/dataset-lock.json').is_file():
        raise ValueError('先に4Aのコーパス取込を完了してください')
    local = ensure_local(Path(tempfile.gettempdir()) / 's2st-prep-inputs' / digest(str(data))[:16])
    stage_audio(data / 'common', data / '.drive-audio-packs', local,
                languages=('en', 'ja'), materialize=False)
    print('音声ZIP準備完了。GPUへ切り替え、同じ設定でセル1 → 4Bを実行してください。', flush=True)


@operation('preparation: resumable workflow')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['4a', 'audio-packs', '4b'], required=True)
    parser.add_argument('--profile', default='smoke')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    data = Path(os.environ['EXPERIMENT_DATA_ROOT']).resolve()
    corpus = Path(os.environ['CORPUS_ROOT']).resolve()
    if data.is_relative_to(corpus) or corpus.is_relative_to(data):
        raise ValueError('preparation outputs must be outside CORPUS_ROOT')
    if args.stage == 'audio-packs':
        prepare_audio_packs(data)
        return
    repository = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    common, phones = data/'common', data/'translatotron2/phonemes'
    tt2, units, s2ut = data/'translatotron2/fairseq', data/'s2ut/units', data/'s2ut/fairseq'
    accepted = corpus/'production/manifests/releases/accepted.jsonl'
    from .hashing import sha256_file
    configuration = {p.relative_to(repository).as_posix(): sha256_file(p)
                     for p in sorted((repository/'configs').rglob('*')) if p.is_file()}
    if args.stage == '4a':
        steps = [('corpus', 'import', [accepted], [common]),
                 ('corpus', 'validate', [accepted, common], [common]),
                 ('translatotron2', 'phonemize', [common], [phones]),
                 ('translatotron2', 'prepare', [common, phones], [tt2]),
                 ('translatotron2', 'validate', [tt2], [tt2])]
    else:
        if os.environ.get('S2ST_DRIVE_SAFE') == '1':
            from .drive_staging import require_audio_packs
            # Check BOTH en (HuBERT) and ja (S2UT prepare) before loading models
            # or transferring any WAV ZIPs on this GPU runtime.
            require_audio_packs(common, data / '.drive-audio-packs')
        # Downloads have their own checksum verification; keep this check on each VM.
        subprocess.run([sys.executable, '-m', 'direct_s2st.cli', 's2ut', 'fetch-artifacts',
                        '--profile', args.profile, '--resume'], check=True)
        steps = [('s2ut', 'extract-units', [common], [units]),
                 ('s2ut', 'prepare', [common, units], [s2ut]),
                 ('s2ut', 'validate', [s2ut], [s2ut])]
    for system, action, inputs, outputs in steps:
        command = [sys.executable, '-m', 'direct_s2st.cli', system, action,
                   '--profile', args.profile, '--resume']
        if args.limit is not None and action in ('import', 'extract-units'):
            command += ['--limit', str(args.limit)]
        identity = dict(revision=revision, configuration=configuration, corpus=str(corpus), command=command)
        def execute():
            if os.environ.get('S2ST_DRIVE_SAFE') == '1' and action in ('extract-units', 'prepare'):
                from .drive_staging import stage_audio, active_map, ensure_local
                import tempfile
                local = ensure_local(Path(tempfile.gettempdir()) / 's2st-prep-inputs' / digest(str(data))[:16])
                languages = ('en',) if action == 'extract-units' else (('ja',) if system == 's2ut' else ('ja', 'en'))
                rows = stage_audio(common, data / '.drive-audio-packs', local, languages=languages,
                                   create=args.stage != '4b')
                with active_map(rows, local):
                    subprocess.run(command, check=True)
            else:
                subprocess.run(command, check=True)
        run_stage(data/'.prep-checkpoints/stages'/f'{system}-{action}.json',
                  identity, inputs, outputs,
                  execute,
                  force=os.environ.get('S2ST_PREP_RECHECK') == '1')


if __name__ == '__main__':
    main()
