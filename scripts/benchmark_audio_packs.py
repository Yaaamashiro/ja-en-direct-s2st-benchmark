"""Measure real local WAV packing; this does not predict DriveFS performance."""
import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import threading
import time
import wave

from direct_s2st.drive_staging import raw_sha, stage_audio


def measure(root, common, workers, repeat):
    local = root / f'vm-{workers}-{repeat}'
    packs = root / f'packs-{workers}-{repeat}'
    stop = threading.Event()
    peak = [0]
    def monitor():
        while not stop.wait(.01):
            try:
                size = sum(p.stat().st_size for p in local.rglob('*') if p.is_file())
                peak[0] = max(peak[0], size)
            except FileNotFoundError:
                pass  # A worker has just removed its owned temporary directory.
    thread = threading.Thread(target=monitor)
    thread.start()
    start, cpu = time.perf_counter(), os.times()
    error = None
    try:
        stage_audio(common, packs, local, splits=('train',), languages=('en',),
                    materialize=False, workers=workers)
    except Exception as failure:
        error = repr(failure)
        raise
    finally:
        elapsed = time.perf_counter() - start
        cpu_end = os.times()
        cpu_seconds = cpu_end.user + cpu_end.system - cpu.user - cpu.system
        stop.set()
        thread.join()
    archives = list(packs.rglob('*.zip'))
    size = sum(p.stat().st_size for p in archives)
    return dict(workers=workers, repeat=repeat, seconds=elapsed, packs=len(archives),
                packs_per_sec=len(archives)/elapsed, MiB_per_sec=size/2**20/elapsed,
                process_cpu_seconds=cpu_seconds, process_cpu_percent=100*cpu_seconds/elapsed,
                sampled_temporary_peak_bytes=peak[0], error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files', type=int, default=1024)
    parser.add_argument('--seconds', type=float, default=1.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--work', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.files < 1 or args.seconds <= 0 or args.repeats < 1:
        parser.error('positive files, seconds and repeats required')
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work, prefix='pack-benchmark-') as temporary:
        root = Path(temporary)
        common = root/'common'
        common.mkdir()
        originals = root/'originals'
        originals.mkdir()
        rows = []
        for i in range(args.files):
            path = originals / f'{i:06d}.wav'
            with wave.open(str(path), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(bytes(round(args.seconds*16000)*2))
            rows.append(dict(pair_id=f'p{i}', en_audio=str(path), en_sha256=raw_sha(path)))
        (common/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        results = []
        for repeat in range(args.repeats):
            # Rotate order to reduce warm-cache/order bias. All sources are local.
            order = [1, 2, 4]
            order = order[repeat % 3:] + order[:repeat % 3]
            for workers in order:
                results.append(measure(root, common, workers, repeat))
        summary = []
        for workers in (1, 2, 4):
            selected = [r for r in results if r['workers'] == workers]
            summary.append(dict(workers=workers, **{
                key: statistics.median(r[key] for r in selected) for key in (
                    'seconds', 'packs_per_sec', 'MiB_per_sec', 'process_cpu_percent',
                    'sampled_temporary_peak_bytes')}))
        report = dict(scope='local filesystem only; DriveFS unmeasured', files=args.files,
                      wav_seconds=args.seconds, pack_files=128, results=results, median=summary)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
