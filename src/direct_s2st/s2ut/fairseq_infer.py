"""Run pinned fairseq generation and preserve numeric dataset ID alignment."""
import argparse
from ..progress import operation, track
import subprocess
import sys
import time
from pathlib import Path
from ..manifests.reader import read_common_manifest

from ..io import atomic_write_jsonl, read_jsonl, ExistingOutputError
from .multitask import read_tsv, validate_prepared
from .reduce_units import validate_units


def parse_generated(path: Path, ids: list[str], *, require_complete=True) -> dict[str, list[int]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("D-"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError("invalid fairseq D-line")
        index = int(fields[0][2:])
        if not 0 <= index < len(ids) or ids[index] in result:
            raise ValueError("duplicate or out-of-range fairseq generation ID")
        result[ids[index]] = validate_units([int(x) for x in fields[2].split()], clusters=100)
    if require_complete and set(result) != set(ids):
        raise ValueError("incomplete fairseq generation; sample IDs must match exactly")
    return result


@operation('s2ut: resumable generation shards')
def generate_shards(command, generated, split, ids, identity, *, resume=False, overwrite=False,
                    runner=None, shard_size=32, selected_ids=None):
    import math
    from ..hashing import sha256_file
    from ..preparation import Checkpoints
    runner = runner or subprocess.run
    wanted = ids if selected_ids is None else selected_ids
    if selected_ids is not None and (not 1 <= len(wanted) <= 10 or len(set(wanted)) != len(wanted) or not set(wanted) <= set(ids)):
        raise ValueError('invalid bounded S2UT preview selection')
    count = 1 if selected_ids is not None else max(1, math.ceil(len(ids)/shard_size))
    results, timings = {}, {}
    cache_identity = dict(stage='generation-v1', inputs=identity, num_shards=count)
    if selected_ids is not None:
        cache_identity['preview_sample_ids'] = selected_ids
    with Checkpoints(generated/'.checkpoints', cache_identity, resume=resume and not overwrite, overwrite=overwrite) as cache:
        for shard in track(range(count), 's2ut: generate/reuse shards'):
            key = str(shard)
            saved = cache.get(key)
            if saved is None:
                directory = cache.root/f'part-{shard:05d}'
                args = list(command)
                args[args.index('--results-path')+1] = str(directory)
                # Pinned fairseq generate uses distributed size/rank only to shard
                # its iterator; unlike training, cli_main does not spawn DDP.
                args += ['--distributed-world-size', str(count), '--distributed-rank', str(shard)]
                if selected_ids is not None:
                    import json
                    numeric = {pair_id: i for i, pair_id in enumerate(ids)}
                    if args[1:3] != ['-m', 'fairseq_cli.generate']:
                        raise ValueError('preview requires the pinned fairseq generate entry point')
                    args = [args[0], '-m', 'direct_s2st.s2ut.preview_generate', '--indices',
                            json.dumps([numeric[v] for v in wanted]), '--dataset-size', str(len(ids)),
                            '--split', split, '--', *args[3:]]
                start = time.perf_counter()
                runner(args, check=True)
                seconds = time.perf_counter()-start
                path = directory/f'generate-{split}.txt'
                values = parse_generated(path, ids, require_complete=False)
                if selected_ids is not None and set(values) != set(wanted):
                    raise ValueError('preview generated unexpected or missing sample IDs')
                saved = dict(records=values, seconds=seconds, log_sha256=sha256_file(path))
                cache.record(key, saved)
                cache.flush()  # Every successful subprocess is a durable recovery unit.
            if results.keys() & saved['records'].keys():
                raise ValueError('duplicate ID across generation shards')
            results.update(saved['records'])
            timings.update({key: saved['seconds']/max(1, len(saved['records'])) for key in saved['records']})
    if results.keys() != set(wanted):
        raise ValueError('incomplete sharded generation; no samples silently skipped')
    return results, timings


@operation('s2ut/fairseq_infer: main')
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-selection', choices=('auto', 'last', 'best'), default='auto')
    parser.add_argument('--sample-ids', type=Path, help='preview only: JSON array of 1..10 IDs')
    args = parser.parse_args()
    from ..preview_selection import sample_ids, common_rows
    selected = sample_ids(args.sample_ids)
    validate_prepared(args.data_root, check_audio=selected is None)
    from ..translatotron2.monitor import select_checkpoint
    checkpoint = select_checkpoint(args.run_root, args.checkpoint_selection)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.predictions.exists() and not (args.overwrite or args.resume):
        raise ExistingOutputError(str(args.predictions))
    rows = read_tsv(args.data_root / f"{args.split}.tsv", ("id", "src_audio", "src_n_frames", "tgt_audio", "tgt_n_frames"))
    manifest = args.common_root / f'{args.split}.jsonl'
    examples, full_ids = (list(read_common_manifest(manifest)), None) if selected is None else common_rows(manifest, selected)
    common = {row["pair_id"]: row for row in examples}
    if (len(common) != len(examples) or (full_ids if selected is not None else common.keys()) != rows.keys()
            or any(row["split"] != args.split for row in examples)):
        raise ValueError("common/prepared split ID mismatch")
    generated = args.run_root / "generated" / args.split
    from ..journal import Journal
    from ..hashing import sha256_file
    identity = dict(checkpoint=sha256_file(checkpoint),
        common=sha256_file(args.common_root / f'{args.split}.jsonl'),
        data=sha256_file(args.data_root / f'{args.split}.tsv'), beam=args.beam,
        source={r['pair_id']: sha256_file(Path(r['ja_audio'])) for r in examples})
    if selected is not None:
        identity['preview_sample_ids'] = selected
    journal = Journal(args.predictions, identity, resume=args.resume, overwrite=args.overwrite)
    if journal.rows and all(r.get('status') == 'success' for r in journal.rows.values()) and set(journal.rows) == set(common):
        return
    if generated.exists() and any(generated.iterdir()) and not (args.overwrite or args.resume):
        raise ExistingOutputError(str(generated))
    command = [sys.executable, "-m", "fairseq_cli.generate", str(args.data_root),
               "--config-yaml", "config.yaml", "--multitask-config-yaml", "config_multitask.yaml",
               "--task", "speech_to_speech", "--target-is-code", "--target-code-size", "100",
               "--path", str(checkpoint), "--gen-subset", args.split,
               "--beam", str(args.beam), "--max-len-a", "1", "--batch-size", "1",
               "--results-path", str(generated)]
    if args.device == "cpu":
        command.append("--cpu")
    ids = list(rows)
    import json
    records, timings = generate_shards(command, generated, args.split, ids,
        json.loads(journal.lock.read_text()), resume=args.resume, overwrite=args.overwrite, selected_ids=selected)
    for pair_id in track(selected if selected is not None else ids, 's2ut: write predictions'):
        row = common[pair_id]
        journal.record({"pair_id": pair_id, "system_id": "s2ut", "run_id": args.run_root.name,
                            "source_audio": row["ja_audio"], "reference_audio": row["en_audio"],
                            "reference_text": row["en_text"], "split": args.split,
                            "status": "success", "error": None, "units": records[pair_id],
                            "inference_seconds": timings[pair_id],
                            "timing_scope": "generation shard wall time including model load, amortized per sample"})


if __name__ == "__main__":
    main()
