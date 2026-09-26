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


def parse_generated(path: Path, ids: list[str]) -> dict[str, list[int]]:
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
    if set(result) != set(ids):
        raise ValueError("incomplete fairseq generation; sample IDs must match exactly")
    return result


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
    args = parser.parse_args()
    validate_prepared(args.data_root)
    checkpoint = args.run_root / "checkpoints" / "checkpoint_last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.predictions.exists() and not (args.overwrite or args.resume):
        raise ExistingOutputError(str(args.predictions))
    rows = read_tsv(args.data_root / f"{args.split}.tsv", ("id", "src_audio", "src_n_frames", "tgt_audio", "tgt_n_frames"))
    common_rows = list(read_common_manifest(args.common_root / f"{args.split}.jsonl"))
    common = {row["pair_id"]: row for row in common_rows}
    if len(common) != len(common_rows) or common.keys() != rows.keys() or any(row["split"] != args.split for row in common_rows):
        raise ValueError("common/prepared split ID mismatch")
    generated = args.run_root / "generated" / args.split
    from ..journal import Journal
    from ..hashing import sha256_file
    journal = Journal(args.predictions, dict(checkpoint=sha256_file(checkpoint),
        common=sha256_file(args.common_root / f'{args.split}.jsonl'),
        data=sha256_file(args.data_root / f'{args.split}.tsv'), beam=args.beam,
        source={r['pair_id']: sha256_file(Path(r['ja_audio'])) for r in common_rows}),
        resume=args.resume, overwrite=args.overwrite)
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
    start = time.perf_counter()
    subprocess.run(command, check=True)
    seconds = time.perf_counter() - start
    records = {}
    ids = list(rows)
    for line in (generated / f'generate-{args.split}.txt').read_text(encoding='utf-8').splitlines():
        if not line.startswith('D-'):
            continue
        fields = line.split('\t')
        index = int(fields[0][2:])
        if not 0 <= index < len(ids) or ids[index] in records:
            raise ValueError('duplicate/out-of-range generated IDs')
        try:
            if len(fields) != 3:
                raise ValueError('malformed generated record')
            records[ids[index]] = dict(status='success', error=None,
                units=validate_units([int(x) for x in fields[2].split()], clusters=100))
        except ValueError as error:
            records[ids[index]] = dict(status='failed', error=str(error), units=None)
    for pair_id in track(ids, 's2ut: write predictions'):
        row = common[pair_id]
        journal.record({"pair_id": pair_id, "system_id": "s2ut", "run_id": args.run_root.name,
                            "source_audio": row["ja_audio"], "reference_audio": row["en_audio"],
                            "reference_text": row["en_text"], "split": args.split,
                            **records.get(pair_id, dict(status='failed', error='missing generated record', units=None)),
                            "inference_seconds": seconds / len(ids),
                            "timing_scope": "generation batch wall time including model load, amortized per sample"})


if __name__ == "__main__":
    main()
