# Parallel audio ZIP creation

`drive_staging.stage_audio(..., workers=2)` builds ZIPs in parallel only during CPU staging with `create=True, materialize=False`. Use an integer from 1 to 4. A value of 1 runs jobs sequentially. GPU extraction and restore use one worker.

Each job handles one ZIP. At most `workers` futures are in flight. Within a ZIP, WAVs are read sequentially in manifest order and checked against their existing source stamps and SHA-256 hashes. Each worker writes a ZIP_STORED archive to a separate temporary directory on the VM. The code checks both the combined space reserved for temporary ZIPs and the normal free-space reserve.

The manifest hash, `pack_files=128`, namespace, ZIP names and member order, plan format, and completion receipt format are unchanged. Worker count is excluded from the identity. To reuse a completed ZIP, the code verifies its receipt, whole-archive hash, and manifest members without opening the original WAVs. An unfinished ZIP with a valid plan and hash can have its receipt recovered. If the ZIP is corrupt or its provenance is unknown, the code preserves it and stops.

An OS lock prevents two processes on the same VM from writing the same ZIP. The OS releases the lock on termination. Use only one CPU VM per Drive pack destination. DriveFS does not provide a distributed lock for this operation.

Worker failures and interruptions propagate to the caller and stop new submissions. Running workers receive a cancellation signal. The caller waits for them to finish before cleaning up temporary files. Published ZIPs and receipts are retained.

Progress goes to stderr every 10 seconds and at start, completion, and failure. The checked/completed counts represent ZIPs that finished verification. The log records built/reused/failed counts, ZIP/s, MiB/s, and cumulative worker seconds for ZIP creation, source reading and verification, Drive copying, readback, and reuse verification. ZIP creation already includes source reading, so summing these timings would overstate wall time. MiB/s is based on archive bytes processed and does not measure new upload speed for reused ZIPs. Heartbeats indicate liveness.

## Resume

Rerun with the same common manifest, `pack_files`, and persistent destination. Changing only the worker count reuses completed ZIPs. Keep the original WAVs, ZIPs, plans, and receipts.

A recovery test used the old `4361e55` code to complete 11 receipts and leave the 12th unpublished. After making the 24 corresponding WAVs unreadable, the test resumed with four workers. All 12 existing ZIPs retained their bytes and modification times, and the run finished with 17 ZIPs.

## Local measurements (2026-11-05)

The test used a Windows CPU, 1,024 one-second WAVs (16 kHz, mono, PCM16), and eight ZIPs with 128 WAVs each. Each row is the median of three runs. Run order was rotated, with identical inputs and a fresh destination. Temporary ZIP size was sampled every 10 ms. Process CPU utilization is process CPU seconds divided by wall seconds, so it can exceed 100% when multiple cores are used.

| Workers | Seconds | ZIP/s | MiB/s | Process CPU % | Peak temporary ZIP size (MiB) |
|---|---:|---:|---:|---:|---:|
| 1 | 3.591 | 2.228 | 8.754 | 104.3 | 3.93 |
| 2 | 2.181 | 3.668 | 14.413 | 187.4 | 7.86 |
| 4 | 1.585 | 5.046 | 19.831 | 304.9 | 15.72 |

There were zero errors on the local filesystem. The speedup on Google Drive has yet to be measured. The conservative default is two workers.

Repeat the measurement on a local filesystem.

```bash
python scripts/benchmark_audio_packs.py --files 1024 --seconds 1 --repeats 3 --work /tmp/pack-measure --output /tmp/pack-measure.json
```

All 36 tests in `tests/unit/test_audio_pack_parallel.py` and `tests/unit/test_drive_staging.py` passed. They verify byte-identical archives, plans, and receipts with 1/2/4 workers, concurrent execution, bounded futures, resume from 1 to 4 workers, propagation and cleanup for hash mismatches, copy failures, and interruptions, free-space reservation, and sequential GPU restore.

The broader regression run took 130.19 seconds and reported 638 passed, 7 skipped, 5 GPU tests excluded, and 31 failed. The same 31 tests failed on the unchanged `4361e55` package. The main causes were a missing fairseq submodule in fixture checkouts, existing source-hash overlay checks on Windows, and mock timings so short that `seconds=0`. The I/O change introduced no new failures. The two-process Windows Gloo test stalled and was excluded separately. Colab GPU behavior and DriveFS performance remain unverified.
