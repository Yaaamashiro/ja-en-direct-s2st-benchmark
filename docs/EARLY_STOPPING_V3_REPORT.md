# Early stopping v3 implementation report

The 2026-10-15 follow-up raised only the S2UT absolute cap to 3000. TT2 retained 1000, and the stopping criteria stayed the same. S2UT checkpoints that recorded the previous v3 cap of 1000 restore their stop state, anchors, counter, best values, and history. This report describes the initial v3 implementation, when the S2UT cap was 1000. See `COLAB.md` for current procedures.

## Stopping, selection, and update cap

| Model | Early stopping metrics | Metric for selecting checkpoint_best |
|---|---|---|
| TT2 | loss, mel_loss, phone_loss, duration_loss | Absolute minimum total dev loss |
| S2UT | loss, unit_loss, source_letter, target_letter, decoder_target_ctc | Absolute minimum dev unit_loss |

Policy version 3 uses `interval=10`, `patience=5`, and `min_relative_improvement=0.01`. At least 1% improvement over any metric's independent anchor resets the shared `bad_evaluations` counter. Only improved metrics update their anchors. Five consecutive evaluations with insufficient improvement on every metric trigger early stopping. Exactly 1% counts as improvement. A zero anchor cannot improve, so 0 to 0 does not reset patience.

During warmup, the policy records each metric's minimum dev value as its anchor and does not stop. The warmup remains 400 updates. The standard TT2/S2UT cells and v3 trainers enforce an absolute cap of 1000 optimizer updates. A run at 610 can add at most 390 updates. A run already at 1000 performs no more updates. Resetting the patience counter does not extend the cap. Vocoders retain the existing default of 10000 updates through `VOCODER_TOTAL_UPDATES`.

The implementation retains the model, loss definitions and weights, optimizer, scheduler, batch, validation split and interval, and the meaning of `best_value`, `best_update`, and `best_sha256` for `checkpoint_best`.

## Persistence and migration

The last checkpoint's monitor payload and `training-monitor.json` store `stopping_metrics`, `anchors`, `bad_evaluations`, `best_values`, `best_updates`, warmup and origin, `policy_history`, and existing best-checkpoint metadata. CSV and SVG retain all train and dev loss records. Validation logs the improvement rate for each metric, marks significant improvements with `*`, and reports the shared counter and stop state on stderr.

To load a v1/v2 state, validate its history and counter under the original policy. Explicit adoption of v3 builds anchors from the minimum values in past dev history and resets the shared counter. Record the adoption point, old policy, and new policy in `policy_history`. V3 applies only to future evaluations. Preserve prior history, optimizer, best checkpoint, and stopping decisions. Explicit adoption allows a v1/v2 run that stopped early to continue. Resume with the same v3 policy restores the full state, including a previous stop.

The standard notebook requires v3. Migrate through standard Cells 1 → 5 with the same experiment, saved run, and warmup of 400. If the launcher pins an older SHA, set `TRAIN_CODE_REVISION` to the complete new published SHA. Reuse the existing Drive data and experiment without a migration cell. Only SHA-reviewed monitor and wrapper changes are applied to the pinned execution copy. Scientific settings and snapshot and recipe identifiers are retained. Modified copies and unreviewed source are rejected.

## Files changed

| File | Change |
|---|---|
| src/direct_s2st/translatotron2/monitor.py | V3 independent anchors, shared counter, best values for all metrics, migration and history checks, logging, and cap helper |
| src/direct_s2st/translatotron2/train.py | V3 cap of 1000, exit without updates at the cap, and monitor metadata |
| src/direct_s2st/s2ut/monitor.py | V3 for all five metrics, save compatibility, cap and stop guards, and retained v2 support |
| src/direct_s2st/s2ut/fairseq_train.py | Cap and monitor metadata, with direct execution exiting normally before training if the cap is already reached |
| src/direct_s2st/colab.py | Absolute cap for the requested and saved policies |
| src/direct_s2st/session_runtime.py | Resident trainer cap and no trainer launch once the cap is reached |
| scripts/colab/training_session.py | Standard-cell TT2 and S2UT cap of 1000 |
| scripts/colab/tt2_schedule_overlay.py | SHA-verified insertion of the v3 monitor and stopping logic into older pinned code |
| scripts/colab/s2ut_runtime_overlay.py | SHA-verified insertion of S2UT and common monitoring and the wrapper into older pinned code |
| notebooks/colab_training.ipynb | V3 settings and description, TT2/S2UT default of 1000, and separate vocoder default |
| docs/COLAB.md | Stopping, migration, standard-cell instructions, and cap |
| DESIGN.md | V3 decisions, preserved training conditions, and safety cap |
| tests/unit/test_all_loss_stopping.py | V3 stopping, migration, restore, real optimizer, and cap tests |
| tests/unit/test_colab_runtime.py | Notebook v3, cap, and retained vocoder defaults |
| tests/unit/test_tt2_schedule_overlay.py | Older TT2 migration to v3 with v1/v2 history retained |
| tests/unit/test_s2ut_runtime_overlay.py | Older S2UT migration to v3 with other source and config retained |
| docs/EARLY_STOPPING_V3_REPORT.md | This report |

## Added tests

The new `test_all_loss_stopping.py` has 31 cases including parametrized tests.

1. Improvement in one metric updates only its anchor (TT2/S2UT).
2. Five evaluations with every loss improving by less than 1% cause a stop (TT2/S2UT).
3. Improvement in a different loss after two insufficient evaluations resets the shared counter (TT2/S2UT).
4. An auxiliary improvement resets the counter without replacing the best checkpoint unless the selection metric sets a new best (TT2/S2UT).
5. JSON and checkpoint payload round trips preserve anchors, counter, best values, and update counts. Altered states are rejected (TT2/S2UT).
6. Migration from stopped v1/v2 preserves history and best checkpoint bytes and resets the counter (each model and both old versions).
7. Use each warmup minimum, even when the last observation is higher (TT2/S2UT).
8. Check the exact 1% threshold and zero anchor (TT2/S2UT).
9. CSV/SVG retain all train and dev metrics (TT2/S2UT).
10. The absolute v3 cap preserves independent vocoder and earlier policy conditions (TT2/S2UT).
11. A CPU fairseq hook fixture and real Adam verify S2UT optimizer preservation and prevent update 1001.
12. A real TT2 smoke model and checkpoint verify optimizer, best checkpoint, and fingerprint preservation and the 1000 cap.
13. Restoring a verified snapshot at 1000 skips trainer startup (TT2/S2UT).
14. V3 adopted at 610 starts counting evaluations at 620 (TT2/S2UT).
15. Repeated auxiliary improvements cannot extend the 1000 cap (TT2/S2UT).
16. The standalone S2UT wrapper exits at 1000 before starting the model, optimizer, or empty gradient audit.

## Verification and limitations

The final CPU run reported 495 passed, 2 skipped, and 4 GPU tests deselected. Command: `.venv313/Scripts/python.exe -m pytest -m "not gpu" -q -p no:cacheprovider`. Modified Python modules and notebook code cells passed Python 3.10 syntax checks. CPU tests verified migration from old checkpoints, optimizer and best-checkpoint preservation, exact state restoration, and the update cap boundary. Long Fisher training on an actual A100 and recovery after a real forced Colab disconnection were outside this local verification.
