# Early stopping v3 実装報告

後続変更（2026-10-15）: S2UTのみ絶対上限を3000へ変更。TT2は1000、早期終了条件は維持。
旧v3の上限1000を記録したS2UT checkpointも停止状態を含めて復元し、anchor/counter/best/historyはリセットしない。
以下はv3初回実装時の報告（当時のS2UT上限は1000）。現在の運用手順はCOLAB.mdを参照。

## 判定・選択・上限

| モデル | early stoppingの監視指標 | checkpoint_bestの選択指標 |
|---|---|---|
| TT2 | loss, mel_loss, phone_loss, duration_loss | Dev loss（合計）の絶対最小値 |
| S2UT | loss, unit_loss, source_letter, target_letter, decoder_target_ctc | Dev unit_lossの絶対最小値 |

policy version=3、interval=10、patience=5、min_relative_improvement=0.01。
各指標の独立anchorから1%以上改善した指標が1つでもあれば、共有bad_evaluationsを0に戻す。
改善した指標だけanchorを更新し、その他のanchorは維持する。
全指標が改善不足の評価が5回連続した場合に停止する。境界のちょうど1%改善を含む。
anchor=0は改善不能として扱い、0から0への変化でpatienceをリセットしない。

warmup中は停止せず、指標別のwarmup内最小Dev値をanchorにする。現在のwarmup400は維持。
通常セルのTT2/S2UTおよびv3 trainerは絶対optimizer更新数1000を上限とする。
610からなら最大追加390更新、1000到達済みなら追加更新しない。
改善によるcounter resetはこのhard limitを延長しない。
Vocoderの従来既定10000更新はVOCODER_TOTAL_UPDATESへ分離して維持する。

モデル・loss定義/重み・optimizer・scheduler・batch・validation split/間隔は変更していない。
checkpoint_bestのbest_value/best_update/best_sha256の意味は変更していない。

## 保存・旧policyからの移行

last checkpointのmonitor payloadとtraining-monitor.jsonに、stopping_metrics、anchors、
bad_evaluations、best_values、best_updates、warmup/origin、policy_history、既存best情報を保存する。
CSV/SVGは既存の全train/dev loss記録を維持する。validation時のstderrに指標別の改善率、
有意な改善の指標（*）、共有counter、停止状態を出力する。

v1/v2の保存状態は従来判定で履歴とcounterを検証してから読み込む。
v3の明示採用時だけ各anchorを過去Dev履歴の指標別最小値から構築し、共有counterを0に戻す。
policy_historyへ採用時点・旧policy・新policyを保存し、未来の評価だけをv3で判定する。
過去のhistory、optimizer、best checkpointを書き換えず、過去の停止判断も再解釈しない。
旧v1/v2で早期終了済みの場合も明示採用時は続行できるが、同じv3で停止済みなら停止を継承する。
同じv3のresumeではanchor/counter/bestをリセットせず完全復元する。

通常ノートブックはv3を要求する。同じ実験名・保存run・warmup400で通常セル1→5から移行する。
launcherの固定SHAが旧版なら、公開後の新しい完全SHAをTRAIN_CODE_REVISIONへ指定する。
データ再生成・新しい実験名・追加の移行セルは不要。既存Driveデータは今回変更していない。
旧コードの実行コピーはSHAで確認したmonitor/監視wrapperの範囲だけを変更し、科学的設定と
snapshot/recipeの識別子を保持する。変更済みコピー・未確認ソースは上書きせず拒否する。

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| src/direct_s2st/translatotron2/monitor.py | v3独立anchor・共有counter・全指標best・移行/履歴検証・ログ・上限helper |
| src/direct_s2st/translatotron2/train.py | v3上限1000・到達済み無更新終了・監視metadata |
| src/direct_s2st/s2ut/monitor.py | 全5指標v3・保存互換・上限/停止済み無更新guard・従来v2維持 |
| src/direct_s2st/s2ut/fairseq_train.py | 上限/監視metadata・到達済みの直接実行を学習前に正常終了 |
| src/direct_s2st/colab.py | 要求policy/保存policyの絶対上限を適用 |
| src/direct_s2st/session_runtime.py | 常駐trainerの上限・到達済みtrainer非起動 |
| scripts/colab/training_session.py | 通常セルのTT2/S2UT上限1000 |
| scripts/colab/tt2_schedule_overlay.py | 旧固定コードへv3 monitorと終了判定をSHA確認して組込み |
| scripts/colab/s2ut_runtime_overlay.py | 旧固定コードへS2UT/共通monitor・監視wrapperをSHA確認して組込み |
| notebooks/colab_training.ipynb | v3の設定/説明・TT2/S2UT既定1000・Vocoder既定分離 |
| docs/COLAB.md | 判定・移行・通常セル手順・上限の説明 |
| DESIGN.md | v3判定・保持する学習条件・安全上限の仕様 |
| tests/unit/test_all_loss_stopping.py | 新規v3判定/移行/復元/実optimizer/上限テスト |
| tests/unit/test_colab_runtime.py | ノートブックのv3/上限/既存Vocoder既定を検証 |
| tests/unit/test_tt2_schedule_overlay.py | 旧TT2へv3を組込み・v1/v2履歴を保持するテスト |
| tests/unit/test_s2ut_runtime_overlay.py | 旧S2UTへv3を組込み・監視以外のsource/config保持を検証 |
| docs/EARLY_STOPPING_V3_REPORT.md | 本報告 |

## 追加テスト一覧

新規test_all_loss_stopping.pyは、パラメータ化を含め31ケース。

1. 1つだけ改善し、そのanchorだけ更新する（TT2/S2UT）。
2. 全lossが1%未満の改善を5回繰り返すと停止する（TT2/S2UT）。
3. 2回の改善不足後に別lossが改善すると共有counterを0に戻す（TT2/S2UT）。
4. 補助lossの改善でcounterが戻っても、選択指標がbestでなければbest checkpointを変えない（TT2/S2UT）。
5. JSON/checkpoint payload往復後にanchor/counter/best値/更新数が一致し、改変状態を拒否する（TT2/S2UT）。
6. v1/v2の終了済み状態からv3へ移行し、履歴・best bytesを保持、counterだけリセットする（各モデル×旧2版）。
7. warmup中の指標別最小値を使用し、最後の観測値で置換しない（TT2/S2UT）。
8. ちょうど1%の境界とanchor=0を検証する（TT2/S2UT）。
9. CSV/SVGに全train/dev指標が残る（TT2/S2UT）。
10. 絶対上限はv3に適用し、Vocoder/旧policyの独立条件を変更しない（TT2/S2UT）。
11. S2UTのCPU用fairseq hook fixture＋実Adamで、移行時のoptimizer保持と1001更新目の禁止を検証する。
12. TT2の実smokeモデル/checkpointで、移行時のoptimizer/best/fingerprint保持と1000上限を検証する。
13. 確認済み1000のsnapshotを復元してもtrainerを起動しない（TT2/S2UT）。
14. 610で採用したv3は620から評価を数え、過去を遡らない（TT2/S2UT）。
15. 補助lossが繰り返し改善しても1000の上限は延びない（TT2/S2UT）。
16. S2UTの直接wrapperも1000到達済みならmodel/optimizer/空gradient auditを起動しない。

## 検証結果・限界

最終全CPUテスト: 495 passed, 2 skipped, 4 GPU tests deselected。
実行: `.venv313/Scripts/python.exe -m pytest -m "not gpu" -q -p no:cacheprovider`。
変更したPythonモジュールとノートブックのコードセルはPython 3.10構文チェックも成功。
旧checkpointからの移行・optimizer保持・best保持・状態完全復元・上限の境界をCPUで検証した。
実A100上での長時間Fisher学習・Colab強制切断後の実機復元は、このローカル環境では未実施。
