# 音声ZIP作成の並列化

`drive_staging.stage_audio(..., workers=2)` はCPUでの `create=True, materialize=False` のときだけZIPを並列作成する。整数1〜4を指定でき、1なら逐次実行。GPUでの展開・復元は1 workerを使用する。

1 ZIPが1 jobで、投入済みFutureは最大workers個。ZIP内のWAVはmanifest順に逐次読み込み、従来のsource stampとSHA-256を照合する。各workerはVM内の独立した一時ディレクトリでZIP_STOREDを作成する。一時ZIPの合計予約容量と通常の空き容量reserveを検査する。

manifest hash、pack_files=128、namespace、ZIP名・内部メンバー順、planと完了receiptの形式は変更しない。workersはidentityへ含めない。完成済みZIPはreceipt・ZIP全体のhash・manifest上のメンバーを照合して再利用し、原本WAVへアクセスしない。未完了ZIPは従来どおり、正しいplanとhashがあればreceiptを補完し、出所不明・破損したZIPは保持して停止する。

VM上のOS lockで、同じZIPへの同一VM内の二重書き込みを拒否する。OSがlockを解放するため、強制終了後に永続的なlockを残さない。Driveの同一pack保存先を複数のCPU VMから同時に処理しないこと。DriveFS越しの分散lockは提供しない。

workerの失敗・中断は呼び出し元へ伝播し、新しいjobの投入を停止する。実行中workerには中止を通知し、終了を待って一時ファイルを片付ける。その時点までに公開できたZIPとreceiptは保持する。

進捗は10秒間隔と開始・終了・失敗時にstderrへ出す。checked/completedは検証を完了したZIP数。built/reused/failed、ZIP/s、MiB/sと、ZIP作成・原本読み込み検証・Driveコピー・readback・再利用検証の累積worker秒を記録する。原本読み込み時間はZIP作成時間に含まれるため、時間項目を足してwall timeとして扱わない。MiB/sは処理済みarchive byteに基づき、再利用時は新規アップロード速度を意味しない。生存通知だけでは前進を示さない。

## 再開

同じcommon manifest、pack_files、persistent保存先を使って再実行する。workersだけを変更しても完成済みZIPを再作成しない。原本・既存ZIP・plan・receiptを削除する必要はない。

旧4361e55のコードで11 receipt完成・12個目のreceipt未発行を再現し、対応する24 WAVを読めない状態にした後、4 workersで残りを作成した。既存12 ZIPは内容とmtimeが一致し、合計17 ZIPになった。

## ローカル測定（2026-11-05）

Windows CPU、16 kHz/mono/PCM16の1秒WAV 1,024個、128 WAV/ZIP（8 ZIP）、各3回の中央値。順序を回転し、同じ入力と新規保存先を使用。VM一時ZIPのサイズは10 ms間隔でsampling。CPU使用率はprocess CPU秒 / wall秒で、複数core使用時は100%を超える。

| workers | 秒 | ZIP/s | MiB/s | process CPU % | 一時ZIP peak MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 3.591 | 2.228 | 8.754 | 104.3 | 3.93 |
| 2 | 2.181 | 3.668 | 14.413 | 187.4 | 7.86 |
| 4 | 1.585 | 5.046 | 19.831 | 304.9 | 15.72 |

エラー0。ローカルファイルシステムでの測定であり、Google Drive上の改善率は未検証。安全側の既定値は2。

再測定例:

```bash
python scripts/benchmark_audio_packs.py --files 1024 --seconds 1 --repeats 3 --work /tmp/pack-measure --output /tmp/pack-measure.json
```

テスト: `tests/unit/test_audio_pack_parallel.py` と既存 `tests/unit/test_drive_staging.py` の36件成功。1/2/4で全archive・plan・receiptのbyte一致、真の同時実行、bounded Future、1→4再開、hash不一致・copy障害・中断の伝播とcleanup、空き容量予約、GPU復元の非並列化を検証した。

広い回帰検査は638成功・7 skip・5 GPU除外・31失敗（130.19秒）。31件は旧4361e55の無変更packageでも同じnodeが失敗することを再確認した。主因はfixture checkoutのfairseq submodule不足、Windowsでの既存source-hash overlay検査、短すぎる模擬測定のseconds=0。今回のI/O変更で新たに失敗したテストはない。WindowsのGloo 2-process試験は待機して進まなかったため別途除外した。Colab GPU・DriveFS性能は未検証。
