# 実装範囲と3方式の実行手順

## 今回追加した実装

- unit／mel vocoderのtrain専用学習。固定fairseq CodeGenerator／Generatorを使い、
  MPD・MSD、least-squares GAN、feature matching、mel L1で実際にoptimizer更新する。
  unit durationは元のHuBERT列のrun lengthに対するlog(1+duration) MSEで学習する。
  推論では既存の学習済みduration predictorを使用し、等間隔展開で代用しない。
- generator、識別器、両optimizer、RNG、入力・音声hashをatomic checkpointへ保存。
  再開時に設定・元音声・unit内容を照合。dev/testでのvocoder fittingは拒否する。
- TT2は相対位置Conformer・2D subsampling・kernel32・masked BatchNorm、
  reference preset、独立source/target mel仕様、beam推論、勾配蓄積、DDP、
  rank別checkpoint再開、dev検証を実装。既定は小型CPU・2updatesである。
- S2UTは固定criterion/modelを変更せず、主loss・3補助loss・5parameter groupの
  有限かつ非ゼロの勾配を監査する実行wrapperを追加した。
- TT2・S2UT・vocoder・Cascadeの失敗記録と再開、評価の入力内容照合。
  成功したサンプルを再利用する際にも出力hashを確認する。
- 同一test splitの3方式を23段階で実行し、WAV・ASR結果・評価条件・checkpointを
  確認してから比較reportを生成する。単なる終了コード0をE2E成功としない。

## 環境が決まってから実行する

`.env.example`を参考にCORPUS_ROOT、EXPERIMENT_DATA_ROOT、RUNS_ROOT、CACHE_ROOTを指定する。
corpusはread-only、派生物はbenchmark側でのみ作成する。固定Docker imageを先にbuildする。
以下はリポジトリrootから実行する。最初のコマンドは計画を表示するだけである。

```sh
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2 --execute
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2 --execute --resume
```

`--docker-context`で接続先を選択できる。remote Dockerの場合、corpus・出力・repoの
bind mount元はdaemon側にも存在する必要がある。接続先へのファイル転送は行わない。
コンテナのconfig mountを変えるなら`.env`のS2ST_CONTAINER_CONFIG_ROOTと
`--container-config-root`を同じ値にする。

既定では両vocoderをtrain splitでfitする。互換性確認済み重みを使用する場合は
`--vocoder-mode pretrained`を指定し、configs/vocoder/{unit,mel}.yamlのcheckpoint／configを設定する。
学習したvocoderを使う場合、生成されるconfigはrun配下のvocoder-unit／vocoder-melを参照する。
コーパスの再生成は行わない。設定済みモデルは各stageで必要に応じてdownloadされ、
unit artifactも専用stageで取得する。未指定の代替モデルへ切り替えることはない。

生成設定はconfigs/local/<name>、各方式の成果物はRUNS_ROOT/<name>-<system>に置く。
完了判定はEXPERIMENT_DATA_ROOT/results/e2e-acceptance.json、比較結果は同じresults以下。
plan、設定、環境ファイルhashが変わった再開は拒否する。成功stageは繰り返さず、失敗stageを再開する。
初回checkpoint保存前にtrainが停止した場合は自動的に既存runを消去しない。
ログを確認し、新しいname／派生data rootを使うか、個別コマンドの明示的--overwriteで再実行する。

`--device cpu`はdirect model/vocoder用で、Cascadeとevaluationは各モデル実装の設定を使う。
GPUを利用しない全パイプラインを保証するオプションではない。
大規模・multi-GPU学習の性能検証や自動クラウド環境構築は今回の完了判定に含めない。

## 実際に検証できたもの／できていないもの

CPU回帰テストは **92 passed, 3 deselected**。TT2 forward/backward・optimizer更新・checkpoint保存と完全一致再開、
固定fairseq HiFi-GAN生成器＋公式period識別器のoptimizer更新、duration教師、
失敗再開・hash拒否・無音WAVの完了拒否・23段階計画を確認した。
CPU Glooの実2プロセス学習とrank別再開の完全一致も確認した。
GANのCPUテストはメモリを抑えるため1個のperiod識別器を使う。productionのMPD/MSD全体、
CodeGeneratorの実音声学習、固定fairseq frontend、Linuxコンテナ統合は未実行である。
テスト用合成テンソルや制御したEOS出力を実音声E2E成功と扱わない。

このworkspaceには実行可能なDocker／対象corpus／学習済み重みが揃っていないため、
3方式の実音声E2E成功、収束、音声明瞭度、BLEU結果は未確認である。
2updatesは実行経路の確認用であり、TT2がEOSを出すことやvocoderの品質を保証しない。
EOS未到達や出力上限超過は失敗として記録し、強制音素・無音・random出力で補わない。
原論文との対応と数値未検証箇所は[REFERENCE_PARITY.md](REFERENCE_PARITY.md)に残す。
独立実装であり原著者の完全再現とは主張しない。

## 外部コードとライセンス

生成器は変更していない固定fairseq commit
`3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`を使用する。
識別器／lossは[公式HiFi-GANの固定ソース](https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/models.py)
から必要部分を取り込み、importのみ調整した。MITライセンス全文は
src/direct_s2st/vocoders/HIFIGAN_LICENSEに同梱している。
学習レシピはsingle-sample segment、固定LR AdamW、mel weight45・duration weight1の
benchmark実装であり、元の全training scheduleとの一致を主張しない。
torch／torchaudio 2.7.1のpinは変更していない。weight_normの非推奨警告は互換性維持のため残す。
