# Japanese → English Direct S2ST Benchmark

**実装・検証途中です。3方式の実E2E完了はまだ確認していません。**
S2UTは3つの補助タスクを必須にし、推論・vocoder adapterを追加しました。
Translatotron 2のduration-based coreと学習・mel推論を実験的に実装しました。
CPUでloss・勾配・checkpoint再開を検証済みですが、実音声E2Eは未実施です。
[本体の構成と原論文との差分](docs/TRANSLATOTRON2.md)を参照してください。
環境は後から指定できます。[実行環境の設定とsmoke手順](docs/REPRODUCTION.md)と
[監査・検証報告](docs/IMPLEMENTATION_STATUS.md)を参照してください。
unit／mel vocoder学習と3方式の一括実行・再開・完了判定も実装しています。
[3方式の実行手順と未検証範囲](docs/PIPELINE_COMPLETION.md)を参照してください。

同一コーパス・同一 test split・同一評価条件で、次の3系統を比較する実験基盤です。

- S2UT: 日本語音声 → 英語離散unit → unit vocoder
- Translatotron 2: 日本語音声 → 英語音素/内部状態 → Mel → Mel vocoder
- Cascade: 日本語ASR → 日英MT → 英語TTS

コーパス生成はこのリポジトリでは行いません。外部の
`ja-en-direct-s2st-corpus` が生成した `accepted.jsonl` と16 kHz音声を読み取り、
外部コーパスを変更せずに派生データを作ります。詳細な不変条件は
[DESIGN.md](DESIGN.md) を参照してください。

## 安全な既定値

- 既定 profile は `smoke`。`full` 学習には設定内の `confirm_full: true` が必要です。
- 既存出力は `--overwrite` なしで置換しません。
- 長時間処理は `--resume`、stable shard、atomic write に対応します。
- model ID、revision、dataset/artifact checksum を run metadata に保存します。
- 音声、特徴量、checkpoint、cache、推論結果は Git 対象外です。

## 本番環境（Dockerのみ）

本番の前処理・学習・推論・評価はホストのPythonから実行せず、必ず
`docker compose run` を使います。clone時はfairseq submoduleも取得します。

```powershell
git submodule update --init --recursive

$env:CORPUS_ROOT = 'D:\data\ja-en-direct-s2st-corpus'
$env:EXPERIMENT_DATA_ROOT = 'D:\data\ja-en-direct-s2st-benchmark'
$env:RUNS_ROOT = 'D:\runs\ja-en-direct-s2st-benchmark'
$env:CACHE_ROOT = 'D:\cache\ja-en-direct-s2st-benchmark'

docker compose build common fairseq
docker compose build cascade evaluation
```

`cascade` と `evaluation` はローカルのfairseq imageを基底にするため、上記の順で
buildします。GPUサービスはNVIDIA Container Toolkitを前提とします。

HuBERT layer 6 / KM100の公式fairseq joblib artifactは、コンテナ内の明示コマンドで
SHA-256検証付きダウンロードを行います。

```powershell
docker compose run --rm fairseq s2ut fetch-artifacts --profile smoke
```

本番データを処理する前に、fairseq公式checkpointと実音声1件を指定し、同じ音声から
Transformers経路とfairseq公式経路が完全に同じunit列を出すことをGPUテストします。

```powershell
$env:FAIRSEQ_HUBERT_CHECKPOINT = '/cache/models/s2ut/hubert_base_ls960.pt'
$env:S2ST_KMEANS_ARTIFACT = '/cache/models/s2ut/hubert_base_l6_k100.bin'
$env:S2ST_PARITY_AUDIO = '/corpus/production/audio/16k/en/example.wav'
docker compose run --rm --entrypoint python3 fairseq -m pytest -q -m gpu tests/integration/test_hubert_fairseq_parity.py
```

## 本番前の5文スモーク順序

```powershell
docker compose run --rm common corpus import --profile smoke --limit 5
docker compose run --rm common corpus validate --profile smoke

docker compose run --rm fairseq s2ut extract-units --profile smoke --limit 5 --resume
docker compose run --rm fairseq s2ut prepare --profile smoke

docker compose run --rm fairseq s2ut train --profile smoke
docker compose run --rm fairseq s2ut train --profile smoke --resume

docker compose run --rm cascade cascade run --profile smoke --split test --limit 5 --resume
```

Translatotron 2は既定で縮小モデル・CPU・2updatesです。実音声frontendには固定fairseq環境が必要です。
S2UTについても実checkpointを用いた一連のGPU検証は未実施です。
新しい `scripts/smoke/run.py` は最大100件/split・最大10updatesに制限し、
実行環境とcorpus保存先を指定した後にprepare→train→infer→vocode→evaluateを実行します。
GPU実機ですべて成功するまで`pilot`や`full`へ進めません。

モデルやGPUをロードせず、解決される処理だけ確認する場合は `--dry-run` を付けます。
S2UT k-meansは `${CACHE_ROOT}/models/s2ut/hubert_base_l6_k100.bin` に保存され、
設定済みSHA-256と一致しないartifactは拒否されます。

`full` は既定では開始されません。smoke/pilot確認後、使用imageのIDを明示してから
Docker内でのみ開始できます。

```powershell
$env:S2ST_DOCKER_IMAGE_DIGEST = docker image inspect --format '{{.Id}}' ja-en-direct-s2st-benchmark-fairseq:locked
docker compose run --rm fairseq s2ut train --profile full
```

学習と推論は設定内の明示的な引数配列を実行します。`RUNS_ROOT/run_id` には
resolved config、環境、dataset lock、command、log、checkpoint、prediction、metrics が保存されます。

## 評価

すべての方式で同じ固定ASRと正規化を使います。生成失敗や評価失敗は削除されず、
per-sample JSONL に残ります。

```powershell
# run_id を追加した評価設定を指定
docker compose run --rm evaluation evaluate run --config my-evaluation.yaml --resume

# run_ids を列挙した集計設定を指定
docker compose run --rm evaluation evaluate aggregate --config my-comparison.yaml
```

BLASER の実行環境がない場合でも ASR-BLEU、音声品質、RTF は評価できます。

## テスト

TT2のreference preset、分散学習、frontend仕様と検証範囲は
[REFERENCE_PARITY.md](docs/REFERENCE_PARITY.md)を参照してください。

ホストPythonは開発時のunit testにだけ使用します。本番データ処理には使用しません。

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q -m "not gpu"
python -m pytest -q -m gpu  # 明示実行のみ
```

S2UTの新しい補助タスクは `ja_text` / `en_text` のUnicode文字単位です。
ASCII whitespaceは `<space>` に畳み、それ以外は大小文字・句読点・結合文字を保持します。
辞書はtrainのみで作成し、未知文字・空列・ID欠落・TTS textとの内容差を拒否します。
`source_letter` / `target_letter` / `decoder_target_ctc` の接続層とloss weightは
パッケージ内 `src/direct_s2st/s2ut/multitask.yaml` に明示し、data-lockにも保存します。
`configs/s2ut/prepare.yaml` の `multitask` マッピングで全3タスクを指定して上書きできます。
