# Japanese → English Direct S2ST Benchmark

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

## 環境

ホスト固有パスはコードや共有設定に書かず、次の環境変数を使います。

```powershell
$env:CORPUS_ROOT = 'D:\data\ja-en-direct-s2st-corpus'
$env:EXPERIMENT_DATA_ROOT = 'D:\data\ja-en-direct-s2st-benchmark'
$env:RUNS_ROOT = 'D:\runs\ja-en-direct-s2st-benchmark'
$env:CACHE_ROOT = 'D:\cache\ja-en-direct-s2st-benchmark'
```

インストールと CLI 確認:

```powershell
python -m pip install -e ".[dev]"
s2st-benchmark --help
```

## 最短の処理順

```powershell
s2st-benchmark corpus import --profile smoke
s2st-benchmark corpus validate --profile smoke

s2st-benchmark s2ut extract-units --config configs/s2ut/prepare.yaml --profile smoke --resume
s2st-benchmark s2ut prepare --config configs/s2ut/prepare.yaml --profile smoke

s2st-benchmark translatotron2 phonemize --config configs/translatotron2/prepare.yaml --profile smoke --resume
s2st-benchmark translatotron2 prepare --config configs/translatotron2/prepare.yaml --profile smoke

s2st-benchmark cascade run --config configs/cascade/default.yaml --profile smoke --split test --resume
```

モデルやGPUをロードせず、解決される処理だけ確認する場合は `--dry-run` を付けます。
S2UT k-means はライセンスを確認したローカル artifact を
`${CACHE_ROOT}/models/s2ut/hubert_base_l6_k100.npy` に置き、
`configs/s2ut/prepare.yaml` の SHA-256 を実ファイル値へ更新してから抽出します。

学習と推論は設定内の明示的な引数配列を実行します。`RUNS_ROOT/run_id` には
resolved config、環境、dataset lock、command、log、checkpoint、prediction、metrics が保存されます。

## 評価

すべての方式で同じ固定ASRと正規化を使います。生成失敗や評価失敗は削除されず、
per-sample JSONL に残ります。

```powershell
# run_id を追加した評価設定を指定
s2st-benchmark evaluate run --config my-evaluation.yaml --resume

# run_ids を列挙した集計設定を指定
s2st-benchmark evaluate aggregate --config my-comparison.yaml
```

BLASER の実行環境がない場合でも ASR-BLEU、音声品質、RTF は評価できます。

## テスト

```powershell
python -m pytest -q -m "not gpu"
python -m pytest -q -m gpu  # 明示実行のみ
```
