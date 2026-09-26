# Japanese → English Direct S2ST Benchmark

日本語から英語への音声翻訳を、同一コーパス・同一test split・同一評価条件で比較するための実験基盤です。
データの取り込みから前処理、学習、推論、音声生成、評価までを扱います。

## 比較する方式

| 方式 | 処理経路 |
| --- | --- |
| S2UT | 日本語音声 → 英語離散unit → unit vocoder |
| Translatotron 2 | 日本語音声 → 音素・内部状態 → mel → mel vocoder |
| ASR→MT→TTS Cascade | Whisper large-v3-turbo → NLLB-200 distilled 600M → Qwen3-TTS |
| S2T→TTS Cascade | Whisper large-v3（translate）→ Qwen3-TTS |

Translatotron 2はnative PyTorchによる独立実装です。原著者の実装や学習済み重みではなく、
canonical-speaker出力を対象とし、話者性の保持は行いません。
実音声での4方式のE2E成功・翻訳品質・原論文の数値再現は未確認です。
構成と適用範囲は[モデルの説明](docs/TRANSLATOTRON2.md)を参照してください。

## 必要なもの

- 外部の `ja-en-direct-s2st-corpus` が生成した `accepted.jsonl` と16 kHz音声
- Docker Compose、およびGPU処理用のLinux・NVIDIA GPU・対応ドライバ・NVIDIA Container Toolkit
- 実行計画の作成と開発テスト用のPython 3.10以上
- 設定で固定されたモデル・artifactを取得できるネットワーク、または準備済みのcache

コーパスや学習済みcheckpointは同梱しません。コーパスの生成もこのリポジトリの対象外です。
入力コーパスは読み取り専用で扱い、特徴量などの派生物は別の保存先に作成します。
通常のデータ処理はDocker内で実行します。Colabでは専用の固定Python環境を使用します。

## Google Colabで使う

用途別のノートブックをColabへアップロードしてください。セル4Aのデータ準備まではCPUランタイムを使えます。
セル4BのHuBERT Unit抽出と学習・比較はGPUを使用します。切り替え後は同じ設定でセル1を再実行して目的のセルへ進むと、ローカル環境を自動構築します。

- [テスト用](notebooks/colab_smoke.ipynb)：各split 5件・2更新。まず動作確認します。
- [通常用](notebooks/colab_training.ipynb)：全件データで学習・4方式比較。長期学習は明示確認が必要です。

最初の設定フォームを編集して、番号順に実行します。テストと通常実験は保存先も分離します。
Dockerなしで環境構築・前処理・TT2／S2UT／vocoder学習を実行できます。
Colabの学習用Pythonは3.10.18です。旧版からは新しいランタイム・実験保存先で開始してください。
学習を短い区間に分けてGoogle Driveへcheckpointを検証付きで保存し、切断後は最後の保存点から再開します。
学習対象をtt2 / s2ut / unit / melから選び、各対象の学習後に4方式比較をONにします。
時間予算、保存頻度、再開、注意事項は[Colab手順](docs/COLAB.md)を参照してください。
80GB GPU用の調整開始値はノートブックの `PERFORMANCE='gpu80'` で選択できます。
バッチ量・CPU並列読み込み・先読み・保存間隔を指定できますが、実機での速度・最大メモリ使用量は未検証です。

## セットアップ

以下は実行先ホストで操作する例です。

```sh
git clone --recurse-submodules https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark.git
cd ja-en-direct-s2st-benchmark
python -m pip install -e .
```

取得済みのリポジトリでは `git submodule update --init --recursive` で固定fairseqを準備します。
以降のコマンドはリポジトリのルートから実行してください。

### 保存先を設定する

[.env.example](.env.example)をコピーして `.env` を作成し、次のパスを設定してください。
`.env` はGit管理対象外です。

| 変数 | 保存先 |
| --- | --- |
| `CORPUS_ROOT` | `production/manifests/releases/accepted.jsonl` と音声を含む既存コーパス |
| `EXPERIMENT_DATA_ROOT` | benchmarkの派生データ。コーパスの外にある専用ディレクトリ |
| `RUNS_ROOT` | 学習ログ、checkpoint、推論結果 |
| `CACHE_ROOT` | モデルとartifactのcache |

パスはDocker daemonが動くホスト上のものを指定します。smoke用の派生データは本学習用と分けてください。
リモート接続は実行ドライバの `--docker-context NAME` で指定できますが、ファイル転送は行いません。
詳しくは[実行環境の設定](docs/REPRODUCTION.md)を参照してください。

### Docker imageをbuildする

```sh
docker compose --env-file .env build common fairseq
docker compose --env-file .env build cascade evaluation
```

`cascade` と `evaluation` はfairseq imageを基底にするため、この順でbuildします。

## 小規模に実行する

まず4方式の実行計画を確認します。このコマンドだけではモデル処理は開始しません。

```sh
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2
```

計画を確認したら `--execute` を付けて実行します。停止した処理は同じ設定で `--resume` を付けて再開します。

```sh
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2 --execute
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2 --execute --resume
```

`--limit 5` はtrain/dev/testの各splitから最大5件を選びます。3つのsplitが必要で、
文字・音素辞書はtrainだけから作成します。入力条件の詳細は[smoke手順](docs/REPRODUCTION.md)を確認してください。

一括実行は前処理、各方式の学習・推論、unit／mel vocoderの学習、評価、比較を順に行います。
2updatesは処理経路を確認するための設定であり、意味の通る音声やE2E成功を保証しません。
学習済みvocoderの指定、個別実行、失敗時の扱いは[パイプラインの実行手順](docs/PIPELINE_COMPLETION.md)に記載しています。

## 設定と出力

- `configs/`：各方式、vocoder、評価の設定。モデルIDとimmutable revisionを固定します。
- `scripts/smoke/`：小規模実行の計画・実行・再開ドライバ。
- `src/direct_s2st/`：CLI、前処理、学習、推論、評価の実装。
- `tests/`：単体テストと統合テスト。
- `docs/`：環境構築、モデル仕様、再現条件、検証記録。

一括実行の生成設定は `configs/local/<name>`、方式別の成果物は `RUNS_ROOT/<name>-<system>` に保存します。
完了判定は `EXPERIMENT_DATA_ROOT/results/e2e-acceptance.json`、比較結果は同じ `results` 以下に出力します。
評価は共通の固定ASRと正規化を使い、失敗したサンプルも記録に残します。

既定profileは `smoke` です。`full` 学習には明示的な設定と確認が必要です。
既存出力は明示的な `--overwrite` なしでは置換せず、再開時には設定・入力・成果物を照合します。
音声、特徴量、checkpoint、cache、推論結果はGitに追加しないでください。

## 開発・テスト

```sh
python -m pip install -e ".[dev]"
python -m pytest -q -m "not gpu"
```

モデルのCPUテストには追加でtorch／torchaudio 2.7.1が必要です。
GPUテストは対応環境と指定artifactを準備して明示的に実行します。
HuBERT unit列のfairseq経路との一致確認を含む手順は[再現手順](docs/REPRODUCTION.md)を参照してください。

## ドキュメント

- [設計とデータの不変条件](DESIGN.md)
- [実行環境・artifactの準備](docs/REPRODUCTION.md)
- [4方式の実行・再開・完了判定](docs/PIPELINE_COMPLETION.md)
- [Translatotron 2のモデル構成](docs/TRANSLATOTRON2.md)
- [reference preset・分散学習・原論文との対応](docs/REFERENCE_PARITY.md)
- [実装状況と検証記録](docs/IMPLEMENTATION_STATUS.md)

S2T→TTSの実行・resume・固定revisionは[実行ガイド](docs/S2T_TTS.md)を参照してください。
Colabでも同一モデル条件で4方式を実行できます。
