# S2T→TTS baseline

## モデルと出力

第4方式のIDは`s2t_tts`、CLIは`s2st-benchmark s2t-tts run`です。
S2UT、Translatotron 2、ASR→MT→TTS Cascadeと同じcommon/testを使います。

- S2T: `openai/whisper-large-v3`、Japanese speech → English text、`task=translate`。
  revisionは`06f233fe06e710322aca913c1bc4249a0d71fce1`。
  実装時に[Hugging Face model API](https://huggingface.co/api/models/openai/whisper-large-v3)で確認したSHAです。
- TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`、revision
  `f00cf133b78d3c2c35857faba3b1be9b98c4f971`、`Ono_Anna`、English。
- `configs/s2t_tts/default.yaml`から条件を読み込みます。`main` revisionは拒否します。
- Whisperは`do_sample=False, condition_on_prev_tokens=False`。
  既存`QwenTTS`を再利用し、`do_sample=False, subtalker_dosample=False`です。
  決定的デコード設定ですが、異なるGPU/runtime間のbitwise一致は保証しません。

既存CascadeはWhisper large-v3-turboの`transcribe` → NLLB-200 distilled 600M
→ QwenTTSのままです。Whisper loaderとjournal処理だけを共通化しています。
以前のCascade journal identityと予測形式は維持しています。

`RUNS_ROOT/s2t_tts-smoke/predictions/`（configの`run_id`で変更可能）に
`predictions.jsonl`、`audio/*.wav`、`predictions.lock.json`、`run-metadata.json`を保存します。
翻訳は`s2t_en_text`、時間は`s2t_seconds`と`tts_seconds`に残します。
その他は共通prediction形式で、同じ評価器・normalization・referenceを使用します。

## 実行

先に4つのroot環境変数を設定し、common manifestを生成・検証してください。
Docker imageは共有Cascade runtimeを使います。新しいimageは不要ですが変更後の再buildは必要です。

```bash
docker compose run --rm cascade s2t-tts run --profile smoke --split test --dry-run
docker compose run --rm cascade s2t-tts run --profile smoke --split test --limit 5
docker compose run --rm cascade s2t-tts run --profile smoke --split test --limit 5 --resume
```

Colab/nativeでは同じ引数を`python -m direct_s2st.cli`へ渡せます。
`--dry-run`はconfig・manifestパス・run ID・出力先・shard計画を表示し、モデルをロードしません。
`--limit`はshardごとの上限です。全件は省略します。
`--shard-index 0 --num-shards 2`で既存方式と同じstable shard分割ができます。
shard出力は`predictions/shard-00000-of-00002`等です。評価前に全shardを統合してください。
単体CLIは自動統合しません（既存Cascadeと同じ）。

`--resume`は同じ入力・model identity・run ID・split・limit・shard条件が必要です。
成功済み音声をSHA照合して再利用し、失敗sampleを再試行します。
成功済み音声が欠落・改変されていれば停止します。確認後に明示的な`--overwrite`で
再生成するか、新しいrun IDを使ってください。`--overwrite`は選択runのjournalを再作成します。
CUDA/OOM等のfatal errorは記録してprocessを停止します。

共通評価は`configs/evaluation/default.yaml`を複製し、`run_id: s2t_tts-smoke`を設定して実行します。
評価モデル等をS2T→TTSだけ変更しないでください。4方式まとめての計画・実行は以下です。

```bash
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --execute
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --execute --resume
```

vocoder学習ありでは25段階です。`s2t-tts`と`s2t-tts-evaluate`が追加されています。
最終acceptanceは全common/test ID・実WAV・評価条件と全4方式を要求します。
以前の3方式の個別結果は引き続き読めますが、3方式だけでは新しい完了判定に通りません。

## Colab

[通常用ノートブック](../notebooks/colab_training.ipynb)と
[テスト用ノートブック](../notebooks/colab_smoke.ipynb)に4方式推論・共通評価・比較セルがあります。
Drive mount → 固定Python 3.10 setup → corpus/common確認 → 学習またはsnapshot復元
→ direct推論/vocoder → Cascade → S2T→TTS → 共通評価 → verify/aggregateの順です。
設定・保存先・再開方法は[Colabガイド](COLAB.md)を参照してください。

`setup.py --inference`は`colab-inference.txt`を`colab310.txt`の制約下で追加します。
torchを別versionに交換しません。ffmpegも導入します。各重い処理は独立したsubprocessで
終了時にGPUメモリを解放します。S2Tには学習checkpointは不要です。

**BLASERの制約:** SONAR/fairseq2の依存は復帰したPython 3.10環境では
別途互換性確認が必要です。Colabでは既定のBLASER無効設定でASR-BLEU、ECAPA、duration/silence、
RTF等を使います。`evaluation-core.txt`はこの共通評価依存だけを分離したものです。
BLASERを有効化して黙って省略する処理は追加していません。
必要時は対応したDocker評価runtimeで全4方式を評価してください。
Dockerの既存`evaluation.txt`はSONARを引き続き含みますが、実build・BLASER実行は未検証です。

## 実装・検証報告（2026-09-26）

主な追加ファイル:

- `configs/s2t_tts/default.yaml`
- `src/direct_s2st/s2t_tts/{__init__,s2t,pipeline}.py`
- `src/direct_s2st/cascade/whisper.py`
- `src/direct_s2st/colab_compare.py`
- `requirements/{colab-inference,evaluation-core}.txt`
- `tests/unit/test_s2t_tts_pipeline.py`

CLI、SYSTEMS、acceptance、smoke suite、Colab notebook/setup/restore、Docker依存、
既存テストとREADME/DESIGN/関連文書を更新しました。Qwen実装・既存Cascadeモデル設定は変更していません。

検証コマンド:

```powershell
.venv\Scripts\python.exe -m pytest -m "not gpu" -q -p no:cacheprovider
```

Python 3.10復帰後のローカルPython 3.12.14によるCPUテストは126 passed、GPUテスト3件はdeselected。checkpoint検証をmockした
4方式acceptance、3方式拒否、S2T/TTS呼び出し順、SHA・identity不一致拒否、resume、
overwrite、shard、共通評価受入、dry-run非ロード、notebook Python構文、復元のみの動作を確認しました。
実音声E2E成功を意味しません。
25段階smoke計画の`PLAN_ONLY`成功、`git diff --check`も確認しました。

Python 3.10復帰後に`uv pip compile requirements/colab310.txt requirements/colab-inference.txt
--python-version 3.10
--python-platform x86_64-unknown-linux-gnu`で依存解決を確認しています。
これはColab実機へのinstall・import・GPU動作確認ではありません。

未検証: 実コーパスでの英語WAV生成、A100での時間/VRAM、Colab実機setupとVM切断復帰、
Docker build/run、実モデルによる4方式評価、BLASER、音声明瞭度・翻訳品質・論文数値再現。
