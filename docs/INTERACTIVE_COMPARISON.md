# 任意文から2方式のS2STを比較する（セル10）

## 手順

1. BF16対応GPUへ接続し、対応コードの通常ノートブックを開く。基本設定はセル1。
2. 同じ実験名を設定し、セル1→**セル10だけ**を実行する。セル4〜8・再学習は不要。
3. セル10の`INTERACTIVE_JA_TEXT`へ日本語1文を入力する（1〜500文字、改行不可。まず短文を推奨）。
4. `INTERACTIVE_MODEL`で`s2ut` / `tt2` / `s2t_tts`を選ぶ。
   前2つだけ保存済みrunが必要。`INTERACTIVE_RUN_NAME=''`なら既存のモデル別選択pointerを使用し、
   一意でない場合は候補名を表示して停止する。明示名も指定可能。
5. Qwen生成日本語音声、Cascadeと選択方式の英語音声・ASR文字起こし・参考BLEU・性能表を確認する。

同じ日本語WAVを両方式へ入力する。入力音声はQwen CustomVoiceで`language=Japanese`、
既存のOno_Anna、固定SHA、公式samplingと文ごとseedを用いて生成する。
その24kHz音声を既存学習入力に合わせて`scipy.signal.resample_poly(2/3)`で16kHzへ変換する。
両方のWAVと変換方法を保存し、2方式が別々の日本語音声を合成することはない。
合成音声での1例比較であり、自然な日本語音声・元のtest分布上の性能を表すものではない。

## 比較と自動参照

```
入力日本語文 → Qwen日本語TTS → 共通の16kHz WAV
                                 ├─ Cascade: Whisper日本語ASR → NLLB英訳 → Qwen英語TTS
                                 └─ 選択方式: S2UT / TT2 / Whisper S2T→Qwen英語TTS
各英語音声 → 固定英語ASR → CascadeのNLLB英訳を自動参照にしたsentence BLEU
```

**Cascade英訳は正解ではなく機械的な擬似参照。**
このBLEUはCascadeへの一致度であり、翻訳精度として順位付けできない。
Cascadeが誤訳すると、その誤訳と一致する方式の点数が高くなる。
Cascade自身のASR-BLEUは主にTTS/評価ASRの整合性を見る値であり、自己評価による翻訳精度ではない。
1文BLEUは表現の違いや文の長さにも敏感。正式test BLEUと混ぜない。

擬似参照は評価にだけ使用する。候補モデルの関数は日本語音声と保存済みモデルしか受け取らず、
Cascade ASR/MT文、入力日本語の正解文字列、英語の教師を与えない。
評価ASRも擬似参照文をプロンプトとして使用しない。

## 既存設定を変えない範囲

| 部品 | 採用する設定 | 理由・検証上の制約 |
|---|---|---|
| S2UT | 保存dev-best、既存fairseq frontend/非train変換、beam10、max_len_a=1、FP32推論 | 既存`generation_command`の評価条件を維持。公式datasetのsource-only経路を使い、dummy教師・新しい辞書・train/dev/test TSVを作らない |
| Unit音声化 | SHA照合した公式LJSpeech Code HiFi-GAN、学習済みduration predictor | 既存paper vocoderと同じ。声はLJSpeech寄りで元話者保持の評価ではない |
| TT2 | 保存dev-best/較正情報、checkpointに結び付いたsource frontend、max_phones400/max_frames3000/beam1 | 既存native推論条件を維持。EOSが出ないモデルを強制打切りで成功扱いにしない |
| Mel音声化 | Griffin–Lim、32反復、位相seed1 | 少数新規入力の明示的工学設定。原論文の未公開反復数を再現したとは主張しない。既存Mel specと逆変換を使用 |
| Cascade/S2T→TTS | 既存configのSHA固定モデル、同じ英語Qwen音声化 | 日本語Qwen入力以外のモデル設定・重みを変更しない |
| Qwen | 共通の`official_sampling_eos_v1` | sampling・EOS監視・上限・seedの差分根拠は[監査記録](CASCADE_TTS_AUDIT.md)を継承 |

推論のみでoptimizer更新・訓練はしない。選択run以外のTT2/S2UTや自前Unit/Mel runは復元しない。
元の依存環境を厳密復元し、その中で固定Qwen等のimportを確認する。
既存環境に無断で依存を追加したり、保存SHAを差し替えたりしない。

## 速度・メモリ

GPU推論は**逐次・別プロセス**。同時ロードのOOMや相互干渉を避ける。
これはユーザーが選択した実行方式であり、本セルの逐次ロードで必要なピークを測る。
全モデルを同時常駐させる既存CLI一括経路のVRAM必要量と同一視しない。
表示値は単一発話・batch1の**初回推論**で、ウォームアップ・反復測定・統計検定はない。
モデル間の順番、カーネル初期化、GPUクロック、キャッシュ、他のColab負荷に影響される。

- 翻訳・音声化秒: CascadeはASR+MT+英語TTS、候補は本体推論+音声化を合算。
  source frontendとローカル音声/feature serializationは含む。CUDAを同期して計時する。
- 共通日本語合成、プロセス起動、モデル取得/ロード、Drive保存、評価用英語ASRは上記に含めない。
- worker取得・ロード秒は別表示。HF取得が含まれる場合がある一方、親プロセスで行う
  Drive snapshot復元は含まれないため、コールドダウンロード速度の公平な比較ではない。
- 入力RTFは翻訳・音声化秒/共通日本語秒数。JSONには出力音声長で割るRTFも残す。
- RAMはworker process RSSの50msサンプリングpeak。Colab全体や別プロセスの使用量ではなく、
  短いpeakを取り逃す可能性がある。psutilがない/権限等で取れない場合は取得不可。
- VRAMは`torch.cuda.max_memory_allocated/reserved`。ロードと推論を含むピーク、
  allocated/GPU総容量の割合を表示する。CUDA context・他プロセス・Torch外の使用量は含まない。
  nvidia-smiのdevice全体使用量やGPU計算利用率と同じではない。
- 逐次ステージのメモリは**最大値**を取り、合計しない。失敗・未計測を0や高速として表示しない。

各測定にセッションIDとGPU情報を保存する。中断再開で異なるセッション/GPUの値が混ざった場合は
そのまま元の測定値として保持し、同一条件の速度比較ではない旨を画面/レポートへ表示する。
正確な性能評価には、同一GPU/入力集合・ウォームアップ・複数回反復を行う別の正式プロトコルが必要。

## 保存・再開

`<実験>/interactive/<ID>/`へplan、段階receipt、入力/出力WAV、中間Mel/Unit、ASR文、
`comparison.json`、`report.md`を保存。学習・test評価・コーパスには書き込まない。
音声は途中段階ごとにSHA照合して保存し、成功済みステージは再利用する。
失敗attemptも`history/`へ残す。破損があれば停止し、黙って音声を再生成しない。

切断後は同じ設定のセル1→10。全段階完了済みなら保存音声・過去の測定値を再表示し、
学習checkpointの再ダウンロードやGPU推論は不要。
同じrun名の学習が後から進んだ場合、既存の完了比較は保存snapshotに固定したまま表示する。
新しいsnapshotや新しい性能測定には`INTERACTIVE_REMEASURE=True`を選ぶ。
これは毎回別IDを作り、旧結果は削除しない。中断した再測定を続けるときはFalseへ戻す。
TT2等の候補が失敗しても独立なCascade結果は残り、エラーと「BLEU/性能は算出不可」を表示する。

新規依存は追加しない。既存SciPy/SoundFile/Torch/SacreBLEU/Transformers/fairseq/Qwenを使用する。
CPU/モックでsource-only入出力・日本語生成/resample・教師非入力・保存/再開・失敗分離・
メモリ集計・通常セル導線を検証する。本物の保存checkpointとQwenを使ったColab/GPU E2Eは未検証。

実機テストは`tests/gpu/test_interactive_comparison_gpu.py`に分離する。
`S2ST_INTERACTIVE_GPU_PLAN`へ検証済みS2UT用planを指定したときだけ、別の一時保存先で
実Qwen日本語合成→2方式翻訳→音声化→ASR/参考BLEUを実行する。モデル取得・CU消費があるため、
通常CPUテストでは実行しない。TT2の翻訳品質/EOS問題の解消をこの機能の実装完了条件と混同しない。

今回の検証: 新機能26件、全CPUテスト **775 passed / 2 skipped / 8 GPU tests deselected**。
Python 3.10構文互換性と`git diff --check`も確認した。実行環境はローカルPython 3.13/CPUであり、
Colab Python 3.10/GPU上の実重みE2E・実VRAM値を検証した結果ではない。
