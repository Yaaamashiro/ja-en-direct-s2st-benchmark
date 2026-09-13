# ja-en-direct-s2st-benchmark design

## Scope

対象は Japanese speech → English speech の一方向のみである。同一の完成済み
corpus manifest を使い、S2UT、Translatotron 2、Cascade S2ST を比較する。
コーパス生成、逆方向翻訳、UnitY、streaming、複数話者、architecture 改良、
300時間学習の自動開始は対象外とする。

## Data ownership and roots

`CORPUS_ROOT` は読み取り専用であり、unit、phoneme、Mel、fairseq data、checkpoint、
prediction を書いてはならない。派生データは `EXPERIMENT_DATA_ROOT`、run は
`RUNS_ROOT`、download/cache は `CACHE_ROOT` に分離する。入力は
`production/manifests/releases/accepted.jsonl` と `production/audio/16k/{ja,en}` である。

共通 manifest は corpus の train/dev/test をそのまま保持する。random split は行わず、
QC ASR 文を教師ラベルに使用しない。pair ID、音声存在、SHA-256、16 kHz、mono、
PCM16、duration、必須 text を検証し、入力 manifest の hash を dataset lock に残す。

## Reproducibility and recovery

長時間処理は sample または shard 単位で atomic write し、`--resume` では成果物を
検証して再利用する。既存出力は `--overwrite` なしで破壊しない。モデル、fairseq、
Python、PyTorch、CUDA、Docker、G2P、評価器を immutable revision/version/digest で
固定する。GPU OOM や device-side assert は process-fatal とし、sample 固有エラーは
failed record として保持する。

本番の前処理、学習、推論、評価はDocker Compose経由だけで実行する。`common`、
`fairseq`、`cascade`、`evaluation`を分離し、GPU処理には明示的なdevice reservationを
設定する。`full`学習はDocker実行マーカーと実際のimage digestがない場合は拒否する。

## Phases

1. Common manifest、HuBERT layer 6 / fairseq joblib k=100 unit、reduced unit、S2UT fairseq data、unit vocoder。
2. 固定版 espeak-ng の英語音素と分割非依存の固定音素辞書、80-bin Mel、Translatotron 2用派生data、Mel vocoder、Cascade。既存fairseq `s2spect2_conformer` はduration-based coreを満たさず使用しない。
3. S2UT/Translatotron 2 train/infer、共通 test 出力、共通評価と比較 report。

規模は smoke（5〜100文）、pilot（約5〜10時間）、full（約300時間）の順とする。
full は smoke/pilot 確認後、設定で明示許可された場合だけ開始する。

## Standard prediction and evaluation

各 prediction は pair/system/run ID、source/reference、output audio/duration、inference time、
RTF、status、error を持つ。失敗 sample も `status=failed` で残す。

評価は全 system で同じ test split、ASR、text normalization、reference を使う。
ASR-sacreBLEU、BLASER、source/reference に対する ECAPA 類似度、duration、silence、
clipping、ASR成功率、失敗率、RTF と段階別 runtime を保存する。BLASER が利用できない
環境でも ASR-BLEU を先に完結できなければならない。

## Completion criteria

同じ日本語 test 音声を3方式へ入力し、英語音声と共通 prediction を得られること。
direct 2方式が train split で学習・checkpoint 再開でき、Cascade が同じ test を処理し、
BLEU、BLASER、speaker similarity、RTF を同条件で比較できること。すべての設定、revision、
dataset lock が記録され、test leakage がないこと。

## S2UT main baseline

主タスクは日本語音声→英語HuBERT Base layer 6、KM100のreduced units。
特徴抽出器のrevision、layer、k-means hash/cluster数をunit-lockとdata-lockに記録する。
連続重複の削除結果は元unit列から再計算して検証する。
fairseq commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99` の
`examples/speech_to_speech/docs/direct_s2st_discrete_units.md` にあるFisher reduced設定に従い、
source_letter (encoder_layer=6, weight=8)、target_letter (encoder_layer=8, weight=8)、
decoder_target_ctc (decoder_layer=3, weight=1.6)を必須にする。
layer番号はfairseq SingleTaskConfigの1-based設定規約をそのまま使用する。
decoder CTCの内部state indexingも固定fairseqの挙動に従う。

`unicode-codepoint-v1` は日本語・英語とも1 codepoint/labelで、subword化しない。
ASCII whitespaceのみ畳み、NFKC・case foldingを行わない。受理文字判定は固定コードポイント
集合を使用する。辞書頻度はtrainからだけ数え、dev/test未知文字をerrorとする。
TTS投入textと正解textのtoken列が異なる場合はレビューを要求し、音声の再生成は行わない。
これはtext同士の一致検証であり、実音声の発話内容を保証する強制alignmentではない。

Fisherとの相違は日英文字ラベル・大小文字/句読点保持、既存base architectureの512次元を
維持していること。Fisherの256次元設定そのものの再現とは称さない。
CTCのzero_infinityはfalseとし、不可能alignmentで補助lossが黙って0になることを避ける。
train直前にID、辞書、unit範囲、CTC最小長、runtime architectureと接続層を確認する。

## Translatotron 2 implementation boundary

[原論文v5 §3](https://arxiv.org/html/2107.08661v5)のConformer encoder、LSTM linguistic decoder、
共有attention、decoder状態とattention contextを入力するduration-based NAT synthesizer、
発話全体のdurationのL2 objective、mel再構成と残差refinementが実装対象である。
Conformerの採用自体は原論文に含まれる。per-phoneme duration教師を必須としない。

固定fairseqの `s2s_conformer_translatotron2.py` はregister済みだが、Transformer言語decoderと
TTSTransformerDecoderを接続した別構造である。duration predictor/objectiveを持たない。
従来configのspeech_to_spectrogram criterionもprev_output_tokens_mtを渡しておらず不整合だった。
データ準備を再利用し、独立したnative PyTorch coreと学習・推論を追加した。
既存fairseqのduration-freeモデルは使用しない。差分はdocs/TRANSLATOTRON2.mdに記録する。
voice preservationはdisabled。speaker similarityはtarget referenceとの話者類似度であり、
voice preservation性能とは解釈しない。
2026-09-13にユーザーが「本体の実装を先に進める」と明示したため、
実装順序のみ変更し、S2UT実E2Eに先行してTT2 coreを実装・CPU検証する。
実データE2Eの完了条件自体は免除しない。

## Portable data and predictions

corpus absolute pathが存在すれば保持し、存在しない場合のみaudio/16k/{ja,en}/... suffixを
corpus_rootおよびcorpus_root/productionへ照合する。一意でなければ停止し、basename検索しない。
共通manifestにcorpus-relative音声pathも保存し、移動後はCORPUS_ROOTで再解決する。
元corpus manifestは変更しない。fairseq/unitの派生pathは実行環境依存なので、移動先で再準備する。

S2UT中間出力はunits.jsonl、Mel中間出力の契約はmels.jsonlとし、共通識別子・参照情報を保持する。
vocoderが成功してから既存Prediction schemaのpredictions.jsonlへ変換する。
unit adapterは固定fairseq CodeGeneratorの学習済みduration predictorを使い、均等展開しない。
Mel adapterは全melパラメータ・log/normalizationの完全一致を要求する。
checkpoint/config/inputのSHA-256をvocoder-lockに記録する。
generation batch wall timeのサンプル平均をS2UTのruntimeに保存するため、これを
純粋な1発話のモデル推論latencyとみなさない。Cascadeとの比較ではtiming_scopeを確認する。

実行・未実行の境界と残課題は [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) に記載する。
