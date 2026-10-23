# ja-en-direct-s2st-benchmark design

## Scope

対象は Japanese speech → English speech の一方向のみである。同一の完成済み
corpus manifest を使い、S2UT、Translatotron 2、ASR→MT→TTS Cascade、S2T→TTS Cascade を比較する。
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

学習済みS2UTだけの正式test評価は4方式acceptanceから分離し、通常セル6/7/8で
保存済み実run・dev unit-loss best・元の依存lockを検証して実行する。
beam10→公式SHA固定LJSpeech Unit vocoder→固定英語Whisper→sacreBLEUとし、
test全IDを保持し、生成/ASR失敗は空仮説としてcorpus BLEUの母数から除外しない。
train/devによるモデル選択を変えず、testによるcheckpoint選択・再学習は行わない。
Unit shard・32件WAV ZIP・ASR journalをDriveへatomic保存し、切断後は検証再利用する。
WAVの個別大量配置を避け、ID/ZIP/member/SHA index・ASR文を永続化する。
通常セル9の保存音声再生はモデル・GPUを使わず、元コーパスを変更しない。
この経路ではTT2・自前vocoder・train音声・BLASER・話者類似度は不要である。

カスケード単独test評価も通常セル6/7/8/9で選択する。学習runを復元せず、固定モデルの
日本語ASR→日英MT→英語TTSを別プロセスで順次実行し、参照文を推論入力へ混入しない。
段階別のjournal・WAV ZIP・実依存manifestを別のbenchmarks/cascade領域へ保存し、
既存S2UT評価を変更しない。ASR-BLEUの正規化・全件母数・評価ASRはS2UT単独経路と同じとする。
現在のQwen BF16設定は維持し、非対応GPUへの暗黙の精度/モデル切替はしない。

Qwen SDKの内部revision取り落としを避けるため、固定revisionの完全snapshotを取得・
必須ファイル検査し、SDKへローカルpathだけを渡す。再開時は保存ASR/MTを読み取り専用で
先に検証し、不足・破損なら再生成せず停止する。元journal/pointer/ZIP/timingを保持する。
CUDA/BF16対応をロード前に確認し、実モデルで固定非参照文を試験合成してWAVのmono・
有限値・非無音・durationを検証後のみ本番TTSへ進む。試験音声は評価ID/母数に含めない。
初版cascadeがTTSロード前に停止した場合だけ、レビュー済みloader差分と同一モデル/testを
確認して元plan・元依存identity・ASR/MT journalを保持した実行overlayを採用できる。
worker差分もASR/MT/identity関数のAST不変を検証し、試験合成とTTS保存検証に限定する。
採用前のTTS WAVや未公開ZIPを暗黙再利用せず、採用後の実行revisionは別receiptとreportへ残す。
既知の完全snapshot修正c741fcaの採用記録があれば、そのplan/receipt/Git系統とTTSロード・
デコード引数不変を検証して生成済みWAVを保持した移行を許可し、親receipt SHAを記録する。

Colab学習は既定で時間上限なし（SESSION_MINUTES=0 / --session-seconds=0）とし、
明示確認された目標更新数まで続行する。時間による自動一時停止は行わない。
Colab側の切断は防げないため、atomic snapshot保存・検証・再開機構は維持する。
ユーザーが正の時間予算を指定した場合だけ従来の制限を適用する。

長時間処理は sample または shard 単位で atomic write し、`--resume` では成果物を
検証して再利用する。既存出力は `--overwrite` なしで破壊しない。モデル、fairseq、
Python、PyTorch、CUDA、Docker、G2P、評価器を immutable revision/version/digest で
固定する。GPU OOM や device-side assert は原則 process-fatal とし、sample 固有エラーは
failed record として保持する。

Colabの明示的な適応microbatch学習だけは、optimizer更新前のCUDA OOMについて
RNG/buffer/勾配を戻して同一サンプルを縮小再試行できる。optimizer中のOOMと
device-side assertは再試行しない。分割はBatchNorm等の統計を変えるため既定OFFであり、
混合精度と同様に実験設定へ記録する。GANの2 optimizer更新は分割しない。

常駐S2UTのFP16は、固定fairseqがoptimizer実行前の勾配overflowを検出して
loss scaleを下げた場合だけ、同じ論理バッチを再試行する。更新数を進めず、
乱数・model/criterion buffer・勾配・loss/gradient証拠を試行前へ戻す。
低下したdynamic loss scaleは保持する。バッチ・学習率・精度は変更しない。
optimizer呼び出し後、原因不明の未更新、最小scale到達、32回の再試行上限では停止する。

S2UTはデータ取得前に実CTCのtrain/backward/dev経路を確認し、評価だけで使う依存不足を
最初のdevまで持ち越さない。旧環境の不足したeditdistanceはSHA固定wheelの別置きとし、
元venv/環境ロックを偽装・上書きしない。追加依存は別manifestで版・SHA・基礎環境を記録する。
通常セル5は新しい未公開ローカルS2UT状態を入力/recipe/optimizer/monitorまで検証し、
Driveへ救済してから復元する。dev後・異常終了後もbackupし、VM強制削除時の未保存分は保証しない。
ローカルの中間checkpointは有界に保持し、last/best・不変Drive snapshotを残す。

S2UT recipe照合では起動moduleのargv[0]を固定名へ正規化する。旧lockは内容と元の
fingerprintを検証し、argv[0]以外の引数・学習条件・入力SHAが完全一致する場合だけ
元fingerprintを継承する。既存lock/checkpointは書き換えない。通常セル5は音声ZIPの
全件取得より前に、保存状態の復元・recipe・checkpoint内fingerprintを照合する。
この先行検査だけはWAV存在確認を省き、入力取得後の通常検査・payload SHA検証は維持する。

Early stopping policy v3はTT2の4 Dev lossと旧S2UT (`legacy_short`) の5 Dev lossすべてに独立anchorを持つ。
いずれかが自身のanchorから1%以上改善すれば共有patienceを0とし、その指標だけanchorを更新。
全指標が改善不足の評価が5回連続すれば停止する。全devを10更新ごとに評価し、warmup中は停止せず、
各指標のwarmup内最小値をanchorとする。この短縮条件の絶対上限はTT2=1000、旧S2UT=3000とする。
anchor/counterの改善はこのhard limitを延長しない。各モデルの上限到達済みの再開では追加更新しない。
旧v3 S2UTの上限1000の保存状態も検証して読み込み、停止policy・anchor・counterは変更せず上限metadataのみ更新する。
旧v1/v2の履歴・optimizer・bestは先に検証し、v3採用後の未来の停止判定だけを変更する。
採用時は過去Devの指標別最小値でanchorを構築し、共有counterを0とする。旧policyでの早期終了も
この明示採用時だけ解除できる。同一v3のresumeは判定状態を完全継承する。policy_historyを保存する。
checkpoint_bestの選択はTT2=Dev total loss最小、S2UT=Dev unit_loss最小のまま、停止判定から独立である。
モデル・loss定義/重み・optimizer・scheduler・batch・validation split/間隔は変更しない。

新規S2UT `paper_fisher_v1` は独立runとして元のwarmup10000・最大400000更新を使い、
独自early stoppingを無効にする。旧runのoptimizerや停止履歴は移植・再解釈しない。
全Devの5 lossと固定dev500件の補助text BLEUを1000更新ごとに記録し、固定dev100件の
音声ASR-BLEUも学習中に1k/5k/10k/25k/50k/100k/200k/400kで測る。
生成へ正解を渡さず、失敗を分母から除外せず、2 BLEU系列を分離して保存する。
checkpoint_bestはDev unit_lossで選び、BLEUで停止/選択しない。モデル・loss・optimizer・batchは維持。
日英コーパス・LJSpeech公開vocoder・固定Whisper ASRの差異を明記する。詳細はdocs/COLAB.md。

通常Colabの全学習targetは、別プロセスの開始前VRAM測定を共通に使用する。
測定はtrainのみ、試行weights/optimizerを本学習へ持ち込まず、update 0から新規開始する。
TT2は実効batch1024を維持して物理batchを選び、GANは実batchを選ぶ。
S2UT paper_exactは論文Table 4のGPUごとの20k予算と公式1GPU蓄積4回を固定し、
VRAM収容確認だけ行う。追加分割・拡大はせず、不足時は停止する。
S2UTの物理分割選定はpaper_practicalだけに限定し、公式バッチとの差異を記録する。
試行記録は候補ごとに保存し、選定後の本学習・再開では固定する。GPU変更時は同じ
選定値の安全性だけ検査し、勝手に再選定しない。数値的同一性は保証しない。

標準の前処理、学習、推論、評価はDocker Compose経由で実行する。`common`、
`fairseq`、`cascade`、`evaluation`を分離し、GPU処理には明示的なdevice reservationを
設定する。`full`学習はDocker実行マーカーと実際のimage digestがない場合は拒否する。
Colabは別途、固定Python 3.10環境と明示的な学習確認を使うnative実行経路を持つ。

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

同じ日本語 test 音声を4方式へ入力し、英語音声と共通 prediction を得られること。
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

source/target letter CEの`unicode-codepoint-v1` は日本語・英語とも1 codepoint/label。
decoder CTCはtrain英語文だけで学習したSentencePiece Unigram要求語彙1000とし、
モデルSHA・版・train文hash・実語彙数を保存する。letterとCTCのbyte一致を要求せず、
同じ文から再符号化したCTC piece列の一致を検証する。旧CTC文字ラベルを黙って使用しない。
ASCII whitespaceのみ畳み、NFKC・case foldingを行わない。受理文字判定は固定コードポイント
集合を使用する。letter辞書は全splitの文字を登録し、頻度はtrainからだけ数える。
dev/test専用文字の頻度は0とし、正解ラベルを`<unk>`へ置換しない。
辞書の登録と重みの学習を区別し、モデルのoptimizer更新にはtrainだけを使う。
TTS投入textと正解textのtoken列が異なる場合はレビューを要求し、音声の再生成は行わない。
これはtext同士の一致検証であり、実音声の発話内容を保証する強制alignmentではない。

通常S2UTは公式Fisherの256次元を使用し、実registryのencoder12層/4heads、decoder6層/8headsを検証する。
Fisherとの相違は日英letter補助ラベル・大小文字/句読点保持とコーパスである。
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
通常Colabはpaper_exact（TT2実効batch1024、S2UT max_tokens20000/update_freq4）を既定とし、
paper_practicalは明示選択した別runとする。既存target16kHz/80-bin Melは再利用し、
sourceのみruntimeで16kHz/80-bin/125–7600Hzへ適応する。旧checkpointへ新条件を上書きしない。
BLEU用TT2はGriffin-Lim、代替HiFi-GANは別比較ID。WaveRNN/MOSは未実装。
公式S2UT Code HiFi-GANはSHA固定で取得しduration predictionを必須とする。
論文との差異・unknown値・実特徴設定・optimizer・Unit artifactはresearch metadataに記録する。
2026-09-13にユーザーが「本体の実装を先に進める」と明示したため、
実装順序のみ変更し、S2UT実E2Eに先行してTT2 coreを実装・CPU検証する。
実データE2Eの完了条件自体は免除しない。

## Portable data and predictions

TT2のColab学習は準備済みMel SHA256を識別子として、必要なWAV/Mel ZIPをオンデマンド取得し、
最大2転送/16待ちで学習順に先読みする。取得済み順の学習・未取得サンプルのスキップは禁止。
完全なZIPと各WAVのSHA256を照合してからローカルパスを公開する。実効batch・sampler・
optimizer/RNG復元の条件は変えない。旧単一巨大ZIPは最初の取得待ちを要し、S2UT/vocoderの
全件事前入力検証は維持する。原本・既存準備成果物を変更せず、取得方式はrunの学習識別子に含めない。

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

## TT2の実験的安定化recipe（2026-10-15）

ユーザーの診断・修正承認に基づき、`stabilized_seconds_v1`を新規TT2 run限定の
paper_practical条件として追加する。duration head/発話長MSEを秒単位とし、target hopで
frame座標へ換算、80ms/tokenで初期化する。LR上限0.0003/warmup400を最初から固定する。
Fisher寸法、loss係数1/10/1、Adam/L2、実効batch1024、監視/best/上限1000は維持する。
これは公式TT2の未公開実装との一致や日英翻訳品質の改善を保証しない。
旧frames方式を既定に残し、旧checkpointの新設定への自動移植・再解釈を禁止する。
prepare成果物は再利用、通常セル5のrun選択・環境/optimizer/RNG復元・VRAM測定を使用する。
S2UTとvocoderは変更しない。検証範囲と数値は別途実測結果として区別して報告する。

### 評価専用postnet較正（2026-10-17）

`stabilized_pooled_v2`は別の新規paper_practical runに限定する。v1の学習objective・
optimizer・logical batchを保持し、検証前にtrain IDのSHA256順で固定128件（不足時は全train）を
batch 4でteacher-forced eval実行する。raw MelをCPUへ一時保存してpostnetのみを再生し、
5層を順にvalid frameのpooled平均・不偏分散で較正する。dev/testによるfitは禁止。
較正と検証の終了/例外時に全学習buffer・RNG・module modeを復元する。
較正値は更新数・parameter SHA・train IDと共にnative checkpoint内へ格納する。
optimizer再開は従来のnative bufferを使い、推論のみ対応する較正値を厳密照合して適用する。
較正のない中間lastは推論を拒否し、検証済みbestを使用する。最終updateでも較正するが、
監視の評価間隔（通常10）は変更しない。
旧recipeへの自動適用や過去dev履歴の変更は禁止。通常セル5で新規/再開を選択する。
初版はsingle process/FP32限定。少数の実例による局所試験であり、128件や全コーパスでの
翻訳品質・汎化・原論文との一致を保証するものではない。

## S2T→TTS baseline

`system_id=s2t_tts`。Whisper large-v3のJapanese→English translationを使い、
既存QwenTTSを共有する。ASR→MT→TTSのturbo/transcribe/NLLBは変更しない。
configにimmutable revision・language・taskを保存し、共通journalのidentityに含める。
S2T出力は`s2t_en_text`、時間は`s2t_seconds`に記録する。
同一common/testと評価configで4方式を検証する。Colabは固定Python 3.10のnative subprocess経路を持つ。
