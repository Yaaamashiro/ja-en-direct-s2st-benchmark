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
2. 固定版 espeak-ng の英語音素と分割非依存の固定音素辞書、80-bin Mel、`s2spect2_conformer` data、Mel vocoder、Cascade。
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
