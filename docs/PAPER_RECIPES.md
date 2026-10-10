# 論文寄せの通常実験（16 kHzコーパスへの適応）

通常用Colabは `paper_exact`、TT2 Fisher、BLEU用Griffin-Lim、対象コーパスのtrain音声で
学習するS2UT Code HiFi-GANを既定とする。**exactは公開された主要レシピ条件を固定する
モードであり、独立実装・別コーパスでの完全再現や論文の数値再現を意味しない。**
TT2は公開された実効バッチ、S2UTはGPUごとのtoken予算も維持する。
コーパスは変更せず、既存16 kHz/80-bin target MelとUnit抽出結果を再利用する。
smokeは小型TT2・2更新の配線確認で、品質の証拠ではない。

## 選択項目

| 設定 | 通常値 | 変更時の扱い |
| --- | --- | --- |
| REPRODUCTION_MODE | paper_exact | 負担が大きければpaper_practicalを明示選択。別runとして管理 |
| TT2_BATCH_SIZE | 8 | 自動測定OFF時の同時処理数。exactはupdate_freq=1024/batch |
| TRAIN_CALIBRATE_BATCH | True | 共通の開始前測定。S2UT exactは公式batchの収容確認のみ。試運転後、初期重みから本学習 |
| TRAIN_CALIBRATION_MAX_BATCH | 1024 | 選定対象の上限。S2UT exactには影響しない。実際の収容数ではない |
| TRAIN_VRAM_RESERVE_RATIO | 0.10 | 試運転のピークVRAMに対する安全余裕（5%以上） |
| TRAIN_CALIBRATION_STEPS | 3 | 候補ごとの実forward/backward/optimizer測定回数（3以上） |
| PRACTICAL_UPDATE_FREQ | 1 | practicalのみ使用。実効batch=物理batch×update_freq×world_size |
| TT2_VOCODER_MODE | griffin_lim | hifiganは別の比較run。Mel vocoder学習が追加で必要 |
| S2UT_VOCODER_MODE | trained | unit学習が必要。paperは公式LJSpeech配布重みの代替でありFisher学習重みではない |
| TOTAL_UPDATES | 研究者が指定 | TT2 Appendix Aの総更新数はunknown。S2UT公式目安400000は自動実行しない |
| S2UT_TOTAL_UPDATES | 400000 | S2UT専用。RUN_TRAININGとCONFIRM_TRAININGは既定OFFのまま。短期変更時は論文の学習量に未達と記録 |
| TRAIN_CALIBRATION_OBJECTIVE | throughput | 安全な候補から試運転速度で選ぶ。capacityは収まる最大数。S2UT exactは収容確認のみ |
| TRAIN_SAVE_INTERVAL | 1 | exactの1更新が長いため毎更新ローカル保存。Drive公開は別の時間間隔・容量制限 |

`paper_exact` は**本学習中**の適応バッチ分割・精度変更を禁止する。TT2は開始前の明示的な
VRAM測定で物理batchを選定し、蓄積回数とともに記録・固定する。
S2UTは開始前も追加分割・拡大禁止で、公式token-budget batchの収容確認のみ行う。OOM時にbatchや精度を黙って
変えず停止する。物理batch8はVRAM収容の保証ではない。新runで物理batchを下げても
exactの実効1024は維持できるが、microbatch-localなBatchNorm統計は同じにならない。
practicalでもFisherモデル・LR・warmup・L2は保持し、資源設定の差を記録する。

## 共通の開始前VRAM測定（Colabセル5）

TT2・S2UT・学習するMel/Unit HiFi-GANに同じ測定・保存機構を使用する。
Griffin-Limと公式学習済みvocoderは学習しないため対象外。Cascadeも既製モデルの推論のみ。
S2UT paper_exact以外では1→2→4→8…の候補を、上限またはOOM/安全余裕不足まで試し、合格した最大の同時処理数を
選ぶcapacity方式と、安全な候補の処理数/秒で選ぶthroughput方式がある。通常はthroughput。
初回optimizer確保の1ステップ目は速度採点から除き、安全性の判定には含める。
最良速度から3%以内では大きいbatchを選ぶ。最悪padding負荷での短い実測なので、
実データ全体の最大速度・GPU占有率を保証しない。
S2UT paper_practical/GANは最後の合格・不合格候補の間も二分探索で測る。TT2は実効batchを割り切れる
候補のみ（1024では2の累乗）を試す。

- 各候補は独立した新規プロセスで初期重みから試す。optimizerの状態確保も測定に含む。
  試行checkpointを本学習へロード・Drive公開しない。本学習は別プロセスでupdate 0から開始。
- TT2はtrainの最大source/target/phone長を組み合わせた保守的なpadding負荷を使用。
  合成した負荷は測定専用であり、翻訳教師ペアとして本学習には使わない。
  exactは**物理batch×蓄積回数＝1024**を維持。practicalも設定済み実効batchを維持する。
- S2UTはtrainの正規token-budget batchからattention負荷・最長音声・件数が大きいものを試す。
  paper_exactはmax_tokens=20000/update_freq=4のまま追加分割なしで1回の試運転を行う。
  MAX_BATCHの設定は無視し、収まらなければ停止する。recordのselected=0は未選定ではなく
  「公式batch・追加capなし」を意味する。paper_practicalのみ物理分割上限を選定・固定する。
  practicalも既存max_tokens/update_freq/サンプル順は変更しないが、分割は公式実行方式との差異として扱う。
  可変長なので常に同じ件数を同時に処理するわけではない。1024発話に変更しない。
- GANは長いtrain音声で、generator/discriminator両optimizerを測定する。勾配蓄積を新たに
  導入せず**実バッチ数**を選定するため、1更新あたりの発話数は選択値になる。
- `checkpoints/<RUN_NAME>/batch-calibration.json`にGPU/PyTorch/CUDA、各試行のピーク
  allocated/reserved/全体使用量・時間・実処理数・選択結果をatomic保存する。
  中断時は完了した候補を再利用し、未完了候補だけ再測定。本学習snapshotにも記録を含める。
- 再開時は選択値を再利用。同じGPU条件では測定不要。別GPUでは同じ選択値だけ再検査し、
  収まらなければ停止する（checkpointに合わせたbatchを勝手に変えない）。
- 学習済み旧runへ後付けしない。設定・データの変更も新run名が必要。準備済みMel/Unitは共有可。

この測定もGPU時間を使う。Colab切断時に未保存の試行が失われる場合はその候補を再実行する。
短い測定は全実行条件のOOM防止を保証しない。試行中の辞書/入力/数値エラーはOOMとみなして
隠さず停止する。本学習のOOMも自動でbatchを変えず、最後の完成snapshotを保持する。
CPU上のテストは選定・分割・保存・再開の検証であり、実GPUの測定結果ではない。

TT2は1更新分の1024件を一度にRAMへ載せず、物理batchごとに読み込み、validな
frame/phone/utterance数でlossを重み付けして累積する。部分的な勾配は保存しない。
切断時は最後のDrive上の完成snapshotから再開し、未保存の更新分は再計算する。
exactの1更新が長い場合、セッション終了猶予に収まらずその更新を失う可能性がある。

## 実際の学習条件

Colabの生成コマンドはTT2で以下を含む（物理batch8の場合）。

```text
--model-size fisher --reproduction-mode paper_exact
--learning-rate 0.0042 --warmup-updates 10000 --l2-regularization 0.000001
--batch-size 8 --update-freq 128 --vocoder-mode griffin_lim
```

S2UTは固定fairseqの `s2ut_transformer_fisher`。実コードからencoder
12層/256次元/4heads、decoder6層/256次元/8headsを確認する。

```text
--arch s2ut_transformer_fisher --lr 0.0005 --warmup-updates 10000
--warmup-init-lr 1e-7 --adam-betas (0.9,0.98) --clip-norm 10.0
--max-tokens 20000 --update-freq 4 --fp16
```

S2UT論文Appendix A・Table 4は**max tokens per GPU=20k、GPU台数=4**を明記している。
update_freq4は公式レシピが案内する4GPU相当を1プロセスで累積する設定であり、
4GPUでの実行そのものと数値的に同一だとは主張しない。
paper_exactでは追加の発話数cap、固定分割、オンライン適応、精度変更を拒否する。
旧版で分割選定したS2UT exactの記録/checkpointは新方式へ自動移植せず、新runを要求する。
token予算に基づく可変batchなので「発話数1024」とは解釈しない。
HuBERT layer6、k=100、固定model revisionとk-means SHA、reduced units、
source/target letter CEとdecoder CTCを維持する。CTCは原論文§4.2どおり、
train英語文のみで学習したSentencePiece Unigram（要求語彙1000）とする。
smokeの小コーパスでは1000 pieceを作れないため、実際の語彙数も記録する。
モデル・train文hash・SentencePiece版・SHA-256をprepared lockへ保存する。
target-letterとCTCは別辞書であり、同じ英語文から再符号化して一致を検証する。

旧4B完了データはセル5のS2UT選択時に `fairseq-unigram-v1` へ再開可能に移行する。
Unit TSV・設定を再利用し、文字補助ラベル/CTCラベルだけ生成する。元のfairseq、
Mel、HuBERT Unit、学習checkpointは変更しない。学習runはrecipe-v2の新名になる。
CTCモデルを変更した旧S2UT checkpointは継続学習しない。

## 特徴・音声復元・評価の差異

- TT2 sourceは学習・推論で16 kHz/80 bins、25 ms window、10 ms hop、125–7600 Hz、
  utterance CMVN。prepared音声やtarget Melは書き換えない。
- targetは既存metadataの16 kHz/80 bins、25 ms/10 ms、0–8000 Hz、natural-log magnitude Mel。
  原論文の24 kHz/128 bins、50 ms/12.5 ms、20–12000 Hzとの差を記録する。
- G2PはeSpeak NG 1.52.0。Google独自G2Pではない。日英合成コーパス・音素辞書は
  原論文と異なる。voice preservationは無効。
- TT2原論文§5.1のBLEU経路に合わせてGriffin-Limを実装。Slaney Mel擬似逆行列、
  magnitude power1、Hann、center、momentum0、32反復、発話別固定seed。
  Googleの反復数・位相初期値はunknownで、こちらの選択を記録する。
- HiFi-GANは代替方式として残す。比較IDにvocoderモードを含め、WAV・RTF・結果を混ぜない。
  同じ学習済みTT2を別vocoderで評価するためだけにTT2を再学習する必要はない。
- S2UT paper vocoderは公式LJSpeech HuBERT-100のcheckpoint/configをHTTPS取得し、
  固定SHA-256で検証する。duration prediction必須で、独自重みへ自動置換しない。
  初回取得にはネットワークと配布元の重み利用条件の確認が必要。
- 通常はtrainedを選び、対象コーパスの英語train音声と元のUnit列で学習する。
  KM100 embedding、duration predictor（128/kernel3/dropout0.5）、log-duration MSE重み1、
  MPD/MSD・GAN/FM/Mel損失を使用する。ただし現在の独立trainerはMel frontend、
  crop/実batch、学習率schedule等が原論文のFisher vocoder条件と同一ではない。
  配布重みの「paper」という旧モード名も完全再現を意味しない。差異はmetadataに記録する。
- 共通主指標はWhisperによるASR-sacreBLEU。Google ASRによる原論文BLEUとの数値同等性は
  保証しない。MOS評価とWaveRNNは今回未実装。speaker similarityはvoice preservation評価ではない。

モデル構造も独立PyTorch実装で、初期化・microbatch BatchNorm・dropout等の数値同等性は
未確立。LRや形状が一致しても、論文と同じ品質が出たとは主張しない。

## GPU実装・測定の境界

TT2はSpecAugmentの一様整数乱数をcheckpoint保存対象のCPU RNGで生成し、
教師強制prenetを全フレームまとめて処理する。LSTMの自己回帰とBatchNormは維持する。
旧CUDA RNG/逐次dropoutと乱数列が異なるためengine-v2をrun fingerprintへ含め、
旧runには混ぜない。勾配/parameter異常チェックは維持して同期を集約する。
S2UTの勾配auditもGPU上で集約し、直近128 lossだけ保持する（全lossはfairseqログ）。

HuBERTは必要なL6より後の層を実行しない。前方層のhidden stateは変えず、
CPUの小型モデルでは削減前後の完全一致を検証する。最大256件/推定128MiBの先読みで
同じ音声長をまとめ、Unit順・paddingなしを維持する。長さが全て異なる場合はbatch1のまま。

performanceログはVRAM・処理数/秒・データ待ち時間と、取得可能ならnvidia-smiのGPU使用率
サンプルを表示する。取得不可はnullで、0%とは扱わない。GPU使用率は瞬間のhardwareサンプルであり
更新全体の平均ではない。A100実測・長時間OOM・品質の確認はColabで別途必要。

## 保存・既存実験

`research-metadata.json` は実際のmodel寸法、optimizer、precision、batch、特徴設定、
辞書/G2P設定、データfingerprint、repository revision、Unit artifact、更新数を記録する。
実際のvocoder重み/config/input SHAは推論後の `vocoder-lock.json` に記録する。
Griffin-Limの重みSHAはnull（学習済み重み不要）。metadataもsnapshotに含まれる。

通常用の学習run名は `tt2-training-実験名-paper_exact-recipe-v2` 等。
古いbatch/LRのcheckpointを新レシピへ自動移植しない。prepared dataは削除しない。
同じ条件の新レシピrunは、同じ実験名・出力先・設定で再実行して再開する。
practicalへ切り替えると別runになり、学習は新規に開始する。

既存Colabは `repository-revision.txt` に古いコードが固定されている場合がある。
ノートブック更新だけではコードは更新されない。既存学習snapshotを保持したまま、
使用revisionを明示更新する移行が必要。旧runに新設定を上書きしてはならない。
本修正のpushは別途ユーザー承認後に行う。

push後、新しい完全なcommit SHAを確認してから、Colabで以下を明示実行する。
`NEW_REVISION` の値は推測しない。旧snapshotは保持されるが旧モデルの学習は移行しない。

```python
import sys
NEW_REVISION = 'ここにpush後の40桁SHA'
ensure_workspace()
run('git', '-C', REPO, 'fetch', 'origin', NEW_REVISION)
run('git', '-C', REPO, 'checkout', '--detach', NEW_REVISION)
run(sys.executable, REPO / 'scripts/colab/use_paper_recipe.py',
    '--persistent', PERSISTENT, '--revision', NEW_REVISION, '--overwrite',
    '--separate-recipe-v2-runs')
# 更新版ノートブックのセル1 → 5。初回setupは自動再構築。
# 4A/4Bが未完了なら、その工程だけ先に再開する。
```

旧paper runがある場合も `--separate-recipe-v2-runs` の明示指定で保存したまま移行できる。
この指定がなければ停止する。すでにrecipe-v2設定がある実験は別のcode/run移行が必要。
データ、旧config、旧checkpointは削除しない。旧checkpointからの学習再開はしない。

## 根拠と検証範囲

[TT2 Appendix A・§5](https://arxiv.org/html/2107.08661v5)、
[S2UT Appendix A・Table 4](https://aclanthology.org/2022.acl-long.235.pdf#page=12)、
[固定fairseq Fisherレシピ](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/speech_to_speech/docs/direct_s2st_discrete_units.md)。

CPUテストは生成コマンド、固定fairseqの実architecture関数、逐次累積と通常累積の
実Adam更新・再開一致、Griffin-Limの実音声復元と再利用、SHA固定を確認する。
Drive/Colab実データmetadata、GPU本学習、翻訳品質、MOSは未検証。
