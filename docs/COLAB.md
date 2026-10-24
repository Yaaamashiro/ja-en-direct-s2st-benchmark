# Google Colab：テストと通常実験

## 新規S2UT：公式Fisherレシピと学習中のBLEU推移

`S2UT_RECIPE='paper_fisher_v1'` は旧短縮runとは別の新規学習です。
warmup **10,000**、最大 **400,000 optimizer updates（絶対値）**、独自early stoppingなし。
モデル、補助タスク、loss係数、Adam、inverse-sqrt、LR 0.0005、FP16、seed 1、
max_tokens 20,000 / update_freq 4は公式レシピを維持します。
1 GPUで4 GPU分を勾配蓄積する公式の単一GPU設定であり、4 GPU実行の数値的同一性は保証しません。

根拠は[原論文](https://arxiv.org/abs/2107.05604)と
[fairseq公式レシピ](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/direct_s2st_discrete_units.md)。
Fisherではなく日英コーパス、公式配布LJSpeech vocoder（Fisherでfitしたものではない）、
固定Whisper（原論文のwav2vec2 ASRではない）を使用する点をmetadataへ明記します。
この変更で翻訳品質の改善を保証するわけではありません。

通常セル1で以下を設定し、セル5を実行します。4A/4Bの再作成は不要です。

```python
TRAIN_TARGET = 's2ut'
S2UT_RECIPE = 'paper_fisher_v1'
S2UT_WARMUP_UPDATES = 10000
S2UT_TOTAL_UPDATES = 400000
S2UT_VOCODER_MODE = 'paper'
REPRODUCTION_MODE = 'paper_exact'
TRAIN_RUN_NAME = ''
RUN_TRAINING = True
CONFIRM_TRAINING = True
SESSION_MINUTES = 0
```

初回の新recipe取得時は、push後の対応する完全SHAを`TRAIN_CODE_REVISION`へ指定してください。
未対応の古いlauncherは実行前に拒否します。
保存名は `s2ut-training-<実験名>-paper-fisher-v1`。
旧runのoptimizer/checkpoint/configは保持し、新runへ移植しません。
以後は同じ設定で**セル1→5**により保存済み環境・optimizer・評価結果を復元します。
旧runを続けるときだけ`legacy_short`、元のwarmup/モード/実保存名を選択します。
新recipeから旧runを明示選択すると不一致で停止し、update 0へ誤再開しません。

### 学習中の測定（すべてdev、testは不使用）

| 測定 | 更新数 | 内容 |
|---|---|---|
| 全Dev loss | 1,000ごと | 全件teacher-forced検証、5種類すべて保存 |
| 補助text BLEU | 1,000ごと | 固定dev最大500件、日本語音声→target-letter自由生成→英文 |
| 音声ASR-BLEU | 1k / 5k / 10k / 25k / 50k / 100k / 200k / 400k | 固定dev最大100件、日本語音声→beam10 Unit自由生成→公式Unit vocoder→英語ASR |

補助text BLEUはS2STの正式指標ではありません。2系列を混ぜずに記録します。
固定IDはseed 1とpair_idのSHA256順で選び、音声100件は500件の部分集合。
生成モデルへ正解Unit・正解テキスト・previous-output targetを渡しません。
英文正解はスコア計算専用です。生成失敗も空hypothesisとして全選択件数の分母へ含め、
failure数・エラーを保存します。Whisperは`openai/whisper-large-v3-turbo`の
`60be3615a4d667e1258e8ad29130467587c489aa`を固定し、英語transcribeで使用します。
Unit vocoderの重み/configもSHA256検証します。

評価は同じGPUで学習と交互に行うため、その間optimizer更新は待ちます。
別GPU/別Colabを要求しません。初回モデル取得の通信・VRAM・ディスク、推論/ASRの時間は追加です。
項目間で30分の時間予算を確認し、超過後の未処理例は失敗として記録します。
1件の処理・モデル取得の強制timeoutではありません。評価全体の失敗は欠測（null）として残し、
偽のBLEUや改善値は作らず、完了した学習更新をlast checkpointへ保存します。
CUDA OOM・device-side assertは例外で、安全性のためprocess-fatalとし、直前の永続snapshotから復旧します。
学習modelのmode・buffer・Python/NumPy/Torch RNGを評価後に戻し、optimizerには触れません。

**BLEUでearly stop/最良モデル選択はしません。**
bestは全Dev `unit_loss`最小の完全checkpoint、再開はlast checkpointです。
lossはtrain100更新/Dev1000更新の間引き履歴として保存し、400k分の巨大な履歴再生を避けます。
400k到達済みcheckpointから追加optimizer更新はしません。

### グラフ・保存・計算資源

Driveの `PERSISTENT/checkpoints/<実保存run名>/` に
`training-monitor.svg/.csv/.json`（loss）と `training-bleu.svg/.csv/.json`（2系列）を公開します。
小さなper-update identity・固定ID・成功/失敗hypothesis・採点signatureも検証済みsnapshotへ同梱します。
評価重みのSHAとpolicyが一致する成功例だけ再利用し、失敗例は再試行します。
ASR/vocoderモデル本体や生成WAVを毎回の学習backupへ複製しません。

`S2UT_PAPER_SAVE_INTERVAL=100`でlast checkpointを保存し、通常の非同期backup（既定600秒間隔）、
Dev評価地点・終了地点の強制backupを使います。評価中のVM強制削除は最後の永続snapshotまで戻ります。
評価途中のjournalがローカルにしかない場合も同様で、そこからの学習/評価を再実行します。
全Dev/BLEU測定地点のsnapshotは保持しますが、全更新を永久保存するものではありません。
Drive既定予算1024GBを超えると安全に停止し、自動削除・無断の予算拡大はしません。
400kは非常に長いため、保存使用量と実測の更新速度を確認してください。
短い上限で試運転した場合は原論文相当の学習量を完了したとは扱いません。

## TT2の安定化試験recipe（新規run限定）

`TT2_RECIPE='stabilized_seconds_v1'` をセル1で明示選択すると、通常セル5から
新しいTT2 runを開始する。既定の`legacy_frames`は旧checkpointの意味を維持する。
新recipeは自動的に`paper_practical`として記録され、論文完全再現とは扱わない。

- duration headの出力・発話長MSEを秒単位に統一。初期値は音素列の各音響tokenあたり80ms。
  upsampling・Gaussian幅・推論のmax_framesはフレーム単位で、実際のtarget hopから換算する。
- peak LR=0.0003、warmup=400を初回から固定。`TT2_WARMUP_UPDATES=400`を使用し、
  途中短縮のoverlayは適用しない。再開時のLR/warmup/objective変更は拒否する。
- Fisherのモデル寸法、Adam/L2、loss係数1/10/1、実効batch1024、10更新ごとの全dev評価、
  policy v3、best=Dev total loss、絶対上限1000は維持する。
- 開始前VRAM測定は新recipeそのものを使い、同時処理数×蓄積回数=1024を維持する。
  VRAM上限まで埋めることや最大速度を保証するものではない。

使用順序は、更新版ノートブックのセル1で`TRAIN_TARGET='tt2'`、
`TT2_RECIPE='stabilized_seconds_v1'`、`TT2_WARMUP_UPDATES=400`、
`TRAIN_RUN_NAME=''`、`RUN_TRAINING=True`、`CONFIRM_TRAINING=True`を設定してセル5。
既存のlauncher pinがある場合は、**push後に確定した対応40桁SHA**を
`TRAIN_CODE_REVISION`へ指定する（未pushの変更はColabから取得できない）。

保存名は`tt2-training-実験名-paper_practical-stabilized-seconds-v1`。
自動選択は同じTT2 recipe内でのみ行い、旧frame runを誤って再開しない。
旧run名を明示指定すると不一致で停止する。切断後は同じ設定でセル1→5。
4A/4B・音声・Mel・音素の再作成、旧runの削除、追加の再開セルは不要。
duration/total lossの単位・尺度が旧runと違うため、異なるrecipeのloss値を直接比較しない。
旧runを再開する場合は`legacy_frames`と元の設定・run名を使う。
S2UTおよびvocoderの学習条件は変更しない。

これはPC診断に基づく実験的な安定化案。TT2論文はdurationの単位・初期化の詳細を
確定できるほど記述していないため、秒単位を「公式実装の修正」とは主張しない。
実データ2件の短いCPU検証は、全コーパスの収束・翻訳品質やA100上の性能の証明ではない。

## 任意：評価専用のpostnet較正を使う新規TT2

セル1の`TT2_RECIPE='stabilized_pooled_v2'`は、上記v1と同じ学習objective・初期化・
LR/warmup・実効batch1024を使い、**検証・推論時だけ**postnetの統計を較正する別recipe。
既定は変更しない。モデルの重みを修復したり、phone_lossを直接改善する処理ではない。

- train IDのSHA256順で最大128件を固定選択し、batch4でteacher-forced evalのraw Melを取得。
  5層を順に、paddingを除く全frameの平均・不偏分散で較正する。dev/testはfitに使わない。
  最初のraw MelだけCPUに一時保持し、encoder/decoderを5回実行する方式にはしない。
- 検証後は学習用buffer・RNG・module modeを復元する。optimizer・勾配は変更しない。
  較正値は更新数・parameter SHA・固定train IDと共にcheckpoint内へ保存するため、
  通常のsnapshot/セル5の再開に含まれる。学習再開にはnativeのbufferを使用する。
- single-process、FP32限定。`TT2_WARMUP_UPDATES=400`、`TRAIN_PRECISION='default'`を維持。
  bestは較正後のDev total lossの最小値で選ぶ。評価は10更新ごと、停止policy v3、
  上限1000は変更しない。最終updateは較正を保存するが評価間隔外にdev評価を追加しない。
- 推論は選択checkpointに埋め込まれた較正値を照合して使用する。未較正の途中lastや
  別の重みの較正値は拒否する。通常は検証済み`checkpoint_best.pt`を選ぶ。

セル1で`TRAIN_TARGET='tt2'`、`TT2_RECIPE='stabilized_pooled_v2'`、`TRAIN_RUN_NAME=''`を選択し、
通常セル5を実行する。保存名は`tt2-training-実験名-paper_practical-stabilized-pooled-v2`。
切断後は**同じ設定でセル1→5**。既存v1/legacy runの自動移植はせず、過去lossも書き換えない。
対応コードが未pushの間はColabでは利用できない。push後の対応40桁SHAを`TRAIN_CODE_REVISION`へ指定する。
4A/4Bの成果物はそのまま使い、旧runは保存する。S2UTは変更しない。

少数の実例によるCPU試験ではMelの改善と学習更新への非干渉を確認したが、
128件較正・Fisher寸法・全コーパス・A100・翻訳品質は未検証。論文再現ではなく実験条件の変更である。
較正によってDev lossの測定条件が変わるため、旧runの未較正Dev値との大小を
そのままモデル重みの品質改善とは解釈しない。旧history・anchor・bestの書換えは行わない。

## セル5の統合再開

更新済みノートブックでは、切断後は同じ実験名でセル1（設定）→セル5を実行する。
Drive接続・コード取得・ランタイム構築もセル5から呼ぶ。追加の再開セルは不要。
`TRAIN_RUN_NAME`は空欄なら、選択したモデルの学習済みrunが1つの場合に自動選択する。
例えば`experiment-001`のTT2は空のrecipe-v2 runを避け、snapshotを持つ
`tt2-training-experiment-001-paper_exact-recipe-v2-probe-fix-v1`を選ぶ。
複数の学習済みrunがある場合は、セル1の`TRAIN_RUN_NAME`に保存名を指定する。
読み取り可能なmarkerがない既存snapshotや選択ミスからupdate 0へ自動開始しない。
snapshot本体のSHA/更新数/設定の検証は従来どおり行い、壊れた直近コピーは検証済みの古い保存地点へ戻す。

再開するrunは保存済みconfig・batchを使う。セル1の性能設定を変えて既存configを上書きしない。
学習率scheduleと監視状態はcheckpointから継承する。モデルの総更新上限とセッション時間は今回の指定を使う。
旧48fcdb7のTT2は既に確認済みのd70516a実行コードを使い、元の依存環境・snapshot identityを維持する。
S2UTなど新規学習はlauncherの固定SHAを記録して開始し、4A/4Bの準備済みデータを使う。

初回launcher取得はorigin/mainを完全SHAに解決して`training-launcher-revision.txt`へ保存する。
次回から同じSHAを使用し、学習runごとの`<RUN_NAME>.launch.json`にも実際のコード・環境を記録する。
launcherを明示更新する場合だけ`TRAIN_CODE_REVISION`へ対応する新しい完全SHAを指定する。
launcherの更新自体で保存済みモデルの実行コード・optimizer条件は変更しない。

TT2の`TT2_WARMUP_UPDATES`は既定400（選択肢400/500/1000/10000）。通常セル5から
明示的に学習率スケジュールを短縮し、`paper_practical`の変更履歴をcheckpointへ保存する。
旧1,000/500短縮済みrunも、既存ウォームアップ終了前なら同じrunで400へ変更できる。
現在400更新・保存済みwarmup500の場合も、次の更新から400設定を使える。
400は学習の強制終了点ではなく、dev lossによる早期終了判定の開始目安。
既定は10更新ごと・patience 5・最低改善率1%。400時点では過去へ遡らず、
ウォームアップ終了後の410/420/430/440/450で十分な改善がなければ450で終了する。
停止判定の基準はウォームアップ中の各lossの最小Dev値。1%未満のbest更新も保存するが、
停止カウンターはリセットしない。いずれかのlossの累計改善が自身のanchorから1%以上になれば
共有カウンターをリセットし、そのlossのanchorだけを現在値へ更新する。
重み・Adam状態・絶対更新数・過去loss/学習率・best checkpointは維持する。
新しい学習率は次の更新から適用するため、過去を400設定で学習したことにはしない。
再開時の10000選択は保存済みスケジュールを継承し、途中で延長はしない。
旧固定コードへは、検証済みのmonitor実装に対してスケジュール/監視class/履歴復元の範囲だけを
変更したローカル実行コピーを作り、全ソース/configのSHAと監査記録を検証する。
未確認のコード差分や改変済みコピーは拒否し、元コード・snapshotは上書きしない。
S2UTの通常セルのウォームアップも400に揃える。原論文の設定ファイルは変更しない。

通常セル1はTT2・S2UTの監視policy v3を明示要求する。旧policyからの再開時は、旧履歴・
旧カウンター・bestのSHAをまず検証し、変更地点と旧/新policyを`policy_history`に記録する。
過去のlossや学習率、best、モデル/optimizerは変更しない。各anchorを過去のDev最小値から構築し、
新条件の共有停止カウントだけを0にし、
max(ウォームアップ終了点, policy変更地点)より後の評価から始める。
例えば400で変更すれば最短450、430で変更すれば最短480（改善がない場合）。
再開のたびにカウンターをリセットせず、同じpolicyはそのまま継承する。
評価間隔の変更やpolicyのダウングレードは拒否する。同一policyでの終了状態は継承するが、
旧v1/v2で早期終了したrunからv3への明示採用時だけ、停止状態を解除して未来の評価へ進める。

### Early stopping v3（全Dev loss・絶対上限TT2=1000 / S2UT=3000）

TT2は `loss / mel_loss / phone_loss / duration_loss` の4種類、S2UTは
`loss / unit_loss / source_letter / target_letter / decoder_target_ctc` の5種類を監視する。
各指標のanchorを `A_i`、現在値を `L_i` とし、`L_i <= A_i * (1 - 0.01)` なら有意な改善。
1つでも改善すれば共有 `bad_evaluations=0` とし、改善した指標のanchorのみ更新する。
どの指標も改善しない評価が5回連続すれば停止する。anchor=0はそれ以上改善できない値として扱う。
全指標を除外せず監視し、合計lossも含める。評価は従来どおり全devを10更新ごとに行う。

Every monitored validation loss has an independent anchor. Training continues while at least one
monitored validation loss improves by at least 1%. Early stopping is triggered after five consecutive
validation evaluations in which none of the monitored losses achieves a 1% improvement.

warmup中は停止せず、各指標のwarmup内最小値をanchorとする。旧policyからの移行時は
保存済みDev履歴全体の各指標の最小値を使うが、過去の停止判定はv3で再解釈しない。
`policy_history`に移行更新数・旧policy・新policyを記録し、再開では同じanchor/counterを復元する。
新しい通常ノートブックで同じ実験名・同じ保存run・同じwarmup400を指定し、必要なlauncher更新の
完全SHAを`TRAIN_CODE_REVISION`へ指定してセル1→5。4A/4Bやデータの再生成、追加の移行セルは不要。

`training-monitor.json`とlast checkpoint内に `stopping_metrics / anchors / bad_evaluations /
best_values / best_updates / policy_history` を保存する。既存の `best_value / best_update / best_sha256`
は推論用bestの意味を維持する。CSV/SVGには従来どおり全lossのtrain/dev履歴を残す。
評価ログではloss別の改善率、改善した指標（`*`）、共有counter、停止状態をstderrへ表示する。
checkpoint_bestはTT2=Dev total loss最小、S2UT=Dev unit_loss最小のまま、停止判定とは独立である。

TT2の `TOTAL_UPDATES=1000`、S2UTの `S2UT_TOTAL_UPDATES=3000` を安全上限とする。
通常セル5およびv3 trainerは指定をモデル別上限へ制限する。上限はoptimizer更新の絶対値で、
S2UTの610からの再開なら最大2390更新、3000到達済みなら追加更新しない。TT2は1000のまま。
旧v3 S2UTの上限1000を記録したcheckpointも復元し、anchor・counter・best・履歴はリセットしない。
上限変更だけで早期終了済みのrunを再開することはない。改善やcounter resetは上限を延長しない。
vocoderの既定更新数は独立の `VOCODER_TOTAL_UPDATES=10000` に移して従来値を保つ。
元論文のscheduler・optimizer・batch・loss重み・validation split/間隔は変更しない。

依存環境の完全なJSONを`.runtime-environments/<SHA256>.json`へatomic保存・readbackする。
新しいランタイムではその全依存版をconstraintとして構築する。
異なるrunを同じローカルvenvで順番に使った場合は、版だけでなくパッケージ構成も復元する。
完全なSHA検証済みmanifestにない余分な通常パッケージは、専用runtime venvであることを確認し、
変更前のpip freezeと削除対象をローカルreceiptに保存してから、そのvenv内だけで除去する。
ノートブック本体のPython、コーパス、保存済み環境JSON、checkpointは変更しない。
旧session.logしかない場合は記録が不完全な可能性があるため、この除去は行わない。
復元後もSHAが一致しなければ停止し、`/content/s2st-runtime/dependency-recovery-difference-*.json`へ
不足・余分・版違い・metadataの差分を記録する。照合を無効化して続行することはない。
復元用constraintは元リポジトリのrequirementsを絶対パスで参照し、
入れ子の`preparation.txt`等も元ファイルのディレクトリから解決する。
旧復元用ファイルは通常セル5の構築時に再生成するため、手動コピー・データ削除は不要。
旧TT2で環境JSONがない場合は、`session.log`の成功した実行記録から依存版の候補を抽出し、
実際にインストールして再計測したSHAがsnapshotと完全一致した場合だけ採用する。
一致しない場合は学習を開始せず停止する。保存済みハッシュや環境JSONを偽装しない。
モデルごとの`tt2-session.log`、`s2ut-session.log`等に構築・復元・学習ログを残す。
初回のlauncher取得後は旧手動移行セルや通常セル3を追加実行する必要はない。
CPUテストで選択・復元・中断時の保持を検証しているが、実Colabでの全手順は未検証。

S2UTの`S2UT_WARMUP_UPDATES`は既定400更新（選択肢400/500/1000/10000）です。
短縮時は`paper_practical`と記録し、学習名に`-warmup400`等を付けます。
`REPRODUCTION_MODE=paper_exact`のままこの短縮を選んだ場合、token予算20,000・
勾配蓄積4・FP16を維持し、追加バッチ分割の事前測定は無効にします。
論文設定の10,000を選べば従来の学習名・条件へ戻ります。
準備済み4A/4Bの再生成は不要です。既存S2UT checkpointのwarmupを途中で変更する機能は
未対応です。通常セル5は構築/ダウンロード前に不一致を拒否します。
既存runは元のwarmupで再開し、異なるスケジュールは別実験で開始してください。
S2UTの`S2UT_MONITOR=True`は全devを10更新ごとに評価し、Unit予測の
teacher-forced dev lossで最良checkpointを選択します。policy v3ではウォームアップ終了後、
全5指標のどれも自身のanchorから1%以上改善しない評価が5回連続した場合に停止します。
いずれかが改善すれば共有counterを0に戻し、改善した指標のanchorだけを更新します。
400時点の過去5回を遡って停止せず、最短で410/420/430/440/450の評価後に停止します。
各指標の基準はwarmup中の最小値です。小さなbest更新も保存し、基準から累計1%以上
改善した時点だけ共有counterをリセットします。各lossのbest値/更新数も別途記録しますが、
checkpoint_bestの選択はUnit lossのままです。設定は
`S2UT_MONITOR_INTERVAL=10`、`S2UT_MONITOR_PATIENCE=5`、
`S2UT_MONITOR_MIN_RELATIVE_IMPROVEMENT=0.01`です。
旧policy v1はUnit-only、v2はUnit/totalの別counterという従来判定を保持して読み込みます。
v3への明示変更時は履歴・bestの検証後、変更地点を記録して共有counterをリセットします。
同じpolicyでの再開はcounterを継承します。既存runの固定実行コードを自動更新はしません。
合計loss、Unit loss、source_letter、target_letter、decoder_target_ctcのtrain/dev推移を
`training-monitor.json`・`.csv`・`.svg`へ記録します。
lossは各タスクのsample_sizeで重み付き平均を取りbits単位で表示し、合計lossには
補助lossの重みが入ります。Dev lossによる停止判定はBLEUによるモデル選択とは異なります。
履歴・判定counter・最良checkpointのSHA256をlast checkpoint内に保持し、Drive snapshotに
グラフ・最良モデルも含めます。再開はlast、推論・比較は検証済みbestを優先します。
常駐単一プロセス、固定バッチ、全dev、同期checkpoint保存が必要です。
VRAM事前測定の間は監視を実行しません。同じpolicyで停止後の再開は更新せず`EARLY_STOPPED`を返します。
旧v1/v2からv3の採用は未来の判定へ続行でき、モデル別のhard limit（TT2=1000、S2UT=3000）は常に維持します。
Driveの保存先には確認用のグラフとCSVもコピーします。初回有効化以前の履歴は復元しません。

### S2UTの初回FP16 overflow

固定fairseqが`gradient overflow detected`でscaleを128→64等へ下げると、
その試行はoptimizerを更新せず`None`を返します。常駐S2UTはこの場合のみ同じ論理バッチを
再試行し、`[fp16-retry] update=... retry=... loss_scale=128->64`を記録します。
optimizer.step未呼び出し・更新数不変・scaleの正常な低下をすべて確認します。
乱数・model/criterionのbuffer・勾配を復元し、失敗した試行の監視lossとgradient証拠を破棄します。
低下後のscaleは保持し、成功時だけ更新数、loss履歴、probe完了数を進めます。
OOMやoptimizer実行後の失敗はこの処理では再試行せず、最小scale到達や上限32回でも停止します。
FP32/BF16へ黙って切り替えず、token予算・蓄積回数・warmup・Adam設定は維持します。

2486164/e56ca1cで固定済みのrunは、新launcherの通常セル5で検証済みFP16 runtime、
TrainingAudit、checkpoint検証・保存・異常終了時退避を差し替えた実行コピーを作ります。元checkout・config・基礎依存環境・
recipe fingerprint・snapshotは保持し、全source/configのSHAを`execution-overlays/`へ記録します。
未レビューの旧コードや改変されたコピーは拒否します。初回overflowで更新0・snapshotなしの
場合も、同じ実験名・設定でセル1→5を再実行できます。4A/4Bの再生成は不要です。
launcherの固定SHAは自動更新しないため、反映時はセル1の`TRAIN_CODE_REVISION`に
修正を含む公開済みSHAを指定してください。CPUの制御フロー/Adamテストで確認済みですが、
実A100のFP16オーバーフロー再試行は未検証です。

### 初回dev評価と未公開checkpointの復旧

S2UTのCTCはdev評価時に`editdistance`を必要とします。新規Colab/Docker環境は
`editdistance==0.8.1`を固定導入します。通常セル5はデータ取得の前に、実際のCTC
train forward/backwardとdev forward（文字/単語編集距離を含む）を小さなCPU入力で実行します。
これは実験の学習ではなく、モデル重み・optimizer・データsplitを使用/変更しないランタイム確認です。

旧環境を復元する場合は保存済みpip freezeとenvironment.jsonをそのまま検証・維持し、
公式PyPIのCPython3.10/Linux-x86_64 wheelをSHA256検証して評価用パッケージを別置きします。
配布wheel/manifestは保持し、import用パッケージだけをPYTHONPATHへ追加します。
基礎venvへ追加インストールせず、追加依存のversion・wheel/hash・基礎環境SHAを
`.runtime-extras/`へ記録します。CER/WERを省略したり、未知文字へ置換したりはしません。

Driveへ未公開の新しいS2UT `checkpoint_last.pt`が同じVMにある場合、通常セル5が
先に入力hash、checkpointの有限なmodel/Adam状態、recipe fingerprint、監視履歴/カウンター、
bestのSHAとreportの一致を検証します。すべて一致する場合だけ別の不変コピーをDriveへ公開し、
その地点から再開します。元ローカルと旧Drive snapshotは残します。VMを削除済みなら
ローカル状態は回収できず、最後の検証済みDrive地点から再開します。4A/4Bの再生成は不要です。

通常の10分間隔に加えて、S2UTはdev評価後と早期終了/完了時にもbackupを強制します。
trainerがエラー終了した場合も、終了後に最後のローカルcheckpointを検証して退避します。
Driveへ公開できない場合はローカルの`.pending-recovery-*`等を保持し、保存成功とは表示しません。
ColabがVMを強制削除すると退避コード自体を実行できないため、毎更新の永続化保証ではありません。
`learned_at_least`はローカル保存済み進捗も反映し、`durable_updates`はDrive公開済みだけを示します。

デフォルトの無制限ローカル保存は、直近2個のinterval checkpointと直近1epochへ制限します。
last・bestとDriveの過去snapshotは保持します。fairseqが古いローカル中間checkpointを整理しますが、
実験データ・コーパス・Driveの復旧地点は削除しません。学習のbatch/精度/LR/warmupは変えません。

## 使うノートブック

| 用途 | ノートブック | データ・学習 | Drive保存先 |
| --- | --- | --- | --- |
| 最初の動作確認 | [colab_smoke.ipynb](../notebooks/colab_smoke.ipynb) | 各split 5件・2更新・小型TT2 | OUTPUT_BASE/smoke/実験名 |
| 通常の学習・比較 | [colab_training.ipynb](../notebooks/colab_training.ipynb) | 全件・更新数を明示指定・Fisher paper_exact | OUTPUT_BASE/training/実験名 |

コーパス側のノートブックと同様、最初の設定フォームを編集して番号順に実行します。
コーパスのコード・データは変更しません。smokeの数値を品質評価として使わないでください。
通常用はFisher形状・optimizer・実効batchを論文に寄せますが、16kHz target Mel・eSpeak・日英コーパスを維持するため完全再現ではありません。
新しい設定と移行上の注意は[論文レシピ](PAPER_RECIPES.md)を参照してください。

## 手順

`OUTPUT_BASE` の既定値は `/content/drive/MyDrive/ja-en-direct-s2st-benchmark-data` です。
既存実験を再開するときは、その実験で使用した出力先を引き続き指定してください。
既定値の変更によって旧保存先のデータが自動移動されることはありません。

1. ColabでCPUランタイムを選択します（最初からGPUでも実行できます）。
2. セル1でコーパスの保存先・出力先・実験名を指定します。
3. セル4AでCPUデータ準備を実行します。Drive接続・コード取得・環境構築（セル2・3相当）は自動実行されます。
   通常用は続けてCPUのままセル4A.5で音声ZIPを準備します。完了・Driveへの保存を確認後、GPUに切り替えて同じ設定でセル1→4Bを実行し、HuBERT Unit抽出とS2UT準備を行います。
4. セル1のTRAIN_TARGETをtt2 / s2ut / unit / melから選び、セル5で学習します。
   通常設定ではtt2・s2ut・unitの3runが必要です。unitはこのコーパスの英語train音声で学習します。S2UT_VOCODER_MODE=paperは公式LJSpeech重みの代替であり、原論文のFisher学習条件とは異なります。TT2をHiFi-GANで復元する場合はmelも学習します。smokeは従来の4runです。
   通常用は各targetの初回にVRAM事前測定を行い、結果を保存してから初期重みで本学習を開始します。
   `TRAIN_CALIBRATE_BATCH=True`、安全余裕10%が既定です。TT2実効1024は維持し、再開時は選択値を固定します。
5. 学習が終わったらRUN_BENCHMARKをONにし、セル6〜8で復元・4方式推論・共通評価・比較を実行します。

通常用はCONFIRM_FULL_DATAが既定でONであり、セル4A・4Bでは全コーパスを前処理します。
OFFにした場合は準備開始前に停止します。
RUN_TRAININGは初期状態でOFFです。学習する場合だけONにし、
TOTAL_UPDATES（累計の更新数）とCONFIRM_TRAININGを設定してください。
通常TT2/S2UTはそれぞれTOTAL_UPDATES=1000 / S2UT_TOTAL_UPDATES=3000を絶対上限とします。
vocoderはVOCODER_TOTAL_UPDATES=10000を使用します。長期学習の自動開始はしません。
旧4Bの補助ラベルは別のfairseq-unigram-allchars-v1へ移行し、S2UTはrecipe-v3の新しい学習runを使用します（TT2はrecipe-v2のまま）。
既存のMel・Unit・旧checkpointを削除する必要はありません。
データの全件使用と長期学習の許可は別々に確認します。勝手にfull学習を開始しません。
SESSION_MINUTESは既定で0（時間上限なし）です。セル5は目標更新数まで学習を続け、
60分ごとの自動停止はしません。Colab側の切断・実行上限を無効にする設定ではありません。
切断後は同じrunの検証済みDrive snapshotから再開し、未保存の更新だけ再計算します。
任意に正の値を指定した場合だけ学習subprocessに時間上限を設けます。
その時間には学習中のdev検証とsubprocess内の保存処理も含まれます。
準備・事前測定・入力preflightは時間予算の外であり、Driveバックアップ完了待ちも別です。
CLIの--session-secondsも既定0で同じ動作です。学習開始の明示確認は引き続き必要です。

### 4Bが未知文字エラーで停止した場合

`unknown source_letter tokens ... ['汎']` 等は、trainにないdev/test文字を旧辞書が拒否したエラーです。
新コードでは日本語・英語の文字辞書にtrain/dev/testの文字を登録し、最適化はtrainのみで行います。
CTCのUnigramモデルは引き続きtrain英語文だけで学習し、未知pieceは正しく`<unk>`へ符号化します。
この語彙登録方針は実験lockに明記します。原論文が全split登録を指定しているという意味ではありません。

保存済みrevisionを明示更新した後、同じ出力先・実験名で次の追加セルを実行できます。
CPUで実行可能です。Unit抽出・モデルダウンロード・4A/Mel生成・学習は実行しません。
保存済みUnitと日本語音声ZIPが必要で、欠けていれば自動で再抽出せず停止します。
未完了の準備・検証には時間がかかります。正常終了後の同じセル再実行は完了receiptを再利用します。

```python
ensure_runtime(require_gpu=False)
configure_preparation()
run(PYTHON, '-m', 'direct_s2st.preparation_workflow',
    '--stage', '4b-finish', '--profile', PROFILE)
```

更新前のS2UT学習checkpointは、変更した語彙のモデルへ継続学習しません。
学習へ進む前に最新ノートブックのセル1を同じ設定で実行し、S2UT recipe-v3のrun名を使用してください。

### 4Bの検証でsample ID alignment failureが出た場合

旧検証コードには、補助ラベル先頭の引用符`"`をCSVの囲み文字として解釈し、
後続サンプルを同じ行に取り込む不具合がありました。fairseqと同じQUOTE_NONEへ修正しています。
生成済みTSVを書き換える修正ではなく、読み取り側の修正です。
実際のID不一致は引き続き拒否し、全splitのID確認を音声の存在確認より先に行います。
エラーには欠落・余分なIDの件数、例、読み取りファイルを表示します。

今回のように生成後の検証で停止し、preparedのdata-lock.jsonが保存されている場合は、
明示的にコードrevisionを更新した後、保存済みTSVだけを検証して読み取り不具合の解消を確認できます。
次は読み取り専用です。音声の存在確認は省略し、ID・辞書・Unit列・CTC整合性を確認します。
学習前のpreflightでは通常の音声確認も実行します。未知文字エラーで生成前に止まった場合の代替ではありません。

```python
ensure_runtime(require_gpu=False)
run(PYTHON, '-c', '''
import json, sys
from pathlib import Path
from direct_s2st.s2ut.migrate_labels import paper_data_root
from direct_s2st.s2ut.multitask import validate_prepared
root = paper_data_root(Path(sys.argv[1]))
assert (root / 'data-lock.json').is_file(), 'prepared data-lockがありません。4b-finishを再開してください'
print(json.dumps(validate_prepared(root, check_audio=False), ensure_ascii=False))
''', DATA)
```

検証成功後はセル5へ進めます（4A・HuBERT抽出・補助ラベルを再生成しません）。
4B工程の完了receiptも残す場合は、通常の4b-finishを再実行してください。
新コードでもID不一致が出る場合は、表示されたmissing/extra/pathを確認し、検証を無効化して学習しないでください。

smoke用は5件・2更新に固定され、通常用は件数制限を設けません。
通常用のPROFILE=pilotはCLI設定選択用であり、コーパスを10時間に切り詰めるものではありません。
前処理ではS2UTの補助ラベルやTT2の音素・melも生成します。

### CPU準備 → Unit抽出 → A100学習

| 工程 | セル | 内容 |
| --- | --- | --- |
| CPU（GPU不要） | 1 → 4A → 4A.5（通常用） | 全件取込・WAV/SHA検証・音素化・TT2 Mel作成と検証、英語・日本語WAVの永続ZIP準備 |
| GPU | 同じ設定で1 → 4B | 完成ZIPをローカルへ取得・展開してHuBERT Unit抽出、S2UT用データ作成と検証 |
| A100など学習用GPU | 同じ設定で1 → 5 | 選択モデルの学習／checkpoint再開 |

GPUの変更はColabのランタイム設定から手動で行ってください。切り替えるとPython変数も消えるため、
**セル1は再実行が必要**です。CORPUS_PATH・OUTPUT_BASE・EXPERIMENTは同じ値を使います。
以後のセルはDrive接続（認証操作が必要な場合あり）、保存済みrevisionのコード取得、Python/fairseq/eSpeak環境の
再構築、環境変数の復元を自動実行します。既存の構築済み環境は再インストールせず動作確認して使います。
セル2・3は手動で先に実行したい場合の入口として残しています。

4A完了後に4Bや5へ進む際、4Aは自動でやり直しません。準備データはDrive上の同じ保存先を参照します。
4BをCPUで実行するとGPU必須のエラーで停止し、黙ってCPU抽出へ切り替えることはありません。
通常用の4Bは必要な英語・日本語ZIPが全て公開済みかをモデル取得前に確認します。
不足時はセル4A.5へ戻る案内を出して停止し、GPU上で原本からZIPを作り始めません。
ZIP取得・展開、k-means、S2UT整備・検証はGPU使用率が低い区間として残ります。
通常用のgpu80設定は学習向けであり、4Bの抽出には使いません。他のGPUでの速度・VRAM使用量は未検証です。
TT2/melだけを学習する場合、Unitを使わないため4Bは不要です。4方式比較にはUnit準備も必要です。

CPU/Drive待ちの間にGPUを確保しないことでGPU利用時間を減らせます。ただし切り替えのたびに環境構築が必要で、
総所要時間が短くなる保証はありません。CPU準備でも互換性維持のためCUDA対応PyTorchをインストールします。
料金や利用枠の削減量は契約・処理時間によるため固定値では示せません。

## 保存・切断後の再開

同じノートブック、実験名、設定でセル1を実行し、必要な工程のセルへ進みます。
学習snapshotがあれば自動でresumeします。snapshotが壊れていれば停止し、
黙って初めから学習し直すことはありません。最初の保存区間が完了するまでは復元点はありません。
学習対象を切り替えても、run名とローカル作業先は実験名・用途・対象別に分離されます。

- data/: common manifest・前処理データ
- checkpoints/: hash検証付き学習snapshot
- runs/: 各方式の予測音声・JSONL・metrics
- data/results/: 比較レポート・acceptance
- session.log: コマンドの標準出力とエラー（セルにも逐次表示）

切断時の学習再計算は最後の保存区間以降です。
推論は成功済みpairのSHAを照合して再利用します。
比較セルは不足するローカルcheckpointをDriveから復元します。
既存の比較レポートを更新する場合だけ、セル6のOVERWRITE_COMPARISONをTrueにしてください。

コードrevision、学習設定、データ、環境lockの不一致は再開を拒否します。
条件変更時は新しい実験名を使ってください。過去のcheckpointやsnapshotは削除しません。
旧版のsmoke-dataや手動run名は自動移動しません。旧実験を継続する場合は、そのrevisionの
ノートブックを使ってください。旧保存先に新しいlockを上書きしてはいけません。

## ランタイムと性能

Colab標準Pythonとは独立したPython 3.10.18とtorch/torchaudio 2.7.1を使用します。
3.13で構築途中のVMを再利用せず、新しいGPUランタイムで開始してください。
requirements/colab310.txtで従来の依存を固定し、setup.py --inference（CPU準備では追加で--allow-cpu）で
Cascade・S2T→TTS・共通評価の依存も導入します。

通常用はPERFORMANCE=gpu80（TT2 batch8、vocoder batch16、workers4、
S2UT max-tokens20000）を使います。VRAM使用量の保証ではありません。
paper_exactのTT2はupdate_freq128で実効1024、S2UTはupdate_freq4を保持します。
smoke用は小さいbatchとworkers0です。
詳細設定が必要な場合はdirect_s2st.colab.make_configの引数を変更しますが、
既存runの条件は途中変更しないでください。

モデルはsubprocessの終了時にGPUメモリを解放します。
BLASERは既定で無効です。SONARの依存は別途検証が必要であり、
有効化時に黙って省略しません。全方式で同じ評価条件を使ってください。

## 検証範囲

### 長時間処理の進捗ログ

データ取込・WAV/SHA-256検証・音素化・Mel/Unit作成・モデル読込・学習・推論・評価・
checkpoint保存/検証/復元は、開始/終了時と実行中およそ10秒ごとに`[progress]`を出します。
Colabでは画面と`session.log`に記録されます。学習・推論の外部プロセス出力も
画面へ転送し、従来の`runs/.../logs/process.log`にも保存します。

例（数値は説明用）:

```text
[progress] corpus: validate WAV and SHA256 status=running checked=120/5000 no_advance=2.1s elapsed=30.0s current=pair-121 activity=SHA256: pair-121.wav bytes=1048576
```

- `checked`はループで処理済みの件数（再利用/対象外判定/失敗記録も含み、成功件数ではありません）。総件数を事前取得しないストリームは`?`を表示します。
- `elapsed`はその工程の経過秒、`no_advance`は件数が最後に進んでからの秒数です。
- `current`は処理中のID/ファイル/更新番号、`activity`はハッシュ検証やダウンロード中のファイルと読み取りバイト数などです。
- モデル読込など件数を定義できない工程は、工程名と経過時間を表示します。全体と内側の工程のログが並ぶ場合があります。
- `running`は監視スレッドが動いていることを示すだけで、処理が前進した保証ではありません。件数やバイト数が変化しているかを確認してください。ランタイム切断やプロセス停止中はログを出せません。

進捗はstderrへ即時出力するため、CLIのstdoutのJSONは従来どおり利用できます。
成果物の形式は変更していません。4A/4Bの`--resume`時の検証再利用は下記を参照してください。
修正前から動いているプロセスには自動適用されません。新コードを取得して起動した処理から有効です。

### 4A/4Bの並列処理と中断再開

新規Mel抽出はVMのローカル一時ディスクで処理し、最大128件・目安256MiBずつの
非圧縮ZIPにまとめ、読戻しSHA256を確認してDriveへ保存します。
Driveへ14万個のNPYを個別保存する方式ではありません。
ZIPと不変receiptは`packs-v1/<先頭2桁>/`へ保存し、次回はZIP単位で検証・再利用します。
旧版の`features/`・`features-v2/`やcheckpointは削除・移動しません。
[既存MelのZIP救済手順](RESUME.md#大量のmel-npyでdriveのinputoutput-errorが出た場合)を使えます。
以前に完成・作成開始した旧形式の学習用ZIPは互換経路で維持します。

セル1に次の設定があります。セル5の学習設定・バッチサイズは変更しません。

- `PREP_MAX_WORKERS=32`: CPU/I/Oワーカー数の上限（1〜32、実際にはCPU数以下）。2以下から開始し、約10秒ごとに処理速度・CPU使用率・空きRAMから増減します。Driveが律速で速度が落ちる場合は減らします。
- `HUBERT_MAX_BATCH=32`: HuBERTの最大バッチ（1〜32）。空きVRAMと測定速度で調整し、OOM時は縮小して再試行します。1件でもOOMなら停止して記録済み部分を残します。32固定で処理する設定ではありません。
- `RECHECK_AUDIO=False`: 再開時、前回SHA-256/WAV検証に成功し、期待SHA・パス・サイズ・更新時刻・行の内容が一致する音声は検証を再利用します。**毎回全音声のバイト列を再検証するにはTrue**にしてください。サイズと更新時刻を保った改変は通常の再利用では検出できません。コーパスを不変に保つ前提です。

4Aではパス解決・WAV/SHA検証・eSpeak音素化・Mel抽出・音声ヘッダー読込を並列化します。
4Bでは音声読込とS2UT用変換を並列化し、HuBERTモデルは1個を使います。
HuBERTの畳み込み正規化にパディングが影響するため、**同じサンプル数の音声だけ**を
`max(16, HUBERT_MAX_BATCH)`件（上限32）の読込ウィンドウ内でまとめます。長さが異なる音声は単独推論になるため、
GPU使用率100%や最大速度、特定の高速化倍率を保証するものではありません。
精度変更（FP16/TF32の強制、有音部分の切捨て）はしません。

保存先は実験のDriveデータ領域です。コーパス側には書き込みません。

- WAV検証: `data/.prep-checkpoints/corpus/`
- 音素: `data/translatotron2/phonemes/.checkpoints/`
- Mel: `data/translatotron2/.prep-checkpoints/mel/`（新方式は`packs-v1/`のZIP。学習用ZIPの分も別途容量が必要。旧NPYは保持）
- Unit: `data/s2ut/units/.checkpoints/` と既存の`.units`ファイル

Mel以外の完了記録を64件または約10秒ごと（次の結果受取時）に小さなJSONへ原子的に保存します。
通常の例外・中断でも受取済み結果を保存します。VMの強制終了では未保存チャンクと
処理中の小さなウィンドウをやり直します。最終TSV/ZIPの作成中に止まった場合は、
抽出済みデータを使ってその集約工程をやり直します。破損した公開済み記録・成果物は
黙って上書きせず停止します。モデル・入力が変わった場合は旧結果を無条件再利用しません。

新方式のMelは最大128件/256MiBのZIPごとに保存し、32件以上たまった場合は
約60秒経過時（次の結果受取時）にも保存します。正常な例外・中断では受取済みの
小さな残りZIPも保存します。VM強制終了では未公開ZIPと処理中のワーカー分だけを
やり直します。`persisted`がDrive保存済み件数、`pending_local`がローカルだけの件数です。
`S2ST_MEL_LOCAL_WORK`でローカル作業親ディレクトリを指定できます（既定はローカルtemp）。
Driveパスやコーパス内を作業場所に指定しないでください。

中断後は**同じ実験名・同じコードrevision・同じ設定でセル1→4Aまたは4B**を実行してください。
`--resume`はセルに設定済みで、VM消失/GPU変更後の環境構築も自動実行します。
旧コードで既に走っている処理には適用されません。旧処理が保存していない途中結果は復元できません。
同じ出力に複数の準備プロセスを同時実行しないでください。

ログには`checked`・`remaining`（総件数が既知の場合）、CPUワーカー数・速度・空きRAM、
HuBERTの実バッチ数・次の上限・空きVRAM、終了時のcheckpoint保存/再利用件数が出ます。
CLIでは対応する環境変数`S2ST_PREP_WORKERS`、`S2ST_HUBERT_BATCH_MAX`、
`S2ST_PREP_RECHECK=1`を指定できます。`S2ST_PREP_ADAPTIVE=0`はCPUワーカーを上限に固定します。

### Unit / Mel ボコーダーの早期終了

通常セル1→5で`VOCODER_MONITOR=True`（既定）を使用します。
`VOCODER_MONITOR_INTERVAL=10`、`VOCODER_MONITOR_PATIENCE=5`、
`VOCODER_MONITOR_MIN_RELATIVE_IMPROVEMENT=0.01`です。
MelはdevのMel再構成L1、Unitはそれに加えてduration MSE（log1p run length）を監視します。
各指標の独立anchorからいずれか1%以上改善すればカウンターを0に戻し、
全指標が5評価連続で改善しなければ停止します。GANのdiscriminator/adversarial/feature/total
lossは敵対学習で単調減少を期待できないため停止指標には使いません。従来のtrain lossログは維持します。

`VOCODER_MONITOR_START_UPDATES=400`までは停止せず、各dev lossの最小値をanchorにします。
これは停止判定のguardであり、**学習率warmupを新設する設定ではありません**。
dev全件から固定の中央segment（既定32 frames）を評価し、発話ごとの平均を取ります。
Unitのdurationは発話全体のoriginal Unitを使用します。testは使用せず、devでoptimizer更新しません。
乱数・モデルbuffer・train/eval状態を評価前後で復元します。
音声の自然さ、翻訳精度、MOSを直接測る指標ではありません。

再開用は従来どおり`generator.pt`（generator/discriminatorと両optimizer、乱数）です。
監視状態・dev入力identityも同梱し、切断再開でanchor/patience/historyをリセットしません。
最良のdev Mel L1のfull checkpointを`checkpoints/checkpoint_best.pt`へ別保存し、比較用音声生成は
SHA確認済みbestを使用します。best未作成の旧runはlastを使用します。
JSON/CSV/SVGは確認済みsnapshotに含め、通常のDriveダッシュボードにも公開します。

監視なしの既存ボコーダーrunには、対応する新launcherが検証済み旧コードへ監視のみをoverlayし、
保存済みoptimizer/configを維持して、再開した更新以降から記録します。過去lossは捏造しません。
未知の旧実装や監視開始後のpolicy/guard/dev入力変更は拒否します。設定OFFでも保存済み監視は継承します。
初回に最新launcherを明示選択する必要がある場合は`TRAIN_CODE_REVISION`へ対応する完全SHAを指定します。
以後は通常セル1→5で再開できます。追加セルは不要です。
`VOCODER_TOTAL_UPDATES=10000`の絶対上限は独立して維持します（再開後に10000追加ではありません）。

#### VRAM測定後に`ExistingOutputError`になった場合

93b04bf以前では、launcherが保存した`batch-calibration.json`だけでも初回の
ボコーダーが「既存の学習出力」と誤判定する場合がありました。修正版は通常セル5の
private preflight receiptで保存先・許可済み準備ファイル名・SHA-256を照合してから開始します。
generator/checkpoint・監視履歴・未知ファイルがあれば従来どおり保護し、
`--overwrite`を自動追加したり、チェックを丸ごと無効化したりしません。
更新済みlauncherを選び、同じ実験名でセル1→5を再実行してください。
保存済みVRAM測定と入力ZIP/検証cacheは再利用します。未完のローカル出力は通常の
auto-resumeが`.interrupted-...`に保持します。Driveのデータや既存checkpointの削除は不要です。

### セル5の常駐学習・開始前VRAM測定

通常ノートブックの `TRAIN_OPTIMIZE=True` は、1 GPU・1学習プロセスをセッション中
常駐させます。従来の100更新ごとの再起動は行わず、`CHUNK_UPDATES` は旧方式だけに適用します。
APIの `make_config` は互換性のため `optimize=False` が既定です。

通常用は `TRAIN_CALIBRATE_BATCH=True` で、TT2/S2UT/Mel・Unit vocoderごとに開始前の
別プロセス試運転を行います。選択値・ピークVRAM・時間・GPU条件をDriveに記録し、
試運転weightsを捨ててから本学習をupdate 0から開始します。TT2は実効1024を維持して
物理batchを選び、GANは実batch数を選びます。S2UTのpaper_exactは公式の
max_tokens20000/update_freq4のVRAM収容確認だけを行い、追加分割・拡大はしません。
不足時は停止し、より大きいGPUまたは別runのpaper_practicalを明示選択します。
S2UTの分割上限の選定はpaper_practical限定です。再開時は保存した設定を固定します。
`TRAIN_CALIBRATION_MAX_BATCH=1024`、`TRAIN_VRAM_RESERVE_RATIO=0.10`、
`TRAIN_CALIBRATION_STEPS=3`が既定です。MAX_BATCHはS2UT paper_exactには影響しません。
最大容量と最大速度は同じとは限りません。
TRAIN_CALIBRATION_OBJECTIVE=throughput（既定）は安全な候補から試運転の処理数/秒で選びます。
capacityは収まる最大数を選ぶ旧方針です。いずれもS2UT paper_exactのbatchは変更しません。
GPU使用率サンプル、VRAM、処理数/秒、TT2のstream読み込み待ちをログで確認できます。
GPU変更時の扱いと安全余裕は[論文レシピ](PAPER_RECIPES.md)を参照してください。

試運転は`python -m direct_s2st.batch_probe`で起動します。終了通知は全trainerで
共通の型を使用し、指定回数の有限・正の測定値が揃った場合だけ成功とします。
通常エラー・測定不足はOOMと区別して停止し、不完全な測定を再利用可能として保存しません。
既存の測定結果とGPU変更時の確認にも同じ検証を適用します。
S2UTの処理数/秒は、勾配蓄積した全サンプル数をoptimizer更新全体の時間で割ります。
ログのmicrobatchは実際の最大同時処理数であり、全処理数とは区別します。
途中で停止した候補は再測定し、それより前に完了・保存された候補だけ再利用します。

`ProbeComplete`で試運転後に停止した旧コードから更新する場合は、
`scripts/colab/use_paper_recipe.py --separate-code-fix-runs --overwrite`で
保存済みrevisionをバックアップして更新し、学習run名に`-probe-fix-v1`などの別名を付けます。
実験名と準備済みデータの場所は変更しません。旧設定・checkpointは保持されますが、
旧runのoptimizer状態は新runへ引き継ぎません。本学習前の失敗なら、新runで測定から開始できます。
ランタイム切替後も同じ別名を設定してからセル5を実行してください。
選択済みの同時処理数を黙って変更したり、試運転の重みを本学習へ持ち込んだりしません。
TT2/vocoderの試運転ログは最後のoptimizer更新も数えて完了表示します。
実際の`-m`起動はCPUのCUDA/trainerモックで回帰テストし、TT2の3回のbackward/Adamと
試運転checkpoint非公開は実CPUモデルでも確認します。実GPU収容確認とは別のテストです。

- `TRAIN_MAX_WORKERS=8`: 読込み並列数の上限。TT2/vocoderはCPU負荷・空きRAM・測定速度で約10秒ごとに調整。S2UTはfairseqのepoch境界でのみ調整します。0は逐次読込みです。
- `TRAIN_CACHE_GB=8`: 旧方式のローカルLRUキャッシュ上限。新しいDrive安全モードでは必要な音声とMel ZIPを先に全てローカルへ取り込むため、全データを8GiBに制限する設定ではありません。ローカル容量不足時は学習開始前に停止し、Driveへの読込みに黙って戻りません。
- `TRAIN_SAVE_INTERVAL=1`: 通常用のoptimizer更新単位の保存間隔。最終更新・時間予算到達時にも保存します。ローカル固定コピーは同期、その後のDriveコピー・SHA-256検証は別スレッドで1件ずつ実行。未確認コピーは最大2件で、遅いDriveには待ち合わせます。
- `TRAIN_BACKUP_MIN_SECONDS=600`: 常駐学習でDriveへ転送する最短間隔。ローカルの50更新ごとの保存は維持し、最初・最終・時間予算到達時の保存は間隔によらず転送します。強制切断では前の`durable_updates`から再開し、未確認更新を再実行します。
- `TRAIN_BACKUP_MAX_GB=1024`: そのrunのDriveバックアップ総容量上限（1024GiB = 1TiB）。旧snapshot・未完コピーも数えます。予約容量ではなく上限です。上限到達時は既存復旧点を削除せず停止します。Drive全体の空き容量も別に必要です。明示的に設定済みの値は自動変更しません。
- `TT2_MONITOR=True`: policy v3で全4指標を監視。`TT2_MONITOR_INTERVAL=10`更新ごとに全devを評価し、いずれかが自身のanchorから1%以上改善すれば共有counterをリセット。全指標が改善不足の評価が`TT2_MONITOR_PATIENCE=5`回連続すれば終了。best保存はDev total lossの絶対最小値で選択。warmup既定400中は停止せず、各指標の最小値をanchorにする。v3の絶対上限は1000更新で、学習率・バッチ・loss定義/重みは変更しない。
- TT2監視は`training-monitor.json`（全履歴・policy・指標別anchor/best・共有カウンタ・条件変更履歴）、CSV、SVGをローカルrunに保存し、確認済み学習snapshotへ同梱。`checkpoints/checkpoint_best.pt`はDev total loss最小のモデルとoptimizer、`checkpoint_last.pt`は再開位置。切断後はsnapshotを復元すると履歴・最良モデル・カウンタが戻る。旧checkpointからの監視初回有効化はその更新から履歴を開始し、過去の指標を捏造しない。既存v1/v2からv3への明示採用は記録済みDev最小値をanchorとし、将来の評価だけに適用する。詳細はセル5の統合再開節を参照。
- 確認済みsnapshotの公開後、`PERSISTENT/checkpoints/<実際のrun名>/training-monitor.svg`・CSV・JSONへ閲覧用コピーを保存する。Driveからグラフを開けるため、学習セルの流れるログを追う必要はない。履歴のJSONは最後の永続保存まで、グラフ/CSVは直近の評価時点までを表示する。閲覧用コピーにI/Oエラーが出ても検証済みsnapshotは保持し、監視データはsnapshotから復元可能。
- TT2推論は既定で検証済みdev-bestを選択し、旧runや初回評価前はlastを使用。`--checkpoint-selection last/best`で明示指定可能。bestが記録されているのに欠落・破損した場合はlastへ黙って戻らない。**これはdev BLEUによるモデル選択や原論文の停止規則の再現ではない**。自由生成の音声・ASR BLEU評価は別途必要。最良モデルもsnapshotへ同梱するためバックアップ容量は増える。古い復旧点は自動削除しない。
- `TRAIN_ADAPTIVE_BATCH=False`: ONはTT2/S2UT限定。論理バッチ・サンプル順・更新回数を維持して分割を調整します。ただしTT2のBatchNorm統計やdropoutは変わるため、同一学習結果は保証しません。別の実験条件です。vocoderは2 optimizerを一組として扱い、分割・OOM再試行はしません。
- `TRAIN_PRECISION='default'`: 既存精度を維持（TT2/vocoder FP32、S2UTのレシピFP16）。明示的にFP32/BF16を選べます。BF16非対応GPUでは停止し、黙って精度を変更しません。

自動分割ON時だけ、optimizer更新前のCUDA OOMを捕捉して勾配・RNG・モデルbufferを戻し、
同じサンプルを小さい分割で再試行します。1件でもOOM、optimizer中のOOM、device assertは停止します。
この旧オンライン適応は`paper_practical`でのみ選択でき、開始前測定とは併用しません。
開始前測定をOFFにして明示的に選択した場合、校正は短い試行後に固定し、以後のOOMでは縮小します。GPUが変わると分割校正をやり直します。
これは最大速度を保証する探索ではありません。GANのOOMは保存済みcheckpointから復旧してください。

`[training-performance]` は処理時間の累計、frames/units等の速度、CPU・RAM・VRAMを表示します。
CUDA時間には区間内の待ちも含まれ、GPU稼働率そのものではありません。
`[recovery] learned_at_least` はローカル保存済み更新、`durable_updates` はDriveマウントへの
コピー検証まで完了した更新です。Driveサーバー側同期の完了までは保証しません。
切断直前の未保存/未確認更新は再実行します。時間予算はupdate境界で判定し、保存処理の完了待ちは
予算を超える場合があります。30秒の猶予後も学習が止まらなければ終了を要求し、最後の確認済み保存を使います。

旧runのconfig/revisionを上書きして移行しないでください。新設定は新runで利用します。
同一設定の再開は従来どおりセル1→5。実GPUでの速度・BF16品質・実Drive切断復旧は未検証です。

S2UTの通常セル5は、音声ZIPを全件取得する前に保存snapshotを検証・復元し、
recipeとcheckpoint内の識別子を照合します。先行検査ではWAV原本の存在確認だけを省き、
入力取得後の全件検査と使用前のSHA検証は省略しません。学習条件が違う場合は
`recipe.warmup_updates`や`command[番号]`などの差分項目を表示して停止します。
修正版の実行コピーへ切り替わってPythonファイルの場所だけが変わった場合は、
元のrecipe-lockとcheckpointのfingerprintを変更せず再開します。実験名の変更や4A/4Bの再生成は不要です。

セル5はフォルダの存在ではなく**検証済みsnapshotの有無**で自動再開します。
初回コピーが途中で止まり、まだ公開済みsnapshotがなければ、未完コピーを保持して
`starting at update 0`と明示して開始します。未公開ローカルファイルは
`.interrupted-<ID>`へ退避します。公開済みsnapshotがすべて破損している場合は、
勝手に0から開始せず停止します。CLIの明示的な`--resume`は復旧点がない場合も停止します。

Vocoderの全train音声/Unitの入力検証は、学習セッションの時間予算を開始する**前**に行います。
検証は4Aと同じ適応ワーカーを使い、`data/.prep-checkpoints/vocoder-inputs/`に
サンプル単位で保存します。再開では期待SHA、パス、サイズ、更新時刻が一致する検証を再利用します。
`RECHECK_AUDIO=True`（CLIでは`S2ST_PREP_RECHECK=1`）なら再ハッシュします。
サイズ・更新時刻を保った改変は高速再利用では検出できないため、コーパスは不変にしてください。
学習に渡す検証証明はセッション専用の一時ファイルで、snapshotには混ぜません。
サンプル読込み時にもサイズ・更新時刻を確認し、検証後の変更を検出したら停止します。

音素辞書と既存ラベルの移行は[音素の移行手順](RESUME.md#音素生成と辞書の新方式への移行)を参照してください。

### Driveの大量ファイル・学習I/O対策

通常の学習ノートブックは`S2ST_DRIVE_SAFE=1`を設定します。`direct_s2st.colab`のCLIも
既定で有効です（Python APIの既定は互換性維持）。コーパス原本・旧Mel・旧Unitは削除しません。

- 4Bの新Unitはoriginal/reduced列をJSONLへまとめ、途中結果は既存のUnit checkpointへ保存します。
  1件ごとの`.units`ファイルを新たに大量作成しません。旧ファイル方式は読み取り可能です。
  旧manifestからの移行は`manifest.inline.shard-...jsonl`という別ファイルへ保存し、
  同じモデル・入力のcheckpointがあれば再抽出しません。S2UT準備・Unit vocoderが両形式に対応します。
- 安全モードのcheckpoint JSONは256バケットへ分散します。旧直下のチャンクも読み取ります。
- 4AのMel準備は必要なWAVを128件単位の不変ZIPへまとめ、通常用のセル4A.5はCPUで
  全splitの英語・日本語ZIPを完成させます。4A.5では音素・Mel・Unitを再生成せず、
  全音声をローカルに展開しないため、作業ディスクは作成中のZIPを置ける容量で足ります。
  保存先は`data/.drive-audio-packs/`であり、コーパス内には書きません。
  初回は既存WAVを個別に読みSHA256を照合する必要があります。次のVMでは完成ZIPを取得して
  ローカルへ展開するため、元WAVの個別読込みを繰り返しません。
  4Bは完成ZIPの利用専用です。既存の55a5a90形式と同じ名前・識別子を使うため、
  旧4Bで作成した途中までのZIPもCPUセルで再利用します。
  原本は不変とする契約です。変更時は新しいcommon manifestを作り、古いZIPを無条件再利用しないでください。
- ZIP公開は計画・コピーのSHA256検証・完了記録の順です。VM強制終了でやり直すのは未公開ZIP分のみ。
  コピー後・完了記録前に止まったZIPも計画とSHA256が一致すれば復旧できます。
- セル5のTT2は既定の`TRAIN_STREAM_INPUTS=True`で、メタデータだけ先に取り込みます。
  学習順に必要なWAV ZIP・Mel ZIPを取得し、学習と並行して次のZIPを先読みします。
  転送は最大2件、待ち行列は最大16件です。取得済み順にサンプルを選んだり、未取得サンプルを
  飛ばしたりしません。必要なZIPが間に合わない時は待つため、GPUの待ち時間が完全になくなる保証はありません。
  実効batch1024、optimizer、サンプル順、試運転後の固定batchは変更しません。
  開始時のMel識別子は準備済み`data-lock.json`のSHA256を使い、未使用ZIP全体を事前に再読込みしません。
  **各ZIPの実データは取得時にSHA256を照合し、WAVも個別に照合してから学習に渡します。**
  同じVMでは完了記録とローカルSHA256を確認して再利用、新VMでは再開位置以降に必要なZIPから取り直します。
  初回のVRAM測定もtrainの測定用サンプルに必要なZIPだけを取得します。
  旧単一巨大Mel ZIPは最初の使用前にそのZIP全体の取得が必要です。既存ZIPは勝手に再分割しません。
  保存済みMel SHA256のない旧形式は`TRAIN_STREAM_INPUTS=False`で従来方式を使ってください。
  S2UT・vocoderは入力の全件事前検証があるため、今回も全件事前取り込みのままです。
  `TRAIN_STREAM_INPUTS=False`のTT2も必要なWAVと**Mel ZIP全体**を学習開始前に取り込みます。
  元パス・SHA256・入力識別子は維持し、実際の読込みのみローカルへ切り替えます。
  ローカルの途中コピーと完成コピーはVMが残っていれば再利用できます。
  VM消失時はDriveのZIPから取り直します。ローカル取り込み容量は`TRAIN_CACHE_GB`とは別です。
  先読みは全データをメモリに載せる仕組みではありません。使用済みZIPはローカルに保持するため、
  1周分のローカルディスク容量は必要です。容量不足・通信失敗は未検証入力で続行せず停止します。
  `[stream-download] ready=...`は取得/照合完了ZIP数で、学習update数ではありません。
  設定JSON/optimizerの識別子には取得方式を加えないため、同じコード・学習設定ならON/OFFで再開互換です。
  ただし保存済みrevision/環境lockへの新コードの移行は別問題です。稼働中の旧実験を無断で更新しません。
- Drive snapshotは多数のファイルではなく1個のZIPで保存します。旧snapshotも復元できます。
  SHA256 readback検証を維持し、容量不足・保存上限では前の復旧点を保持します。

DriveのAPI/転送量・容量制限そのものを解除するものではありません。
最初の原本取得、ZIP転送、checkpoint保存で制限が出る可能性は残ります。
空き容量確認にはマウントの報告を使いますが、Driveサーバー側の正確な残容量・同期完了は保証しません。
容量エラーやquotaエラーで停止しても、既存データを削除せず同じ設定で再開してください。
実GPU・実Drive上での性能と切断復旧は未検証です。

### fairseqコピーで停止した場合

`kaldi_self_train/st/utils`・`steps`の参照先が存在しないことによるコピー失敗は、
リンクを参照先の内容ではなくリンク自体としてコピーすることで回避します。
一時ディレクトリでコピーを完了してから公開し、完了マーカーがない既存コピーは
`/content/s2st-runtime/fairseq.incomplete-<ID>`へ退避します。削除はしません。
コピー中断時の一時ディレクトリも診断用に保持し、次回は新しいコピーを作ります。

修正版setupを取得した後は、同じVMでセル3を再実行できます。コーパス・checkpointや
インストール済みPython環境の削除は不要です。学習中にsetupを同時実行しないでください。
コード取得は保存済みGit revisionに戻るため、新コードを使う場合は最新コードを取得した新しいVMで新しい実験名を使い、
旧実験のrevision/環境lockを無理に上書きしないでください。

### 学習後、まず3件を試聴する（通常セル6・7）

選択した方式で必要な学習が終わったら、ノートブックを開き直し、セル1で
`RUN_TRAINING=False`、`RUN_PREVIEW=True`、`RUN_BENCHMARK=False`、
`PREVIEW_SAMPLES=3`（1〜10件）を設定して **セル1 → 6 → 7** を実行します。
旧launcherを保存している実験では、`TRAIN_CODE_REVISION`に試聴機能対応の
完全SHAを明示してください。学習用の保存revisionや実験名を変える必要はありません。

セル6・7はセル1の`TT2_VOCODER_MODE`・`S2UT_VOCODER_MODE`を試聴にも適用します。

| 設定 | 試聴で使う音声化方式 | 自前vocoderの学習snapshot |
|---|---|---|
| `TT2_VOCODER_MODE='griffin_lim'` | 保存済みTT2 Mel仕様によるGriffin–Lim（32反復・seed=1） | Melは不要 |
| `TT2_VOCODER_MODE='hifigan'` | 自前Mel HiFi-GANの検証済みbest | Melが必要 |
| `S2UT_VOCODER_MODE='paper'` | SHA256固定の公式LJSpeech Unit HiFi-GAN | Unitは不要 |
| `S2UT_VOCODER_MODE='trained'` | 自前Unit HiFi-GANの検証済みbest | Unitが必要 |

公式Unitとの切り分けは`griffin_lim`＋`paper`で行えます。必要な学習済みrunはTT2とS2UTのみです。
公式Unitは初回に重み・configを取得し、SHA256を検証して再利用します。
保存済みS2UTデータのHuBERT revision・層・クラスタ数・k-means SHA256も照合します。
`paper`はLJSpeech話者の配布モデルで、Fisherや対象コーパスで学習した重みではありません。
音声の話者が変わり得るため、対象話者の保持性能とは区別してください。

セル6は、必要な`last-training-run-{tt2,s2ut,unit,mel}.json`を優先して実際の保存名を選びます。
ポインタがなければ学習済みrunが一意のときだけ自動選択し、複数候補は
`PREVIEW_RUN_NAMES`で指定します。例えばTT2の`-probe-fix-v1`も実名のまま復元し、
更新ゼロのデフォルト名を使って新規学習することはありません。
snapshot ZIP・ファイル・保存configの整合性を照合して専用ローカル領域に復元します。
学習入力の全件取り込みはせず、同じdev IDの日本語/英語WAVだけを検証コピーします。
学習済みcheckpoint ZIPの取得は必要です。モデルによっては数GBあり、Driveの転送制限は残ります。

セル7はGPUを使用します。各runの保存されたコード位置・依存環境を順に復元してから、
現在の固定revisionの推論コードでdevの選択件数だけを生成します。
同じVMでの再実行も各runの完全な依存manifestに合わせて復元するため、Unit/Mel用の追加依存が
旧TT2環境に残ってSHA不一致になる場合も、通常セルの復元処理で調整します。
環境identityはGPU機種を含みません。GPU変更だけで保存lockを変更する必要はありません。
再学習、optimizer更新、4A/4Bのやり直し、ASR/スコア評価は行いません。
原音（日/英）、正解Melの再構成、正解の短縮Unitの再構成、TT2/S2UT翻訳音声を並べます。
**Unit再構成もdurationはvocoderの予測値であり、正解の時間長を復元するものではありません。**
試聴の表示名・`listening.json`には実際に選んだ音声化方式と公式重みの識別情報を記録します。
方式や重みが異なる場合は別の試聴識別子で保存し、以前の自前vocoderの音声は上書きしません。
セル6の後に方式を変更した場合、セル7は停止してセル6の再実行を案内します。
旧試聴planは自前HiFi-GANの意味のまま扱い、公式重みへ暗黙変換しません。
旧launcherに固定されている場合は対応版の`TRAIN_CODE_REVISION`を指定してください。
正式評価や学習済みモデルのconfig・checkpointは、この試聴の切替では変更しません。
この比較で、vocoder自体の再構成品質と翻訳モデルの出力品質を分けて確認できます。
生成失敗はエラー表示のまま残し、代替音声は作りません。
試聴launcher・各推論子プロセスは通常のINFOログを使用し、固定fairseqが有効化する
Numba等のDEBUGダンプを抑制します。進捗、INFO、警告、エラー、例外、結果JSONは残します。
JITコンパイルやモデルの計算自体を無効化する変更ではありません。

試聴成果物は`<実験>/preview/<識別子>/listening.json`と`audio/`へ保存します。
既存checkpoint・元コーパス・全件testのpredictions/resultsは変更しません。
切断後は同じ設定で通常のセル1 → 6 → 7を再実行します。
専用ローカルコピーは失われますが、Driveの保存snapshotと試聴成果物は保持されます。
セル8は試聴を正式比較として評価しません。正式な全件test評価へ進む場合は
`RUN_PREVIEW=False`、`RUN_BENCHMARK=True`としてセル6から準備します。
実Colab/GPUでのこの新しい試聴E2Eは未検証です。
音声化方式の4組合せ、不要vocoderの復元省略、設定変更時の別保存、旧planの意味の維持、
SHA/Unit IDの不一致拒否、生成失敗の保持、保存依存環境への振分けはCPUテストで確認します。
公式Unit重み・configの実ファイルもSHA256検証し、同じ推論adapterで正解Unit3件と予測Unit3件を
CPU音声化できることを確認しました。これはWindows/Python 3.13で固定fairseqの生成器・duration予測器を
そのまま分離読込した確認であり、Colab/Python 3.10の全fairseq import・GPU E2Eや音質評価の代わりではありません。
今回の切替のために追加パッケージは導入せず、既存のTT2/S2UTの保存済み依存環境を使用します。

S2UT推論は1件ずつ処理し、`--required-batch-size-multiple 1`も明示します。
既定値8のままだと、固定fairseqのバッチ生成で最後の1件が除外されるためです。

### S2UTだけの正式test評価と生成音声の保存（通常セル6～9）

TT2をまだ再学習していない場合も、S2UTだけを評価できます。セル1は
`RUN_TRAINING=False`、`RUN_PREVIEW=False`、`RUN_BENCHMARK=True`、
`BENCHMARK_TARGET='s2ut'`、`S2UT_VOCODER_MODE='paper'`、`DATA_LIMIT=None`、
`CONFIRM_FULL_DATA=True`とします。`BENCHMARK_RUN_NAME=''`は保存済み
`last-training-run-s2ut.json`を優先し、なければ一意な実保存runを選択します。
複数候補があれば実名を指定してください。`TRAIN_TARGET`は評価対象を決めません。
保存launcherが旧版の場合は対応版の完全SHAを`TRAIN_CODE_REVISION`に明示指定します。
元の学習revision・依存lock・snapshotは書き換えません。

1. **セル6**: S2UTの保存snapshot、元の依存環境identity、prepared metadataを照合。
   `checkpoint_best.pt`をdev unit lossの最良値・SHAと照合し、選択updateを表示します。
   学習最終400000 updateとdev-bestのupdateは異なり得ます。test全ID・正解英語は
   checkpoint-boundのtarget文字ラベルと照合します。取得対象WAVはtestの日英のみ。
   既存の音声ZIPは再利用し、欠けたtest ZIPだけ元コーパスを読んで別領域へ作成します。
2. **セル7**: 固定fairseq、beam10でtest全件のUnit生成、公式HuBERT L6/KM100
   LJSpeech Unit HiFi-GAN（SHA固定、duration予測）で音声化。
   TT2・自前Unit/Mel・train音声・新たな学習/optimizer更新は不要です。
3. **セル8**: 元の保存依存環境を復元し、固定revisionの
   `openai/whisper-large-v3-turbo`で英語ASR、既存の英語正規化＋sacreBLEUでcorpus BLEU。
   生成/ASR失敗も空仮説として**全test件数を母数に含めます**。失敗件数、ASR成功率も保存します。
   BLASER・話者類似度はこの単独経路では実行せず、4方式acceptanceとは別の報告です。

生成Unitの完了shard、WAVの32件グループ、ASRの32件ごとのjournalを、SHA検証付き
不変ZIPとしてDriveへ保存します。切断後は同じ設定で**セル1→6→7→8**。
ローカル環境/入力が失われても、完了receiptとZIPを検証して復元します。
保存前に失われた実行中グループは再処理され得ます。Drive転送制限やマウント切断を
回避できる保証はありませんが、生成WAVをDriveに1件ずつ置く方式は使いません。
保存済みZIPは削除せず、追記・完了indexのatomic更新で公開します。

保存先は`<実験>/benchmarks/s2ut/<評価ID>/`です。

- `plan.json`: 実run名、checkpoint/環境/metadata SHA、ASR revision、全test ID。
- `audio/part-....zip`: 生成WAV（32件ごと）。隣接JSONはZIPと各WAVのSHAを持つreceipt。
- `audio-index.json` / `audio-index.csv`: pair ID→ZIP名・ZIP内WAV・正解英語・成否。
- `predictions.jsonl`: 全件の生成結果（ローカルパスは記録当時のもの。再生はZIP/indexを使用）。
- `results/metrics.json` / `report.md`: 全件S2UT単独ASR-BLEU。
- `results/per_sample.jsonl` / `.csv`: 正解文、ASR文、sentence BLEU、失敗理由。

**後から聴くには、CPUでセル1→9だけ**実行します。
`BENCHMARK_LISTEN_ID`にpair ID、または空欄のまま`BENCHMARK_LISTEN_INDEX`に0始まりの番号を指定。
保存ZIPから選択WAVをSHA検証して取り出し、日本語原音・英語正解・S2UT翻訳・ASR文を表示します。
モデル/venv/GPUは不要、再推論や再学習もしません。元音声は元コーパスから読み取るため、
日本語/英語原音も聴くにはコーパスを保持してください。原音が取得できなくても保存済み
翻訳音声の試聴は継続し、原音の取得失敗を表示します。生成WAV自体はZIPをPCへ取得して
展開するだけでも再生できます。試聴は評価結果・既存のpreviewを変更しません。

新しい経路はCPUテストで保存・SHA拒否・再開・母数・通常セル分岐を確認します。
実Colab/GPUでの全test E2Eや品質・実行時間の保証ではありません。
従来の4方式比較は`BENCHMARK_TARGET='all'`で維持します。
この設定は少数試聴・全件推論の双方に適用し、学習batchは変えません。
旧設定の不完全な生成シャードは保持しますが、新しいキャッシュ識別子で再生成します。
`preview generated unexpected or missing sample IDs`が出た旧版は、対応する
`TRAIN_CODE_REVISION`に更新し、通常のセル1 → 6 → 7で再実行してください。
checkpointや準備済みデータの削除・再学習は不要です。

### 保存済みCascade結果のKFTT/JESC別診断（セル8A）

Cell 8A is a diagnostic-only analysis that recomputes Cascade metrics separately
for KFTT/JESC from saved Cell 8 results. It performs no inference and does not
modify the official benchmark result.

セル8の直後の**8A「Cascade KFTT-only 診断」**で、All / KFTT / JESCの
日本語ASR CER・MT corpus BLEU・最終ASR-BLEUを表示します。分類はmanifestの
`corpus`、結合は`pair_id`を使用します。AllのASR-BLEUが正式保存値と一致することを
確認し、元の評価と同じ失敗サンプルの扱いを維持します。結果行の欠落・重複・参照不一致は
記録済みの生成失敗と区別して停止します。CERはNFKC正規化＋空白除去（句読点保持）の
文字編集距離合計÷正解文字数合計です。MT BLEUはTTS失敗時にも保存済みMT文を利用します。

通常は`CASCADE_DIAGNOSTIC_TARGET='auto'`で保存済みCascade評価を選択します。
`BENCHMARK_TARGET='all'`の場合は4方式比較内のCascade、それ以外（セル1の既定の`s2ut`を含む）は
Cascade単独評価の`latest.json`を使用し、学習・評価対象の設定自体は変更しません。別の保存済みCascade評価を
確認するときだけ`CASCADE_DIAGNOSTIC_EVALUATION_ROOT`を指定します。CPUで処理し、既存の
`sacrebleu==2.5.1`環境を優先します。既存の`PYTHON`、ローカルvenv、Colab Pythonを順に
実import・短いCPU BLEU計算で確認し、`all`ではPyYAMLも確認します。利用可能な環境がなければ、
不足依存とPythonのパスを表示します。`CASCADE_DIAGNOSTIC_INSTALL_CPU_DEPS=True`の場合だけ、
既存環境が使えなければ`/content/s2st-diagnostic-deps/`へ固定版のCPU評価依存を導入・検証して
再利用します。`requirements/diagnostic-cpu.txt`のwheelだけを`pip --isolated --no-deps --target`で
導入し、NumPyはPython 3.13にもwheelがある2.1.3をこの隔離フォルダだけで使用します。
sacreBLEU 2.5.1・正規化・tokenizerは正式評価と同じで、Allの保存値との照合も必須です。
既存の学習venv・システムのパッケージ・Driveは変更しません。自動導入をOFFにした場合は不足時に停止します。
GPU・モデルロード・保存済みWAV/評価ZIPの読取り・結果保存は行いません。`CASCADE_DIAGNOSTIC_FETCH_CODE=True`は不足する診断コードだけを
実験/コーパス外のローカルGit作業ツリーへ取得する許可です。学習用checkoutや保存revisionは
変更しません。KFTTの低スコア例を表で表示し、全件はノートブック変数
`cascade_corpus_diagnostic['kftt_samples']`に保持します。
KFTT/JESCの差だけでコーパス品質が原因と断定するものではありません。
子プロセスの進捗・エラー本文はColabのセル出力へ中継します。単なる`CalledProcessError`で
原因が見えなくなるのを防ぎ、失敗時にも保存結果の自動修復・再推論は行いません。

### 正解日本語とASR日本語の翻訳比較（セル8B、GPU）

正式なCascade test評価が保存済みなら、GPUランタイムで同じ実験設定の**セル1 → 8B**を
実行します。8Aの実行変数には依存せず、8B内でも保存済みAll ASR-BLEU・ID・参照・SHAを
検証します。セル3・6〜8の再実行やWhisper/TTS/音声再生成は不要です。
`TRAIN_CODE_REVISION`は現在のままで、8B対応コードがなければ診断コードだけ別のローカル
worktreeに取得します。セル1〜8Aの挙動と、既存の正式評価結果は変更しません。

保存済みplanの同一NLLBで、各testサンプルの`ja_text`（正解日本語）と`asr_ja_text`
（保存済み日本語ASR）をそれぞれ翻訳します。英語参照文はBLEU計算にだけ使います。
All / KFTT / JESCごとに、保存済みMT BLEU・正解日本語からのMT BLEU・ASR日本語の再翻訳BLEU・
両者の差・再翻訳と保存済みMTの一致件数・失敗件数を表示します。すべてcorpus BLEUで、
失敗した翻訳は空仮説として選択サンプル全件の分母に残します。KFTT/JESC分類はmanifestの
`corpus`を使用し、結合と再開は`pair_id`で検証します。

`MT_DIAGNOSTIC_LIMIT=0`は全test、正数はpair_idのハッシュ順で固定抽出する診断用subsetです。
`MT_DIAGNOSTIC_TARGET='auto'`は8Aと同じ対象選択です。`MT_DIAGNOSTIC_RUN_LABEL`を変更すると
元の診断を残した別runになります。結果は`<実験>/diagnostics/cascade-gold-mt/<ID>/`の
`metrics.json`・`per_sample.csv`・`report.md`に保存します。NLLBの言語トークン・dtype・実生成設定・
依存バージョン・GPUも記録します。モデル・入力・実環境・コードが一致する場合だけ再開し、
8件の翻訳分岐ごとに小さなJSONチャンクを永続化します。通常の中断では未保存分もflushし、
ランタイム強制終了では最後の未保存最大7分岐を再処理する可能性があります。
完了済みの診断はNLLBを再ロードせず再集計できます。CUDA/OOM等は全件failedとして流さず停止します。

GPUと固定版のNLLB依存を事前に実import・CUDA演算で検証します。既存環境を優先し、
`MT_DIAGNOSTIC_INSTALL_DEPS=True`で不足時だけ`/content/s2st-mt-diagnostic-runtime/`に
専用Python 3.10.18・torch 2.7.1+cu126・Transformers 4.57.3等を構築します。
`requirements/diagnostic-mt.txt`の固定版を使い、学習venv・Colabのシステムパッケージは変更しません。
新しいNLLB重みとキャッシュは`/content/s2st-mt-diagnostic-cache/`だけに保存します。
GPUがない場合は依存を導入せず停止し、CPUにはfallbackしません。

生成条件は、[SHA固定の公式NLLB設定](https://huggingface.co/facebook/nllb-200-distilled-600M/blob/f8d333a098d19b4fd9a8b18f94170487ad3f821d/generation_config.json)と
[Transformers 4.57.3 NLLB文書](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/nllb)を確認しています。
元のadapterと同じ日本語`src_lang=jpn_Jpan`・英語`forced_bos_token_id`を使い、モデル既定の
最大生成長200・beam/sampling/repetition設定・FP32・1文ずつの呼び出しを維持します。
反復を見て都合のよい生成設定へ変更しません。EOS・生成長上限到達は診断情報として残し、
元のMT出力条件を黙って修正しません。再翻訳が保存済みMTと異なる場合は、実環境・生成設定の
記録を先に確認してください。正解日本語で低BLEUでも、対訳の破損を単独で証明するものではありません。
これは原因切り分け用であり、正解テキストを使う値を正式な音声翻訳スコアへ置き換えません。

### カスケードだけの正式test評価（通常セル6～9）

セル1で`RUN_TRAINING=False`、`RUN_PREVIEW=False`、`RUN_BENCHMARK=True`、
`BENCHMARK_TARGET='cascade'`、`CONFIRM_FULL_DATA=True`、`DATA_LIMIT=None`にします。
対応版の完全SHAを`TRAIN_CODE_REVISION`に指定してセル1→6→7→8を実行します。
`TRAIN_TARGET`、`BENCHMARK_RUN_NAME`、TT2/S2UTのvocoder設定はこの経路では使いません。

- **セル6**: common/test全件・dataset lockを確認し、testの日英WAVだけを既存ZIPから
  取得・検証。TT2/S2UT/Unit/Melの保存runやcheckpointを探さず、再学習・4A/4B再処理もしません。
- **セル7**: `configs/cascade/default.yaml`のSHA固定モデルによる
  Whisper日本語ASR → NLLB日英MT → Qwen英語TTS。モデルは段階ごとに別プロセスで動かし、
  同時にGPUへ載せません。モデル/設定/共通test以外の入力は不要です。
  参照日本語・参照英語をASR/MT/TTSの入力へ渡さず、ASR出力をMTへ、MT出力をTTSへ渡します。
  現在の既存Qwen adapterはBF16固定なので、生成にはBF16対応CUDA GPUが必要です。
  CPUへ暗黙fallbackしたり、別の音声合成モデルに置き換えたりはしません。
- **セル8**: S2UT単独評価と同じ固定英語Whisper・英語正規化・sacreBLEUを使用。
  失敗例は空仮説としてtest全件の母数に残します。BLASER/話者類似度は実行しません。
- **セル9**: `BENCHMARK_TARGET='cascade'`のままID/番号を指定し、保存WAVをCPUで試聴。

学習環境とは別の評価用環境receipt（`runtime.json`）と元の実依存manifest
（`environment.json`）を保存します。切断後は通常のセル1→6→7→8で実依存を復元し、
ASR/MTの32件ごとのjournal ZIP、32件グループのTTS WAV ZIP、評価ASR journalを検証・再利用します。
保存前の実行中グループは再処理され得ます。モデル・参照・コードを変えた場合は別の評価IDです。
CUDA OOM/device-side assertはprocess-fatalとし、残りを全件failed扱いで流しません。
サンプル固有の失敗は原因と中間文を残して評価へ進めます。
全件再利用できる段階ではモデルをロードせず、参照テキストで出力を補完することもありません。

保存先は`<実験>/benchmarks/cascade/<評価ID>/`です。既存`benchmarks/s2ut/`は変更しません。
`results/report.md`・`results/metrics.json`に単独評価レポート、
`results/per_sample.csv`に正解英語・生成音声の英語ASR・sentence BLEUに加えて
日本語ASR文（`asr_ja_text`）・MT英語文（`mt_en_text`）を保存します。
`predictions.jsonl`には段階別の処理時間・失敗理由、`audio/`にはWAV ZIP、
`audio-index.csv/json`にはIDとZIP内WAVの対応を残します。
4方式acceptanceとは別のレポートです。従来の4方式比較は`BENCHMARK_TARGET='all'`で維持します。

新規依存は追加せず、既存の`requirements/colab-inference.txt`に含まれる
`qwen-tts==0.1.1`、`transformers==4.57.3`などの固定依存を使用し、モデル取得前にimportを確認します。
CPU/モックでは通常セルの対象分岐・参照漏洩防止・ZIP復元・段階再利用・失敗の母数保持・
依存receipt一致・S2UT保存先の非変更をテストします。実Colab/GPU・実Qwen全件生成は未検証です。

#### Qwenトークナイザー取得エラーと、ASR/MTを保持する再開

以下のloader-only overlayは旧版の経緯です。現在の公式sampling/EOS対策では
**旧WAVへ合流せず、完了ASR/MTだけを共有する別評価**を作ります。
通常セル1→6→7→8を使用し、旧plan/receiptを書き換えないでください。
設定変更の理由・公式との差・検証範囲は[Qwen TTS監査記録](CASCADE_TTS_AUDIT.md)を参照してください。
新policyはmain/subtalker sampling=True、文ごとseed固定、120秒相当1501 token上限、
600秒の協調的時間保護を明示します。EOS未確認音声を切り詰めて成功扱いにはしません。
Qwen実GPUで問題が解消したことはまだ未確認です。

`qwen-tts==0.1.1`は指定revisionを受け取っても、内部の音声トークナイザーの
`cached_file`やprocessor取得でそのrevisionを引き継がない経路があります。
本体を`f00cf13...`から取得した一方、`speech_tokenizer`を別snapshotから参照して
`preprocessor_config.json`が見つからなくなる場合があります。
FlashAttention未導入の警告とは別の問題です。

修正版は固定SHAの**完全snapshotを先に取得し、root processor・tokenizer・重み・
speech_tokenizerの必須ファイル/JSONを確認してから、SDKへローカルディレクトリだけを渡します**。
SDKの内部取得で`main`へ戻ることを避け、既存のHub blobは再利用します。
セル7はGPU確認→保存済みASR/MTの読み取り専用検証→固定snapshot→実Qwenロード→
固定文 `This is a synthesis test.` の試験合成→未生成TTS、の順で進みます。
CUDA/BF16非対応時はロード前に停止し、GPU名・対応状況・bfloat16/SDPAをログへ残します。
試験合成はWAV保存・読み戻し・mono・有限値・非無音・durationを検証し、失敗なら
正式test音声を1件も生成しません。成功記録とWAVは別の`preflight/`へ保存し、
正式predictions/index・BLEU母数へ混ぜません。試験前後で乱数状態も復元します。
旧loader-only修正では依存version・モデルSHA・話者・言語・BF16/SDPA・正式デコード条件を変更していません。
新しいsampling policyは正式デコード条件の変更なので、上記の別評価方式を使用します。

`f9f1520047bca2c0e977b581cab70168892091e2`で**最初のTTSロード前に停止した評価**は、
対応修正版の`TRAIN_CODE_REVISION`で通常のセル1→6→7→8を実行してください。
セル6はモデル設定・test全ID/manifest SHAが同じなら元planの再利用を検討します。
レビュー済みGit差分（TTS loader/試験合成・通常launcher・説明/テスト）と、
worker内ASR/MT/identity関数のAST不変を確認し、
初回のloader修正採用はTTS WAVがない場合に限定し、`execution-overlays/<修正revision>.json`を保存します。
元plan・ASR/MT journal・完了pointer・runtime SHAを上書きしません。
元と同じeditable build位置から実依存を復元し、修正ソースだけを別のPYTHONPATHから使用します。
したがってASR/MT完了分を再計算せず、TTSから続行できます。
`ASR: reuse 1196/1196`・`MT: reuse 1196/1196`と表示し、Whisper/NLLBはロードしません。
pointer・identity・journal・ZIP・SHA・全ID・成功状態・入力文/時刻の対応を検証し、
不足や破損は明確に停止させます。旧結果を再生成したり、元ファイルを更新したりしません。
新規runで初めてASR/MTを生成する経路は残しますが、保存途中のtext段階がある場合も
自動再推論しません。不完全なtext段階をやり直す場合は別途原因・復旧方法の確認が必要です。
同じ修正revisionを再開する場合は、その採用後に生成したWAVも再利用します。
すでに`c741fca0c8408d67750a480cb989370fbcab9071`の修正採用記録がある場合は、
元plan SHA・採用receipt・Git系統・固定snapshot処理・モデルロード/合成引数のAST不変を
確認した場合に限り、生成済みWAVを保持して試験合成/再開対応版へ移行できます。
新しいoverlayへ親receiptのSHAを記録し、旧receiptやWAVを変更しません。
最終metrics/reportに元revisionと実行修正revisionを記録します。

検証済み修正採用記録のない旧版が既にWAVを生成していた場合、不変ZIPが未公開の場合、元plan/依存receiptが不一致の場合、
ASR/MT/モデル設定/依存を含む未レビュー差分がある場合は自動合流せず停止します。
古いsnapshotから音声トークナイザーだけを手作業でコピーしたり、`environment.json`や
保存planのrevisionを書き換えたり、コーパス/ASR/MT結果を削除したりしないでください。
CPUテストは実TransformersのEncodec feature extractorによるローカル設定読込も含みますが、
Qwen本体重みを使う実GPU合成は未検証です。

TTSは先に全保存グループのjournal/identity・WAV SHA/形式を確認し、
`TTS: saved=... remaining=...`、`[cascade-tts] reused=... computed=... remaining=...`を表示します。
成功済みWAVは再生成せず、`mt_en_text`だけを入力します。同じVMでの中断なら
未公開グループの成功済みWAVも再利用します。VMごと消失した場合はDrive保存済み32件単位から
再開するので、最後の最大31件は再生成され得ます（500件処理、480件公開なら480件を保証）。
全成功WAVが保存済みならQwenロード・試験合成も不要です。セル8の評価定義は変更しません。
ASR/MTバッチ化・TTSバッチ化・FlashAttention追加は今回実施していません。

実モデル用GPUテストは`tests/gpu/test_cascade_qwen_preflight.py`に分離しています。
固定Colab環境のL4で`S2ST_RUN_QWEN_GPU_TEST=1`を明示した場合のみ、モックなしで
Qwen本体/音声tokenizerのロードとWAV生成を行います。CPUテスト成功はL4実機成功を意味しません。
運用確認は同じ実験の通常セル1→必要なら6→7で、試験合成成功後の正式音声を確認し、
セル7を中断→再実行して保存済み件数と再利用ログを確認してください。追加セルは不要です。

### セル10: 任意日本語文の2方式試聴・参考BLEU・性能比較

セル1→10で使用します。Qwen生成の同じ日本語音声をCascadeと、
`INTERACTIVE_MODEL`で選ぶS2UT/TT2/S2T→TTSへ入力し、音声・ASR文・参考BLEU・速度・RAM/VRAMを並べます。
Cascade英訳は自動参照であり正解保証はありません。正式ベンチマークとは別の`interactive/`に保存し、
GPU推論は逐次実行します。性能値は初回1文の簡易測定で、キャッシュ再利用時は元の測定値です。
設定根拠、再開、性能の計測範囲は[任意文比較の説明](INTERACTIVE_COMPARISON.md)を参照してください。

### テストの範囲

工程別の保存単位・再開手順・旧データの制約は[中断・再開の範囲](RESUME.md)を参照してください。

ノートブック構文、テスト/通常の設定分離、全件処理の確認ガード、ログ表示コード、
既存の学習snapshot・再開処理、CPU/GPU工程分離、VM消失後の自動構築・環境再利用をCPU/モックでテストします。
実Colabでの環境構築・GPU学習・Drive切断復帰・実音声4方式E2Eは未検証です。
Python 3.10/Linux向けの学習＋推論依存解決は確認済みです。
