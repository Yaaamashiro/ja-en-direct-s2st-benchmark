# Google Colab：テストと通常実験

## 使うノートブック

| 用途 | ノートブック | データ・学習 | Drive保存先 |
| --- | --- | --- | --- |
| 最初の動作確認 | [colab_smoke.ipynb](../notebooks/colab_smoke.ipynb) | 各split 5件・2更新・小型TT2 | OUTPUT_BASE/smoke/実験名 |
| 通常の学習・比較 | [colab_training.ipynb](../notebooks/colab_training.ipynb) | 全件・更新数を明示指定・reference TT2 | OUTPUT_BASE/training/実験名 |

コーパス側のノートブックと同様、最初の設定フォームを編集して番号順に実行します。
コーパスのコード・データは変更しません。smokeの数値を品質評価として使わないでください。
通常用のreferenceモデルも原論文の完全な学習レシピや収束を保証するものではありません。

## 手順

`OUTPUT_BASE` の既定値は `/content/drive/MyDrive/ja-en-direct-s2st-benchmark-data` です。
既存実験を再開するときは、その実験で使用した出力先を引き続き指定してください。
既定値の変更によって旧保存先のデータが自動移動されることはありません。

1. ColabでCPUランタイムを選択します（最初からGPUでも実行できます）。
2. セル1でコーパスの保存先・出力先・実験名を指定します。
3. セル4AでCPUデータ準備を実行します。Drive接続・コード取得・環境構築（セル2・3相当）は自動実行されます。
   完了・Driveへの保存を確認後、GPUに切り替えて同じ設定でセル1→4Bを実行し、HuBERT Unit抽出とS2UT準備を行います。
4. セル1のTRAIN_TARGETをtt2 / s2ut / unit / melから選び、セル5で学習します。
   4方式比較には4つの学習済みrunが必要なので、各対象について繰り返します。
5. 学習が終わったらRUN_BENCHMARKをONにし、セル6〜8で復元・4方式推論・共通評価・比較を実行します。

通常用はCONFIRM_FULL_DATAをONにすると全コーパスを前処理します。
RUN_TRAININGは初期状態でOFFです。学習する場合だけONにし、
TOTAL_UPDATES（累計の更新数）とCONFIRM_TRAININGを設定してください。
データの全件使用と長期学習の許可は別々に確認します。勝手にfull学習を開始しません。
SESSION_MINUTESは一回の学習subprocess予算です。準備・保存・評価時間は含まれません。

smoke用は5件・2更新に固定され、通常用は件数制限を設けません。
通常用のPROFILE=pilotはCLI設定選択用であり、コーパスを10時間に切り詰めるものではありません。
前処理ではS2UTの補助ラベルやTT2の音素・melも生成します。

### CPU準備 → Unit抽出 → A100学習

| 工程 | セル | 内容 |
| --- | --- | --- |
| CPU（GPU不要） | 1 → 4A | 全件取込・WAV/SHA検証・音素化・TT2 Mel作成と検証 |
| GPU | 同じ設定で1 → 4B | HuBERT/クラスタリングによるUnit抽出・S2UT用データ作成と検証 |
| A100など学習用GPU | 同じ設定で1 → 5 | 選択モデルの学習／checkpoint再開 |

GPUの変更はColabのランタイム設定から手動で行ってください。切り替えるとPython変数も消えるため、
**セル1は再実行が必要**です。CORPUS_PATH・OUTPUT_BASE・EXPERIMENTは同じ値を使います。
以後のセルはDrive接続（認証操作が必要な場合あり）、保存済みrevisionのコード取得、Python/fairseq/eSpeak環境の
再構築、環境変数の復元を自動実行します。既存の構築済み環境は再インストールせず動作確認して使います。
セル2・3は手動で先に実行したい場合の入口として残しています。

4A完了後に4Bや5へ進む際、4Aは自動でやり直しません。準備データはDrive上の同じ保存先を参照します。
4BをCPUで実行するとGPU必須のエラーで停止し、黙ってCPU抽出へ切り替えることはありません。
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

セル1に次の設定があります。セル5の学習設定・バッチサイズは変更しません。

- `PREP_MAX_WORKERS=8`: CPU/I/Oワーカー数の上限（1〜32、実際にはCPU数以下）。2以下から開始し、約10秒ごとに処理速度・CPU使用率・空きRAMから増減します。Driveが律速で速度が落ちる場合は減らします。
- `HUBERT_MAX_BATCH=8`: HuBERTの最大バッチ（1〜32）。空きVRAMと測定速度で調整し、OOM時は縮小して再試行します。1件でもOOMなら停止して記録済み部分を残します。
- `RECHECK_AUDIO=False`: 再開時、前回SHA-256/WAV検証に成功し、期待SHA・パス・サイズ・更新時刻・行の内容が一致する音声は検証を再利用します。**毎回全音声のバイト列を再検証するにはTrue**にしてください。サイズと更新時刻を保った改変は通常の再利用では検出できません。コーパスを不変に保つ前提です。

4Aではパス解決・WAV/SHA検証・eSpeak音素化・Mel抽出・音声ヘッダー読込を並列化します。
4Bでは音声読込とS2UT用変換を並列化し、HuBERTモデルは1個を使います。
HuBERTの畳み込み正規化にパディングが影響するため、**同じサンプル数の音声だけ**を
最大16件の読込ウィンドウ内でまとめます。長さが異なる音声は単独推論になるため、
GPU使用率100%や最大速度、特定の高速化倍率を保証するものではありません。
精度変更（FP16/TF32の強制、有音部分の切捨て）はしません。

保存先は実験のDriveデータ領域です。コーパス側には書き込みません。

- WAV検証: `data/.prep-checkpoints/corpus/`
- 音素: `data/translatotron2/phonemes/.checkpoints/`
- Mel: `data/translatotron2/.prep-checkpoints/mel/`（抽出済みnpyも保持するためZIPとは別に容量が必要）
- Unit: `data/s2ut/units/.checkpoints/` と既存の`.units`ファイル

完了記録を64件または約10秒ごと（次の結果受取時）に小さなJSONへ原子的に保存します。
通常の例外・中断でも受取済み結果を保存します。VMの強制終了では未保存チャンクと
処理中の小さなウィンドウをやり直します。最終TSV/ZIPの作成中に止まった場合は、
抽出済みデータを使ってその集約工程をやり直します。破損した公開済み記録・成果物は
黙って上書きせず停止します。モデル・入力が変わった場合は旧結果を無条件再利用しません。

中断後は**同じ実験名・同じコードrevision・同じ設定でセル1→4Aまたは4B**を実行してください。
`--resume`はセルに設定済みで、VM消失/GPU変更後の環境構築も自動実行します。
旧コードで既に走っている処理には適用されません。旧処理が保存していない途中結果は復元できません。
同じ出力に複数の準備プロセスを同時実行しないでください。

ログには`checked`・`remaining`（総件数が既知の場合）、CPUワーカー数・速度・空きRAM、
HuBERTの実バッチ数・次の上限・空きVRAM、終了時のcheckpoint保存/再利用件数が出ます。
CLIでは対応する環境変数`S2ST_PREP_WORKERS`、`S2ST_HUBERT_BATCH_MAX`、
`S2ST_PREP_RECHECK=1`を指定できます。`S2ST_PREP_ADAPTIVE=0`はCPUワーカーを上限に固定します。

### セル5の常駐・適応学習（任意）

通常ノートブックの `TRAIN_OPTIMIZE=True` は、1 GPU・1学習プロセスをセッション中
常駐させます。従来の100更新ごとの再起動は行わず、`CHUNK_UPDATES` は旧方式だけに適用します。
APIの `make_config` は互換性のため `optimize=False` が既定です。

- `TRAIN_MAX_WORKERS=8`: 読込み並列数の上限。TT2/vocoderはCPU負荷・空きRAM・測定速度で約10秒ごとに調整。S2UTはfairseqのepoch境界でのみ調整します。0は逐次読込みです。
- `TRAIN_CACHE_GB=8`: ローカル音声/ZIP範囲キャッシュの総容量上限。複数worker・終了済workerの残存分も含みます。空きディスクを1 GiB残し、収まらないデータは元ファイルから読みます。コーパスやDrive原本には書きません。セッション終了時に自分の一時キャッシュのみ削除します。
- `TRAIN_SAVE_INTERVAL=50`: optimizer更新単位の保存間隔。最終更新・時間予算到達時にも保存します。ローカル固定コピーは同期、その後のDriveコピー・SHA-256検証は別スレッドで1件ずつ実行。未確認コピーは最大2件で、遅いDriveには待ち合わせます。
- `TRAIN_ADAPTIVE_BATCH=False`: ONはTT2/S2UT限定。論理バッチ・サンプル順・更新回数を維持して分割を調整します。ただしTT2のBatchNorm統計やdropoutは変わるため、同一学習結果は保証しません。別の実験条件です。vocoderは2 optimizerを一組として扱い、分割・OOM再試行はしません。
- `TRAIN_PRECISION='default'`: 既存精度を維持（TT2/vocoder FP32、S2UTのレシピFP16）。明示的にFP32/BF16を選べます。BF16非対応GPUでは停止し、黙って精度を変更しません。

自動分割ON時だけ、optimizer更新前のCUDA OOMを捕捉して勾配・RNG・モデルbufferを戻し、
同じサンプルを小さい分割で再試行します。1件でもOOM、optimizer中のOOM、device assertは停止します。
校正は短い試行後に固定し、以後のOOMでは縮小します。GPUが変わると分割校正をやり直します。
これは最大速度を保証する探索ではありません。GANのOOMは保存済みcheckpointから復旧してください。

`[training-performance]` は処理時間の累計、frames/units等の速度、CPU・RAM・VRAMを表示します。
CUDA時間には区間内の待ちも含まれ、GPU稼働率そのものではありません。
`[recovery] learned_at_least` はローカル保存済み更新、`durable_updates` はDriveマウントへの
コピー検証まで完了した更新です。Driveサーバー側同期の完了までは保証しません。
切断直前の未保存/未確認更新は再実行します。時間予算はupdate境界で判定し、保存処理の完了待ちは
予算を超える場合があります。30秒の猶予後も学習が止まらなければ終了を要求し、最後の確認済み保存を使います。

旧runのconfig/revisionを上書きして移行しないでください。新設定は新runで利用します。
同一設定の再開は従来どおりセル1→5。実GPUでの速度・BF16品質・実Drive切断復旧は未検証です。

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

### テストの範囲

ノートブック構文、テスト/通常の設定分離、全件処理の確認ガード、ログ表示コード、
既存の学習snapshot・再開処理、CPU/GPU工程分離、VM消失後の自動構築・環境再利用をCPU/モックでテストします。
実Colabでの環境構築・GPU学習・Drive切断復帰・実音声4方式E2Eは未検証です。
Python 3.10/Linux向けの学習＋推論依存解決は確認済みです。
