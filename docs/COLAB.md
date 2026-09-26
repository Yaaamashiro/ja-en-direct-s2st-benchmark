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

1. ColabでGPUランタイムを選択します。
2. セル1でコーパスの保存先・出力先・実験名を指定します。
3. セル2〜4でDrive接続、コード取得、Python 3.10環境構築、データ準備を実行します。
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

## 保存・切断後の再開

同じノートブック、実験名、設定で番号順に再実行します。
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
requirements/colab310.txtで従来の依存を固定し、setup.py --inferenceで
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
全件検証や`--resume`の意味、成果物の内容は変更していません。
修正前から動いているプロセスには自動適用されません。新コードを取得して起動した処理から有効です。

### fairseqコピーで停止した場合

`kaldi_self_train/st/utils`・`steps`の参照先が存在しないことによるコピー失敗は、
リンクを参照先の内容ではなくリンク自体としてコピーすることで回避します。
一時ディレクトリでコピーを完了してから公開し、完了マーカーがない既存コピーは
`/content/s2st-runtime/fairseq.incomplete-<ID>`へ退避します。削除はしません。
コピー中断時の一時ディレクトリも診断用に保持し、次回は新しいコピーを作ります。

修正版setupを取得した後は、同じVMでセル3を再実行できます。コーパス・checkpointや
インストール済みPython環境の削除は不要です。学習中にsetupを同時実行しないでください。
セル2は保存済みGit revisionに戻るため、新コードを使う場合は新しい実験名で開始し、
旧実験のrevision/環境lockを無理に上書きしないでください。

### テストの範囲

ノートブック構文、テスト/通常の設定分離、全件処理の確認ガード、ログ表示コード、
既存の学習snapshot・再開処理をCPUテストします。
実Colabでの環境構築・GPU学習・Drive切断復帰・実音声4方式E2Eは未検証です。
Python 3.10/Linux向けの学習＋推論依存解決は確認済みです。
