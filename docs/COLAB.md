# Google Colabでの学習と再開

[notebooks/colab_training.ipynb](../notebooks/colab_training.ipynb)をColabへアップロードし、
GPUランタイムで上から実行します。このノートブックはコーパスの受け入れ・前処理と、
TT2／S2UT／unit・mel vocoderの学習を扱います。Dockerは使いません。

## 保存先

- コーパス：Google Drive上の既存コーパス。変更しません。
- 派生データ：Drive上の専用DATAディレクトリ。smokeと本学習を分けます。
- 作業用checkpoint：`/content/s2st-work/<run-name>`。
- 世代バックアップ：Driveの`checkpoints/<run-name>/<update>-<uuid>`。
- ランタイム：`/content/s2st-runtime`。VMを作り直すたびにセットアップします。

初回のrepository revisionをDriveへ固定し、Python 3.10.18、torch/torchaudio 2.7.1、
fairseqとeSpeakの既存固定commitで独立環境を構築します。Colab標準のPython/torchは
学習に使いません。依存一覧も保存・照合します。依存解決結果やコードが変わった場合は
再開を拒否します。使い始める前に、このColab対応コードが含まれるrevisionを取得してください。

通常のDocker専用`full` CLIガードを偽装・解除する仕組みではありません。
Colabでは専用ドライバがnative trainerを明示的に呼び出します。
既定は2更新、累計10更新超は`--confirm-training`が必要です。
Colab実機での依存インストール・GPU実音声学習は別途検証が必要です。

## 切断に備えた仕組み

1. `chunk-updates`だけ学習し、trainerを正常終了させます。
2. checkpoint内の累計更新数とoptimizer状態を確認します。
3. checkpoint・設定・ログを新しいDrive世代へコピーし、SHA-256を照合します。
4. 全ファイルのコピー後に`snapshot.json`を書き、その世代を再開可能にします。
5. 次の区間は既存checkpointから続行します。

TT2はモデル・optimizer・RNG・rank別buffer、vocoderは生成器・識別器・両optimizer・RNG、
S2UTはfairseq標準の学習状態を保存します。S2UTの区間再開が連続GPU実行とbitwise一致する
とは主張しません。単一GPU用で、複数Colabから同じrunへ同時書き込みしないでください。

切断後は同じ保存先で上からセットアップし、学習セルの`RESUME=True`にして実行します。
`TOTAL_UPDATES`は追加回数ではなく累計到達点です。増やして続行できます。
データ・モデル構造・学習率等は変えず、変更時は新しいrunを開始してください。
復元前のローカル作業ディレクトリは`.interrupted-<uuid>`へ退避し、勝手に消しません。

コピー途中や破損した最新世代は採用せず、正常な以前の世代へ戻ります。
保存完了前の区間は再計算します。最初のバックアップ前に切れた場合は再開点がないため、
ローカル残骸を確認し、新しいrun名で開始してください。
ディスク容量不足・学習エラー時も既存バックアップを保持して停止します。
世代の自動削除はしません。容量を監視し、検証済み世代を残して手動整理してください。

## 時間・容量の調整

Colabのidle timeout、利用可能GPU、最大稼働時間は変動します。固定の時間まで動く保証は
ありません。[公式FAQ](https://research.google.com/colaboratory/faq.html)を確認してください。
自動再接続、keep-aliveなどの制限回避処理はありません。

`SESSION_SECONDS`は学習subprocessへ渡す残り時間です。上限で実行中の区間を停止し、
最後にDrive保存が完了した区間を再開点にします。初期化・データ照合・コピーには別途
時間がかかるため、サービスの上限より十分短く設定してください。
Driveへの書き込みがクラウドへ同期する時刻までは保証できません。

まず`CHUNK_UPDATES=1`で保存まで確認し、更新時間と保存時間を測って調整します。
短い区間は再計算量を抑えますが、モデル再ロード・保存のコストが大きくなります。
バックアップはrun全体を世代ごとに保存するため、大規模モデルではDrive容量を多く使います。
不要な推論音声・巨大データをwork-rootへ置かないでください。

Driveの小ファイルI/Oは遅くなる場合があります。最初はDrive上の派生データを使い、
必要なら同じ内容を毎回同じ`/content`パスへ復元してから学習してください。
prepared manifestに絶対パスが含まれるため、途中で保存先だけを変えて再開しないでください。

## 本学習と評価

ノートブックのsmoke設定は品質評価用ではありません。新しいDATAとrun名で本学習データを
準備し、モデルサイズ・学習率・batch size・更新数を明示的に決めます。
`make_config`のTT2はsmoke／接続確認用レシピで、model-sizeだけを変えても論文の全設定には
なりません。[reference recipe](REFERENCE_PARITY.md)を参照してください。
S2UTの`max-tokens=2000`もメモリ使用量の保証ではありません。

推論・vocoder・評価は既存CLIを使用し、復元先のcheckpointを推論設定に指定します。
Cascade／評価を使う場合は固定Python環境へ`requirements/cascade.txt`／
`requirements/evaluation.txt`を追加インストールしてください。学習開始後に依存を変更せず、
変更が必要なら環境lockを更新して別runとして扱います。
3方式Docker一括実行の`suite.py`はこのノートブックでは使いません。
実音声のE2E成功は学習区間の終了とは別に確認します。
