# Colab中断・再開の範囲

同じ実験名、同じ出力先、同じコードrevision・設定でセル1を実行し、
止まった工程のセルを再実行します。ローカル環境が消えていればセル3相当が自動実行されます。
`RECHECK_AUDIO=False`、上書きはOFFのままにしてください。
「再開」は最後に永続保存された単位からの再実行であり、メモリ上の処理位置の復元ではありません。

## 工程別の確認結果

| 工程 | 保存・再利用する単位 | 中断時に再実行する範囲 |
|---|---|---|
| セル1〜3・ランタイム構築 | 保存済みrevision/環境lock、構築済みコンポーネント | 新VMではPython環境やライブラリの構築が必要。ビルド途中の命令は再実行 |
| 4A・コーパス取込 | 音声パス解決、WAV/SHA256検証をレコード単位で記録 | 未保存レコード。マニフェスト読込み・メタデータ確認は必要 |
| 4A・音素化 | サンプル単位の音素列 | 未保存サンプル。最終TSV/inventoryは保存結果から再構成 |
| 4A・Mel準備 | 入力WAVヘッダ、生成済み分散NPY、分割ZIP | 未保存NPYと作成途中のZIP。NPYは256フォルダへ分散、新規ZIPは原則512件または256MiBで分割（単一の巨大NPYは例外） |
| 4A・最終検証 | サンプル単位の検証成功記録 | 未保存サンプル。最終fingerprintのファイルハッシュ処理は再実行 |
| 4B・HuBERT | 抽出済みUnitとサンプル単位の記録 | 未保存サンプル/バッチ。モデル読込みは再実行 |
| 4B・学習データ準備 | サンプル単位のUnit読込み・WAVヘッダ結果、最終TSV/辞書 | 未保存サンプル。テキストラベル作成・TSV整合性確認は再走査 |
| 5・学習（TT2/S2UT/vocoder） | 確認済みの学習checkpoint/snapshot | 最後の永続保存後の更新。初回保存前なら最初から。ローカル読込みキャッシュは新VMで再構築 |
| 学習成果の復元・モデル取得 | 完成した成果物・ファイルのハッシュ確認 | コピー/ダウンロード途中のファイルは再取得する場合あり。バイト位置からの再開は保証しない |
| 推論（TT2/Cascade/S2T）・音声合成 | 成功したサンプルと結果journal | 未保存・失敗サンプル。モデル読込み・入力ハッシュ確認は再実行 |
| S2UT推論 | fairseq生成を約32サンプルずつのシャードに分け、成功シャードを保存 | 実行途中のシャード。各シャードでモデルを読み直すため、その分のオーバーヘッドあり |
| 評価 | サンプル単位の評価journal | 未保存・失敗サンプル。有効な追加指標が失敗したサンプルも再試行（そのサンプルのASRを含む） |
| 完了判定・集計 | WAV検証成功記録、同一内容のCSV/JSON/Markdown | 未検証WAV。整合性・学習checkpoint確認と集計計算は再実行。途中まで書かれた同一内容の出力は再利用 |

サンプルcheckpointは通常64件または記録時点で約10秒経過するごとにまとめて保存します。
強制終了では未公開分が失われます。正常な例外終了では残りを保存します。
ZIPとS2UT生成は成功したシャードごとに記録します。
学習は設定された更新間隔で保存されます。`durable_updates`が復旧基準です。
Driveマウントでの書込み/読戻し確認は行いますが、Google Driveサーバー側同期の完了は保証できません。

## 4A・4Bを完了後に再実行した場合

ノートブックは`direct_s2st.preparation_workflow`を呼び、正常終了した個々の工程を
`data/.prep-checkpoints/stages/`に記録します。
revision、設定内容、入力/出力のファイル一覧・サイズ・更新時刻が一致すれば重い処理を省略し、
`[preparation-stage] ... status=reused`を表示します。
ファイル数の多いDriveでは、このメタデータ確認にも時間がかかります。即時終了の保証ではありません。
途中までの場合は、その工程内のサンプルcheckpointから再開します。

コーパスは公開済みの不変データであることが前提です。高速な再利用判定では毎回全WAVをハッシュしません。
同じサイズ・更新時刻を維持したまま内容を変更すると検出できないため、原本は編集しないでください。
音声を再検証する場合は`RECHECK_AUDIO=True`を指定します。工程完了記録を迂回し、WAV検証をやり直します。
音素・HuBERT・Melの抽出結果まで無条件に全再生成するオプションではありません。
checkpointや完了記録の破損、設定不一致は黙って無視せず停止します。

## 既存の成果物について

- 旧形式の単一Mel ZIPは再分割せず、既存の参照位置とlock形式を維持します。ZIP索引の検証にも途中保存を使います。
- 新しい工程完了記録がない旧データでは、最初の1回は工程の確認が必要です。既存の互換checkpointは再利用します。
- 旧版が保存していなかった進捗は、今回の修正でさかのぼって復元できません。
- 実行中のColabに修正は自動反映されません。セル1は保存済みrevisionを使うため、リモート更新だけでも反映されません。
  既存実験のrevision/lockを手動で上書きせず、旧成果物を保持した上で移行してください。

## 固定eSpeak音素辞書の不足でMel準備が停止した場合

eSpeak NG 1.52.0の見本語由来の辞書には、trainで確認された
`a‍ɪ‍ə`, `a‍ɪ‍ɚ`, `o`, `r`, `ɐ`, `ɑ̃`, `ɔ`が含まれていませんでした。
準備処理はこの7種類を固定の補足集合として学習用辞書に加えます。
補足集合・学習用辞書のハッシュを新しいdata-lockに記録します。
元のphonemes/TSV、inventory、metadata、checkpointは保持するため音素化をやり直す必要はありません。
dev/testの未知音素を自動追加することはありません。すべてのsplitの辞書範囲をMel抽出前に確認します。
既に学習を始めたrunにこの新辞書を適用して再開することはできません。

切断後はセル1で同じ実験を指定してから、修正済みコードを取得し、
`scripts/colab/resume_preparation.py --persistent <実験保存先> --revision <新しい完全SHA> --profile pilot --overwrite`
を実行します。このスクリプトは保存済み音素化のhash/count/IDを全splitで確認し、
旧repository-revisionを退避して固定revisionだけを更新します。
この`--overwrite`はrevision固定情報の変更だけを許可し、音素・Melには`--resume`を使います。
環境を構築してMel準備と最終検証から続け、成功時に工程完了記録を保存します。
学習config/checkpointがある実験は停止します。終了後はGPUへ切り替えてセル1→4Bです。

環境再構築では、host Pythonへのuvインストールに旧`PIP_CONSTRAINT`などを継承しません。
Python 3.10の依存制約は仮想環境の構築後に適用します。

## Mel抽出開始直後のEOFErrorについて

`mel: extract/reuse features checked=0`の直後に`EOFError: No data left in file`が
出る旧版には、一時NPYを先に作成する処理と、固定fairseqの「既存ファイルは省略」
という既定動作の不整合がありました。公式抽出の呼出しでは新規の非公開一時ファイルに
限って`overwrite=True`を指定し、空ファイルのまま抽出が省略されないようにしています。
公開済み成果物・キャッシュを上書きする許可ではありません。
修正版へ移行し、同じ復旧スクリプトを再実行できます。
音素化・保存済み入力WAVヘッダの互換checkpointは保持しますが、
ログで`saved=0`だったMel特徴量自体には再利用できる抽出結果がありません。

## 音素生成と辞書の新方式への移行

新規の音素生成では、見本語だけでなく固定版eSpeak NG 1.52.0の
`base1 → en → en-us`音素定義からIPAを取得して、分割非依存の辞書を作ります。
他のG2P辞書との混合や、dev/testの出力に応じた自動追加はしません。
`don't`、`prefecture's`、`3.14`、略語、語内ハイフン、時刻などを分断せずeSpeakへ渡します。
生成方式と辞書方式のversion、出力TSVのSHAをmetadataに記録します。
辞書外の音素は、そのサンプルの生成直後にID/本文付きで停止し、不正ラベルを保存しません。
保存済みラベルもMel抽出前に全splitの辞書範囲・ID・metadataを検証します。

WAVを読まずに確認する場合:

```bash
python -m direct_s2st.cli translatotron2 validate-phonemes --profile pilot
```

通常の`--resume`は、完了済みの旧metadataがあれば旧音素生成方式を維持して再利用します。
旧/新方式のラベルは混在させません。完了metadataがない旧形式の途中結果で
辞書方式が一致しない場合は、明示的な移行を求めて停止します。
単なる7音素の補足は旧ラベルを再生成しません。短縮形・数字の生成も修正したい場合は、
学習開始前に修正済みコードを取得し、次の**明示的な再生成**を使います:

```bash
python scripts/colab/resume_preparation.py \
  --persistent <実験保存先> --revision <修正コードの完全SHA> \
  --profile pilot --overwrite --regenerate-phonemes
```

旧音素、旧学習用辞書、旧data-lockは`data/translatotron2/phoneme-migrations/<ID>/`へ
退避して保持します。音素生成はやり直すため時間がかかりますが、入力コーパス、保存済みMel、
Melの途中checkpoint、公開済みZIPは保持し、互換な特徴量を再利用します。
移行自体も途中保存されるため、切断したら同じコマンドを再実行できます。
学習config/checkpointがある実験への自動移行は拒否します。

## 検証範囲

CPUテストで中断・再開、完了結果の再利用、変更/破損の検出、ノートブック呼出し、
学習snapshot復元、推論/評価journal、集計の再実行を検証します。
実Colabの24時間切断、Drive同期障害、全14万件と実GPUによる全工程E2Eは未検証です。

## 大量のMel NPYでDriveのInput/output errorが出た場合

旧版は1個の`features/`フォルダに全NPYを保存していました。
[Colab公式FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html#drive-timeout)では、
フォルダ内の大量ファイルが`Input/output error`を招く場合があり、各フォルダを約1万件未満に
抑えることを推奨しています。同じエラーは操作/帯域quotaなどでも発生するので、
このログだけでファイルの破損・消失とは判定できません。

新方式はローカル作業ディスクでNPYを扱い、最大128件/目安256MiBずつのZIPを
`<Mel checkpoint root>/packs-v1/<先頭2桁>/pack-<ID>.zip`へ保存します。
フォルダを分けるだけでは個別読書き回数は減らないため、NPY単位のDrive書込みをやめました。
ZIP保存後にreceiptを公開した分だけが再開時に再利用されます。
抽出設定・checkpoint identity/keyは変えず、旧checkpointも上書きしません。
既存`features-v2/`のコピーも元SHA256で照合してZIPに取り込みます。
旧`features/`・`features-v2/`・chunkは削除・再生成・移動しません。

各NPYは元checkpointのSHA256と照合し、ZIPは書込み後の読戻しSHA256を確認します。
次回はZIPのパス/サイズ/更新時刻を保存した不変receiptが一致すれば内容読込みを省略します。
同じサイズ/更新時刻のまま内容を変更することは検出できません。必要時は`S2ST_PREP_RECHECK=1`で
内容を再検証します。これは既存Melの再抽出を指定するものではありません。

### 旧フォルダがマウント経由で読めないときの救済

`scripts/colab/repair_mel_cache.py`は、Drive APIで**旧フォルダを読み取り専用で**一覧取得し、
ファイルIDを使ってダウンロードできます。旧NPYのマウントパスを開くことはありません。
ダウンロード・ZIP作成はローカルで行い、保存済みZIPだけDriveマウントへ書き込みます。
元ファイル・checkpointは残すので、**救済ZIPに加え、その後の学習用ZIPの空き容量が必要**です。
APIの権限、ファイルの同期、Drive/APIのquota、保存先の書込み障害は別途解消する必要があります。
この手順で必ず復旧する・短時間で終わるという保証はありません。

### 推奨: ダウンロード済みフォルダZIPから救済する

DriveのWeb画面等で旧`features`をZIPとしてダウンロードできる場合、そのZIPを
Colabのローカルディスクへ置き、次の引数で取り込めます。分割ZIPは`--source-zip`を
繰り返してすべて指定します。ZIPを展開してDriveへNPYを戻す必要はありません。
Webからの大量ダウンロード自体も成功・短時間完了を保証するものではありません。

```python
MEL_CACHE = PERSISTENT / 'data/translatotron2/.prep-checkpoints/mel/2b2a74082e7462215d32ceca'
repair_args = [sys.executable, REPO / 'scripts/colab/repair_mel_cache.py',
               '--cache-root', MEL_CACHE, '--storage', 'packed',
               '--source-zip', '/content/features-part-1.zip',
               '--source-zip', '/content/features-part-2.zip']
run(*repair_args, '--limit', '3')
run(*repair_args)
```

各NPYのSHA256が保存済みcheckpointと一致したものだけを保存します。
ZIPにない未救済分は既存分散コピー、読める旧ファイルの順に参照します。
旧マウントが読めず不足分がある場合は下記API引数も追加できます。
ZIPはこのスクリプトが取得・削除するものではありません。VM消失後、未救済分の
読込みに必要な元ZIPは再び配置してください。保存済みpackは再利用されます。

### ZIPを用意できない場合: APIによる一度だけの旧データ移行

旧データをAPIで1ファイルずつ読む回数そのものは減らせません。batch APIもmediaの
一括ダウンロードには使えません。初回救済が長時間になる可能性は残ります。
一覧/ダウンロードを既定1秒間隔に抑え、403のrateLimitExceeded/userRateLimitExceeded・
429・一時的な5xxだけを最大8回バックオフ再試行します。制限に当たると要求間隔も
増やし、同じプロセス内で維持します。権限不足や日次制限等は無条件再試行しません。
1秒間隔がすべての割当に対して安全という保証はありません。

同じ実験のセル1を実行後、処理を実行していない状態で修正版checkoutを取得します。
`REVISION`にはこの修正を含む公開済みcommitの完全SHAを指定します。
保存済みrevisionはまだ手で編集せず、音素・Mel・checkpointも削除しないでください。

```python
import sys
ensure_workspace()
configure_preparation()
REVISION = '<修正版の完全SHA>'
run('git', '-C', REPO, 'fetch', 'origin', REVISION)
run('git', '-C', REPO, 'checkout', '--detach', REVISION)
```

DriveのWeb画面で、ログに出たcheckpointフォルダ内の**旧`features`フォルダ**を開き、
URLの`/folders/`より後のIDを控えます。例のcache rootは実際のエラーログのものに合わせます。
`features-v2`、`mel`、実験フォルダ全体のIDではありません。
認証はノートブック上で、マウントと同じアカウントに対して行います。

```python
from google.colab import auth
auth.authenticate_user()
MEL_CACHE = DATA / 'translatotron2/.prep-checkpoints/mel/2b2a74082e7462215d32ceca'
FEATURES_FOLDER_ID = '<旧featuresフォルダのID>'
# host Pythonを使用。仮想環境のPYTHONではありません。
repair_args = [sys.executable, REPO / 'scripts/colab/repair_mel_cache.py',
               '--cache-root', MEL_CACHE, '--storage', 'packed',
               '--drive-folder-id', FEATURES_FOLDER_ID, '--api-interval', '1']
run(*repair_args, '--limit', '3')  # まずSHA一致・保存先への書込みを3件で確認
```

成功したら同じセル内、または次のセルで上限なしでコピーします。

```python
run(*repair_args)
```

`[mel-recovery] packed=... reused=... persisted=... pending_local=... remaining=...`で
実際の進捗を確認できます。`packed`は今回取り込んだ件数、`persisted`はZIP保存済み、
`pending_local`は未公開のローカル件数です。強制終了時は未公開分だけやり直します。
3件の試行だけでは全件のquota/書込み成功は保証できません。
中断したら同じ認証・同じ引数で再実行します。保存済みpackは再ダウンロードしません。
分散コピー済み分は再ダウンロードせず、元を保持したままZIPに取り込みます。
完全なDriveファイルID一覧は同じcheckpoint rootへ保存し、再利用します。
一覧の途中で停止した場合は一覧取得をやり直します。APIの一覧が古い場合のみ
`--refresh-index`を追加し、旧一覧を退避して取り直します（旧NPY/chunkは変更しません）。
ファイルが欠落・重複している場合やSHA256が違う場合は黙って先へ進みません。
API認証/権限不足は`authenticate_user()`だけでは解消しない場合があります。
再試行上限でもquotaが続く場合は連続実行せず、要求間隔（`--api-interval 2`等）、
公式FAQの対処とアクセス権・空き容量を確認してください。バックオフはquotaを増やしません。

コピー完了後、保存済みrevisionを専用スクリプトで更新してMel準備と最終検証を再開します。
この`--overwrite`はrevision固定情報の更新を許可するだけで、Mel再生成ではありません。
既に音素移行が完了していれば`--regenerate-phonemes`は付けません。

```python
run(sys.executable, REPO / 'scripts/colab/resume_preparation.py',
    '--persistent', PERSISTENT, '--revision', REVISION,
    '--profile', PROFILE, '--overwrite')
```

これで互換な保存済みMelを使い、未保存分の抽出とZIP/TSV/最終検証へ進みます。
元の音声・音素・学習用辞書はこのストレージ修復で変更しません。
