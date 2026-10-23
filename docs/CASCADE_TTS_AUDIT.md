# Cascade Qwen TTS 設定監査・EOS対策（2026-10-24）

## 判断の前提

公式との差分は「公式値・従来値・理由・妥当性・検証範囲」を確認してから採用する。
公式値が不明な場合は推測を公式仕様と呼ばない。品質に影響する変更は別評価とし、
結果がよかった生成だけを選ぶことや、設定の自動フォールバックは行わない。
この変更はCascade/S2T→TTS共通のQwen adapterが対象。TT2/S2UT本体・学習設定は変更しない。

Colabの観測では、短い英訳 `The victim"s name is Sebastian Jones.` に対し、
main talkerのgeneration_stepが7566→7621と進み、8192上限へ近づいていた。
停止していたのではなく、長すぎる音声コード生成を続けていたことは確認できる。
これは**まだ最終的なEOS例外の観測ではなく、EOS未生成の原因を断定する証拠でもない**。
内側の`max_new_tokens=15`は残り15 codebookを出すsubtalkerであり、主系列の長さではない。

## 公式との差分と採否

基準は[固定モデルのgeneration_config.json](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/f00cf133b78d3c2c35857faba3b1be9b98c4f971/generation_config.json)、
[model config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/f00cf133b78d3c2c35857faba3b1be9b98c4f971/config.json)、
[codec config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/f00cf133b78d3c2c35857faba3b1be9b98c4f971/speech_tokenizer/config.json)、
固定`qwen-tts==0.1.1` wheel内のソース。変更可能なupstream mainだけを互換性の根拠にしない。

| 設定 | 公式/従来 | 採用する設定 | 理由と妥当性の判断 |
|---|---|---|---|
| main/subtalker sampling | 公式は両方True。従来は両方False | 両方True | 従来は決定的評価のためgreedy固定だったが、音声モデルでの適性確認が不足。公式へ戻す。今回のEOS問題を必ず解消するとは未確認。旧WAVとは別評価にする |
| samplingの他パラメータ | top_k=50、top_p=1、temperature=0.9、repetition_penalty=1.05。subtalkerも50/1/0.9 | 公式と同じ値を明示 | 任意のtemperature/penalty探索による品質変更を避ける。固定artifactと実際のtalker呼出引数の両方を検査 |
| RNG | 公式モデルの生成設定に文ごとのseed固定はない | base seed=17と正確なMT入力文のSHA256から文ごとに導出 | 再開・順序・試験合成による乱数消費の影響を避ける工学的追加。参照文は使わない。周囲のPython/NumPy/Torch/CUDA RNGは復元。GPU/ライブラリを跨ぐbitwise同一性までは保証しない |
| max_new_tokens | 公式8192、従来も8192 | 既存の受理上限120秒に対応する1500音声frame＋EOS=1501 | codecの実rateは24000/1920=12.5Hz。「12Hz」という名前から12で計算しない。従来も120秒超WAVは最終的に失敗だったため、その上限を生成前に適用。EOS強制挿入・切り詰めWAVの成功扱いはしない |
| main生成時間 | 公式に本ベンチマーク用の時間上限なし | 600秒、トークン生成後の協調的チェック | 12秒で55 main tokenという今回の観測なら1500 tokenは約330秒。余裕を持たせた運用上の保護であり、最適値の実測ではない。遅い環境で正当な生成を失敗にし得る差分なのでplanに保存する |
| dtype | 公式でBF16使用 | BF16を維持 | 今回は変更しない。非対応GPUで黙ってFP16/CPUへ落とさず、ロード前に停止 |
| attention | 公式例はFlashAttention 2を推奨、従来はSDPA | SDPAを維持 | 追加ビルド依存を避け、既存固定環境の互換性を保つ。FA2とbitwise同一とは主張しない。今回FA2追加による速度・数値変更は行わない |
| voice/language | CustomVoiceのOno_Annaは日本語話者、英語はサポート言語 | Ono_Anna/Englishを維持 | 話者変更と生成方式変更を同時に行わず、比較条件を保つ。英語のアクセント/音質が最適とは保証しない。これがEOS問題の原因という証拠はない |
| モデル・SDK | 固定CustomVoice revision、qwen-tts 0.1.1 | 同じSHA/依存を維持 | 別モデルや新版SDKを混ぜない。root・speech_tokenizerを含む完全snapshotをローカルからロード |

公式へ戻すsamplingと、独自の安全上限・seed固定は区別する。
このpolicyを「公式設定と完全同一」「EOS問題を実機で解消済み」とは呼ばない。
[upstream issue #341](https://github.com/QwenLM/Qwen3-TTS/issues/341)にもgreedyの退化報告があるが、
別モデル種別/環境の報告であり、今回の原因の証明には使わない。

## 実装上の確認事項

`qwen-tts==0.1.1`は公開SDKから受けた任意のkwargsを、そのままmain talkerへ転送しない。
単に`stopping_criteria`を公開APIへ渡す対策では監視が動かない。
adapterの呼出中だけ、**そのインスタンスのmain talker.generate**にobserverを接続する。
SDK全体の書き換えやsubtalkerへの誤接続はしない。

- audio EOSは固定model configの2150。text EOS 151673と混同しない。
- 実際に渡されたsampling引数・上限・audio EOSを検査する。
- Hugging Face本来のEOS停止判定を維持し、実トークン数とno_advanceをstderrへ表示。
  `checked=N/1501`の分母は生成予定量ではなく安全上限。EOSが出れば1501未満でも正常終了し、残りtoken数やETAの予測には使わない。
- 戻り系列末尾のaudio EOSを**codec decode前**に確認。未確認ならdecode/成功WAV保存をしない。
- 上限到達・EOS未確認は個別失敗。CUDA/依存/API契約不一致は全体停止。
- main token数、seed、policy、EOS有無、停止理由、生成秒数を`tts_generation`へ記録。
- 成功済みWAVの再利用でも、SHA/形式だけでなく同じpolicy/入力seed/EOS記録を要求する。
- 例外・KeyboardInterruptでもインスタンスのgenerate、RNG、adapter lockを復元する。

時間保護は**hard timeoutではない**。CUDA kernelが返らない場合や最初のtoken前、
codec decode中は即時割込みできない。kernelが戻った後のmain token境界で検査する。
正常EOSが出て終了した場合は、それ以前に時間上限を超えていてもEOS成功を優先する。
OS子プロセスの強制killと復旧は今回追加していない。

## 旧評価を保持する通常セルの流れ

ユーザーの「公式設定で別評価を作る」承認に基づく。

1. 対応コードを取得した通常セル1→6で、新policyを含む別の評価ID/planを作る。
2. 完了済みASR/MTのplan・pointer・runtime SHAを新planの`text_source`に固定する。
   同じtest ID、manifest、ASR/MTモデル条件であることを確認する。
3. セル7では旧ZIP/journal/入力対応/依存環境を読み取り専用で検証し、Whisper/NLLBはロードしない。
   欠落・破損・依存不一致なら停止し、旧textを自動再生成しない。
4. 固定非参照文で試験合成後、新しいTTSを全件生成する。旧WAVは再利用しない。
5. セル8で新WAVを評価、セル9で試聴。失敗を空仮説として全test件数を母数に含める従来BLEU定義は維持。

旧plan・ASR/MT・runtime・WAV・レポートは元フォルダに残る。
`latest.json`だけは新評価を指すよう更新し、直前のpointer bytesを新評価の`previous-latest.json`へ保存する。
途中再開は同じ新policyのセル1→必要なら6→7→8。新評価内の成功WAVのみ再利用する。
失敗行は通常の再実行時に再試行するが、sampling条件を自動変更したり、成功候補を選別したりしない。

新policyへ旧greedy WAVを取り込むため、旧loader-only overlayの安全ガードを緩めることは禁止。
元のeditable依存build位置も維持する。旧build復元の明示allowlistは従来の
`f9f1520047bca2c0e977b581cab70168892091e2`のまま。
別の未知revisionの依存環境を再利用する場合は自動許可せず、追加レビューが必要。
依存のreceiptやplanを書き換えて検査を回避しない。

## 再開・表示の追加修正

- グループ進捗にASR文字列を含む巨大なtupleを表示せず、part IDだけを表示する。
  同じASR文がログに出続けること自体はASR再実行/停止の証拠ではない。
- Driveへ公開済みpartと、その後のローカル再試行journalを照合して復元する。
  旧成功行/WAVは不変。旧失敗行の新結果と追加ローカル成功だけを保持する。
- 先頭partを再開しただけで、すでに保存済みの後続partをaudio-indexから消さない。
- 32件単位ZIPを維持。VM消失時は未公開の最後の最大31件が再生成され得る。

## 検証と未確認の境界

追加CPUテストは、実Transformersの小型モデルによるEOS/StoppingCriteria動作、
固定SDK wheelのkwargs転送構造、EOSなし/最終tokenのEOS/時間保護/不正引数、
RNGと例外復元、PCM変換前の異常検出、公式artifact契約、旧評価のbytes/mtime不変、
ASR/MT破損時のfail-closed、ZIP復元、再試行中断、index非縮小、通常セル分岐を含む。
小型モデルとSDK境界のdoubleを使ったテストは、本物のQwen推論を意味しない。

実GPUテストは`tests/gpu/test_cascade_qwen_preflight.py`へ分離。
`S2ST_RUN_QWEN_GPU_TEST=1`で固定L4/重みのEOS付き試験合成、
`S2ST_RUN_QWEN_AB_TEST=1`で非参照の固定3文についてbounded greedy/sampling診断を行う。
A/Bは両条件を同じseed・20秒相当token/180秒協調時間上限で実行し、成功・失敗を両方記録する。
短い診断budgetで失敗しても正式120秒budgetで失敗するとは限らない。
診断の旧greedy設定はテスト内だけで使用し、正式policy/評価には混ぜない。

今回、実Colabの操作・CU消費・Qwen実重みのGPU試験は行っていない。
実機の長時間生成/音質改善は別途確認が必要。依存追加・pip更新・CUDA/attention実装切替はしていない。
固定SDK wheelソースと、実Transformers 4.57.3のgeneration APIを検査し、
workerは固定依存環境receiptとの一致を引き続き要求する。

最終CPU検証: `pytest -q -m 'not gpu'`で **749 passed / 2 skipped / 7 deselected**。
Python 3.13/CPU上での結果で、7件のGPUテストは実行していない。
Swig/weight_normの既存非推奨警告6件、失敗0件。`git diff --check`も成功。
この結果はColab Python 3.10/L4での実重みE2E確認の代わりにはならない。
