# Unified Timeline Video — Experimental 1

## 状態

`main`向けExperimental機能です。実装基準ソース: `085943ad9cb8023a2042d4d731c83e638361c06e`。

ローカル受入はPASSです。Full CPU suiteは`1424 passed / 1 skipped / 0 failed`、実ComfyUIのブラウザ保存・再読込、Follow／RepeatのGPU生成、A/B mode分離、Manifest整合を確認しました。両GPU runは704×416、24fps、240フレーム、映像10秒、32kHz stereo音声10秒で完走し、OOM／NaN／allocation failure／crashはありません。Production標準への昇格やRelease/tag／Registry公開を意味せず、機能名どおりExperimentalです。

## 接続と設定

```
Core Load Video
    ↓ VIDEO
H3 Continuum Video Adapter  (Force Rate = 24)
    ↓ images / IMAGE
H3 Continuum Sampler V3.8
    └ Timeline Video Frames   [内部名 reference_video_1 を維持]
```

Video AdapterとCore Load Videoは既存のままです。AudioはCore Load Audioを利用し、この機能が動画素材の音声を勝手に追加することはありません。

動画入力を有効にすると、Samplerに **Video Reference Mode** が表示されます。
この設定は **Prompt Format = Timelineとは別**です。設定を増やすのはこの1項目のみで、新しい動画ソケット、Loader、Enableは追加しません。

### Follow Timeline

生成全体の時間位置に対応する参照区間を渡します。20秒素材を2×10秒で使うと、前半グループは先頭側、後半グループはその続きです。
厳密な境界はH3の17k+5フレーム整列と既存の出力計画に従います。例えば最初の10秒グループは243フレームとなり、次の参照開始は243/24=10.125秒です。単純な10.000秒への巻戻しではありません。

対象は **実際に新しく出力へ追加される自然長** です。Continuationの過去フレームprefixは二重に消費しません。
Terminal Mergeでは、統合された1回のSamplingが担当する可視区間をまとめて使用し、Merge自体やSeedの決定を変更しません。
Resume／Reviewでは保存済みのnet_framesを積み上げる既存カウンタを使い、Queueのたびに参照開始位置を0へ戻しません。

素材が終わったら、以後は動画参照なしで続行します。終端をまたぐ場合は残りを使い、残りが5フレーム未満ならそのグループの動画参照をスキップして診断を表示します。
自動ループ、長い静止画保持、Stretchは行いません。通常のH3整列に必要な最大16フレームの短い終端補完だけは残します。

最後のグループを「今回要求した総尺」で参照クロップしない理由: 総Chunksを増やしたとき、保存済みグループの本来の参照入力が変わってしまうのを防ぐためです。実際の最終映像の尺調整は既存Assemblyのままです。

### Repeat Reference

従来のVideo Guide Frames処理をそのまま使います。毎グループで同じ先頭側の、1チャンク上限内の素材を参照します。
短い素材を必要尺まで何回もコピーする「ループ再生」ではありません。
旧ワークフローや旧APIに新設定がない場合はこのモードになります。短い／長いという素材長だけでモードを自動変更しません。

### できないこと／保証しないこと

- 元動画の動き、画素、人物、カメラを完全コピーする制御ではありません。参照映像は外観にも影響します。
- 正しい区間を渡すことと、生成結果が意図した動きに一致することは別の評価項目です。
- 単独15秒生成がstaticになるという報告やIssue #13の長尺driftの修正ではありません。
- 長い入力のRAM削減機能ではありません。Followは全24fpsフレームの正規化済みCPUコピーを保持します。
- 真のVFRタイムスタンプ追従や補間は追加していません。入力IMAGE列は24fpsという既存契約です。

## 内部構成

- `video_reference_modes.py`: Follow source、物理可視区間選択、Qwen/VAE共通の既存エンコーダ呼出し、slice identity。
- `v3/driving_nodes.py`: V38へ省略可能なscalar widgetを末尾追加。V34入力境界でモードを消費し、Repeatは元の関数と元の引数を保ちます。
- `v2/sequence.py`: 新sourceは既存の`reference_video_source`引数で運びます。旧`timeline_video_source`は使わず、Terminal Merge無効化を回避します。
- 物理グループごとに別のprompt cache辞書を作り、同じ文章でも前の区間のconditioningを再利用しません。
- VAE cacheの`source.combined_hash`にslice identityを入れます。共通キャッシュの実装は変更しません。
- Run identityはFollow／Repeat、素材内容、サイズ、chunk_seconds、前処理・終端方針を区別します。Chunks総数は含めません。
- 新規Followグループのplanに`_h3_continuum_video_reference_v1`を追加し、実参照範囲を記録します。Terminalの論理2区間には共通の物理slice情報が付きます。
- UIは既存のネイティブwidgetを保存し、独自配列入替えを追加しません。古いワークフローの読み込み時は新モードをRepeatへ復元します。

## 配布ワークフロー

適用ツールは元の公式JSONを変更せず、次の比較用コピーを生成します。各JSONと同じ内容のZIPも作成します。

- `MiniMax_H3_Continuum_V38X2_Timeline_Experimental_Follow.json`
- `MiniMax_H3_Continuum_V38X2_Timeline_Experimental_Repeat.json`

変更はSamplerの新設定と動画入力の表示名のみです。元の画像、プロンプト、Seed、Run名、バイパス状態、外部ノード設定・接続は維持します。
元の動画入力グループがBypassなら、そのままです。試験時にグループ全体を有効化してください。Core Load Videoだけを無効にしてAdapterを有効のままにしないでください。

## 受入結果と残る境界

1. Full CPU suite: `1424 passed / 1 skipped / 0 failed`。
2. ブラウザ: Followを保存し、再読込後も`Video Reference Mode`と`Timeline Video Frames`を保持。
3. Follow GPU: Chunk 1とChunk 2で別の参照区間を使用し、区間ごとに別conditioning／VAE cache identityを確認。
4. Repeat GPU: 2 physical groupsで従来のpersistent参照を維持し、VAE conditioningはrun内で1回。
5. A/B: Chunk 1はほぼ一致し、Chunk 2から明確に分岐。2 modeが同じconditioning経路へ潰れていないことを確認。
6. Integrity: 出力仕様、AV duration、Manifest／Registry Manifest、OOM／NaN／crashなしを確認。

未検証の全モデル・解像度・長尺・source exhaustion・Resume／Retry組合せを一括保証しません。参照区間が正しく選ばれることと、生成映像が意図したmotionを忠実に再現することは別の評価です。Production標準への昇格、Release/tag、Registry公開は別判断です。
