# Step 6: プレビュー・レビュー反映・最終出力

## やること
- プレビューは `npx remotion preview` で実行し、起動ログを必ず保持する。
- プレビューで可読性、情報密度、テンポ、同期ズレを確認する。
- 文字が小さすぎないか、背景/図形/枠線/文字の色トークンが全シーンで一貫しているかを確認する。
- レビュー指摘を分類して、対応ステップへ戻して修正する。
- 最終レンダリング条件を満たして出力する（例: 480p / 12fps、要件優先）。

## プレビュー依頼時のサーバ確認
1. 依頼前に `pgrep -fl "remotion preview"` で Remotion サーバの起動有無を確認する。
2. 未起動なら `npx remotion preview` で起動する。
3. 起動直後は不安定なため、5秒待ってから再度 `pgrep -fl "remotion preview"` で生存確認する。
4. 5秒後にプロセスが消えている場合は、依頼より先に起動ログを確認して原因を処理する。

## 白画面時の優先確認
1. プレビュー実行ターミナルで `font` / `bundle` エラーを確認する。
2. `@remotion/google-fonts` 利用時は `subset` / `weight` 指定の不整合を確認する。
3. `remotion` コマンド未解決なら `@remotion/cli` 未導入を疑い、解消後に再確認する。

## はみ出し検査ルール（全シーン共通）
- セーフ領域を先に確定する。`availableHeight = compHeight - padTop - padBottom`、`availableWidth = compWidth - padLeft - padRight`。
- 縦積みは合計高さで判定する。`requiredHeight = Σ(textBlockHeight) + Σ(gap) + visualHeight + motionOvershoot`。
- 合格条件は `requiredHeight <= availableHeight * 0.92`。余白8%未満はレビューで要修正とする。
- 固定 `marginTop` / `marginBottom` の積み上げで高さを作らない。縦方向は `gap` と `flex` で管理する。
- `AnimatedText` 等の `translateY` / `scale` は最大オフセット時も判定対象に含める。

## フレーム検査ポイント（最小）
1. `frame 0`（初期状態）
2. 各要素の出現直後（delay通過直後）
3. 中盤
4. 終了直前
- 4点のいずれかでセーフ領域を越えたらNG。ズームで「見えている」は無効。

## 可変レイアウト指示（固定値回避）
- `pad` / `gap` / `fontSize` は解像度比率で算出し、`clamp(min, preferred, max)` で上下限を持たせる。
- 見出しは幅基準、本文は箱基準でオートスケールする。実装は `references/remotion-autoscale-text.md` を参照する。
- 可読下限を割る場合は縮小で粘らない。文分割かシーン分割に戻す。

## 作成・更新するファイル
- `progress.md`
- `review/review-log.md`
- `deliverables/`（最終出力）

## 参照
- モーション見直し: `references/motion-transition-patterns.md`
- 配置/可読性見直し: `references/infographic-patterns.md`, `references/color-patterns.md`
- 文字オートスケール: `references/remotion-autoscale-text.md`

---
レビューを儀式にすると品質は上がらない。差分を残して、原因と対処を記録する。
