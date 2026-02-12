# Step 5: コンテンツ制作（音声・素材・Remotion実装）

## 実装前チェックリスト（Remotion）
- [ ] `remotion-best-practices/rules/transitions.md` を確認した。
- [ ] `remotion-best-practices/rules/timing.md` を確認した。
- [ ] `remotion-best-practices/rules/sequencing.md` を確認した。
- [ ] `remotion-best-practices/rules/measuring-text.md` を確認した。
- [ ] `@remotion/cli` が導入済みで、`npx remotion preview` を実行できる。
- [ ] Google Fonts 利用時、指定する `subset` と `weight` の実在を確認した。
- [ ] 遷移・要素モーションの採用パターンを `design/visual-spec.md` に記録した。
- [ ] テキストサイズ計測に使うフォントを先にロードする実装方針を決めた。
- [ ] テキストの改行やはみ出しを防ぐための `remotion-autoscale-text.md` を確認した。
- [ ] `availableHeight` / `requiredHeight` の式で、主要シーンの縦方向予算を事前確認した。
- [ ] `pad` / `gap` / `fontSize` を固定値ではなく `clamp(min, preferred, max)` で設計した。
- [ ] `translateY` / `scale` などモーションの最大オフセットを見積もった。

## やること
- `brief.md` の `制作前チェック` 欄で Step 5 チェックリスト完了を明示し、`progress.md` に実施記録を残す。
- シーンごとにナレーションを*分割*して音声ファイル（例: `scene-01-narration.wav`）を作成する。
- 音声ファイルの各ファイルの長さ(秒数)を記録してシーンのトランジションのタイミングを確定する。
- Remotionでシーンを実装する。
- 実装時は各シーンで `requiredHeight <= availableHeight * 0.92` を満たす。
- 固定 `marginTop` / `marginBottom` の積み上げで高さを作らない。縦方向は `flex` + `gap` で管理する。
- 表紙・オープニング・クロージングを必ず実装し、本文シーンと視覚文法を統一する。
- グラフィカル的リッチなコンテンツを目指す。リッチなアイコン、グラフを積極的に模索、王道パターンに少しの工夫を加える。
- 文字サイズは固定値で決め打ちしない。コンポーネント内でオートスケールさせる。`remotion-autoscale-text.md`を参照
- Google Fonts 失敗時は `Noto Sans JP, Hiragino Sans, sans-serif` にフォールバックする。
- 素材配置と命名を統一し、差し替え容易性を確保する。

## 実装中の最小検査（シーンごと）
1. `frame 0`
2. 各要素の出現直後（delay通過直後）
3. 中盤
4. 終了直前
- 上記4点のどこかでセーフ領域を越えたら、そのシーンは未完了扱いにする。

## 作成・更新するファイル
- `progress.md`
- `src/`（Remotionコンポーネント）
- `public/`（素材）

## 参照
- オートスケール実装指針: `references/remotion-autoscale-text.md`
- トランジション/モーション実装: `references/motion-transition-patterns.md`
- 命名規則: `references/project-file-conventions.md`
- 適切なテキスト配置方法(はみ出しと改行を防げ): `remotion-autoscale-text.md`

---
チェック未完了で音声生成に入ると、だいたいシーン境界が壊れる。
