# Step 5: コンテンツ制作（音声・素材・Remotion実装）

## 実装前チェックリスト（Remotion）
- [ ] `remotion-best-practices/rules/transitions.md` を確認した。
- [ ] `remotion-best-practices/rules/timing.md` を確認した。
- [ ] `remotion-best-practices/rules/sequencing.md` を確認した。
- [ ] `remotion-best-practices/rules/measuring-text.md` を確認した。
- [ ] 遷移・要素モーションの採用パターンを `design/visual-spec.md` に記録した。
- [ ] テキスト計測に使うフォントを先にロードする実装方針を決めた。

## やること
- ナレーション音声を作成し、実再生時間で尺を確定する。
- Remotionでシーンを実装する。
- 文字サイズは固定値で決め打ちしない。コンポーネント内でオートスケールさせる。
- 素材配置と命名を統一し、差し替え容易性を確保する。

## 作成・更新するファイル
- `progress.md`
- `src/`（Remotionコンポーネント）
- `public/`（素材）

## 参照
- オートスケール実装指針: `references/remotion-autoscale-text.md`
- トランジション/モーション実装: `references/motion-transition-patterns.md`
- 命名規則: `references/project-file-conventions.md`

---
固定フォントサイズは、解像度差分で壊れる前提の設計。
