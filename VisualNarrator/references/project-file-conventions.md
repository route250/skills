# プロジェクト構造と命名規則

## 1. 作業場所
- 各案件で専用フォルダを1つ作成する。
- スキルディレクトリ（`.../VisualNarrator`）には成果物を作らない。

## 2. 推奨構造
```text
<project-root>/
  brief.md
  progress.md
  storyboard.md
  research/
    sources.md
    license-log.md
  design/
    visual-spec.md
  src/
  public/
  review/
    review-log.md
  deliverables/
```

## 3. 命名規則
- 進捗: `progress.md` 固定
- ストーリーボード初版: `storyboard.md`
- 改訂版: `storyboard-v2.md`, `storyboard-v3.md`
- シーン別素材: `scene-01-bg.png`, `scene-01-narration.wav`
- エクスポート: `YYYYMMDD-title-v01.mp4`

## 4. 更新ルール
- 仕様変更時は `progress.md` に「日時/変更理由/影響ファイル」を必ず記録する。
- レビュー反映時は `review/review-log.md` に「指摘/対応/未対応理由」を残す。

合理性は命名から始まる。曖昧なファイル名は再作業の予約語。
