---
name: VisualNarrator
description: 図形・レイアウト・動きで情報を視覚的な物語に変換し、理解と意思決定を支援するスキル。スライドや動画プレゼンの構成設計、インフォグラフィック中心の表現設計、事実確認、Remotion実装、レビュー反映が必要なときに使う。
---

# VisualNarrator

## 実行原則
- 作業開始時にプロジェクト専用フォルダを1つ作成する。
- 成果物はプロジェクトフォルダにのみ作成する。`VisualNarrator` スキルディレクトリに成果物を作らない。
- 進捗は常に `progress.md` に記録し、仕様変更時は関連ファイルを同時更新する。
- 不足情報がある場合は推測で埋めず、課題とゴールを質問して確定する。
- インフォグラフィックを優先し、文字は補助として使う。
- Remotion実装は `remotion-best-practices` スキルを前提にする。

## 標準ワークフロー
1. 目的と対象を定義する（`step-1.md`）
2. メッセージとシーン構成を設計する（`step-2.md`）
3. 事実・出典・ライセンスを確認する（`step-3.md`）
4. 視覚設計（インフォグラフィック・色彩・モーション）を決める（`step-4.md`）
5. 音声・素材・Remotion実装を行う（`step-5.md`）
6. プレビュー、レビュー反映、最終出力を行う（`step-6.md`）

## 参照資料
- インフォグラフィック型・画面占有率・典型レイアウト: `references/infographic-patterns.md`
- 色彩パターン（用途別・視覚効果別）: `references/color-patterns.md`
- トランジションとモーション典型: `references/motion-transition-patterns.md`
- Remotion文字オートスケール実装: `references/remotion-autoscale-text.md`
- プロジェクト構造と命名規則: `references/project-file-conventions.md`
- Remotionルール群（前提）: `remotion-best-practices` の `rules/transitions.md`, `rules/timing.md`, `rules/sequencing.md`, `rules/measuring-text.md`

## 品質ゲート
- 目的と対象が1文で説明できない場合、次工程に進まない。
- 1シーン1メッセージを守る。
- 根拠のない主張を採用しない。
- レビュー未反映で完了扱いにしない。
