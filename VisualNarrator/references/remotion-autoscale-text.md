# Remotion文字オートスケール指針

固定フォントサイズは解像度差分で破綻しやすい。コンポーネント内で自動調整する。

## 1. 基本方針
- テキストは「箱」に入れて扱い、箱サイズに合わせてフォントサイズを計算する。
- 幅だけでなく高さ制約も評価する。
- 最小/最大フォントサイズを必ず設定する。
- 前提は `remotion-best-practices` の `rules/measuring-text.md`。

## 2. 推奨アプローチ
1. `@remotion/layout-utils` の `fillTextBox()` を利用し、
   箱に収まるフォントサイズを算出する。
2. 計算値を `fontSize` に反映し、オーバーフロー時は行数制限で制御する。
3. 長文は先に文を分割し、1ボックス1メッセージを守る。

## 3. 実装テンプレート
```tsx
import {fillTextBox} from '@remotion/layout-utils';

const fit = fillTextBox({
  text,
  box: {width: boxW, height: boxH},
  fontFamily: 'Noto Sans JP, Hiragino Sans, sans-serif',
  fontWeight: '700',
  maxFontSize: 96,
  minFontSize: 24,
  lineHeight: 1.25,
});

return <div style={{fontSize: fit.fontSize, lineHeight: 1.25}}>{text}</div>;
```

## 3.1 計測フォントを固定し、失敗時フォールバックを持つ
```tsx
import {measureText} from '@remotion/layout-utils';
const FONT_STACK = 'Noto Sans JP, Hiragino Sans, sans-serif';

const measured = measureText({
  text,
  fontFamily: FONT_STACK,
  fontSize: 48,
  validateFontIsLoaded: true,
});
```
- Google Fonts を使う場合、`subset` と `weight` の実在を先に確認する。
- `loadFont()` 失敗時は Google Fonts 依存を即時解除し、`FONT_STACK` にフォールバックする。

## 3.2 幅優先の見出しと箱優先の本文を分ける
- 見出し: `fitText()` で幅基準の最大化を行う。
- 本文/注釈: `fillTextBox()` で高さ込みの収まりを保証する。

## 4. 運用ルール
- 画面全体で固定値を使い回さない。コンポーネント単位で計算する。
- 可読下限を割る場合は、フォント縮小ではなく文を分割する。
- 日本語長文は禁則処理より先に文圧縮を検討する。
- 計測時と描画時で `fontFamily` / `fontWeight` / `letterSpacing` を一致させる。
- テロップ連続表示では `Sequence` の `premountFor` を使い、レイアウト揺れを抑える。
- 白画面発生時は UI 操作より先にプレビューログの `font` / `bundle` エラーを確認する。

## 5. 参考ソース
- Remotion layout-utils (`fillTextBox`): https://www.remotion.dev/docs/layout-utils/fill-text-box
- Remotion layout-utils index: https://www.remotion.dev/docs/layout-utils/
