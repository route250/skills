# トランジションとモーション典型パターン

目的に対して動きを選ぶ。派手さは選定基準にならない。

## 1. シーン切り替え（Transition）
| パターン | 使う場面 | 視覚効果 | 過用リスク |
|---|---|---|---|
| Fade | 文脈が連続する切り替え | 視線に負担をかけない | 単調化 |
| Slide（左右） | 因果・時間進行を示す | 方向性を明示 | 多用で忙しい |
| Wipe | セクション境界を明確化 | 区切りが強い | 情報番組風に寄りすぎる |
| Zoom | 重要点へのフォーカス移動 | 焦点集中 | 酔いやすい |

## 2. 要素モーション（Within Scene）
| パターン | 使う場面 | 視覚効果 | 推奨 |
|---|---|---|---|
| Stagger（段差表示） | リスト・比較要素 | 読む順序を制御 | 要素間40-120ms差 |
| Emphasis pulse | 重要値の強調 | 注意を一点集中 | 1シーン1-2回まで |
| Position morph | 状態変化の説明 | 関係変化が直感的 | 始点終点を固定 |
| Opacity reveal | 補助情報の追加 | 情報密度を段階制御 | 本文より遅く出さない |

## 3. 時間とイージング指針
- UIモーションの基準として 200-500ms を中心に使う。
- ほとんどの進入/退出は ease-in-out 系で十分。
- 速すぎる切替より、意図が読める速度を優先する。

## 4. Remotionでの実装ヒント
- `@remotion/transitions` でシーン遷移を定義し、全体で一貫した文法を保つ。
- 複雑な組み合わせより、同一動画内で2-3パターンに制限する。

### 4.1 最小実装パターン（TransitionSeries）
```tsx
import {TransitionSeries, linearTiming} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';

<TransitionSeries>
  <TransitionSeries.Sequence durationInFrames={90}>
    <SceneA />
  </TransitionSeries.Sequence>
  <TransitionSeries.Transition
    presentation={fade()}
    timing={linearTiming({durationInFrames: 12})}
  />
  <TransitionSeries.Sequence durationInFrames={90}>
    <SceneB />
  </TransitionSeries.Sequence>
</TransitionSeries>;
```

### 4.2 方向付きスライド
```tsx
import {slide} from '@remotion/transitions/slide';

presentation={slide({direction: 'from-right'})}
```

### 4.3 スプリング遷移の無難な設定
```tsx
import {springTiming} from '@remotion/transitions';

timing={springTiming({
  config: {damping: 200},
  durationInFrames: 18,
})}
```
- 跳ねすぎを抑えるため `damping: 200` を基準にする。
- 説明動画でバウンス過多は、だいたい情報の敵になる。

### 4.4 スタガー表示（要素の段差投入）
```tsx
import {Sequence, useVideoConfig} from 'remotion';

const {fps} = useVideoConfig();
<Sequence from={0} premountFor={1 * fps}><CardA /></Sequence>
<Sequence from={4} premountFor={1 * fps}><CardB /></Sequence>
<Sequence from={8} premountFor={1 * fps}><CardC /></Sequence>
```
- `premountFor` を使って初回描画のチラつきを抑える。

### 4.5 実装上の注意
- `Transition` はシーンを重ねるため、総尺は短くなる。
- `Overlay` は総尺を変えない。切替地点の演出専用として使う。
- `Overlay` を `Transition` に隣接させない（設計制約あり）。

## 5. 参考ソース
- Material motion duration/easing: https://m1.material.io/motion/duration-easing.html
- Remotion transitions docs: https://www.remotion.dev/docs/transitions/
