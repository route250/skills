---
name: sbv2-tts
description: Style-Bert-VITS2 音声生成を行う。uvxが必要
---

# sbv2-tts

Style-Bert-VITS2による音声生成コードと音声モデルをまとめたパッケージです。
参考)https://github.com/litagin02/Style-Bert-VITS2
同梱した `scripts/sbv2.py` を `uvx` で実行する。

## 実行手順

1. `uvx` が使えるか確認する。使えない場合は `uv` を導入してから実行する。
2. 使い方確認が必要な場合は必ず `--help` を実行する。
3. 音声モデルは、docs/models.mdに一覧がある。ライセンス条件に留意すること。
4. 実行は `/fullpath/scripts/run_sbv2_uvx.sh` を使う。音声モデルは最初に自動ダウンロードされるが少し時間がかかるかも。

```bash
/fullpath/script/run_sbv2_uvx.sh --model amitaro --text "おはよう" --output out1.wav  --text "こんにちは" --output out2.wav
```

## 読み上げ前のテキスト前処理の推奨
- アルファベット表記は、できるだけカタカナに変換してから入力してください。
- 読み間違いやすい漢字（例: 「方」「日」など）は、ひらがなに変換してから入力するほうが望ましいです。

## ライセンス注意

- モデルごとに利用条件が異なるため、実行前に `--models` の出力を確認する。
- 再配布・商用利用・クレジット表記・禁止用途の要件を必ず満たす。
- 迷う場合は同梱の `docs/TERMS_OF_USE.md` と各配布元ライセンスURLを優先して確認する。
