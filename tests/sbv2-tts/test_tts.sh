#!/bin/bash

set -euo pipefail

PrjDir=$(cd $(dirname $0)/../..; pwd)
SkillDir=$PrjDir/sbv2-tts
ScriptPy=$SkillDir/scripts/sbv2.py
ProjectDir=$SkillDir/scripts
WorkDir=$PrjDir/tmp/sbv2-tts
VenvDir=$WorkDir/.venv
UvCacheDir=$WorkDir/.uv-cache

function find_uv_bin() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    for p in \
        "/opt/homebrew/bin/uv" \
        "/usr/local/bin/uv" \
        "$HOME/.local/bin/uv" \
        "$HOME/.uv/bin/uv" \
        "$HOME/bin/uv"
    do
        if [[ -x "$p" ]]; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

UV_BIN="${UV_BIN:-$(find_uv_bin || true)}"
if [[ -z "$UV_BIN" ]]; then
    echo "uv が見つかりません。先に uv をインストールしてください。"
    exit 1
fi

mkdir -p "$WorkDir" "$UvCacheDir"

# テスト時は仮想環境とキャッシュを tmp 配下に固定する。
export UV_PROJECT_ENVIRONMENT="$VenvDir"
export UV_CACHE_DIR="$UvCacheDir"

Cmd=("$UV_BIN" run --project "$ProjectDir" "$ScriptPy")
# show help only to verify executable path; silence output
"${Cmd[@]}" -h >/dev/null 2>&1 || true

cd $WorkDir

function test_title() {
    echo ""
    echo "-----------------------------------"
    echo "$*"
    echo "-----------------------------------"
}

test_no=0

test_no=$((test_no+1))
test_title コマンドラインでテキストを与えるテスト
wavfile=$(printf "test%d.wav" $test_no)
infofile="${wavfile%.wav}.info"
logfile=$(printf "test%d.log" $test_no)
rm -f "$wavfile" "$infofile" "$logfile"
"${Cmd[@]}" --model amitaro --text "こんにちは。Hello world 今日は良い天気ですね。" --output "$wavfile" >"$logfile" 2>&1
if [ $? -ne 0 -o ! -f "$wavfile" -o ! -f "$infofile" ]; then
    cat "$logfile"
    echo "Test${test_no} failed"
    exit 1
fi
echo "OK"

test_no=$((test_no+1))
test_title assist_textを指定するテスト
wavfile=$(printf "test%d.wav" $test_no)
infofile="${wavfile%.wav}.info"
logfile=$(printf "test%d.log" $test_no)
rm -f "$wavfile" "$infofile" "$logfile"
"${Cmd[@]}" --model amitaro --assist-text "ニュースキャスター風に" --text "こんにちは。Hello world 今日は良い天気ですね。" --output "$wavfile" >"$logfile" 2>&1
if [ $? -ne 0 -o ! -f "$wavfile" -o ! -f "$infofile" ]; then
    cat "$logfile"
    echo "Test${test_no} failed"
    exit 1
fi
echo "OK"

test_no=$((test_no+1))
test_title "ファイルでテキストを与えるテスト"
txtfile=$(printf "test%d.txt" $test_no)
wavfile=$(printf "test%d.wav" $test_no)
infofile="${wavfile%.wav}.info"
logfile=$(printf "test%d.log" $test_no)
rm -f "$txtfile" "$wavfile" "$infofile" "$logfile"
cat <<'__EOT__' > "$txtfile"
おはようございます。
Ｇｏｏｄ morning！
今日は素晴らしい一日になりそうです。
__EOT__
# 注意: テキストファイルを引数に与える場合、スクリプトは入力ファイル名の拡張子を
# .wav に置き換えた名前で出力ファイルを作成します。従って `--output` を同時に指定
# するとエラーになる実装です（仕様に合わせ、ここでは --output を与えずに実行します）。
"${Cmd[@]}" --model amitaro "$txtfile" >"$logfile" 2>&1
if [ $? -ne 0 -o ! -f "$wavfile" -o ! -f "${wavfile%.wav}.info" ]; then
    cat "$logfile"
    echo "Test${test_no} failed"
    exit 1
fi
echo "OK"

test_no=$((test_no+1))
test_title "jvnv-F1-jp で --style Angry 指定時の style を確認"
wavfile=$(printf "test%d_angry.wav" $test_no)
infofile="${wavfile%.wav}.info"
logfile=$(printf "test%d_angry.log" $test_no)
rm -f "$wavfile" "$infofile" "$logfile"
"${Cmd[@]}" --model jvnv-F1-jp --style Angry --text "こんにちは。テストです。" --output "$wavfile" >"$logfile" 2>&1
if [ $? -ne 0 -o ! -f "$wavfile" -o ! -f "$infofile" ]; then
    cat "$logfile"
    echo "Test${test_no} failed"
    exit 1
fi
grep -q "^style: Angry$" "$infofile"
if [ $? -ne 0 ]; then
    cat "$infofile"
    echo "Test${test_no} failed: style mismatch for --style Angry"
    exit 1
fi
echo "OK"

test_no=$((test_no+1))
test_title "jvnv-F1-jp で style 省略時の style を確認"
wavfile=$(printf "test%d_default.wav" $test_no)
infofile="${wavfile%.wav}.info"
logfile=$(printf "test%d_default.log" $test_no)
rm -f "$wavfile" "$infofile" "$logfile"
"${Cmd[@]}" --model jvnv-F1-jp --text "こんにちは。テストです。" --output "$wavfile" >"$logfile" 2>&1
if [ $? -ne 0 -o ! -f "$wavfile" -o ! -f "$infofile" ]; then
    cat "$logfile"
    echo "Test${test_no} failed"
    exit 1
fi
grep -q "^style: Neutral$" "$infofile"
if [ $? -ne 0 ]; then
    cat "$infofile"
    echo "Test${test_no} failed: style mismatch when omitted"
    exit 1
fi
echo "OK"

test_no=$((test_no+1))
test_title "--speed 0 はエラーになるテスト"
wavfile=$(printf "test%d.wav" $test_no)
infofile="${wavfile%.wav}.info"
logfile=$(printf "test%d.log" $test_no)
rm -f "$wavfile" "$infofile" "$logfile"
set +e
"${Cmd[@]}" --model amitaro --text "こんにちは" --output "$wavfile" --speed 0 >"$logfile" 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
    cat "$logfile"
    echo "Test${test_no} failed: expected non-zero exit"
    exit 1
fi
grep -q -- "--speed は 0 より大きい値を指定してください。" "$logfile"
if [ $? -ne 0 ]; then
    cat "$logfile"
    echo "Test6 failed: expected validation message"
    exit 1
fi
echo "OK"

test_no=$((test_no+1))
test_title "空テキストはエラーになるテスト"
txtfile=$(printf "test%d.txt" $test_no)
wavfile=$(printf "test%d.wav" $test_no)
infofile="${wavfile%.wav}.info"
logfile=$(printf "test%d.log" $test_no)
rm -f "$txtfile" "$wavfile" "$infofile" "$logfile"
cat <<'__EOT__' > "$txtfile"
   
__EOT__
set +e
"${Cmd[@]}" --model amitaro "$txtfile" >"$logfile" 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
    cat "$logfile"
    echo "Test${test_no} failed: expected non-zero exit"
    exit 1
fi
grep -q "入力テキストが空です" "$logfile"
if [ $? -ne 0 ]; then
    cat "$logfile"
    echo "Test${test_no} failed: expected empty text message"
    exit 1
fi
echo "OK"
