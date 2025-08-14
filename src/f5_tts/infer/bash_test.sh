#!/usr/bin/env bash
set -euo pipefail

REF=${1:-ref.wav}                               # 1. arg: referencia WAV
TXT=${2:-"Hello from Bash queue!"}              # 2. arg: felolvasandó szöveg
HOST=${3:-"localhost:7860"}                     # 3. arg: host:port
API="http://${HOST}"
SESSION="bash_$$"

[[ -f "$REF" ]] || { echo "❌  nincs ilyen fájl: $REF"; exit 1; }

# 1) WAV feltöltése (kulcs: files!)
TOKEN=$(curl -s -F "files=@${REF}" "$API/gradio_api/upload" | jq -r '.[0]')
[[ -n "$TOKEN" && "$TOKEN" != "null" ]] || { echo "❌ feltöltési hiba"; exit 1; }

# 2) JSON body (10 input a basic_tts-hez)
BODY=$(jq -n --arg file "$TOKEN" --arg txt "$TXT" --arg sid "$SESSION" '
  {data:[$file,"",$txt,false,true,0,0.15,32,1.0,"None"], session_hash:$sid}')

# 3) Szinkron hívás a queue ellenére (`simple_format=true`)
echo "▶️  /call/basic_tts?simple_format=true"
RESP=$(curl -sS -H 'Content-Type: application/json' -d "$BODY" \
             -X POST "$API/gradio_api/call/basic_tts?simple_format=true")

# 4) Base64 WAV kinyerése
B64=$(echo "$RESP" | jq -r '.data[0][1]?' | cut -d, -f2)
[[ -z "$B64" || "$B64" == "null" ]] && { echo "❌  nincs audio"; echo "$RESP"|jq .; exit 1; }

OUT="tts_$(date +%s).wav"
echo "$B64" | base64 -d > "$OUT" && echo "💾  $OUT → $(du -h "$OUT"|cut -f1)"

# 5) Lejátszás (aplay / ffplay / afplay)
play() { command -v "$1" &>/dev/null && "$@"; }
play aplay -q "$OUT"   && exit
play ffplay -nodisp -autoexit -loglevel quiet "$OUT" && exit
play afplay "$OUT"     && exit
echo "ℹ️  Nyisd meg kézzel a fájlt: $OUT"

