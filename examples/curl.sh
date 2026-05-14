#!/usr/bin/env bash
# kor-minish HTTP API — curl 예시
set -euo pipefail

HOST="${HOST:-http://127.0.0.1:8765}"

echo "=== /health ==="
curl -s "$HOST/health" | python3 -m json.tool

echo
echo "=== /encode ==="
curl -s -X POST "$HOST/encode" \
  -H 'Content-Type: application/json' \
  -d '{"texts":["안녕하세요","반갑습니다"],"normalize":true}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("dim:",d["dim"],"n:",len(d["embeddings"]))'

echo
echo "=== /similarity ==="
curl -s -X POST "$HOST/similarity" \
  -H 'Content-Type: application/json' \
  -d '{"query":"한국 음식 만들기","docs":["김치찌개 레시피","된장국","주식 매수","자동차 보험"]}' \
  | python3 -m json.tool
