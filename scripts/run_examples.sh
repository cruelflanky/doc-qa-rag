#!/usr/bin/env bash
# Реальные примеры запросов для README. Стек должен быть поднят: docker compose up -d --build
# Запуск: scripts/run_examples.sh > examples.md
set -euo pipefail

BASE="${BASE_URL:-http://localhost:8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

show() {  # заголовок, команда, ответ
  printf '\n### %s\n\n```bash\n%s\n```\n\n```json\n' "$1" "$2"
  eval "$2" | python3 -m json.tool --no-ensure-ascii
  printf '```\n'
}

wait_task() {
  local status_url="$1" state
  while true; do
    state=$(curl -s "$BASE$status_url" | python3 -c 'import sys,json;print(json.load(sys.stdin)["state"])')
    [[ "$state" == "SUCCESS" || "$state" == "FAILURE" ]] && break
    sleep 2
  done
}

ask() {  # doc_id, вопрос, debug
  printf "curl -s -X POST %s/documents/%s/ask -H 'Content-Type: application/json' -d '%s'" \
    "$BASE" "$1" "$(python3 -c 'import json,sys;print(json.dumps({"question":sys.argv[1],"debug":sys.argv[2]=="true"},ensure_ascii=False))' "$2" "${3:-false}")"
}

UPLOAD=$(curl -s -F "file=@$ROOT/samples/contract_supply.pdf" -F "title=Договор поставки № 47/2026" "$BASE/documents")
printf '### Загрузка документа\n\n```bash\ncurl -s -F "file=@samples/contract_supply.pdf" -F "title=Договор поставки № 47/2026" %s/documents\n```\n\n```json\n' "$BASE"
echo "$UPLOAD" | python3 -m json.tool --no-ensure-ascii
printf '```\n'
STATUS_URL=$(echo "$UPLOAD" | python3 -c 'import sys,json;print(json.load(sys.stdin)["status_url"])')
DOC=$(echo "$UPLOAD" | python3 -c 'import sys,json;print(json.load(sys.stdin)["doc_id"])')

sleep 3
show "Состояние задачи во время индексации" "curl -s $BASE$STATUS_URL"
wait_task "$STATUS_URL"
show "Состояние задачи после индексации" "curl -s $BASE$STATUS_URL"
show "Список документов" "curl -s $BASE/documents"

show "Вопрос с точным обозначением: номер пункта" "$(ask "$DOC" 'Что установлено пунктом 9.4.2 договора?')"
show "Вопрос с точным обозначением: ИНН" "$(ask "$DOC" 'Какой ИНН у Поставщика?')"
show "Вопрос с точным обозначением: сумма" "$(ask "$DOC" 'Какова общая цена договора?')"
show "Перефразированный вопрос" "$(ask "$DOC" 'Можно ли отказаться принимать товар с неправильными этикетками?')"
show "Вопрос, ответа на который в документе нет" "$(ask "$DOC" 'Какой порядок страхования груза при перевозке?')"
show "Диагностика (debug=true)" "$(ask "$DOC" 'Какой размер неустойки за просрочку поставки товара?' true)"
