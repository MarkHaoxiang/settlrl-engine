#!/usr/bin/env bash
# Dev launcher: API + two one-bot services + frontend, with a pre-registered
# admin account and both bot services already registered, so bots are seatable
# the moment the page loads. Ctrl-C tears the whole thing down.
#
#   ./packages/settlrl-app/dev.sh
#
# Each bot service hosts a single bot (settlrl-bot-service --bot KIND); the
# launcher runs greedy and mcts on adjacent ports. Overridable via env:
# ADMIN_EMAIL, ADMIN_PASSWORD, API_PORT, BOT_PORT.
set -euo pipefail

ADMIN_EMAIL="${ADMIN_EMAIL:-dev@example.com}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-devpassword}"
API_PORT="${API_PORT:-8000}"
BOT_PORT="${BOT_PORT:-8100}"
API="http://localhost:${API_PORT}"

# The bots to serve, each on its own port starting at BOT_PORT.
BOTS=(greedy mcts)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

pids=()
cleanup() {
  echo
  echo "shutting down..."
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. API server. RELOAD=0 so the in-memory admin account + bot registration
#    survive (the file-watch reloader would wipe them on every save); restart
#    the script after editing Python. The Vite frontend hot-reloads regardless.
echo "starting API on ${API} (admin: ${ADMIN_EMAIL}) ..."
SETTLRL_APP_ADMIN_EMAILS="$ADMIN_EMAIL" RELOAD=0 PORT="$API_PORT" \
  uv run settlrl-app &
pids+=($!)

# 2. Bot services — one per bot; the game server delegates moves to them.
for i in "${!BOTS[@]}"; do
  bot="${BOTS[$i]}"
  port=$((BOT_PORT + i))
  echo "starting bot service '${bot}' on http://localhost:${port} ..."
  BOT_PORT="$port" uv run --package settlrl-agents settlrl-bot-service --bot "$bot" &
  pids+=($!)
done

# 3. Wait for everything, then seed the admin account and register the bots.
echo "waiting for services ..."
until curl -sf "${API}/api/bots" >/dev/null 2>&1; do sleep 0.5; done
for i in "${!BOTS[@]}"; do
  until curl -sf "http://localhost:$((BOT_PORT + i))/info" >/dev/null 2>&1; do
    sleep 0.5
  done
done

# Register the admin (a 400 here just means it already exists from a prior run).
if curl -sf -X POST "${API}/api/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"${ADMIN_EMAIL}\",\"password\":\"${ADMIN_PASSWORD}\"}" >/dev/null 2>&1; then
  echo "registered admin ${ADMIN_EMAIL}"
else
  echo "admin ${ADMIN_EMAIL} already exists — continuing"
fi

# Log in (promotes to superuser via ADMIN_EMAILS) and register the bot services
# by base URL (each bot self-identifies via GET /info).
TOKEN="$(curl -sf -X POST "${API}/api/auth/login" \
  -d "username=${ADMIN_EMAIL}&password=${ADMIN_PASSWORD}" \
  | sed -n 's/.*"access_token":"\([^"]*\)".*/\1/p' || true)"
if [ -n "$TOKEN" ]; then
  for i in "${!BOTS[@]}"; do
    base="http://localhost:$((BOT_PORT + i))"
    if curl -sf -X POST "${API}/api/admin/bot-providers" \
      -H "Authorization: Bearer ${TOKEN}" \
      -H "Content-Type: application/json" \
      -d "{\"base_url\":\"${base}\"}" >/dev/null 2>&1; then
      echo "registered bot service '${BOTS[$i]}' -> ${base}"
    else
      echo "bot service '${BOTS[$i]}' already registered — continuing"
    fi
  done
else
  echo "WARNING: could not log in; register the bot services from the menu"
fi

cat <<EOF

  ready:
    frontend  http://localhost:5173   (open this)
    API       ${API}
    bot svcs  ${BOTS[*]} on ports ${BOT_PORT}+
    sign in   ${ADMIN_EMAIL} / ${ADMIN_PASSWORD}  (admin)

EOF

# 4. Frontend dev server in the foreground — Ctrl-C here ends everything.
cd packages/settlrl-app/frontend
[ -d node_modules ] || npm install
npm run dev
