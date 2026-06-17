#!/usr/bin/env bash
# Dev launcher: API + bot service + frontend, with a pre-registered admin
# account and the local bot service already registered, so bots are seatable
# the moment the page loads. Ctrl-C tears the whole thing down.
#
#   ./packages/settlrl-render/dev.sh
#
# Overridable via env: ADMIN_EMAIL, ADMIN_PASSWORD, API_PORT, BOT_PORT.
set -euo pipefail

ADMIN_EMAIL="${ADMIN_EMAIL:-dev@example.com}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-devpassword}"
API_PORT="${API_PORT:-8000}"
BOT_PORT="${BOT_PORT:-8100}"
API="http://localhost:${API_PORT}"

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
SETTLRL_RENDER_ADMIN_EMAILS="$ADMIN_EMAIL" RELOAD=0 PORT="$API_PORT" \
  uv run settlrl-render &
pids+=($!)

# 2. Bot service — hosts the settlrl-agents policies the game server delegates to.
echo "starting bot service on http://localhost:${BOT_PORT} ..."
BOT_PORT="$BOT_PORT" uv run settlrl-render-bot &
pids+=($!)

# 3. Wait for both, then seed the admin account and register the bot service.
echo "waiting for services ..."
until curl -sf "${API}/api/bots" >/dev/null 2>&1; do sleep 0.5; done
until curl -sf "http://localhost:${BOT_PORT}/catalog" >/dev/null 2>&1; do sleep 0.5; done

# Register the admin (a 400 here just means it already exists from a prior run).
if curl -sf -X POST "${API}/api/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"${ADMIN_EMAIL}\",\"password\":\"${ADMIN_PASSWORD}\"}" >/dev/null 2>&1; then
  echo "registered admin ${ADMIN_EMAIL}"
else
  echo "admin ${ADMIN_EMAIL} already exists — continuing"
fi

# Log in (promotes to superuser via ADMIN_EMAILS) and register the bot service.
TOKEN="$(curl -sf -X POST "${API}/api/auth/login" \
  -d "username=${ADMIN_EMAIL}&password=${ADMIN_PASSWORD}" \
  | sed -n 's/.*"access_token":"\([^"]*\)".*/\1/p' || true)"
if [ -n "$TOKEN" ]; then
  if curl -sf -X POST "${API}/api/admin/bot-providers" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"local\",\"base_url\":\"http://localhost:${BOT_PORT}\"}" >/dev/null 2>&1; then
    echo "registered bot service 'local' -> http://localhost:${BOT_PORT}"
  else
    echo "bot service 'local' already registered — continuing"
  fi
else
  echo "WARNING: could not log in; register the bot service from the menu"
fi

cat <<EOF

  ready:
    frontend  http://localhost:5173   (open this)
    API       ${API}
    bot svc   http://localhost:${BOT_PORT}
    sign in   ${ADMIN_EMAIL} / ${ADMIN_PASSWORD}  (admin)

EOF

# 4. Frontend dev server in the foreground — Ctrl-C here ends everything.
cd packages/settlrl-render/frontend
[ -d node_modules ] || npm install
npm run dev
