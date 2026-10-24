#!/usr/bin/env bash
# Sync integration tests against real servers: a WebDAV server (wsgidav, HTTPS with a self-signed
# certificate, basic auth) and paperless-ngx (Docker). Needs: python3 with wsgidav + cheroot
# (pip install wsgidav cheroot), openssl, docker with compose, and the Android SDK for Gradle.
#   tools/sync_it/run.sh            start both, run SyncIntegrationTest, stop both
#   KEEP=1 tools/sync_it/run.sh     leave the servers running afterwards
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$HERE"
mkdir -p work/dav
[ -f work/cert.pem ] || openssl req -x509 -newkey rsa:2048 -nodes -days 30 -subj "/CN=flatscan-test" \
  -keyout work/key.pem -out work/cert.pem 2>/dev/null
python3 -m wsgidav.server.server_cli --config wsgidav.yaml > work/wsgidav.log 2>&1 &
DAV_PID=$!
docker compose -p flatscan_it up -d
cleanup() { if [ -z "${KEEP:-}" ]; then kill $DAV_PID 2>/dev/null || true; docker compose -p flatscan_it down -v >/dev/null 2>&1 || true; fi; }
trap cleanup EXIT
echo "waiting for paperless (first start migrates its database: a few minutes)..."
TOKEN=""
for i in $(seq 1 180); do
  TOKEN=$(curl -sf -X POST -d "username=admin&password=admin-pass" http://127.0.0.1:8000/api/token/ 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])" 2>/dev/null || true)
  [ -n "$TOKEN" ] && break
  sleep 2
done
[ -n "$TOKEN" ] || { echo "paperless did not come up"; docker compose -p flatscan_it logs paperless | tail -30; exit 1; }
export FS_WEBDAV_URL=https://127.0.0.1:8433/dav/ FS_WEBDAV_USER=scanner FS_WEBDAV_PASS=s3cret-pass FS_WEBDAV_ROOT="$HERE/work/dav"
export FS_PAPERLESS_URL=http://127.0.0.1:8000/ FS_PAPERLESS_TOKEN="$TOKEN"
cd "$ROOT"
./gradlew --no-daemon testDebugUnitTest --tests '*SyncIntegrationTest*' "$@"
