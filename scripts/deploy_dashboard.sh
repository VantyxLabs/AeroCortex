#!/usr/bin/env bash
# Sync dashboard/static to the CloudFront origin bucket and invalidate.
set -euo pipefail

BUCKET="${DASHBOARD_BUCKET:?Set DASHBOARD_BUCKET}"
DIST_ID="${CLOUDFRONT_DISTRIBUTION_ID:-}"
API_KEY="${API_KEY:-}"
VERSION="${VERSION:-aws}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# Keep /static/... paths that index.html expects; also publish index at bucket root.
mkdir -p "$TMPDIR/static"
cp -R dashboard/static/. "$TMPDIR/static/"
cp dashboard/static/index.html "$TMPDIR/index.html"
cat > "$TMPDIR/config.json" <<EOF
{
  "api_base_url": "/api",
  "upstream_api_url": "/api",
  "api_key": "${API_KEY}",
  "default_scenarios": [
    "GPS_INTERFERENCE",
    "GPS_LOSS",
    "BATTERY_DEGRADATION",
    "LOW_BATTERY",
    "STRONG_WIND",
    "COMMUNICATION_LOSS",
    "SENSOR_ANOMALY",
    "COMBINED_FAILURE"
  ],
  "poll_interval_ms": 4000,
  "version": "${VERSION}"
}
EOF

aws s3 sync "$TMPDIR" "s3://${BUCKET}" --delete \
  --cache-control "max-age=300" \
  --exclude "config.json"

aws s3 cp "$TMPDIR/config.json" "s3://${BUCKET}/config.json" \
  --cache-control "max-age=60" \
  --content-type "application/json"

if [[ -n "$DIST_ID" ]]; then
  aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/index.html" "/config.json" "/static/*" "/*"
fi

echo "Dashboard synced to s3://${BUCKET}"
