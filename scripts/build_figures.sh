#!/usr/bin/env bash
# Rebuild the README hero images from their HTML sources.
#
# Chrome stamps a creation time into anything it renders, so two renders of the
# same HTML differ in bytes while being identical in content. Any equality check
# over these files has to compare rendered content, never a file digest.
set -euo pipefail

cd "$(dirname "$0")/.."
CHROME=${CHROME:-google-chrome}

command -v "$CHROME" >/dev/null || { echo "need $CHROME on PATH" >&2; exit 1; }

render_png() {  # $1 = light|dark, $2 = background
  local t=$1 bg=$2 d="$PWD/assets/readme"
  "$CHROME" --headless --disable-gpu --no-sandbox \
            --window-size=840,300 --force-device-scale-factor=2 \
            --default-background-color="$bg" \
            --screenshot="$d/hero-$t.png" "file://$d/hero-$t.html" 2>/dev/null
  printf '  %-16s %s\n' "hero-$t.png" "$(identify -format '%wx%h, %b' "$d/hero-$t.png")"
}

echo "readme hero (raster, 2x):"
render_png light FFFFFFFF
render_png dark  0D1117FF
