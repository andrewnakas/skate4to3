#!/usr/bin/env bash
# Export a level's work dir to per-tile GLBs.   tools/export.sh <workdir-name> [export flags]
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); W=$R/work/$1; shift
mkdir -p "$W/glb"
SKATE_WORK=$W PYTHONPATH="$R/sk4;$R/tools" python -u "$R/sk4/export_glb.py" --out "$W/glb" "$@" > "$W/export.log" 2>&1
grep -aE "textures:|decals: drew|tiles over|collision: [0-9]+ planning|Traceback|Error" "$W/export.log" | head
