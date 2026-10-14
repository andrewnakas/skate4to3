#!/usr/bin/env bash
# One command: skate. level -> installable Skate 3 DLC pack.
#   tools/build.sh <skate. level> <work dir name> "<Display Name>" [flags file]
#   e.g. tools/build.sh dingolevel_mpr mpr "Summer Camp"
# Steps: tools/front_end.sh (index, bind, collect, textures) -> tools/export.sh (GLB blocks) ->
# sk3/assemble.py (tile arenas) -> tools/build_pack.py (streams, manifests, vaults, .big, .header).
# Output: work/<dir>/pack/<CONTENTID>/  - copy that folder into the recomp's dlc/ folder.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
LEVEL=$1; DIR=$2; NAME=$3; FLAGS=${4:-$R/work/$DIR/hd_flags.txt}
W=$R/work/$DIR; mkdir -p "$W"
[ -f "$FLAGS" ] || printf '%s\n' "--tile-budget=100000 --tile-vertex-budget=200000 --collision-window-mb=13.0 --collision-stream-mb=2.9 --shared-small-texture-m2=200 --tile-textures --texture-px=256 --hot-texture-px=1024 --tile-texture-px=1024 --tile-stream-mb=10.3 --tile-texture-budget-mb=7.0 --global-pool-mb=9.0 --global-edge-px=512 --decals --render-window-mb=1000 --bake-light" > "$FLAGS"
cd "$R"
bash tools/front_end.sh "$LEVEL" "$DIR"
bash tools/export.sh "$DIR" $(cat "$FLAGS")
python -u -m sk3.assemble "$W/glb" "$W/arenas" | tail -2
python -u tools/build_pack.py "$W/arenas" "$W/glb" "$NAME" --out "$W/pack"
