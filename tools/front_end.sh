#!/usr/bin/env bash
# skate. level -> per-level work dir ready for export.   tools/front_end.sh <level> <workdir-name>
#   e.g. tools/front_end.sh dingolevel_mpr mpr
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
LEVEL=$1; W=$R/work/$2; GW=${GLOBAL_WORK:-work}
PY=${PY:-python}
mkdir -p "$W"
cd "$R"
run() { echo; echo "=== $1 ==="; shift; "$@" 2>&1 | tail -"${TAIL:-12}"; }
# global indices (once per game update)
mkdir -p $GW; [ -f $GW/layer_choice.json ] || cp work/layer_choice.json $GW/ 2>/dev/null || true
[ -s $GW/global_assets.json ] || SKATE_WORK=$GW run index $PY -I sk4/collect.py index
[ -s $GW/partition_all.json ] || SKATE_WORK=$GW run partitions $PY -I sk4/collect.py partitions
[ -s $GW/guid_to_texture.json ] || SKATE_WORK=$GW run guid_index $PY -I sk4/guid_index.py
SKATE_WORK=$GW run bind_keys $PY -I sk4/bind_keys.py
for f in chunk_catalog.json chunk_ranges.json global_assets.json global_chunks.json guid_to_texture.json \
         partition_all.json tex_res.json depot_textures_all.json param_roles.json; do ln -f $GW/$f "$W/$f"; done
export SKATE_WORK=$W
run collect $PY -I sk4/collect.py level "$LEVEL"
run level_keys $PY -I sk4/level_keys.py
run extract_depot_textures $PY -I sk4/extract_depot_textures.py ${EXTRACT_ARGS:-}
run inherit_variants $PY -I sk4/inherit_variants.py
run bind_by_name $PY -I sk4/bind_by_name.py
run build_key_textures $PY -I sk4/build_key_textures.py
run decal_sheets $PY -I sk4/decal_sheets.py
run bake_materials $PY -I sk4/bake_materials.py
echo; echo "=== front end complete: $W ==="
