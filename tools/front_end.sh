#!/usr/bin/env bash
# skate. level -> per-level work dir ready for export.   tools/front_end.sh <level> <workdir-name>
#   e.g. tools/front_end.sh dingolevel_mpr mpr
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
LEVEL=$1; W=$R/work/$2
PY=${PY:-python}
mkdir -p "$W"
cd "$R"
run() { echo; echo "=== $1 ==="; shift; "$@" 2>&1 | tail -"${TAIL:-12}"; }
# global indices (once per game update)
[ -s work/global_assets.json ] || SKATE_WORK=work run index $PY -I sk4/collect.py index
[ -s work/partition_all.json ] || SKATE_WORK=work run partitions $PY -I sk4/collect.py partitions
[ -s work/guid_to_texture.json ] || SKATE_WORK=work run guid_index $PY -I sk4/guid_index.py
SKATE_WORK=work run bind_keys $PY -I sk4/bind_keys.py
for f in chunk_catalog.json chunk_ranges.json global_assets.json global_chunks.json guid_to_texture.json \
         partition_all.json tex_res.json depot_textures_all.json param_roles.json; do ln -f work/$f "$W/$f"; done
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
