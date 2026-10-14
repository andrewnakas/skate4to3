#!/usr/bin/env bash
# Build every map we have into packs, one at a time (RAM), resumable: a map whose pack exists is
# skipped. Intermediates (glb, arenas) are deleted after a successful pack to save disk.
#   tools/build_all.sh [name filter regex]
set -u
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"
MODS="C:/Program Files (x86)/Steam/steamapps/common/Skate/Mods"
FILTER=${1:-.}
mkdir -p work/batch
targets=()
# stock skate. levels: id|level|display|mod root
for l in "ftue|dingolevel_ftue_island|FTUE Island|" \
         "sdm1|dingolevel_sdm/dingolevel_sdm_int_001|SDM Interior 1|" "sdm2|dingolevel_sdm/dingolevel_sdm_int_002|SDM Interior 2|"; do
  targets+=("$l"); done
for m in "$MODS"/*; do
  b=$(basename "$m"); f="$m/reskate-levels.json"; [ -f "$f" ] || continue
  [[ $b =~ (Characters|Clothing|Skate_Shop|Barber|Tattoo|Accessories) ]] && continue
  lvl=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['levels'][0]['asset'].split('/')[2])" "$f")
  id=$(echo "mod_$b" | tr 'A-Z-' 'a-z_'); disp=$(echo "$b" | sed 's/^[A-Za-z0-9]*-//; s/_/ /g')
  targets+=("$id|$lvl|$disp|$m")
done
targets+=("bam|bam_levelroot|Big City|")
for t in "${targets[@]}"; do
  IFS='|' read -r id lvl disp mod <<< "$t"
  [[ $id =~ $FILTER ]] || continue
  W=work/$id/lvl; L=work/batch/$id.log
  if ls $W/pack/*/*.big >/dev/null 2>&1; then echo "skip $id (pack exists)"; continue; fi
  echo "=== $id ($lvl) $(date +%H:%M)" | tee -a work/batch/summary.log
  (
    set -e
    export SKATE_MOD_ROOT="$mod"
    GW=work/$id; [ -z "$mod" ] && GW=work
    GLOBAL_WORK=$GW bash tools/front_end.sh "$lvl" $id/lvl
    [ -f $W/hd_flags.txt ] || cp work/mpr/hd_flags.txt $W/
    bash tools/export.sh $id/lvl $(cat $W/hd_flags.txt)
    python -u -m sk3.assemble $W/glb $W/arenas
    python -u tools/build_pack.py $W/arenas $W/glb "$disp" --out $W/pack
  ) > $L 2>&1 && { echo "  ok $(tail -1 $L | cut -c1-150)" | tee -a work/batch/summary.log; rm -rf $W/glb $W/arenas $W/textures; } \
            || echo "  FAILED: $(grep -a -E 'Error|Traceback' $L | tail -1 | cut -c1-150)" | tee -a work/batch/summary.log
done
