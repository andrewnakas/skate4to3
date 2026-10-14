#!/bin/bash
# tools/variant_boot.sh <tiles dir> <ID> <keep-glob-regex>  -> pack with only streams matching the regex, boot it
set -u
R=$(cd "$(dirname "$0")/.." && pwd); SRC=$1; ID=$2; KEEP=$3; DROP=${4:-^$}
V=$R/work/var/$ID; rm -rf "$V"; mkdir -p "$V"
cp "$SRC/map.json" "$V/"
for d in "$SRC"/c*; do b=$(basename "$d"); [[ $b =~ $KEEP ]] && ! [[ $b =~ $DROP ]] && cp -r "$d" "$V/"; done
python -u "$R/tools/pack.py" "$V" "$ID" --content-id "$ID" --out "$R/work/var/packs" | tail -1
bash "$R/tools/boot_test.sh" "$R/work/var/packs/$ID" 120
