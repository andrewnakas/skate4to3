#!/bin/bash
# Boot a pack on the rig, travel to its first location through the pause-menu map (5th entry when
# it is the only add-on staged), and save a 6-frame contact sheet.  tools/boot_test.sh <pack dir> [secs=210]
set -u
R=$(cd "$(dirname "$0")/.." && pwd); P=$1; ID=$(basename "$P"); S=${2:-210}
bash "$R/tools/smoke_boot.sh" "$P" "$S" "--skate3_demo_path_gameplay_inputs=start,a,rt:1500,rt:1500,down,down,down,down,a:1500,a:1500,a:1500,a" | grep -E "exited|shot $S"
python - "$R/work/smoke" "$ID" "$S" <<'PY'
import sys, os
from PIL import Image
d, i, s = sys.argv[1], sys.argv[2], int(sys.argv[3])
ts = [t for t in range(30, s + 1, 30) if os.path.exists(f'{d}/{i}_{t}.png')][-6:]
ims = [Image.open(f'{d}/{i}_{t}.png') for t in ts]
W, H = ims[0].size; sh = Image.new('RGB', (W * 2, H * 3))
for k, im in enumerate(ims): sh.paste(im, ((k % 2) * W, (k // 2) * H))
sh.resize((W, int(H * 1.5))).save(f'{d}/{i}_sheet.png'); print('sheet', f'{d}/{i}_sheet.png', ts)
PY
grep -a "stream.DIST" "$R/work/smoke/$ID.log" | grep -aoE "c(Pres|Sim|Tex)_[-0-9_]+high|c(Pres|Sim)_Global" | sort -u | wc -l | sed 's/^/distinct streams opened: /'
