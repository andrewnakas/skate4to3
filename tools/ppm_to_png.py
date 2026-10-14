#!/usr/bin/env python3
"""Convert the engine's @capture PPM frames to PNG, newest run first.

skate3_input_capture_dir writes binary PPM (P6). Pillow reads it directly; this just
renames, sorts by capture time and reports which frames actually have content, since a
frame grabbed during a load is a flat colour and worth telling apart from the map.
"""
import sys
from pathlib import Path

from PIL import Image, ImageStat

src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src
out.mkdir(parents=True, exist_ok=True)
rows = []
for ppm in sorted(src.glob('*.ppm'), key=lambda p: p.stat().st_mtime):
    im = Image.open(ppm).convert('RGB')
    png = out / (ppm.stem + '.png')
    im.save(png)
    st = ImageStat.Stat(im)
    spread = sum(st.stddev) / 3
    rows.append((ppm.stem, im.size, round(spread, 1), png))
for name, size, spread, png in rows:
    print(f"{name:22s} {size[0]}x{size[1]}  detail {spread:6.1f}  {'FLAT (load/black)' if spread < 6 else 'has content'}  {png}")
print(f"{len(rows)} frame(s)")
