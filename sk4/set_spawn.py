"""Move the startline to a chosen place, without re-exporting.

The pack's spawn comes from the startline marker in `d2s3_spawns.json`; the loader's
`--index` resolves back to the same target and d2s3's `--spawn` does not override the marker.
Re-running the exporter to look at another part of the map costs minutes, but the GLBs do not
change - only the marker does. This rewrites the marker in place, so a rebuild is all it takes.

Ground is chosen with the same clearance and eight-neighbour support rules the spawn search
uses, and far-LOD proxy plate is refused, so the skater still lands on something real.

    python tools/set_spawn.py <x> <z> [glb_dir]
"""
from __future__ import annotations
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import find_spawns

D2S3 = os.environ.get('D2S3', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'decendersmaptoskate3recomp'))


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    tx, tz = float(sys.argv[1]), float(sys.argv[2])
    glb = sys.argv[3] if len(sys.argv) > 3 else os.path.join(D2S3, 'work', 'IsleOfGrom', 'glb')

    gp, sp = [], []
    for f in sorted(__import__('glob').glob(os.path.join(glb, '*.glb'))):
        for name, P, T in find_spawns.load_glb(f):
            pts, up = find_spawns.surface_samples(P, T)
            sp.append(pts)
            if '__proxylod' in name:
                continue
            if name.split('__')[0] in find_spawns.GROUND:
                gp.append(pts[up])
    ground = find_spawns.Grid(np.concatenate(gp))
    solid = find_spawns.Grid(np.concatenate(sp))

    got = find_spawns.search(ground, solid, tx, tz, 120, [])
    if got is None:
        raise SystemExit(f"no standable real ground within 120 m of ({tx:.0f}, {tz:.0f})")
    x, y, z = got[0], got[1] + find_spawns.STAND, got[2]

    path = os.path.join(glb, 'd2s3_spawns.json')
    spawns = json.load(open(path))
    spawns[0] = {'name': 'Isle of Grom', 'kind': 'startline',
                 'x': round(x, 3), 'y': round(y, 3), 'z': round(z, 3), 'yaw': 0.0}
    json.dump(spawns, open(path, 'w'), indent=1)
    open(os.path.join(glb, 'spawn.txt'), 'w').write(f"{x},{y},{z}\n")
    print(f"startline -> {x},{y},{z}  ({math.hypot(x - tx, z - tz):.1f} m from the target)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
