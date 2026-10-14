#!/usr/bin/env python3
"""What each section of a mesh is bound to, per LOD - the answer to "why is this prop textured wrong".

    SKATE_WORK=work/mpr python tools/dump_mesh_sections.py <name substring> [...]

Per mesh whose res name contains a substring: every LOD (resident or not), and per section the
material, class, depot key and what the exporter's binding chain finds for it - the shader-block
texture (KEY_TEX), the material-name texture (MATERIAL_TEX), the paint (KEY_TINT) and the raw
shader-block record (depot_textures.json: colour, placeholder, normal, mask).
"""
from __future__ import annotations

import json
import os
import sys

import export_glb as E
import meshset
from cas import decompress_many


def _load(name):
    fp = os.path.join(E.WORK, name)
    return json.load(open(fp)) if os.path.exists(fp) else {}


def main():
    pats = [a.lower() for a in sys.argv[1:]]
    if not pats:
        sys.exit(__doc__)
    RES = _load('global_assets.json')['res']
    CH = _load('global_chunks.json')
    key_tex, mat_tex = _load('depot_key_textures.json'), _load('material_textures.json')
    tints, depot = _load('depot_key_tints.json'), _load('depot_textures.json')
    names = sorted(n for n in RES if n.endswith('_mesh') and any(p in n.lower() for p in pats))
    cas = E.Cas()
    for n, b in zip(names, decompress_many([cas.read(*RES[n][1:5]) for n in names])):
        ms = meshset.parse(b)
        print(f"\n{n}")
        resident = [k for k, l in enumerate(ms.lods) if l.chunk in CH]
        pick = (resident[-1] if E.COARSE.search(n) else resident[0]) if resident else None
        for k, lod in enumerate(ms.lods):
            print(f"  LOD{k} {lod.name}  {'resident' if k in resident else 'NOT in level'}"
                  f"{'  <- drawn first choice' if k == pick else ''}")
            for s in lod.sections:
                cls = E.classify(s.material)[0]
                why = ('shadow' if s.is_shadow else 'empty' if s.primitives == 0
                       else 'class dropped' if cls in E.DROP_CLASSES else '')
                key = str(s.depot_key)
                kt, mt, d = key_tex.get(key), mat_tex.get(s.material), depot.get(key) or {}
                print(f"    {s.material:<34} {cls:<9} {s.primitives:>6} tris  key {key}"
                      + (f"  SKIPPED ({why})" if why else ''))
                if why == 'shadow':
                    continue
                print(f"        key tex : {kt['base'] if kt else '-'}"
                      + (f"  {kt['w']}x{kt['h']}" if kt else ''))
                print(f"        name tex: {mt['base'] if mt else '-'}")
                print(f"        tint    : {tints.get(key, '-')}")
                print("        block   : " + (', '.join(
                    f"{c}={v if isinstance(v, bool) else str(v).rsplit('/', 1)[-1]}"
                    for c, v in d.items() if c != 'materials') or '-'))


if __name__ == '__main__':
    main()
