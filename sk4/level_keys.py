#!/usr/bin/env python3
"""The level's slice of the global binding: work/<level>/depot_textures.json.

    SKATE_WORK=work/mpr python sk4/level_keys.py

Keeps the keys any LOD of a placed mesh uses, the decal template keys, and every `preset:`
entry, so the texture chain extracts what the level draws and nothing else.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import meshset  # noqa: E402
from cas import decompress_many  # noqa: E402
from guid_index import LAYOUT  # noqa: E402

WORK = os.environ.get('SKATE_WORK', os.path.join(os.path.dirname(HERE), 'work'))


def main():
    allb = json.load(open(os.path.join(WORK, 'depot_textures_all.json')))
    res = json.load(open(os.path.join(WORK, 'global_assets.json')))['res']
    meshes = sorted({m for m, _T, _src in json.load(open(os.path.join(WORK, 'instances.json'))) if m in res})
    keys, handles = set(), {}
    for s in range(0, len(meshes), 400):
        raw = []
        for n in meshes[s:s + 400]:
            pkg, ci, off, size = res[n][1:5]
            h = handles.get((pkg, ci))
            if h is None:
                h = handles[(pkg, ci)] = open(os.path.join(LAYOUT, pkg, f'cas_{ci:02d}.cas'), 'rb')
            h.seek(off)
            raw.append(h.read(size))
        for b in decompress_many(raw):
            for lod in meshset.parse(b).lods:
                keys.update(str(sec.depot_key) for sec in lod.sections if sec.depot_key)
    dp = os.path.join(WORK, 'decals.json')
    if os.path.exists(dp):
        keys.update(str(d['tkey']) for d in json.load(open(dp)) if d.get('tkey'))
    out = {k: v for k, v in allb.items() if k in keys or k.startswith('preset:')}
    json.dump(out, open(os.path.join(WORK, 'depot_textures.json'), 'w'), indent=0)
    print(f'  level keys {len(keys):,} from {len(meshes):,} meshes; bound entries kept {len(out):,} '
          f'({sum(1 for v in out.values() if v.get("c")):,} with a colour texture)')


if __name__ == '__main__':
    main()
