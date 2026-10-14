"""Give a material variant the texture of the material it is a variant OF.

skate. ships far-LOD copies of a material under the same name plus `_proxy`
(`Skatepark_Surface_proxy` beside `Skatepark_Surface_M`), and the proxy's shader block
carries no colour slot at all - the proxy art is drawn from a baked atlas instead. Where the
atlas is available it wins (tools/build_key_textures.py binds it by depot key); where it is
not, the base material's own texture is a far better answer than a flat class swatch, and it
is already extracted, so this is a join with no CAS access.

Runs after tools/extract_depot_textures.py, on work/material_textures.json in place.
"""
from __future__ import annotations
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texcache import WORK

# tried in order, longest first, so `_proxy_Shadow` is not left as `_proxy`
STRIP = ('_proxy_shadow', '_shadow', '_proxy')
# a base material often carries a suffix the variant drops
SUFFIX = ('', '_m', '_mat', '_m_mat')


def main():
    dst = os.path.join(WORK, 'material_textures.json')
    mt = json.load(open(dst))
    mats = sorted(json.load(open(os.path.join(WORK, 'material_meshes.json'))))
    by_low = {}
    for m in mt:
        by_low.setdefault(m.lower(), m)

    added, unresolved = {}, []
    for m in mats:
        if m in mt:
            continue
        low = m.lower()
        stem = None
        for cut in STRIP:
            if low.endswith(cut):
                stem = low[:-len(cut)]
                break
        if not stem:
            continue
        hit = None
        for suf in SUFFIX:
            cand = by_low.get(stem + suf)
            if cand:
                hit = cand
                break
        if not hit:
            # `Wood_Lumber_03_2x2m_M_BrownWarm_proxy` -> any bound material with that stem
            pre = [v for k, v in by_low.items() if k.startswith(stem)]
            if len(pre) == 1:
                hit = pre[0]
        if hit:
            e = dict(mt[hit])
            e['inherited_from'] = hit
            added[m] = e
        else:
            unresolved.append(m)

    if added:
        mt.update(added)
        tmp = dst + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(mt, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    print(f"variant inheritance: +{len(added)} material(s) "
          f"({len(mt):,} bound in total); {len(unresolved)} variant(s) unresolved")
    for m, e in sorted(added.items())[:14]:
        print(f"   {m[:56]:<56} <- {e['inherited_from'][:40]}")


if __name__ == '__main__':
    main()
