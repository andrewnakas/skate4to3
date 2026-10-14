"""Offline audit: for every material on the island, is the texture the game's own?

Joins three measurements so the answer is weighted by how much of the map each material
actually covers, rather than counting materials:

  work/<out>/uv_density.json   area and tiles/m per material, as exported
  work/material_textures.json  the depot-bound texture, per material  (the game's own)
  work/depot_textures.json     the raw binding, to say WHY a material has no colour

Prints the shortfall in square metres, worst first - the only ranking that matters, because
one unbound cliff material is worth more than two hundred unbound bolts.
"""
from __future__ import annotations
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texcache import WORK


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, 'IsleOfGrom', 'glb')
    rows = json.load(open(os.path.join(out_dir, 'uv_density.json')))
    mt = json.load(open(os.path.join(WORK, 'material_textures.json')))
    depot = json.load(open(os.path.join(WORK, 'depot_textures.json')))
    ct = json.load(open(os.path.join(WORK, 'class_textures.json')))

    chan = {}
    for e in depot.values():
        for m in e.get('materials', ()):
            have = chan.setdefault(m, set())
            have.update(k for k in ('c', 'ny', 'msk') if e.get(k))

    tot = sum(r['m2'] for r in rows)
    exact = [r for r in rows if r['material'] in mt]
    atlas = [r for r in rows if r['atlas']]
    fall = [r for r in rows if r['material'] not in mt and not r['atlas']]
    print(f"island surface measured: {tot:,.0f} m2 over {len(rows)} material/class pairs")
    print(f"  exact game texture   {sum(r['m2'] for r in exact):>11,.0f} m2  "
          f"{100 * sum(r['m2'] for r in exact) / tot:5.1f}%   {len(exact)} pairs")
    print(f"  terrain/atlas (world-projected, texture from the surface token)"
          f" {sum(r['m2'] for r in atlas):>9,.0f} m2  "
          f"{100 * sum(r['m2'] for r in atlas) / tot:5.1f}%   {len(atlas)} pairs")
    print(f"  class fallback       {sum(r['m2'] for r in fall):>11,.0f} m2  "
          f"{100 * sum(r['m2'] for r in fall) / tot:5.1f}%   {len(fall)} pairs")

    res = collections.Counter()
    for r in exact:
        e = mt[r['material']]
        res[f"{e['w']}x{e['h']}"] += 1
    print("  resolution of the bound textures: "
          + ", ".join(f"{k} x{v}" for k, v in res.most_common(6)))
    n_nrm = sum(1 for r in exact if 'normal' in mt[r['material']])
    print(f"  with a valid tangent-space normal map: {n_nrm} of {len(exact)} pairs")

    print("\nmaterials still on a class fallback, worst first:")
    for r in sorted(fall, key=lambda r: -r['m2'])[:20]:
        have = sorted(chan.get(r['material'], ()))
        why = ('no depot binding' if not have
               else 'binding has no colour slot (only ' + '+'.join(have) + ')')
        cls_tex = ct.get(r['class'], {})
        src = cls_tex.get('base', cls_tex.get('png', 'flat swatch'))
        print(f"  {r['m2']:>9,.0f} m2  {r['class']:<9} {r['material'][:52]:<52} {why}")
        print(f"             using class texture: {str(src).rsplit('/', 1)[-1]}")

    print("\nper-class UV density as exported (target 0.50 tiles/m):")
    per = collections.defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        p = per[r['class']]
        p[0] += r['m2']
        p[1] += r['uv_area']
    bad = 0
    for c, (wa, ua) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        if wa < 50:
            continue
        d = (ua / wa) ** 0.5 if wa else 0
        ok = 0.40 <= d <= 0.62
        bad += not ok
        print(f"  {'ok ' if ok else 'OFF'} {c:<10} {wa:>11,.0f} m2  {d:6.3f} tiles/m")
    print(f"\n{bad} class(es) off target")


if __name__ == '__main__':
    main()
