"""Bind textures by ShaderBlockDepot KEY, not by material name.

A material name is not an identity. `m_proxy_atlas` - the material every `projectlod` proxy
mesh shares - is claimed by 237 depot keys carrying 226 DIFFERENT baked atlases, and it covers
296,000 m2 of the Isle of Grom crop: 24% of the island's surface. Keyed by name, all of it
draws with one arbitrary atlas (a wall-cable spacer). Keyed by depot key, each proxy mesh gets
the atlas the game bakes for it.

Every other multi-key material resolves to one or two distinct colours, so this changes nothing
for them and everything for the proxies.

Reads the PNGs tools/extract_depot_textures.py already wrote, so it needs no CAS access and no
Oodle - it is a join, not an extraction.

Output work/depot_key_textures.json   depot key -> {png, w, h, normal, nw, nh, atlas}
"""
from __future__ import annotations
import collections, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image

import textures
from texcache import WORK

# Baked-atlas namespace: UVs index a packed sheet, so they must NOT be rescaled to a tiling
# target the way a 2x2 m library material's are.
ATLAS_TEX = re.compile(r'projectlod|_proxy_c$|_atlas')
# The `projectlod` atlases ARE usable: each is a per-mesh baked unwrap of that plot, in
# pixelFormat 61 (BC3). They only looked like placeholder magenta because the decoder ran
# them through BC7. Their UVs are a per-mesh unwrap, so they must keep those UVs - never
# world-project them.


def slug_of(name: str) -> str:
    return 'MAT_' + re.sub(r'[^A-Za-z0-9_.-]', '_', name)[-80:] + '.png'


def main():
    depot = json.load(open(os.path.join(WORK, 'depot_textures.json')))
    tdir = os.path.join(WORK, 'textures')
    have = set(os.listdir(tdir))

    info: dict[str, tuple] = {}

    def probe(name):
        if name in info:
            return info[name]
        s = slug_of(name)
        if s not in have:
            info[name] = None
            return None
        fp = os.path.join(tdir, s)
        try:
            with Image.open(fp) as im:
                im.load()
                w, h = im.size
                is_nrm = textures.looks_like_normal_map(im)
                ok_nrm = textures.valid_tangent_normal(im)
        except Exception:  # noqa: BLE001
            info[name] = None
            return None
        info[name] = (f'textures/{s}', w, h, is_nrm, ok_nrm)
        return info[name]

    out, skip = {}, collections.Counter()
    for key, e in depot.items():
        col = e.get('c')
        if not col:
            skip['no colour slot'] += 1
            continue
        c = probe(col)
        if not c:
            skip['colour png missing'] += 1
            continue
        if c[3]:
            skip['colour decodes as a normal map'] += 1
            continue

        rec = {'base': col, 'png': c[0], 'w': c[1], 'h': c[2],
               'atlas': bool(ATLAS_TEX.search(col.lower()))}
        if e.get('tint'):
            rec['tint'] = e['tint']
        n = probe(e.get('ny') or '')
        if n and n[4]:
            rec['normal'], rec['nw'], rec['nh'] = n[0], n[1], n[2]
        out[str(key)] = rec

    # A material with no colour texture is not a gap: many are flat-colour blockout and
    # dev-shader materials that state their colour as a vector parameter instead. Without it
    # they fall back to an untinted class swatch, which is more of the same grey.
    tints = {str(k): e['tint'] for k, e in depot.items()
             if e.get('tint') and str(k) not in out}
    tdst = os.path.join(WORK, 'depot_key_tints.json')
    ttmp = tdst + '.tmp'
    with open(ttmp, 'w') as fh:
        json.dump(tints, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(ttmp, tdst)
    print(f"keys with a colour but no texture: {len(tints):,} "
          f"(written to depot_key_tints.json)")

    dst = os.path.join(WORK, 'depot_key_textures.json')
    tmp = dst + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dst)

    n_atlas = sum(1 for r in out.values() if r['atlas'])
    n_nrm = sum(1 for r in out.values() if 'normal' in r)
    print(f"depot keys with a texture: {len(out):,} of {len(depot):,} "
          f"({n_atlas:,} baked atlases, {n_nrm:,} with normal maps)")
    print(f"distinct colour textures: {len({r['base'] for r in out.values()}):,}")
    for k, c in skip.most_common():
        print(f"  skipped: {k} x{c}")


if __name__ == '__main__':
    main()
