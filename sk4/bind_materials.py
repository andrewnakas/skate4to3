"""SUPERSEDED by tools/bind_depot.py - kept only as a cross-check.

Guessing a texture from the material's name gets 64 of 279 island materials; reading the
ShaderBlockDepot gets the game's own binding for 798. This writes
work/material_textures_basename.json so it can be diffed against the depot result, and no
longer overwrites work/material_textures.json - it used to, which made the pipeline's texture
quality depend on which binder ran last.

Resolve each source material to ITS OWN game texture, not a per-class guess.

skate. names many materials by full asset path - `world/materials/metal/iron_rusted_01_2x2m/
metal_iron_rusted_01_2x2m_m_fxpaintgrimecpvrailing01_Grindable` - and the texture sits in that
same directory with a `_c` suffix. Earlier passes missed these because the paths are mixed-case
(`World/Materials/...`) while the res index is lower-cased, so every comparison failed.

Also resolves mesh-local materials by looking in the MESH's own directory, which is where a
prop keeps its texture. Whatever cannot be resolved falls back to the class texture.
"""
from __future__ import annotations
import collections, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textures
from texcache import TexCache, WORK

GLB = os.environ.get('IOG_GLB',
                     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work', 'IsleOfGrom', 'glb'))
QUAL = re.compile(r'(_grindable|_nonegrindable|_proxy)$', re.I)


def candidates(mat: str, mesh_dirs: list[str]) -> list[str]:
    """Texture paths this material might own, most specific first."""
    out = []
    if '/' in mat:
        d, leaf = mat.rsplit('/', 1)
        stem = QUAL.sub('', leaf)
        base = stem.split('_m_')[0].split('_M_')[0]
        own = d.rsplit('/', 1)[-1]
        for s in (base, stem, own):
            out += [f'{d}/{s}_c', f'{d}/{s}_cu']
        # a _TA array directory holds the plain material one level up
        if d.lower().endswith('_ta'):
            out += [f'{d[:-3]}/{base}_c']
    leaf = QUAL.sub('', mat.rsplit('/', 1)[-1])
    base = leaf.split('_m_')[0].split('_M_')[0]
    for md in mesh_dirs:
        out += [f'{md}/{base}_c', f'{md}/{base}_cu']
    return out


def basename_hits(mat: str, by_base: dict) -> list[str]:
    """Textures whose BASENAME matches this material, wherever they live.

    A `_TA` texture-array material names its texture but keeps it in another directory:
    `World/Materials/Array/Concrete_Sidewalk_01_2x2m_TA/Concrete_Sidewalk_01_2x2m` is painted by
    `concrete_sidewalk_01_2x2m_c`, which is not under that path at all. Matching on basename
    finds it; matching on directory never can.
    """
    leaf = QUAL.sub('', mat.rsplit('/', 1)[-1])
    stems = {leaf, leaf.split('_m_')[0], leaf.split('_M_')[0]}
    out = []
    for st in stems:
        if len(st) < 6:
            continue
        for suf in ('_c', '_cu'):
            out += by_base.get((st + suf).lower(), [])
    return out


def main():
    tc = TexCache()
    lower = {k.lower(): k for k in tc.res}
    by_base = collections.defaultdict(list)
    for k in tc.res:
        by_base[k.rsplit('/', 1)[-1].lower()].append(k)
    cs = json.load(open(os.path.join(GLB, 'class_sources.json')))
    mat_mesh = {}
    p = os.path.join(WORK, 'material_meshes.json')
    if os.path.exists(p):
        mat_mesh = json.load(open(p))
    out, hist, fails = {}, collections.Counter(), collections.Counter()
    mats = sorted({m for v in cs.values() for m in v})
    tdir = os.path.join(WORK, 'textures')
    os.makedirs(tdir, exist_ok=True)
    for mat in mats:
        dirs = mat_mesh.get(mat, [])
        picked = None
        tries = [lower.get(c.lower()) for c in candidates(mat, dirs)]
        tries += basename_hits(mat, by_base)
        for real in tries:
            if not real:
                continue
            g = tc.fetch(real)
            if not g or textures.looks_like_normal_map(g[3]):
                continue
            picked = (real,) + g
            break
        if not picked:
            fails['no own texture'] += 1
            continue
        real, i, w, h, im, fmt = picked
        slug = 'MAT_' + re.sub(r'[^A-Za-z0-9_.-]', '_', mat)[-90:] + '.png'
        im.save(os.path.join(tdir, slug))
        out[mat] = {'base': real, 'png': f'textures/{slug}', 'w': w, 'h': h, 'mip': i}
        hist[f'{w}x{h}'] += 1
    json.dump(out, open(os.path.join(WORK, 'material_textures_basename.json'), 'w'),
              indent=1)
    print(f"materials resolved to their OWN texture: {len(out)} of {len(mats)}")
    for k, c in hist.most_common(8):
        print(f"    {k:<10} {c}")
    for k, c in fails.most_common():
        print(f"    {k}: {c}")
    for m, e in list(out.items())[:12]:
        print(f"    {m.rsplit('/',1)[-1][:44]:<46} -> {e['base'].rsplit('/',1)[-1][:34]} {e['w']}x{e['h']}")


if __name__ == '__main__':
    main()
