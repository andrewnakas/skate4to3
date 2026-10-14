"""Fill the gaps the depot cannot: resolve a material to a texture by its own NAME.

Most materials bind through their shader block (sk4/bind_keys.py). A few do not - their
block names a texture whose colour will not decode, or they are engine dev materials with no
real binding at all - and those land on a generic class texture. Two of them,
`DEV_ProxyGrid_World_M_Asphalt` and `Skatepark_Trim_Edges_M`, covered 14% of the skatepark
view, so "a few" is not the same as "a little".

The names are not arbitrary: skate. names a material after the library asset it is built from
(`Decal_Envelope_Trim_01_2x2m_M_DM_mat` -> `world/materials/decal/envelope_trim_01_2x2m/
decal_envelope_trim_01_2x2m_c`), and a dev material still names its SURFACE. So the stem is
tried against the texture index first, then the token-overlap family match sk4/bind_keys.py
uses for the terrain presets.

Runs after tools/inherit_variants.py, on work/material_textures.json in place.
"""
from __future__ import annotations
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textures
from bind_keys import preset_entry, role_of, family_of
from texcache import TexCache, WORK

DROP_TOKENS = ('mat', 'm', 'dm', 'proxy', 'shadow', 'grd', 'sl', 'ta')
# A stem match is exact and always trusted. Family scoring is a guess, and an ungated guess
# put a *nanaimo logo* on `N_Trim_Concrete` - the one-character token "n" matched inside
# "nanaimo" - and vegetation cards on `Orange_Dark_proxy`. So a family match must name a
# family long enough to mean something, score well, and actually appear in what it picked.
MIN_FAMILY_LEN = 4
# Pipeline words, not surface families. `Proxy_Iron_proxy` scored 4 against
# `vegetation_proxy_cards_02_c` purely because both say "proxy".
NOT_A_FAMILY = {'proxy', 'base', 'trim', 'dark', 'light', 'main', 'default', 'temp', 'test'}


def stem_candidates(mat: str):
    """Longest-first stems of the material name, as library basenames."""
    low = re.sub(r'[^a-z0-9_]', '_', mat.lower()).strip('_')
    parts = [p for p in low.split('_') if p]
    while parts:
        yield '_'.join(parts)
        if parts[-1] in DROP_TOKENS or re.fullmatch(r'[a-z]{2,}\d*', parts[-1]):
            parts = parts[:-1]
        else:
            parts = parts[:-1]


def main():
    mt = json.load(open(os.path.join(WORK, 'material_textures.json')))
    mats = sorted(json.load(open(os.path.join(WORK, 'material_meshes.json'))))
    tex = json.load(open(os.path.join(WORK, 'tex_res.json')))
    by_base = {}
    for n in tex:
        by_base.setdefault(n.rsplit('/', 1)[-1].lower(), n)
    families = {}
    for n in tex:
        if n.startswith('world/materials/') and role_of(n) == 'c':
            families.setdefault(family_of(n), {})['c'] = n

    # drop what a previous run of THIS tool added, so its rules can be re-applied cleanly
    for k in [k for k, e in mt.items() if isinstance(e, dict) and 'by_name' in e]:
        del mt[k]
    missing = [m for m in mats if m not in mt]
    print(f"materials with no texture: {len(missing)} of {len(mats)}")

    picks, how = {}, {}
    for m in missing:
        hit = None
        for stem in stem_candidates(m):
            if len(stem) < 6:
                break
            for suf in ('_c', '_c_ta'):
                cand = by_base.get(stem + suf)
                if cand:
                    hit = cand
                    break
            if hit:
                how[m] = 'stem'
                break
        if not hit:
            low = re.sub(r'^dev_proxygrid_world_m_', '', m.lower())
            fam = re.split(r'[^a-z0-9]+', low)[0]
            if len(fam) >= MIN_FAMILY_LEN and fam not in NOT_A_FAMILY:
                r = preset_entry(low, families)
                if r and r.get('c') and fam in r['c'].rsplit('/', 1)[-1].lower():
                    hit = r['c']
                    how[m] = 'family'
        if hit:
            picks[m] = hit
    print(f"resolved by name: {len(picks)}")

    tc = TexCache()
    tdir = os.path.join(WORK, 'textures')
    cache, added, failed = {}, 0, 0
    for m, name in sorted(picks.items()):
        slug = 'MAT_' + re.sub(r'[^A-Za-z0-9_.-]', '_', name)[-80:] + '.png'
        fp = os.path.join(tdir, slug)
        if name not in cache:
            if not os.path.exists(fp):
                g = tc.fetch(name, min_px=16)
                if not g:
                    cache[name] = None
                else:
                    _mip, w, h, im, _fmt = g
                    if textures.looks_like_normal_map(im):
                        cache[name] = None
                    else:
                        im.save(fp)
                        cache[name] = (f'textures/{slug}', w, h)
                    im.close()
            else:
                from PIL import Image
                with Image.open(fp) as im:
                    cache[name] = (f'textures/{slug}', im.size[0], im.size[1])
        got = cache[name]
        if not got:
            failed += 1
            continue
        png, w, h = got
        mt[m] = {'base': name, 'png': png, 'w': w, 'h': h, 'mip': 0, 'by_name': how[m]}
        added += 1
        print(f"   {m[:46]:<46} -> {name.rsplit('/', 1)[-1][:40]}  [{how[m]}]")

    dst = os.path.join(WORK, 'material_textures.json')
    tmp = dst + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(mt, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dst)
    print(f"\nadded {added} material(s) by name ({failed} whose texture would not decode); "
          f"{len(mt):,} bound in total")


if __name__ == '__main__':
    main()
