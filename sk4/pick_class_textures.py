"""One verified albedo + one verified tangent normal per material class."""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textures
from texcache import TexCache, WORK

# Patterns are tried IN ORDER and the first one with a gate-passing texture wins. Pooling
# them and taking the global best score threw the ordering away: Grass came out as
# terrain_grass_SKIRT (the vertical fringe at a grass edge, 256x64) and Wood as wood_WICKER,
# because specialty variants have more contrast than the plain ground material. Canonical
# stems therefore go first, and only genuinely absent ones fall through to the loose pattern.
WANT = {
    'Concrete': [r'concrete_(aggregate|poured|sidewalk|smooth)_\d', r'\bconcrete_'],
    'Asphalt':  [r'asphalt_(debris|road|smooth)_\d', r'asphalt'],
    'Rock':     [r'\bstone_(granite|limestone|slate)_\d', r'terrain_rock', r'\bstone_'],
    'Dirt':     [r'terrain_(mud|dirt|soil)_\d', r'terrain_(mud|dirt|soil)'],
    'Grass':    [r'terrain_grass_\d', r'terrain_grass'],
    'Sand':     [r'terrain_sand_\d', r'terrain_sand'],
    'Wood':     [r'\bwood_(planks|boards|plank)_\d', r'\bwood_(oak|pine|birch)_\d', r'\bwood_'],
    'Metal':    [r'metal_steel_01_\d', r'metal_(steel|iron)_\d'],
    'Rust':     [r'metal_rust_\d', r'metal_rust'],
    'Brick':    [r'masonry_brick_\d', r'masonry_brick'],
    'Tile':     [r'masonry_ceramic_tile_\d', r'masonry_tile_\d', r'masonry_pavingstone_\d'],
    'Plastic':  [r'synthetic_plastic_\d', r'synthetic_plastic'],
    'Fabric':   [r'textile_(canvas|cotton|cloth|denim|burlap)_\d', r'textile_'],
    'Glass':    [r'glass_window_\d', r'glass_opaque_\d', r'glass_'],
    'Paint':    [r'synthetic_paint_\d', r'synthetic_paint', r'\bconcrete_(poured|smooth)_\d'],
    'Sign':     [r'metal_steel_\d', r'signage_\d'],
    'Foliage':  [r'terrain_leaves_\d', r'terrain_leaves|foliage'],
    'Misc':     [r'masonry_pavingstone_\d', r'concrete_(aggregate|poured)_\d', r'masonry_'],
    'Water':    [r'water_(surface|ocean)_\d', r'water'],
    'Vehicle':  [r'vehicle_paint_\d', r'vehicle|veh_'],
    'Emissive': [r'emissive'],
    'Rail':     [r'metal_iron_rusted_\d', r'metal_(steel|iron)_\d'],
}


FLAT_OK = {'Glass', 'Metal', 'Rail', 'Paint', 'Plastic', 'Sign'}
# Multicoloured decorative art is never a tiling surface: masonry_ceramic_MOSAIC tiled
# candy-coloured jewels across every Tile surface on the island.
GARISH_RX = re.compile(r'mosaic|rainbow|graffiti|mural|pieballoon|cutout|logo|pattern|decal')


def colourfulness(im):
    """Mean distance from grey. A material sits low; a mosaic or logo sits high."""
    import numpy as np
    a = np.asarray(im.convert('RGB'), dtype=np.float32)
    return float(np.abs(a - a.mean(axis=2, keepdims=True)).mean())


def quality(im, cls=None, name=''):
    """(ok, score, mean, std) - is this image usable as a tiling surface, and how good?

    Name and resolution are not enough. Picking on those alone gave Concrete AND Misc the
    same white pool plaster (mean 185, std 4.6), Metal a flat 194/std 2.4 and Sign a blank
    224/std 1.5 - nine of nineteen classes came out washed out or featureless, which is why
    the island rendered white. A surface texture has to have contrast and sit in a sane
    brightness band.
    """
    import numpy as np
    a = np.asarray(im.convert('RGB'), dtype=np.float32)
    flat = a.reshape(-1, 3)
    mean = float(flat.mean())
    std = float(a.std())
    # per-channel spread: a real material varies, a fill does not
    detail = float(np.abs(np.diff(a.mean(axis=2), axis=1)).mean())
    h, w = a.shape[0], a.shape[1]
    square = w == h
    # A tiling MATERIAL has moderate contrast. A logo or deck graphic has enormous contrast
    # and a strip aspect: signage_7plymaple_1x4_02_c (64x256, std 87.5) scored top of its
    # class and tiled rainbow candy-stripes across every sign surface in the level. Same trap
    # took Grass to a 256x64 grass-skirt fringe. So require square, and stop rewarding std
    # past the point where it stops meaning "material" and starts meaning "picture".
    # Flat is CORRECT for some materials. Glass, steel and paint really are near-featureless,
    # and EVERY candidate failed a std>=11 gate written for terrain - so Rail fell through to a
    # flat gold debug swatch and Metal to a near-white blank. Judge those on brightness instead.
    chroma = colourfulness(im)
    garish = bool(GARISH_RX.search(name.rsplit('/', 1)[-1].lower())) or chroma > 26.0
    if cls in FLAT_OK:
        ok = (12.0 <= mean <= 192.0 and detail >= 0.05 and square
              and not garish and chroma <= 12.0)
    else:
        ok = (12.0 <= mean <= 200.0 and 11.0 <= std <= 60.0 and detail >= 1.2
              and square and not garish)
    # Resolution was only a tie-break, so a 128px texture with marginally better contrast beat
    # a 256px one every time. Over a 50 m cliff face that difference is the whole look, so give
    # each doubling real weight - while keeping contrast dominant so a sharp blank never wins.
    import math
    res_bonus = 7.0 * math.log2(max(w, h) / 128.0)
    score = min(std, 45.0) + 14.0 * detail - 0.16 * max(0.0, mean - 165.0) + res_bonus
    if not square:
        score -= 25.0
    if garish:
        score -= 60.0
    return ok, score, mean, std


def bad(t):
    b = t.rsplit('/', 1)[-1].lower()
    return ('_proxy' in b or b.startswith('decal_') or '_fx_' in b or b.startswith('t_fx')
            or 'thumb' in b or 'icon' in b)


def main():
    tc = TexCache()
    colour = [n for n in tc.res if n.rsplit('/', 1)[-1].lower().endswith(('_c', '_cu')) and not bad(n)]
    tdir = os.path.join(WORK, 'textures')
    for f in glob.glob(os.path.join(tdir, 'CLASS*_*.png')):
        os.remove(f)
    out = {}
    print(f"{'class':<9} {'albedo':>9}  {'normal':>9}  asset")
    for cls, pats in WANT.items():
        pick = None
        fallback = None
        seen = set()
        for pat in pats:
            rx = re.compile(pat)
            cands = [t for t in colour
                     if t not in seen and rx.search(t.rsplit('/', 1)[-1].lower())]
            seen.update(cands)
            # square tiling assets first, then the plainest (shortest) name
            cands.sort(key=lambda t: (0 if re.search(r'_(2x2m|1x1m|4x4m)', t.lower()) else 1,
                                      len(t.rsplit('/', 1)[-1])))
            passers = []
            for t in cands[:24]:
                got = tc.fetch(t)
                if not got or textures.looks_like_normal_map(got[3]):
                    continue
                ok, sc, mean, std = quality(got[3], cls, t)
                rec = (sc, got, t, mean, std, ok)
                if ok:
                    passers.append(rec)
                elif fallback is None or sc > fallback[0]:
                    fallback = rec
            if passers:
                # best-looking texture WITHIN this priority tier
                pick = max(passers, key=lambda r: (r[0], r[1][1] * r[1][2]))
                break
        pick = pick or fallback
        if not pick:
            print(f"{cls:<9} {'-':>9}  {'-':>9}  none verified")
            continue
        _sc, got, t, mean, std, ok = pick
        best = (t,) + got
        note = '' if ok else ' (best available, still weak)'
        t, i, w, h, im, fmt = best
        im.save(os.path.join(tdir, f'CLASS_{cls}.png'))
        e = {'base': t, 'png': f'textures/CLASS_{cls}.png', 'w': w, 'h': h, 'mip': i}
        stem = t[:-2] if t.lower().endswith('_c') else t[:-3]
        ntxt = '-'
        for suf in ('_ny', '_n', '_nm'):
            g = tc.fetch(stem + suf)
            if g and textures.valid_tangent_normal(g[3]):
                g[3].save(os.path.join(tdir, f'CLASSN_{cls}.png'))
                e['normal'] = f'textures/CLASSN_{cls}.png'
                ntxt = f'{g[1]}x{g[2]}'
                break
        out[cls] = e
        print(f"{cls:<9} {w:>4}x{h:<4}  {ntxt:>9}  std {std:5.1f} mean {mean:5.0f}  "
              f"{t.rsplit('/', 1)[-1][:34]}{note}")
    json.dump(out, open(os.path.join(WORK, 'class_textures.json'), 'w'), indent=1)
    print(f"\n{len(out)} classes, {sum('normal' in e for e in out.values())} with verified normals")


if __name__ == '__main__':
    main()
