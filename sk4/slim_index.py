"""One lean pass: slim on-disk indices + final class textures (albedo + normal).

Every earlier script re-parsed work/global_assets.json (64 MB -> ~1 GB of dicts) and
rebuilt the 24-TOC chunk catalog. With swap saturated that spike is what the OOM killer
kept picking. This loads the world ONCE, writes small indices, frees them, and finishes
the class picks from the slim data.
"""
import collections, gc, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textures, superbundle as sb
from cas import decompress_many

GAME = os.environ.get('SKATE_ROOT',
                      ('C:/Program Files (x86)/Steam/steamapps/common/Skate' if os.name == 'nt' else
                       '/run/media/nakas/Windows/Program Files (x86)/Steam/steamapps/common/Skate'))
LAYOUT = os.path.join(GAME, 'Data', 'Win32', 'configurations', 'layout')
WORK = os.environ.get('SKATE_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work'))
TEXTURE_TYPE = 0x6BDE20BA

def main():
    # ---- 1. slim res index: textures only
    p = os.path.join(WORK, 'tex_res.json')
    if os.path.exists(p):
        texres = json.load(open(p))
        print(f"  tex_res.json cached: {len(texres):,}")
    else:
        big = json.load(open(os.path.join(WORK, 'global_assets.json')))
        texres = {n: v[1:5] for n, v in big['res'].items() if v[6] == TEXTURE_TYPE}
        del big
        gc.collect()
        json.dump(texres, open(p, 'w'))
        print(f"  wrote tex_res.json: {len(texres):,} texture entries")

    # ---- 2. chunk catalog, built once
    p = os.path.join(WORK, 'chunk_catalog.json')
    if os.path.exists(p):
        cat = json.load(open(p))
        print(f"  chunk_catalog.json cached: {len(cat):,}")
    else:
        _o = sb.package_name
        def _tol(a, hi, _f=_o):
            try:
                return _f(a, hi)
            except KeyError:
                return f"pkg_{a:04x}_{hi:04x}"
        sb.package_name = _tol
        cat = {}
        for root, _, files in os.walk(os.path.join(GAME, 'Data')):
            for f in files:
                if not f.endswith('.toc'):
                    continue
                try:
                    S = sb.SuperBundle(os.path.join(root, f), GAME)
                except Exception:
                    continue
                for c in S.chunks():
                    cat.setdefault(c.guid, [c.package, c.cas_index, c.offset, c.size])
        json.dump(cat, open(p, 'w'))
        print(f"  wrote chunk_catalog.json: {len(cat):,} guids")

    loc = json.load(open(os.path.join(WORK, 'chunk_ranges.json')))
    H = {}
    def rd(pkg, ci, off, size):
        h = H.get((pkg, ci))
        if h is None:
            h = H[(pkg, ci)] = open(os.path.join(LAYOUT, pkg, f'cas_{ci:02d}.cas'), 'rb')
        h.seek(off)
        return h.read(size)

    def usable(name):
        v = texres.get(name)
        if not v:
            return None
        try:
            d = textures.parse_descriptor(decompress_many([rd(*v)])[0])
        except Exception:
            return None
        if not d:
            return None
        br = []
        for e in loc.get(d['guid'], [])[:2]:
            pkg, ci, off, size, lo, ls = e
            try:
                br.append((lo, ls, decompress_many([rd(pkg, ci, off, size)])[0]))
            except Exception:
                pass
        cb = []
        c = cat.get(d['guid_rev'])
        if c:
            try:
                cb.append(decompress_many([rd(c[0], c[1], c[2], c[3])])[0])
            except Exception:
                pass
        i, px = textures.best_mip(d, br, cb)
        if i is None:
            return None
        w, h = d['w'] >> i, d['h'] >> i
        if w < 32 or h < 32:
            return None
        try:
            return i, w, h, textures.decode(px, w, h, d['format'])
        except Exception:
            return None

    colour = [n for n in texres if n.rsplit('/', 1)[-1].lower().endswith(('_c', '_cu'))]
    def bad(t):
        b = t.rsplit('/', 1)[-1].lower()
        return ('_proxy' in b or b.startswith('decal_') or '_fx_' in b
                or b.startswith('t_fx') or 'thumb' in b or 'icon' in b)
    WANT = {
        'Concrete': [r'\bconcrete_'], 'Asphalt': [r'asphalt'],
        'Rock': [r'\bstone_', r'terrain_rock'], 'Dirt': [r'terrain_(mud|dirt|soil)'],
        'Grass': [r'terrain_grass'], 'Sand': [r'terrain_sand'], 'Wood': [r'\bwood_'],
        'Metal': [r'\bmetal_(steel|iron|alumin)'], 'Rust': [r'metal_rust'],
        'Brick': [r'masonry_brick'], 'Tile': [r'masonry_(ceramic|tile)'],
        'Plastic': [r'synthetic_plastic'], 'Fabric': [r'textile_'], 'Glass': [r'glass_'],
        'Paint': [r'synthetic_paint'], 'Sign': [r'signage_'],
        'Foliage': [r'terrain_leaves|foliage'], 'Misc': [r'\bstone_|masonry_|\bconcrete_'],
        'Water': [r'water'], 'Vehicle': [r'vehicle|veh_'], 'Emissive': [r'emissive'],
    }
    out = json.load(open(os.path.join(WORK, 'class_textures.json'))) \
        if os.path.exists(os.path.join(WORK, 'class_textures.json')) else {}
    tdir = os.path.join(WORK, 'textures')
    os.makedirs(tdir, exist_ok=True)
    for cls, pats in WANT.items():
        cands = []
        for pat in pats:
            rx = re.compile(pat)
            cands += [t for t in colour if rx.search(t.rsplit('/', 1)[-1].lower()) and not bad(t)]
        seen = set()
        uniq = [t for t in cands if not (t in seen or seen.add(t))]
        uniq.sort(key=lambda t: (0 if re.search(r'_(2x2m|1x1m|4x4m)', t.lower()) else 1, len(t)))
        best = None
        for t in uniq[:8]:
            u = usable(t)
            if not u:
                continue
            if best is None or u[1] * u[2] > best[2] * best[3]:
                best = (t,) + u
            if best[2] >= 512:
                break
        if not best:
            print(f"  {cls:<10} none installed")
            continue
        t, i, w, h, im = best
        im.save(os.path.join(tdir, f'CLASS_{cls}.png'))
        e = {'base': t, 'png': f'textures/CLASS_{cls}.png', 'w': w, 'h': h, 'mip': i}
        stem = t[:-2] if t.lower().endswith('_c') else t[:-3]
        nrm = '-'
        for suf in ('_ny', '_n', '_nm'):
            u2 = usable(stem + suf)
            if u2:
                u2[3].save(os.path.join(tdir, f'CLASSN_{cls}.png'))
                e['normal'] = f'textures/CLASSN_{cls}.png'
                nrm = f'{u2[1]}x{u2[2]}'
                break
        out[cls] = e
        print(f"  {cls:<10} {w}x{h:<5} mip{i}  normal {nrm:<9} {t.rsplit('/',1)[-1][:40]}")
        gc.collect()
    json.dump(out, open(os.path.join(WORK, 'class_textures.json'), 'w'), indent=1)
    print(f"\n  {len(out)} classes, {sum(1 for e in out.values() if 'normal' in e)} with normal maps")

if __name__ == '__main__':
    main()
