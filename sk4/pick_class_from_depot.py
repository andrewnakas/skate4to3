"""Choose each class's fallback texture from what the GAME binds to that class.

The earlier picker scored candidate names out of the whole 23,543-texture index with
hand-tuned heuristics, and it picked `glass_stained_01` for Glass (chroma 31 - stained glass
for window panes) and `textile_felt_01` for Fabric (mean 201 - near white). There is no need
to guess: the ShaderBlockDepot says which texture the game binds to every material on the
island, so the right fallback for a class is simply the texture that the most of that class's
own surface already uses.

Weighted by square metres when a previous export's `uv_density.json` is available, by
material count otherwise. Candidates must survive the same sanity gates the verifier
applies, so a class never falls back to a mask, a placeholder or a baked atlas.

    python tools/pick_class_from_depot.py [uv_density.json]
"""
from __future__ import annotations
import collections, json, os, re, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bind_keys import role_of
from texcache import WORK

MAX_MEAN = 195.0        # near-white: the verifier fails a class texture above this
MAX_CHROMA = 26.0       # garish
MIN_STD = 3.0


def slug_of(name: str) -> str:
    """Same naming tools/extract_depot_textures.py writes."""
    return 'MAT_' + re.sub(r'[^A-Za-z0-9_.-]', '_', name)[-80:] + '.png'


def stats(fp):
    with Image.open(fp) as im:
        im = im.convert('RGB')
        a = np.asarray(im, dtype=np.float32)
        return (im.size, float(a.mean()), float(a.std()),
                float(np.abs(a - a.mean(axis=2, keepdims=True)).mean()))


def main():
    import export_glb as E
    depot = json.load(open(os.path.join(WORK, 'depot_textures.json')))
    tdir = os.path.join(WORK, 'textures')
    have = set(os.listdir(tdir))

    dens = sys.argv[1] if len(sys.argv) > 1 else None
    area = collections.Counter()
    if dens and os.path.exists(dens):
        for r in json.load(open(dens)):
            area[r['material']] += r['m2']

    # class -> colour texture -> weight
    want = collections.defaultdict(collections.Counter)
    norm_of = {}
    for e in depot.values():
        col = e.get('c')
        if not col or role_of(col) != 'c' or '/decal/' in col or col.rsplit('/', 1)[-1].startswith('decal_'):
            continue
        if 'projectlod' in col or '_atlas' in col:
            continue
        if e.get('ny'):
            norm_of.setdefault(col, e['ny'])
        for m in e.get('materials', ()):
            cls = E.classify(m)[0]
            if cls in ('Collision',):
                continue
            want[cls][col] += area.get(m, 0.0) + 1.0

    out, rejected = {}, collections.Counter()
    for cls, cands in sorted(want.items()):
        chosen = None
        for col, w in cands.most_common():
            slug = slug_of(col)
            if slug not in have:
                rejected[f'{cls}: png missing'] += 1
                continue
            fp = os.path.join(tdir, slug)
            (w_, h_), mean, std, chroma = stats(fp)
            if w_ != h_:
                rejected[f'{cls}: non-square'] += 1
                continue
            if mean > MAX_MEAN or chroma > MAX_CHROMA or (std < MIN_STD and mean > 170):
                rejected[f'{cls}: mean {mean:.0f} chroma {chroma:.0f} std {std:.1f}'] += 1
                continue
            e = {'base': col, 'png': f'textures/{slug}', 'w': w_, 'h': h_,
                 'weight': round(w, 1), 'candidates': len(cands)}
            nrm = norm_of.get(col)
            if nrm:
                nslug = slug_of(nrm)
                if nslug in have:
                    with Image.open(os.path.join(tdir, nslug)) as nim:
                        nim = nim.convert('RGB')
                        if np.asarray(nim, dtype=np.float32).reshape(-1, 3).mean(0)[2] >= 150:
                            e['normal'] = f'textures/{nslug}'
                            e['nw'], e['nh'] = nim.size
            chosen = e
            break
        if chosen:
            out[cls] = chosen

    prev = {}
    dstp = os.path.join(WORK, 'class_textures.json')
    if os.path.exists(dstp):
        prev = json.load(open(dstp))
    # A class the depot cannot serve keeps whatever it had rather than losing its texture.
    kept = [c for c in prev if c not in out]
    for c in kept:
        out[c] = prev[c]

    tmp = dstp + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dstp)

    print(f"class fallbacks from the depot: {len(out) - len(kept)} chosen, {len(kept)} kept")
    for cls, e in sorted(out.items()):
        was = prev.get(cls, {}).get('base', '-')
        tag = '' if was == e['base'] else f"   (was {str(was).rsplit('/', 1)[-1]})"
        src = 'kept' if cls in kept else f"{e.get('candidates', '?')} cand"
        print(f"  {cls:<10} {e['w']}x{e['h']:<4} {'nrm' if 'normal' in e else '   '} "
              f"{src:>9}  {e['base'].rsplit('/', 1)[-1][:46]}{tag}")
    if rejected:
        print("\n  rejected candidates:")
        for k, c in rejected.most_common(12):
            print(f"    {k} x{c}")


if __name__ == '__main__':
    main()
