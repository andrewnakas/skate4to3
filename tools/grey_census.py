"""Which surfaces render grey, how much of the island they cover, and which binding path served them.

The user's standing complaint is grey materials. This ranks them by what a player sees - area -
instead of by count. For every rendered material in the staged GLBs it sums the surface area
and measures its base-colour image (mean colour, mean per-pixel chroma, luminance spread), then
names the binding path from the material name the exporter wrote:

    IOG_<Class>[__proxylod]__<texture>[__p<paint>]   'texture' (a real skate. texture)
    IOG_<Class>[__proxylod]__p<paint>                'paint'   (the block's colour, no texture)
    IOG_<Class>[__proxylod]                          'class'   (fallback class swatch)

A material counts as grey when its image has mean per-pixel chroma under --chroma (default 12).
Concrete, steel and asphalt are grey for real, so the list that matters is FLAT: an image with
almost no luminance variation (--flat, default 5) is a swatch or a placeholder, not a texture.

    python tools/grey_census.py [GLB_DIR] [--chroma 12] [--top 30]
"""
import argparse, collections, glob, io, os, sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import verify_build as vb  # noqa: E402


def path_of(name):
    rest = name[4:].split('__')
    parts = [p for p in rest[1:] if p != 'proxylod']
    if not parts:
        return 'class'
    if len(parts) == 1 and parts[0].startswith('p') and parts[0][1:].isdigit():
        return 'paint'
    return 'texture'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('glb', nargs='?', default=os.path.join(os.environ.get('D2S3', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'decendersmaptoskate3recomp')), 'work', 'IsleOfGrom', 'glb'))
    ap.add_argument('--chroma', type=float, default=12.0)
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--flat', type=float, default=5.0, help='luminance std under which an image is a flat swatch')
    args = ap.parse_args()
    vb.GLB = args.glb
    area = collections.Counter()
    stats = {}
    for f in sorted(glob.glob(os.path.join(args.glb, 'block_*.glb'))):
        js, bn = vb.read_glb(f)
        tex = [t['source'] for t in js.get('textures', [])]
        for mesh in js['meshes']:
            for pr in mesh['primitives']:
                mat = js['materials'][pr['material']]
                name = mat.get('name', '')
                if not name.startswith('IOG_') or name.startswith('IOG_Collision'):
                    continue
                P = vb.accessor(js, bn, pr['attributes']['POSITION']).astype(np.float64)
                T = vb.accessor(js, bn, pr['indices']).reshape(-1, 3).astype(np.int64)
                area[name] += 0.5 * np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), axis=1).sum()
                if name in stats:
                    continue
                bc = mat.get('pbrMetallicRoughness', {}).get('baseColorTexture')
                if bc is None:
                    stats[name] = None
                    continue
                im = js['images'][tex[bc['index']]]
                bv = js['bufferViews'][im['bufferView']]
                o = bv.get('byteOffset', 0)
                a = np.asarray(Image.open(io.BytesIO(bn[o:o + bv['byteLength']])).convert('RGB').resize((64, 64)), float)
                lum = a @ [0.299, 0.587, 0.114]
                stats[name] = (a.reshape(-1, 3).mean(0), float((a.max(2) - a.min(2)).mean()), float(lum.std()))
    tot = sum(area.values())
    grey = {n: a for n, a in area.items() if stats.get(n) and stats[n][1] < args.chroma}
    by_path = collections.defaultdict(lambda: [0.0, 0.0])
    by_cls = collections.defaultdict(lambda: [0.0, 0.0])
    for n, a in area.items():
        k = path_of(n)
        c = n[4:].split('__', 1)[0]
        by_path[k][0] += a
        by_cls[c][0] += a
        if n in grey:
            by_path[k][1] += a
            by_cls[c][1] += a
    print(f"rendered surface {tot:,.0f} m2 over {len(area)} materials; grey (chroma < {args.chroma:g}): "
          f"{sum(grey.values()):,.0f} m2 ({100 * sum(grey.values()) / tot:.1f}%)")
    print("by binding path:  " + "   ".join(f"{k} {v[0]:,.0f} m2 ({100 * v[1] / max(v[0], 1):.0f}% grey)" for k, v in sorted(by_path.items())))
    print("by class (grey m2 / total):")
    for c, (t, g) in sorted(by_cls.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"   {c:<10} {g:>10,.0f} / {t:>10,.0f} m2")
    flat = {n: a for n, a in area.items() if stats.get(n) and stats[n][2] < args.flat}
    print(f"FLAT materials (luminance spread < {args.flat:g}: a swatch, not a texture): "
          f"{sum(flat.values()):,.0f} m2 over {len(flat)} materials")
    for n, a in sorted(flat.items(), key=lambda kv: -kv[1])[:args.top]:
        m, ch, sd = stats[n]
        print(f"   {a:>9,.0f} m2  {path_of(n):<7} rgb {tuple(int(x) for x in m)} chroma {ch:4.1f} std {sd:4.1f}  {n[:92]}")
    print(f"largest grey materials:")
    for n, a in sorted(grey.items(), key=lambda kv: -kv[1])[:args.top]:
        m, ch, sd = stats[n]
        print(f"   {a:>9,.0f} m2  {path_of(n):<7} rgb {tuple(int(x) for x in m)} chroma {ch:4.1f} std {sd:4.1f}  {n[:92]}")


if __name__ == '__main__':
    main()
