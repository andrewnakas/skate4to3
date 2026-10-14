"""Ride-surface fidelity audit: does the floor you SEE sit where the floor you RIDE is?

Since the collision copy is skate.'s own (tools/physics.py), the ride surface is right by
construction; what can still drift is the drawn surface over it. A visible floor 5 cm above the
collision is a board sinking into the ground; 5 cm below is a board hovering. Both read as
"not smooth" on screen even though the physics is skate.'s.

Per 100 m tile, both copies are rasterised to their highest up-facing surface at 0.25 m
(d2s3.surface.top_raster: exact half-plane rasterisation), and compared cell by cell where both
exist. Only ride classes are drawn into the render raster. Writes ride_audit.json next to the GLBs.

    python tools/ride_audit.py [--glb DIR] [--cell 0.25] [--tol 0.02]
"""
import argparse, collections, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.environ.get('D2S3', os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                                           'decendersmaptoskate3recomp'))]
import verify_build as vb  # noqa: E402
from export_glb import PROTECTED  # noqa: E402
from d2s3 import surface  # noqa: E402

NEG = surface.NEG


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--glb', default=os.path.join(os.environ.get('D2S3', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'decendersmaptoskate3recomp')), 'work', 'IsleOfGrom', 'glb'))
    ap.add_argument('--cell', type=float, default=0.25)
    ap.add_argument('--tile-size', type=float, default=100.0)
    ap.add_argument('--tol', type=float, default=0.02, help='metres of render/collision disagreement tolerated')
    args = ap.parse_args()
    vb.GLB = args.glb

    rend = collections.defaultdict(list)
    coll = collections.defaultdict(list)
    for _f, mn, P, _uv, T in vb.prims():
        cname = mn[4:].split('__', 1)[0] if mn.startswith('IOG_') else ''
        is_coll = cname == 'Collision'
        if not is_coll and (cname not in PROTECTED or '__proxylod' in mn and False):
            continue
        c = P[T].mean(axis=1)
        tu = np.floor(c[:, 0] / args.tile_size).astype(int)
        tv = np.floor(c[:, 2] / args.tile_size).astype(int)
        for k in set(zip(tu.tolist(), tv.tolist())):
            sel = (tu == k[0]) & (tv == k[1])
            (coll if is_coll else rend)[k].append((P, T[sel], (cname, '__proxylod' in mn)))

    out, allc = {}, collections.Counter()
    by_cls = collections.defaultdict(lambda: [0, 0, 0, []])   # (class, proxy) -> same, above, below, |d| sample
    for k in sorted(set(rend) & set(coll)):
        lo = np.array([k[0] * args.tile_size, -1e4, k[1] * args.tile_size])
        hi = lo + np.array([args.tile_size, 2e4, args.tile_size])

        def raster(parts):
            P = np.concatenate([p for p, _, _ in parts])
            off, Ts = 0, []
            for p, t, _ in parts:
                Ts.append(t + off)
                off += len(p)
            r = surface.top_raster(P, np.concatenate(Ts), args.cell, up_min=0.7, bounds=(lo, hi))
            return None if r is None else r[0]

        hr, hc = raster(rend[k]), raster(coll[k])
        if hr is None or hc is None:
            continue
        groups = collections.defaultdict(list)
        for part in rend[k]:
            groups[part[2]].append(part)
        for g, parts in groups.items():
            hg = raster(parts)
            if hg is None:
                continue
            bg = (hg > NEG) & (hc > NEG)
            dg = (hg - hc)[bg]
            dg = dg[np.abs(dg) < 0.5]
            r = by_cls[g]
            r[0] += len(dg)
            r[1] += int((dg > args.tol).sum())
            r[2] += int((dg < -args.tol).sum())
            r[3].append(np.abs(dg[::max(1, len(dg) // 4000)]))
        both = (hr > NEG) & (hc > NEG)
        d = (hr - hc)[both]
        # Only disagreements small enough to be the SAME surface drawn slightly off: beyond 0.5 m
        # the two rasters are seeing different things (an awning over a floor, a proxy roof).
        same = np.abs(d) < 0.5
        dd = d[same]
        above, below = int((dd > args.tol).sum()), int((dd < -args.tol).sum())
        rec = {'cells_both': int(both.sum()), 'same_surface': int(same.sum()),
               'render_above_collision': above, 'render_below_collision': below,
               'median_abs_cm': round(float(np.median(np.abs(dd))) * 100, 2) if len(dd) else None,
               'p95_abs_cm': round(float(np.percentile(np.abs(dd), 95)) * 100, 2) if len(dd) else None,
               'render_only_cells': int(((hr > NEG) & (hc <= NEG)).sum())}
        out[f'{int(k[0] * args.tile_size + args.tile_size / 2)}_{int(k[1] * args.tile_size + args.tile_size / 2)}'] = rec
        allc.update({'same': int(same.sum()), 'above': above, 'below': below,
                     'render_only': rec['render_only_cells']})

    path = os.path.join(args.glb, 'ride_audit.json')
    json.dump(out, open(path, 'w'), indent=1)
    n = max(1, allc['same'])
    print(f"ride audit over {len(out)} tiles at {args.cell} m: {allc['same']:,} cells where the drawn "
          f"ride surface and the collision are the same surface")
    print(f"  drawn ABOVE collision by >{args.tol * 100:.0f} cm: {allc['above']:,} ({100 * allc['above'] / n:.2f}%)"
          f"   BELOW: {allc['below']:,} ({100 * allc['below'] / n:.2f}%)")
    print(f"  drawn floor with no collision under it: {allc['render_only']:,} cells "
          f"({allc['render_only'] * args.cell ** 2:,.0f} m2)")
    print(f"  {'class':<10} {'proxy':<5} {'m2 same surface':>16} {'above':>7} {'below':>7} {'median':>8} {'p95':>8}")
    for (c, px), (n, a, b, smp) in sorted(by_cls.items(), key=lambda kv: -kv[1][0]):
        if n < 400:
            continue
        smp = np.concatenate(smp) if smp else np.zeros(1)
        print(f"  {c:<10} {'yes' if px else 'no':<5} {n * args.cell ** 2:>16,.0f} {100 * a / n:>6.1f}% {100 * b / n:>6.1f}% "
              f"{np.median(smp) * 100:>6.1f}cm {np.percentile(smp, 95) * 100:>6.1f}cm")
    worst = sorted(out.items(), key=lambda kv: -(kv[1]['render_above_collision'] + kv[1]['render_below_collision']))[:6]
    for k, r in worst:
        print(f"   {k}: above {r['render_above_collision']:,}  below {r['render_below_collision']:,}  "
              f"median {r['median_abs_cm']} cm  p95 {r['p95_abs_cm']} cm  render-only {r['render_only_cells']:,}")
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
