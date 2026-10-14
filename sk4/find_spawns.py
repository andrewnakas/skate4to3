#!/usr/bin/env python3
"""Pick spawn markers that stand on open, flat ground in an exported GLB set.

The first build's startline sat 0.7 m above a floor, 0.5 m under a rock top, with signage
overhead: "the most common height near the skatepark" is not somewhere a skater can stand.
A marker here needs
  * an up-facing ground-class surface under it,
  * that surface at the same height across its 3x3 neighbourhood of 1 m cells,
  * nothing at all between 0.2 m and 2.4 m above it,
  * open air: no roof within 15 m overhead, and at most a third of the ground within 5 m
    blocked by anything taller than 1.2 m. The full-island startline passed every earlier rule
    while sitting under a plot structure 8.2 m up with a wall 3.1 m away, and the player
    spawned stuck inside it.

Writes d2s3_spawns.json (and the spawn counters in d2s3_export.json) beside the GLBs, and
spawn.txt holding the startline as "x,y,z" for d2s3's --spawn.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct

import numpy as np

# The startline's display name; export_glb sets it from the level (work/level.json).
START_NAME = 'Isle of Grom'
GROUND = {'IOG_Concrete', 'IOG_Asphalt', 'IOG_Tile', 'IOG_Brick', 'IOG_Wood', 'IOG_Misc',
          'IOG_Paint', 'IOG_Grass', 'IOG_Dirt', 'IOG_Sand'}
CLEAR_LO, CLEAR_HI, SUPPORT_TOL, STAND = 0.2, 2.4, 0.3, 0.5
ROOF_HI, OPEN_RADIUS, WALL_LO, WALL_HI, MAX_WALL_FRAC = 15.0, 5, 1.2, 4.0, 0.35


def load_glb(path):
    d = open(path, 'rb').read()
    jl = struct.unpack_from('<I', d, 12)[0]
    J = json.loads(d[20:20 + jl])
    b0 = 20 + jl
    B = d[b0 + 8:b0 + 8 + struct.unpack_from('<I', d, b0)[0]]

    def acc(i):
        a = J['accessors'][i]
        v = J['bufferViews'][a['bufferView']]
        dt = {5126: '<f4', 5125: '<u4', 5123: '<u2'}[a['componentType']]
        n = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[a['type']]
        arr = np.frombuffer(B, dtype=dt, count=a['count'] * n, offset=v.get('byteOffset', 0) + a.get('byteOffset', 0))
        return arr.reshape(a['count'], n) if n > 1 else arr

    for m in J['meshes']:
        for p in m['primitives']:
            yield (J['materials'][p['material']]['name'],
                   acc(p['attributes']['POSITION']).astype(np.float64),
                   acc(p['indices']).astype(np.int64).reshape(-1, 3))


def surface_samples(P, T, spacing=0.9):
    """Points on every triangle at <= `spacing`, plus which are on up-facing faces."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    fn = np.cross(b - a, c - a)
    up = fn[:, 1] > 0.85 * np.linalg.norm(fn, axis=1)
    emax = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                              np.linalg.norm(a - c, axis=1)])
    steps = np.clip(np.ceil(emax / spacing), 1, 40).astype(int)
    pts, ups = [], []
    for s in np.unique(steps):
        sel = steps == s
        uv = np.array([(i / s, j / s) for i in range(s + 1) for j in range(s + 1 - i)])
        A, BA, CA = a[sel], (b - a)[sel], (c - a)[sel]
        q = A[:, None, :] + uv[None, :, :1] * BA[:, None, :] + uv[None, :, 1:] * CA[:, None, :]
        pts.append(q.reshape(-1, 3))
        ups.append(np.repeat(up[sel], len(uv)))
    return np.concatenate(pts), np.concatenate(ups)


class Grid:
    def __init__(self, pts):
        ix, iz = np.floor(pts[:, 0]).astype(np.int64), np.floor(pts[:, 2]).astype(np.int64)
        key = (ix + (1 << 20)) * (1 << 21) + (iz + (1 << 20))
        order = np.argsort(key, kind='stable')
        self.key, self.y = key[order], pts[order, 1]

    def heights(self, ix, iz):
        k = (ix + (1 << 20)) * (1 << 21) + (iz + (1 << 20))
        lo, hi = np.searchsorted(self.key, k), np.searchsorted(self.key, k, side='right')
        return self.y[lo:hi]


def open_air(solid: Grid, ix, iz, h):
    """True when a skater standing at height h in cell (ix, iz) can ride away: nothing roofing the
    spot, and the ground around it mostly clear of anything taller than a ledge or rail."""
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            s = solid.heights(ix + du, iz + dv)
            if len(s) and np.any((s > h + CLEAR_HI) & (s < h + ROOF_HI)):
                return False
    walls = total = 0
    for du in range(-OPEN_RADIUS, OPEN_RADIUS + 1):
        for dv in range(-OPEN_RADIUS, OPEN_RADIUS + 1):
            if max(abs(du), abs(dv)) < 2:
                continue
            total += 1
            s = solid.heights(ix + du, iz + dv)
            if len(s) and np.any((s > h + WALL_LO) & (s < h + WALL_HI)):
                walls += 1
    return walls <= MAX_WALL_FRAC * total


def standable(ground: Grid, solid: Grid, ix, iz):
    """Heights h in cell (ix, iz) a skater can stand on, highest first."""
    out = []
    for h in sorted({round(float(v) * 4) / 4 for v in ground.heights(ix, iz)}, reverse=True):
        support = 0
        blocked = False
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                g = ground.heights(ix + du, iz + dv)
                if len(g) and np.any(np.abs(g - h) <= SUPPORT_TOL):
                    support += 1
                s = solid.heights(ix + du, iz + dv)
                if len(s) and np.any((s > h + CLEAR_LO) & (s < h + CLEAR_HI)):
                    blocked = True
        if support >= 8 and not blocked and open_air(solid, ix, iz, h):
            out.append(h)
    return out


def supported(coll: Grid, ix, iz, h, tol=0.35):
    """Collision floor at height h under the cell and most of its neighbours: the spot is real
    ground in the physics, not a render surface over reduced or missing collision."""
    if coll is None:
        return True
    n = 0
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            c = coll.heights(ix + du, iz + dv)
            if len(c) and np.any(np.abs(c - h) <= tol):
                n += 1
    return n >= 7


def highest(ground: Grid, solid: Grid, coll: Grid | None = None, top_n=6000):
    """The highest HEIGHT a skater can stand at, with open air and collision under it.

    This used to return the first cell whose tallest sample was highest - and then the height
    it actually stood at in that cell, which under a tall rock was the floor 60 m below it. Every
    candidate cell is now scored by its best standable height, and the search stops once no
    remaining cell could beat the best found.
    """
    cells = {}
    for k, y in zip(ground.key.tolist(), ground.y.tolist()):
        if y > cells.get(k, -1e9):
            cells[k] = y
    best = None
    for k, ymax in sorted(cells.items(), key=lambda kv: -kv[1])[:top_n]:
        if best is not None and ymax <= best[1]:
            break
        ix = (k // (1 << 21)) - (1 << 20)
        iz = (k % (1 << 21)) - (1 << 20)
        for h in standable(ground, solid, int(ix), int(iz)):
            if supported(coll, int(ix), int(iz), h):
                if best is None or h > best[1]:
                    best = (ix + 0.5, h, iz + 0.5)
                break
    return best


def search(ground, solid, x, z, radius, taken, min_gap=6.0):
    for r in range(0, radius + 1):
        ring = [(u, v) for u in range(-r, r + 1) for v in range(-r, r + 1) if max(abs(u), abs(v)) == r]
        for u, v in ring:
            ix, iz = math.floor(x) + u, math.floor(z) + v
            if any(math.hypot(ix + 0.5 - t[0], iz + 0.5 - t[2]) < min_gap for t in taken):
                continue
            hs = standable(ground, solid, ix, iz)
            if hs:
                return (ix + 0.5, hs[0], iz + 0.5)
    return None


def feature_clusters(feats, radius=25.0, gap=60.0, n=9):
    """Centres of the densest skate-feature clusters, densest first, at least `gap` apart."""
    F = np.asarray(feats, float)
    if not len(F):
        return []
    xz = F[:, [0, 2]]
    d = np.linalg.norm(xz[:, None] - xz[None], axis=2)
    near = d < radius
    score = near.sum(1).astype(float)
    out = []
    for i in np.argsort(-score, kind='stable'):
        c = xz[near[i]].mean(0)
        if all(np.hypot(*(c - o[:2])) >= gap for o in out):
            out.append((float(c[0]), float(c[1]), int(score[i])))
        if len(out) >= n:
            break
    return out


def run(glb_dir: str, checkpoints: int = 8, top: bool = True, features=None) -> list[dict]:
    gp, sp, rp, cp = [], [], [], []
    for f in sorted(glob.glob(os.path.join(glb_dir, '*.glb'))):
        for name, P, T in load_glb(f):
            pts, up = surface_samples(P, T)
            sp.append(pts)
            if name.startswith('IOG_Collision'):
                cp.append(pts[up])
                continue
            # Match the CLASS, not the whole material name: a primitive that found its
            # own game texture is named `IOG_Concrete__<texture>`, and once the depot bound
            # nearly all of the ground, exact-name matching left almost no ground samples
            # and the spawn search reported no standable ground on the whole island.
            if '__proxylod' in name:
                continue      # far-LOD plot silhouette, not a surface to stand on
            if name.split('__')[0] in GROUND:
                gp.append(pts[up])
            elif name.split('__')[0] == 'IOG_Rock':
                rp.append(pts[up])     # rock tops count for the "very top" startline only
    ground, solid = Grid(np.concatenate(gp)), Grid(np.concatenate(sp))
    top_ground = Grid(np.concatenate(gp + rp))
    coll = Grid(np.concatenate(cp)) if cp else None
    rails = json.load(open(os.path.join(glb_dir, 'splines.json')))['splines']
    lines = [np.array(r['points']) for r in rails]
    pts = np.concatenate(lines) if lines else np.zeros((1, 3))
    tx, tz = float(np.median(pts[:, 0])), float(np.median(pts[:, 2]))

    spawns, taken = [], []
    clusters = feature_clusters(features) if features else []
    start = None
    if clusters:
        # skate spots first: the densest cluster of skatepark kit, ramps and rails
        tx, tz = clusters[0][0], clusters[0][1]
        start = search(ground, solid, tx, tz, 40, taken)
        top = False
    if start is None and top:
        start = highest(top_ground, solid, coll)
    if start is None:
        start = search(ground, solid, tx, tz, 150, taken)
    if start is None:
        raise SystemExit('no standable ground within 150 m of the rail cluster')
    spawns.append({'name': os.environ.get('SPAWN_NAME', START_NAME), 'kind': 'startline', 'x': round(start[0], 3),
                   'y': round(start[1] + STAND, 3), 'z': round(start[2], 3), 'yaw': 0.0})
    taken.append(start)
    for cx, cz, _n in clusters[1:]:
        if len(spawns) > checkpoints:
            break
        got = search(ground, solid, cx, cz, 25, taken)
        if got:
            taken.append(got)
            spawns.append({'name': f'Spot {len(spawns)}', 'kind': 'checkpoint', 'x': round(got[0], 3),
                           'y': round(got[1] + STAND, 3), 'z': round(got[2], 3), 'yaw': 0.0})
    lengths = [float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()) for l in lines]
    for i in np.argsort(lengths)[::-1]:
        if len(spawns) > checkpoints:
            break
        l = lines[i]
        d = l[1] - l[0]
        d = d / (np.linalg.norm(d) or 1.0)
        got = search(ground, solid, l[0][0] - 4 * d[0], l[0][2] - 4 * d[2], 4, taken)
        if got:
            taken.append(got)
            spawns.append({'name': f'Rail {len(spawns)}', 'kind': 'checkpoint', 'x': round(got[0], 3),
                           'y': round(got[1] + STAND, 3), 'z': round(got[2], 3),
                           'yaw': round(math.degrees(math.atan2(d[0], d[2])), 2)})
    json.dump(spawns, open(os.path.join(glb_dir, 'd2s3_spawns.json'), 'w'), indent=1)
    ep = os.path.join(glb_dir, 'd2s3_export.json')
    if os.path.exists(ep):
        e = json.load(open(ep))
        e.setdefault('counters', {})['spawns'] = {'markers': len(spawns), 'trail_starts': 1}
        json.dump(e, open(ep, 'w'), indent=1)
    s = spawns[0]
    open(os.path.join(glb_dir, 'spawn.txt'), 'w').write(f"{s['x']},{s['y']},{s['z']}\n")
    below = solid.heights(math.floor(s['x']), math.floor(s['z']))
    print(f"{'skate-spot' if clusters else 'rail'} cluster centre ({tx:.1f}, {tz:.1f}); startline {s['x']},{s['y']},{s['z']} "
          f"({math.hypot(s['x'] - tx, s['z'] - tz):.1f} m away)")
    print(f"  column at startline: surfaces {sorted(set(np.round(below, 2)))[-6:]}")
    print(f"{len(spawns)} markers: " + ", ".join(f"{m['name']}@({m['x']:.0f},{m['y']:.1f},{m['z']:.0f})" for m in spawns))
    return spawns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('glb_dir')
    ap.add_argument('--checkpoints', type=int, default=8)
    args = ap.parse_args()
    run(args.glb_dir, args.checkpoints)


if __name__ == '__main__':
    main()
