"""Recover objects skate. places that our placement data misses, from skate.'s own baked collision.

Every sublevel ships `staticmodelgroup.physics`: the collision of every static object in it, with
its placement already applied, in world space (tools/physics.py). Our instances reproduce it
exactly where they exist (12,397 of 12,803 parts within 5 cm, median 0.1 mm), but skate. bakes
objects our loaders never reach - instances placed through nested prefabs and the like: 30 copies
of one generated rail where we have 1, 29 barrier kits where we have 2.

A baked part shares its shape descriptor byte-for-byte with the part of the mesh's own
`<mesh>.physics` it came from, so the vertex bytes identify the mesh exactly, and the two part
transforms give the instance transform:

    world part:  x = s_w R_w v + t_w      mesh part:  x = s_0 R_0 v + t_0
    instance:    S = s_w / s_0,  R_i = R_w R_0^T,  t_i = t_w - S R_i t_0

Writes work/physics_placements.json: [[mesh, rows 3x3 (exporter convention W = P @ rows + t),
t, sublevel]], only for objects none of our placement sources already has.
"""
import collections, hashlib, json, os, struct, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import export_glb as eg  # noqa: E402
import physics  # noqa: E402
from cas import decompress_many  # noqa: E402

LEVEL_PFX = 'levels/game/dingolevel_isle_of_grom/dingolevel_isle_of_grom/'
OUT = os.path.join(eg.WORK, 'physics_placements.json')


def rel(b, o):
    return o + struct.unpack_from('<I', b, o)[0]


def raw_parts(b):
    nd = struct.unpack_from('<I', b, 0x18)[0]
    d0, p0 = rel(b, 0x28), rel(b, 0x30)
    out = []
    for k in range((rel(b, 0x34) - p0) // physics.PART_SIZE):
        P = p0 + k * physics.PART_SIZE
        di = struct.unpack_from('<I', b, P + 56)[0]
        h = None
        if di < nd:
            D = d0 + di * physics.DESC_SIZE
            nv, _nh, nf = struct.unpack_from('<3H', b, D + 24)
            vo = rel(b, D + 36)
            h = hashlib.sha1(b[vo:vo + nv * 12] + struct.pack('<2H', nv, nf)).hexdigest()
        out.append({'q': struct.unpack_from('<4f', b, P), 't': np.array(struct.unpack_from('<3f', b, P + 16)),
                    's': struct.unpack_from('<f', b, P + 28)[0] or 1.0, 'shape': h})
    return out


def main():
    G = json.load(open(os.path.join(eg.WORK, 'global_assets.json')))
    RES = G['res']
    del G
    cas = eg.Cas()
    per_mesh = [n for n in RES if n.endswith('.physics') and not n.endswith('staticmodelgroup.physics')]
    shape_of = collections.defaultdict(list)     # shape hash -> [(physics name, part record)]
    for n, b in zip(per_mesh, decompress_many([cas.read(*RES[n][1:5]) for n in per_mesh])):
        try:
            for p in raw_parts(b):
                if p['shape']:
                    shape_of[p['shape']].append((n, p))
        except (struct.error, ValueError):
            continue
    print(f"indexed {len(shape_of):,} distinct collision shapes from {len(per_mesh):,} per-mesh assets")

    # every object we already place, as world part centres (all sources, no crop, no drop)
    ours = []
    raw = json.load(open(os.path.join(eg.WORK, 'instances.json')))
    srcs = [(m, np.array(T[:3], float), np.array(T[3], float)) for m, T, _s in raw if m]
    bm = json.load(open(os.path.join(eg.WORK, 'blueprint_meshes.json')))['blueprint_meshes']
    for kind, bp, M, _src, excluded in json.load(open(os.path.join(eg.WORK, 'placements.json'))):
        if kind in ('ObjectReferenceObjectData', 'SpatialReferenceObjectData') and not excluded:
            for m in dict.fromkeys(bm.get(bp, [])):
                srcs.append((m, np.array(M[:3], float), np.array(M[3], float)))
    for m, M in json.load(open(os.path.join(eg.WORK, 'ecs_mesh_instances.json'))):
        if m:
            M = np.array(M)
            srcs.append((m, M[:3, :3], M[3, :3]))
    # one batched decompression (a Wine call per asset made this take over 20 minutes)
    need = sorted({eg.physics_name(m) for m, _R, _t in srcs} & set(RES))
    cache = {n: raw_parts(b) for n, b in zip(need, decompress_many([cas.read(*RES[n][1:5]) for n in need]))}
    for m, R, t in srcs:
        for p in cache.get(eg.physics_name(m), ()):
            ours.append(p['t'] @ R + t)
    grid = collections.defaultdict(list)
    for i, c in enumerate(ours):
        grid[tuple(np.floor(c / 0.05).astype(int))].append(c)

    def placed_already(c):
        k = np.floor(c / 0.05).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for o in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if np.linalg.norm(o - c) < 0.05:
                            return True
        return False

    smg = sorted(n for n in RES if n.startswith(LEVEL_PFX) and n.endswith('/staticmodelgroup.physics'))
    found, seen = [], set()
    stats = collections.Counter()
    for n, b in zip(smg, decompress_many([cas.read(*RES[n][1:5]) for n in smg])):
        sub = n[len(LEVEL_PFX):-len('/staticmodelgroup.physics')]
        for p in raw_parts(b):
            stats['parts'] += 1
            if placed_already(p['t']):
                stats['already placed'] += 1
                continue
            cands = shape_of.get(p['shape'])
            if not cands:
                stats['shape not in any per-mesh asset'] += 1
                continue
            pn, p0 = cands[0]
            mesh = pn[:-len('.physics')] + '_mesh'
            if mesh not in RES:
                stats['no render mesh'] += 1
                continue
            if len({c[0] for c in cands}) > 1:
                stats['ambiguous shape (first taken)'] += 1
            S = p['s'] / (p0['s'] or 1.0)
            Ri = physics._quat(*p['q']) @ physics._quat(*p0['q']).T
            ti = p['t'] - S * (Ri @ p0['t'])
            key = (mesh, *np.round(ti, 2))
            if key in seen:
                stats['further parts of a recovered object'] += 1
                continue
            seen.add(key)
            found.append([mesh, (S * Ri).T.round(6).tolist(), ti.round(4).tolist(), sub])
            stats['recovered objects'] += 1
    json.dump(found, open(OUT, 'w'), indent=0)
    print(', '.join(f"{k} {v:,}" for k, v in stats.items()))
    print('recovered, by mesh:')
    for m, c in collections.Counter(f[0].rsplit('/', 1)[-1] for f in found).most_common(25):
        print(f"   {c:4}x {m}")
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
