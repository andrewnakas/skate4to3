"""Builder validation: retail collision triangles -> sk3.collision_write.build_mesh -> bytes, then
(1) decode our bytes and compare triangles, (2) walk OUR tree + clusters from the bytes (MeshQuery)
for vertical rays and compare with a brute-force search over the source triangles, (3) every leaf's
triangles lie inside its cell and every unit is reached once, (4) compare our edge bytes with
retail's on the same triangles.
`python tests/test_collision_builder.py [World prefix]` prints the numbers per mesh.
"""
import collections, os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import big, sfil, rw4, collision as co, collision_write as cw

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')


def retail_meshes(world, prefix):
    a = big.read(os.path.join(DATA, 'content', f'worldDIST_{world}.big'))
    for e in a.entries:
        b = os.path.basename(e.name)
        if b.endswith('.xsf') and b.startswith(prefix):
            for x in sfil.read(a.read(e))[1]:
                for o in rw4.parse(x.arena).of_type(co.T_MESH):
                    yield o.data


def brute_down(P, x, z, y0=1e9):
    """vectorised version of co.vertical_hit over all source triangles"""
    x0, y0_, z0 = P[:, 0, 0], P[:, 0, 1], P[:, 0, 2]
    x1, y1, z1 = P[:, 1, 0], P[:, 1, 1], P[:, 1, 2]
    x2, y2, z2 = P[:, 2, 0], P[:, 2, 1], P[:, 2, 2]
    d = (x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)
    ok = np.abs(d) >= 1e-14
    d = np.where(ok, d, 1.0)
    l1 = ((x - x0) * (z2 - z0) - (x2 - x0) * (z - z0)) / d
    l2 = ((x1 - x0) * (z - z0) - (x - x0) * (z1 - z0)) / d
    l0 = 1 - l1 - l2
    ok &= (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
    y = l0 * y0_ + l1 * y1 + l2 * y2
    ok &= y <= y0
    if not ok.any():
        return None
    k = int(np.argmax(np.where(ok, y, -np.inf)))
    return float(y[k]), k


def validate(data, nrays=400, seed=1):
    src = co.parse_mesh(data)
    V, T, S, E, _U = src.triangles()
    m, rep = cw.build_mesh(V.astype(np.float64), T, S)
    b = cw.encode_mesh(m)
    r = dict(rep, retail_bytes=len(data), our_bytes=len(b))
    # (1) decode our own output: same triangle set (vertex positions + surface)
    back = co.parse_mesh(b)
    V2, T2, S2, _E2, _ = back.triangles()
    key = lambda VV, TT, SS: collections.Counter(
        (tuple(sorted(map(tuple, np.round(VV[t], 4).tolist()))), int(s)) for t, s in zip(TT, SS))
    r['decode_same_tris'] = key(V, T, S) == key(V2, T2, S2)
    r['reencode_exact'] = cw.encode_mesh(co.parse_mesh(b)) == b
    # (3) leaf containment, every unit reached once
    q = co.MeshQuery(b)
    seen = collections.Counter(); outside = 0
    for c, uid, lo, hi in q.leaves_hit(lambda lo, hi: True):
        for u, P, _s in q.leaf_triangles(c, uid):
            seen[u] += 1
            outside += not (np.all(P >= lo - 1e-6) and np.all(P <= hi + 1e-6))
    r['units_once'] = len(seen) == len(T2) and set(seen.values()) == {1}
    r['tris_outside_cell'] = outside
    # (2) vertical rays through our tree vs brute force over the source triangles
    P = V[T].astype(np.float64)
    rng = np.random.default_rng(seed)
    lo, hi = V.min(0), V.max(0)
    hits = miss = surf_bad = 0; err = 0.0
    for _ in range(nrays):
        x, z = rng.uniform(lo[0], hi[0]), rng.uniform(lo[2], hi[2])
        a, bb = q.ray_down(x, z), brute_down(P, x, z)
        if (a is None) != (bb is None):
            miss += 1
            continue
        if a is not None:
            hits += 1
            err = max(err, abs(a[0] - bb[0]))
            if abs(a[0] - bb[0]) < 1e-6:          # same height: surfaces must agree unless coplanar overlap
                surf_bad += (a[2] if a[2] is not None else -1) != int(S[bb[1]])
    r.update(rays=nrays, ray_hits=hits, ray_hit_mismatch=miss, ray_max_height_err=float(err),
             ray_surface_diff=surf_bad)
    # (4) our edge bytes vs retail on identical triangles
    Eb = cw.edge_bytes(np.round(V.astype(np.float64) / 0.001).astype(np.int64), T, enable_vertices=False)
    r['edge_code_eq'] = round(float(np.mean((Eb & 0x1f) == (E & 0x1f))), 4)
    r['edge_convex_eq'] = round(float(np.mean((Eb & 0x20) == (E & 0x20))), 4)
    r['edge_vertexbit_eq'] = round(float(np.mean((Eb & 0x40) == (E & 0x40))), 4)
    return r


def test_builder_on_retail_tile():
    try:
        data = next(retail_meshes('BlackBoxPark', 'cSim_-50_-50'))
    except (OSError, StopIteration):
        pytest.skip('no retail Skate 3 data')
    r = validate(data, nrays=200)
    assert r['decode_same_tris'] and r['reencode_exact'] and r['units_once']
    assert r['tris_outside_cell'] == 0 and r['ray_hit_mismatch'] == 0 and r['ray_max_height_err'] < 1e-6


if __name__ == '__main__':
    world, prefix = (sys.argv[1:3] if len(sys.argv) > 2 else ('BlackBoxPark', 'cSim_50_50'))
    for d in retail_meshes(world, prefix):
        print(validate(d))


def synthetic_terrain(n=60, size=120.0, seed=4):
    """an unquantised height field (2 n^2 triangles) with a box ledge, surfaces by quadrant"""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-size / 2, size / 2, n + 1)
    X, Z = np.meshgrid(xs, xs)
    Y = 2.0 * np.sin(X / 9.0) * np.cos(Z / 13.0) + rng.normal(0, 0.05, X.shape)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    T = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            T += [(a, a + n + 1, a + 1), (a + 1, a + n + 1, a + n + 2)]
    T = np.array(T)
    C = V[T].mean(1)
    S = np.where(C[:, 0] > 0, 0x0201, 0x2a71) + np.where(C[:, 2] > 0, 0, 0x8000)
    return V, T, S


def test_builder_synthetic_terrain():
    V, T, S = synthetic_terrain()
    m, rep = cw.build_mesh(V, T, S)
    b = cw.encode_mesh(m)
    q = co.MeshQuery(b)
    P = V[T]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        x, z = rng.uniform(-59, 59, 2)
        a, bb = q.ray_down(x, z), brute_down(P, x, z)
        assert a is not None and bb is not None
        worst = max(worst, abs(a[0] - bb[0]))
    assert worst < 2e-3           # quantisation to the 1 mm grid
    assert cw.encode_mesh(co.parse_mesh(b)) == b
