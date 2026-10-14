"""Offline verification of a built Isle of Grom: textures, UVs, geometry, collision, grinds.

Runs against the staged GLBs and the packed streams, with no game running. Every check is a
measurement with a stated threshold, so "perfect" means a number, not an impression.

    python tools/verify_build.py [glb_dir] [dist_dir]
"""
from __future__ import annotations
import collections, glob, json, os, re, struct, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texcache import WORK

D2S3 = os.environ.get('D2S3', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'decendersmaptoskate3recomp'))
GLB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D2S3, 'work', 'IsleOfGrom', 'glb')
DIST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    D2S3, 'work', 'IsleOfGrom', 'DIST_IsleOfGrom')

FAIL: list[str] = []
WARN: list[str] = []

CELL = 2.0               # m, collision coverage grid
HOLE_DY = 1.5            # m, how far below a visual floor its collision may sit
FLOOR_UP = 0.7           # normal.y above which a triangle counts as a floor
UV_LO, UV_HI = 0.40, 0.62
# Stream ceilings, per kind, from what has actually loaded and hung (d2s3/knobs.py, maps.json):
#   cPres (render): the 5 MB figure is soft - Ovanny loads at 9.6-10.2 MB, The Pit shipped 10.5,
#     Summer Camp (2026-09-22) loaded seven tiles of 5.1-6.3 MB. Warn at 7, fail at 8.
#   cSim (collision): Flowy shipped 8.6 MB and hung the loading screen; the largest proven to
#     load is Ovanny's 4.21 MB. So collision is the tight one, and render is not.
#   cPres_Global is always resident: "Igloo boots with 10.9 MB; Favela hangs with 21.8".
PRES_WARN, PRES_MAX = 7_000_000, 8_000_000
# Collision: five Grom boots fit ONE per-stream limit between 3.045 MB (loaded) and 3.091 MB
# (stalled), with the first two streams requested together - see PLAN.md. 2.92 MB is proven.
SIM_WARN, SIM_MAX = 2_920_000, 3_000_000
# Skate 3 keeps the collision of the 3x3 tiles around the player resident together: 14.87 MB
# loaded and 15.05 MB sat on the loading screen (PLAN.md, "The real limit"). Checked on every
# 3x3 window of real streams, because the exporter can only plan it from a byte estimate.
WINDOW_WARN, WINDOW_MAX = 14_500_000, 14_870_000
GLOBAL_MAX = 11_000_000
# Invisible collision (tools/invisible_collision.py), fraction of floor or wall collision area.
# The shipped Summer Camp of 2026-09-22 had 12.7% floor / 5.3% wall.
INVISIBLE_WARN, INVISIBLE_MAX = 0.01, 0.02

_RNG = np.random.default_rng(0)
_CT = {5121: ('u1', 1), 5123: ('<u2', 2), 5125: ('<u4', 4), 5126: ('<f4', 4)}


def read_glb(path):
    b = open(path, 'rb').read()
    off, js, bn = 12, None, None
    while off < len(b):
        ln, ty = struct.unpack_from('<II', b, off)
        off += 8
        if ty == 0x4E4F534A:
            js = json.loads(b[off:off + ln])
        elif ty == 0x004E4942:
            bn = b[off:off + ln]
        off += ln
    return js, bn


def accessor(js, bn, i):
    a = js['accessors'][i]
    bv = js['bufferViews'][a['bufferView']]
    dt, sz = _CT[a['componentType']]
    n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
    base = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    st = bv.get('byteStride') or sz * n
    if st == sz * n:
        return np.frombuffer(bn, dt, a['count'] * n, base).reshape(a['count'], n)
    out = np.empty((a['count'], n), dt)
    for k in range(a['count']):
        out[k] = np.frombuffer(bn, dt, n, base + k * st)
    return out


def prims():
    """(material name, positions, uv|None, triangles) for every primitive in every block."""
    for f in sorted(glob.glob(os.path.join(GLB, 'block_*.glb'))):
        js, bn = read_glb(f)
        names = [m.get('name', '') for m in js['materials']]
        for mesh in js['meshes']:
            for pr in mesh['primitives']:
                mi = pr.get('material')
                mn = names[mi] if mi is not None else ''
                P = accessor(js, bn, pr['attributes']['POSITION']).astype(np.float64)
                ai = pr['attributes'].get('TEXCOORD_0')
                UV = accessor(js, bn, ai).astype(np.float64) if ai is not None else None
                T = accessor(js, bn, pr['indices']).reshape(-1, 3).astype(np.int64)
                yield os.path.basename(f), mn, P, UV, T


def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


def check_textures():
    hdr('TEXTURES')
    mt = json.load(open(os.path.join(WORK, 'material_textures.json')))
    kt_path = os.path.join(WORK, 'depot_key_textures.json')
    kt = json.load(open(kt_path)) if os.path.exists(kt_path) else {}
    dens = os.path.join(GLB, 'uv_density.json')
    rows = json.load(open(dens)) if os.path.exists(dens) else []
    print(f"  bindings available: {len(kt):,} by shader-block key, {len(mt):,} by material name")
    if rows:
        tot = sum(r['m2'] for r in rows)
        ex = sum(r['m2'] for r in rows if r.get('bound'))
        print(f"  island surface on the game's own texture: {ex:,.0f} of {tot:,.0f} m2 "
              f"({100 * ex / tot:.1f}%)")
        if tot and ex / tot < 0.80:
            WARN.append(f"only {100 * ex / tot:.0f}% of surface has the game's own texture")

    ct = json.load(open(os.path.join(WORK, 'class_textures.json')))
    print(f"\n  {'class':<10} {'res':>9} {'nrm':>9} {'mean':>5} {'std':>5} {'chroma':>6}  verdict")
    for c, e in sorted(ct.items()):
        p = os.path.join(WORK, e['png'])
        if not os.path.exists(p):
            FAIL.append(f"class texture {c}: png missing")
            continue
        im = Image.open(p).convert('RGB')
        a = np.asarray(im, dtype=np.float32)
        mean, std = float(a.mean()), float(a.std())
        chroma = float(np.abs(a - a.mean(axis=2, keepdims=True)).mean())
        v = []
        if mean > 195:
            v.append('WASHED OUT')
            FAIL.append(f"class {c}: near-white mean {mean:.0f}")
        if chroma > 26:
            v.append('GARISH')
            FAIL.append(f"class {c}: chroma {chroma:.0f}")
        if im.size[0] != im.size[1]:
            v.append('NON-SQUARE')
            FAIL.append(f"class {c}: {im.size}")
        if std < 3 and mean > 170:
            v.append('blank-ish')
            WARN.append(f"class {c}: std {std:.1f} mean {mean:.0f}")
        nrm = '-'
        if 'normal' in e:
            n = np.asarray(Image.open(os.path.join(WORK, e['normal'])).convert('RGB'),
                           dtype=np.float32)
            nrm = f"{n.shape[1]}x{n.shape[0]}"
            if n.reshape(-1, 3).mean(0)[2] < 150:
                v.append('BAD NORMAL')
                FAIL.append(f"class {c}: normal blue {n.reshape(-1, 3).mean(0)[2]:.0f}")
        print(f"  {c:<10} {im.size[0]:>4}x{im.size[1]:<4} {nrm:>9} {mean:5.0f} {std:5.1f} "
              f"{chroma:6.1f}  {' '.join(v) or 'ok'}")


def check_uv():
    hdr('UV TILING  (target 0.50 tiles/m: skate. authors the *_2x2m library)')
    per = collections.defaultdict(lambda: [0.0, 0.0])
    dens = os.path.join(GLB, 'uv_density.json')
    if os.path.exists(dens):
        # The exporter's own record: it still knows which primitives are baked atlases,
        # whose UVs index a packed sheet and must NOT be judged against a tiling target.
        n_baked = 0
        for r in json.load(open(dens)):
            if r.get('baked'):
                n_baked += 1
                continue
            p = per[r['class']]
            p[0] += r['m2']
            p[1] += r['uv_area']
        if n_baked:
            print(f"  ({n_baked} baked-atlas material/class pair(s) excluded: their UVs "
                  f"index a sheet)")
        report_uv(per)
        return
    for _f, mn, P, UV, T in prims():
        if UV is None or 'Collision' in mn:
            continue
        cls = mn.split('__')[0].replace('IOG_', '') or 'unknown'
        J = T if len(T) <= 8000 else T[::max(1, len(T) // 8000)]
        q = len(T) / len(J)
        a, b, c = P[J[:, 0]], P[J[:, 1]], P[J[:, 2]]
        wa = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        d1, d2 = UV[J[:, 1]] - UV[J[:, 0]], UV[J[:, 2]] - UV[J[:, 0]]
        ua = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        ok = wa > 1e-9
        r = per[cls]
        r[0] += float(wa[ok].sum()) * q
        r[1] += float(ua[ok].sum()) * q
    report_uv(per)


def report_uv(per):
    print(f"  {'class':<12} {'surface m2':>12} {'tiles/m':>9}  verdict")
    for cls, (wa, ua) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        if wa < 50:
            continue
        d = (ua / wa) ** 0.5 if wa else 0.0
        if d < 0.25:
            verdict = 'STRETCHED'
            FAIL.append(f"{cls}: {d:.3f} tiles/m over {wa:,.0f} m2")
        elif not UV_LO <= d <= UV_HI:
            verdict = 'off target'
            WARN.append(f"{cls}: {d:.2f} tiles/m")
        else:
            verdict = 'ok'
        print(f"  {cls:<12} {wa:>12,.0f} {d:>9.3f}  {verdict}")


def check_collision():
    hdr('COLLISION COVERAGE  (a floor the skater can stand on, per 2 m cell)')
    # Classes the exporter deliberately never collides are not floors: the four cells this
    # check first reported as open were all `terrain_grass_skirt_01_2x05m`, the decorative
    # grass fringe, whose cards face up but are not something to stand on.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import export_glb as E
    never = set(E.NEVER_COLLIDE)
    print(f"  not counted as floor: {', '.join(sorted(never))}")
    floor_y, coll_y = {}, {}
    vis_tris = coll_tris = 0
    for _f, mn, P, _UV, T in prims():
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm, axis=1)
        good = ln > 1e-9
        up = np.zeros(len(T), bool)
        up[good] = nrm[good, 1] / ln[good] > FLOOR_UP
        # Area samples, not centroids: a 13,000 m2 lake or terrain triangle marked ONE 2 m cell,
        # and the map-edge "holes" this check reported (2026-09-22: 2,809 cells) were largely that.
        area = 0.5 * ln
        is_coll = 'Collision' in mn
        if is_coll:
            coll_tris += len(T)
            sel = up
        elif mn.split('__')[0].replace('IOG_', '') not in never:
            vis_tris += len(T)
            sel = up & (area > 0.25)
        else:
            continue
        if not sel.any():
            continue
        import invisible_collision as _ic
        q, _w, _r = _ic._sample(a[sel], b[sel], c[sel], 1.0, 10**6, _RNG)
        u = np.floor(q[:, 0] / CELL).astype(np.int64)
        v = np.floor(q[:, 2] / CELL).astype(np.int64)
        tgt = coll_y if is_coll else floor_y
        for k, y in zip(zip(u.tolist(), v.tolist()), q[:, 1].tolist()):
            if y > tgt.get(k, -1e9):
                tgt[k] = y
    # Judged against the cell AND its eight neighbours. The visual copy is deliberately
    # coarser than the collision copy on tiles that would otherwise blow the stream ceiling,
    # so a big visual triangle's centroid can sit a metre above the dense collision beneath
    # it on rocky ground; comparing single columns counted that as a hole.
    def backed(k, y):
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                if coll_y.get((k[0] + du, k[1] + dv), -1e9) >= y - HOLE_DY:
                    return True
        return False

    holes = [k for k, y in floor_y.items() if not backed(k, y)]
    bare = [k for k in holes if not any((k[0] + du, k[1] + dv) in coll_y
                                        for du in (-1, 0, 1) for dv in (-1, 0, 1))]
    cov = 1 - len(holes) / max(len(floor_y), 1)
    print(f"  visual triangles {vis_tris:,}   collision triangles {coll_tris:,}")
    print(f"  cells with a visible floor: {len(floor_y):,} "
          f"({len(floor_y) * CELL * CELL:,.0f} m2)")
    print(f"  of those, backed by collision within {HOLE_DY} m: "
          f"{len(floor_y) - len(holes):,}  ({100 * cov:.2f}%)")
    print(f"  of the rest, cells with NO collision anywhere nearby: {len(bare):,}"
          f"  ({len(bare) * CELL * CELL:,.0f} m2)")
    if holes:
        xs = [k[0] * CELL for k in holes]
        zs = [k[1] * CELL for k in holes]
        print(f"  {len(holes)} uncovered cell(s) ({len(holes) * CELL * CELL:,.0f} m2), "
              f"x {min(xs):.0f}..{max(xs):.0f}  z {min(zs):.0f}..{max(zs):.0f}")
        clumps = collections.Counter((int(k[0] * CELL // 50) * 50, int(k[1] * CELL // 50) * 50)
                                     for k in holes)
        print("  worst 50 m patches: "
              + ", ".join(f"({a},{b}):{n}" for (a, b), n in clumps.most_common(6)))
    if bare:
        FAIL.append(f"{len(bare)} floor cell(s) with no collision anywhere near them "
                    f"({len(bare) * CELL * CELL:,.0f} m2)")
    if cov < 0.99:
        WARN.append(f"collision backs {100 * cov:.2f}% of the visible floor "
                    f"({len(holes)} cells beyond {HOLE_DY} m)")


def check_grinds():
    hdr('GRIND SPLINES')
    p = os.path.join(GLB, 'splines.json')
    if not os.path.exists(p):
        FAIL.append('splines.json missing')
        return
    sp = json.load(open(p))['splines']
    L = [float(np.linalg.norm(np.diff(np.asarray(s['points'], float), axis=0), axis=1).sum())
         for s in sp]
    if not L:
        FAIL.append('no grind splines')
        return
    L.sort()
    print(f"  splines {len(sp):,}   total {sum(L):,.0f} m   median {L[len(L) // 2]:.1f} m   "
          f"longest {L[-1]:.1f} m")
    short = sum(1 for x in L if x < 1.0)
    if short:
        WARN.append(f"{short} spline(s) under 1 m")
    print(f"  under 1 m: {short}   over 20 m: {sum(1 for x in L if x > 20)}")


def check_invisible():
    """Collision with no drawn surface within 0.5 m: invisible walls and floors."""
    hdr('INVISIBLE COLLISION  (collision surface with nothing drawn within 0.5 m)')
    import invisible_collision
    res = invisible_collision.measure(GLB, 0.5)
    frac = res['invisible'] / max(res['total'], 1.0)
    for k in ('floor', 'wall'):
        t, b = res['kinds'][k]
        f = b / max(t, 1.0)
        if f > INVISIBLE_MAX:
            FAIL.append(f"{b:,.0f} m2 of {k} collision ({100 * f:.1f}%) has nothing drawn within 0.5 m")
        elif f > INVISIBLE_WARN:
            WARN.append(f"{b:,.0f} m2 of {k} collision ({100 * f:.1f}%) has nothing drawn within 0.5 m")
    print(f"  overall {100 * frac:.1f}% of {res['total']:,.0f} m2")


def check_tile_textures():
    """Per-tile texture copies: the arena builder keeps them in their tiles by image name ('@u_v' folds
    into the GUID), so identical bytes are fine. The promoted payload is the thing to watch."""
    import hashlib
    per = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(GLB, 'block_*.glb'))):
        js, bn = read_glb(f)
        for im in js.get('images', []):
            n = im.get('name', '')
            if '@' not in n:
                continue
            bv = js['bufferViews'][im['bufferView']]
            o = bv.get('byteOffset', 0)
            per[n.split('@')[0]][n] = hashlib.sha1(bn[o:o + bv['byteLength']]).hexdigest()
    if not per:
        return
    hdr('PER-TILE TEXTURES')
    dup = 0
    for base, copies in per.items():
        dup += len(copies) - len(set(copies.values()))
    n = sum(len(c) for c in per.values())
    print(f"  {n:,} tile copies of {len(per):,} textures; byte-identical across tiles: {dup} "
          f"(kept apart by name)")
    log = os.path.join(os.path.dirname(GLB), 'arenabuild.log')
    if os.path.exists(log):
        m = re.search(r'\[TexPlan\] Promotion: (\d+)/(\d+) candidate\(s\) into cPres_Global \((\d+) KiB\)',
                      open(log, errors='replace').read())
        if m:
            print(f"  the arena builder promoted {m[1]} texture(s) ({m[3]} KiB payload) into cPres_Global")


def check_streams():
    hdr('STREAMS')
    files = [(os.path.getsize(p), os.path.basename(p))
             for p in glob.glob(os.path.join(DIST, '*.xsf'))]
    if not files:
        print(f"  no streams at {DIST} (geometry-only verification)")
        return
    files.sort(reverse=True)
    glob_sz = next((s for s, n in files if n == 'cPres_Global.xsf'), 0)
    tiles = [(s, n) for s, n in files if n != 'cPres_Global.xsf']
    print(f"  {len(files)} stream(s); cPres_Global {glob_sz / 1e6:.2f} MB "
          f"of {GLOBAL_MAX / 1e6:.0f} MB allowed")
    if glob_sz > GLOBAL_MAX:
        FAIL.append(f"cPres_Global.xsf is {glob_sz / 1e6:.2f} MB, over the "
                    f"{GLOBAL_MAX / 1e6:.0f} MB always-resident limit")
    sim = {}
    for s_, n_ in tiles:
        m_ = re.match(r'cSim_(-?\d+)_(-?\d+)_high\.xsf$', n_)
        if m_:
            sim[(int(m_[1]), int(m_[2]))] = s_
    xs = sorted({x for x, _ in sim})
    step = min((b - a for a, b in zip(xs, xs[1:]) if b > a), default=100)   # the tile size these streams use
    wins = sorted(((sum(sim.get((x + dx, z + dz), 0) for dx in (-step, 0, step) for dz in (-step, 0, step)), (x, z))
                   for (x, z) in sim), reverse=True)
    if wins:
        print(f"  collision 3x3 windows: heaviest {wins[0][0] / 1e6:.2f} MB around {wins[0][1][0]}_{wins[0][1][1]} "
              f"(loads at 14.87, stalls at 15.05)")
        for w_, (x, z) in wins:
            if w_ > WINDOW_MAX:
                FAIL.append(f"collision around {x}_{z} is {w_ / 1e6:.2f} MB over its 3x3 tiles, above the "
                            f"{WINDOW_MAX / 1e6:.2f} MB that loaded")
            elif w_ > WINDOW_WARN:
                WARN.append(f"collision around {x}_{z} is {w_ / 1e6:.2f} MB over its 3x3 tiles, close to the "
                            f"{WINDOW_MAX / 1e6:.2f} MB limit")
    print("  largest tile streams:")
    for s, n in tiles[:6]:
        print(f"    {s / 1e6:6.2f} MB  {n}")
    for s, n in tiles:
        sim = n.startswith('cSim')
        mx, wn = (SIM_MAX, SIM_WARN) if sim else (PRES_MAX, PRES_WARN)
        if s > mx:
            FAIL.append(f"{n} is {s / 1e6:.2f} MB, over the {mx / 1e6:.1f} MB "
                        f"{'collision' if sim else 'render'} ceiling")
        elif s > wn:
            WARN.append(f"{n} is {s / 1e6:.2f} MB: above the largest "
                        f"{'collision stream proven to load (2.92 MB)' if sim else '7 MB render warning (Summer Camp loaded 6.3 MB tiles; Ovanny 9.6-10.2)'}")


def main():
    if not glob.glob(os.path.join(GLB, 'block_*.glb')):
        raise SystemExit(f"no GLBs in {GLB}")
    # Streamed twice rather than held: the island is 2.7 M triangles and this box has
    # 4 GB of swap already pinned.
    check_textures()
    check_uv()
    check_collision()
    check_grinds()
    check_invisible()
    check_tile_textures()
    check_streams()
    hdr(f"FAILURES ({len(FAIL)})")
    for x in FAIL:
        print('  X ' + x)
    hdr(f"WARNINGS ({len(WARN)})")
    for x in WARN:
        print('  ! ' + x)
    print('=' * 78)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
