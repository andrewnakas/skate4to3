"""Render the staged GLBs from an eye-level camera, sampling the real textures.

The mean-colour map (tools/topdown_map.py) shows WHICH texture a surface got; it cannot show
whether the UVs put it on correctly. Booting the game shows that, but the pack's spawn is
pinned to the startline marker, so each viewpoint costs a rebuild. This renders any camera in
seconds, so the whole map's texture MAPPING can be audited offline.

Not a lighting model: flat texture sampling with a little normal-based shading, which is what
makes a wrong UV scale or a wrong texture obvious.

    python tools/preview_render.py out.png --eye X,Y,Z --look X,Y,Z [--fov 70] [--glb DIR]
"""
from __future__ import annotations
import argparse, glob, io, json, os, struct, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D2S3 = os.environ.get('D2S3', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'decendersmaptoskate3recomp'))
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


def acc(js, bn, i):
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


def load(glb_dir):
    """-> list of (positions, uvs|None, triangles, texture array|None, material name)"""
    parts = []
    for f in sorted(glob.glob(os.path.join(glb_dir, 'block_*.glb'))):
        js, bn = read_glb(f)
        tex = {}
        for mi, m in enumerate(js['materials']):
            t = m.get('pbrMetallicRoughness', {}).get('baseColorTexture')
            if not t:
                continue
            src = js['textures'][t['index']]['source']
            im = js['images'][src]
            bv = js['bufferViews'][im['bufferView']]
            o = bv.get('byteOffset', 0)
            try:
                with Image.open(io.BytesIO(bn[o:o + bv['byteLength']])) as pim:
                    tex[mi] = np.asarray(pim.convert('RGB'), dtype=np.uint8)
            except Exception:  # noqa: BLE001
                pass
        names = [m.get('name', '') for m in js['materials']]
        for mesh in js['meshes']:
            for pr in mesh['primitives']:
                mi = pr.get('material')
                if mi is None or 'Collision' in names[mi]:
                    continue
                P = acc(js, bn, pr['attributes']['POSITION']).astype(np.float32)
                ai = pr['attributes'].get('TEXCOORD_0')
                UV = acc(js, bn, ai).astype(np.float32) if ai is not None else None
                T = acc(js, bn, pr['indices']).reshape(-1, 3).astype(np.int64)
                parts.append((P, UV, T, tex.get(mi), names[mi]))
    return parts


def render(parts, eye, look, fov, W, H, rng):
    fwd = look - eye
    fwd /= np.linalg.norm(fwd) or 1.0
    up0 = np.array([0, 1, 0], np.float32)
    right = np.cross(fwd, up0)
    right /= np.linalg.norm(right) or 1.0
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd])              # world -> camera
    f = (W / 2) / np.tan(np.radians(fov) / 2)
    img = np.zeros((H, W, 3), np.float32)
    zb = np.full((H, W), np.inf, np.float32)
    idb = np.full((H, W), -1, np.int32)
    sky = np.array([150, 180, 215], np.float32)
    img[:] = sky
    for pi, (P, UV, T, tex, name) in enumerate(parts):
        C = (P - eye) @ R.T
        z = -C[:, 2]
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        keep = (z[a] > 0.3) & (z[b] > 0.3) & (z[c] > 0.3)
        if not keep.any():
            continue
        T2 = T[keep]
        a, b, c = T2[:, 0], T2[:, 1], T2[:, 2]
        sx = f * C[:, 0] / z + W / 2
        sy = -f * C[:, 1] / z + H / 2
        ax, ay, bx, by, cx, cy = sx[a], sy[a], sx[b], sy[b], sx[c], sy[c]
        minx = np.minimum(np.minimum(ax, bx), cx)
        maxx = np.maximum(np.maximum(ax, bx), cx)
        miny = np.minimum(np.minimum(ay, by), cy)
        maxy = np.maximum(np.maximum(ay, by), cy)
        vis = (maxx > 0) & (minx < W) & (maxy > 0) & (miny < H)
        if not vis.any():
            continue
        T2 = T2[vis]
        a, b, c = T2[:, 0], T2[:, 1], T2[:, 2]
        ax, ay, bx, by, cx, cy = sx[a], sy[a], sx[b], sy[b], sx[c], sy[c]
        area = 0.5 * np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        k = np.clip(np.ceil(area * 2.0).astype(np.int64), 1, 20000)
        if k.sum() > 40_000_000:
            continue
        idx = np.repeat(np.arange(len(T2)), k)
        u = rng.random(len(idx), dtype=np.float32)
        v = rng.random(len(idx), dtype=np.float32)
        fl = u + v > 1.0
        u[fl] = 1.0 - u[fl]
        v[fl] = 1.0 - v[fl]
        w = 1.0 - u - v
        X = (ax[idx] * w + bx[idx] * u + cx[idx] * v).astype(np.int32)
        Y = (ay[idx] * w + by[idx] * u + cy[idx] * v).astype(np.int32)
        Z = (z[a][idx] * w + z[b][idx] * u + z[c][idx] * v)
        ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        X, Y, Z, idx2, u2, v2, w2 = X[ok], Y[ok], Z[ok], idx[ok], u[ok], v[ok], w[ok]
        if not len(X):
            continue
        o = np.argsort(-Z)                        # far first, near overwrites
        X, Y, Z, idx2, u2, v2, w2 = X[o], Y[o], Z[o], idx2[o], u2[o], v2[o], w2[o]
        sel = Z < zb[Y, X]
        if not sel.any():
            continue
        X, Y, Z, idx2, u2, v2, w2 = (X[sel], Y[sel], Z[sel], idx2[sel],
                                     u2[sel], v2[sel], w2[sel])
        if tex is not None and UV is not None:
            th, tw = tex.shape[:2]
            au, bu, cu = UV[a][idx2], UV[b][idx2], UV[c][idx2]
            uu = au[:, 0] * w2 + bu[:, 0] * u2 + cu[:, 0] * v2
            vv = au[:, 1] * w2 + bu[:, 1] * u2 + cu[:, 1] * v2
            tx = np.mod((uu * tw).astype(np.int64), tw)
            ty = np.mod((vv * th).astype(np.int64), th)
            col = tex[ty, tx].astype(np.float32)
        else:
            col = np.full((len(X), 3), 160.0, np.float32)
        # cheap directional shading so shape reads
        n = np.cross(P[b] - P[a], P[c] - P[a])
        ln = np.linalg.norm(n, axis=1)
        ln[ln == 0] = 1
        nl = np.abs((n[:, 1] / ln))[idx2]
        col *= (0.55 + 0.45 * nl)[:, None]
        zb[Y, X] = Z
        img[Y, X] = col
        idb[Y, X] = pi
    return np.clip(img, 0, 255).astype(np.uint8), idb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--eye', required=True)
    ap.add_argument('--look', required=True)
    ap.add_argument('--fov', type=float, default=70.0)
    ap.add_argument('--size', default='1280x720')
    ap.add_argument('--glb', default=os.path.join(D2S3, 'work', 'IsleOfGrom', 'glb'))
    args = ap.parse_args()
    W, H = (int(x) for x in args.size.split('x'))
    eye = np.array([float(x) for x in args.eye.split(',')], np.float32)
    look = np.array([float(x) for x in args.look.split(',')], np.float32)
    parts = load(args.glb)
    print(f"{len(parts)} primitive(s)")
    out, idb = render(parts, eye, look, args.fov, W, H, np.random.default_rng(0))
    import collections
    px = collections.Counter(idb[idb >= 0].ravel().tolist())
    tot = sum(px.values())
    print(f"  filled {tot:,} of {W * H:,} pixels; materials by screen area:")
    seen = collections.Counter()
    for pi, n in px.items():
        seen[parts[pi][4]] += n
    for nm, n in seen.most_common(12):
        print(f"    {100 * n / max(tot, 1):5.1f}%  {nm[:74]}")
    Image.fromarray(out).save(args.out)
    print("wrote", args.out)


if __name__ == '__main__':
    main()
