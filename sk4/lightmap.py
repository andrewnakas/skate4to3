"""Baked lighting for the LIGHTMAP channel: sun shadows, sky occlusion and slope shading.

Until 2026-09-30 every surface carried one flat grey swatch as its lightmap (see LIGHTMAP_GREY in
export_glb.py), sampled through the tiling UV0: no shadow under a deck, a tree or a building, no
shading on a hillside. Stock Skate 3 arenas carry their direct sun IN the lightmap, so a converted
map without one reads as flat however sharp its textures are. the arena builder packs a real second UV
set for it (MeshVertexFlattener: TEX1 = the texCoord of Emissive, default TEXCOORD_1, clamped to
0..1; GlbTextureAutoBuilder: lightmap = Emissive, single mip).

One image per render tile:

    rows 0 .. MAP_ROWS-1   top-down map of the tile's bounding box: light on the HIGHEST drawn
                           surface at each spot (sun x shadow x slope + sky x occlusion)
    rows MAP_ROWS .. end   a ramp of LEVELS constant shades

A triangle that faces up and is the highest thing at its spot takes planar UVs into the map. Any
other triangle (walls, the floor under a roof, undersides) cannot be told apart from what is above
it in a top-down map, so it takes ONE shade from the ramp: its own facing to the sun, shadowed by a
march over the height field from its own height. Vertices are split only where a vertex is shared
by triangles that need different UVs.

The height field is the max height of everything drawn (not water, decals or the collision copy)
on a CELL grid over the whole map, so shadows cross tile boundaries.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

CELL = 0.25            # m, height field and lighting grid
PX = 512               # lightmap edge (single mip, BC1: 131 KB per tile)
STRIP = 16             # rows of constant shades at the bottom
MAP_ROWS = PX - STRIP
LEVELS = 16
SHADOW_REACH = 150.0   # m a shadow is followed
AMBIENT = np.array([0.40, 0.43, 0.48], np.float32)   # open ground in full shadow
SUN = np.array([0.48, 0.44, 0.36], np.float32)       # added on flat ground in full sun
COVER_GAP = 0.5        # m of something drawn above a point before it counts as covered
LEVEL_LO, LEVEL_HI = 0.20, 0.92


def _sun_vec(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])


def _march_steps(reach):
    out, t = [], CELL
    while t < reach:
        out.append(t)
        t += CELL if t < 10 else (0.5 if t < 40 else 1.0)
    return out


class LightBake:
    def __init__(self, blocks, skip_classes=(), sun_az=135.0, sun_el=50.0, log=print):
        self.sun = _sun_vec(sun_az, sun_el)
        lo = np.array([np.inf, np.inf])
        hi = -lo
        n_parts = 0
        for classes in blocks.values():
            for key, parts in classes.items():
                if key[0] in skip_classes:
                    continue
                for W, _uv, T in parts:
                    if len(T):
                        lo = np.minimum(lo, W[:, [0, 2]].min(axis=0))
                        hi = np.maximum(hi, W[:, [0, 2]].max(axis=0))
                        n_parts += 1
        self.x0, self.z0 = float(lo[0]) - 2.0, float(lo[1]) - 2.0
        self.nx = int(np.ceil((hi[0] + 2.0 - self.x0) / CELL)) + 1
        self.nz = int(np.ceil((hi[1] + 2.0 - self.z0) / CELL)) + 1
        H = np.full(self.nz * self.nx, -np.inf, np.float32)
        rng = np.random.default_rng(7)
        samples = 0
        for classes in blocks.values():
            for key, parts in classes.items():
                if key[0] in skip_classes:
                    continue
                for W, _uv, T in parts:
                    if not len(T):
                        continue
                    W = np.asarray(W, np.float32)
                    a, b, c = W[T[:, 0]], W[T[:, 1]], W[T[:, 2]]
                    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
                    ns = np.clip(np.ceil(ar / (CELL * CELL / 3.0)).astype(np.int64), 1, 200000)
                    rep = np.repeat(np.arange(len(ar)), ns)
                    u = rng.random(len(rep), dtype=np.float32)
                    v = rng.random(len(rep), dtype=np.float32)
                    f = u + v > 1
                    u[f], v[f] = 1 - u[f], 1 - v[f]
                    q = a[rep] + (b[rep] - a[rep]) * u[:, None] + (c[rep] - a[rep]) * v[:, None]
                    q = np.concatenate([q, W])
                    np.maximum.at(H, self._cell(q[:, 0], q[:, 2]), q[:, 1])
                    samples += len(q)
        H = H.reshape(self.nz, self.nx)
        hole = ~np.isfinite(H)
        # a cell the random samples missed takes the highest neighbour, not a pit
        from scipy.ndimage import maximum_filter
        left = hole
        for _ in range(4):
            if not left.any():
                break
            H[left] = maximum_filter(H, 3)[left]
            left = ~np.isfinite(H)
        H[left] = H[~left].min() if (~left).any() else 0.0
        # the samples are random, so a surface has single-cell pits where every sample missed and
        # the cell kept whatever lies below: close them
        from scipy.ndimage import minimum_filter
        H = minimum_filter(maximum_filter(H, 3), 3)
        self.H = H
        log(f"lightmap: height field {self.nx:,} x {self.nz:,} cells of {CELL} m from {n_parts:,} part(s), "
            f"{samples / 1e6:.0f}M samples, {100 * hole.mean():.0f}% empty")
        self._light(log)

    def _cell(self, x, z):
        ix = np.clip(((x - self.x0) / CELL).astype(np.int64), 0, self.nx - 1)
        iz = np.clip(((z - self.z0) / CELL).astype(np.int64), 0, self.nz - 1)
        return iz * self.nx + ix

    @staticmethod
    def _pair(n, o):
        """Slices (dst, src) so that src index = dst index + o, both inside 0..n."""
        return slice(max(0, -o), n - max(0, o)), slice(max(0, o), n - max(0, -o))

    def _light(self, log):
        H = self.H
        nz, nx = H.shape
        lit = np.zeros(H.shape, np.float32)
        jitter = [(0.0, 0.0), (1.6, 0.8), (-1.6, -0.8)]
        az0 = np.degrees(np.arctan2(self.sun[0], self.sun[2]))
        el0 = np.degrees(np.arcsin(self.sun[1]))
        steps = _march_steps(SHADOW_REACH)
        for daz, del_ in jitter:
            s = _sun_vec(az0 + daz, el0 + del_)
            d = s[[0, 2]] / np.hypot(s[0], s[2])
            tan = s[1] / np.hypot(s[0], s[2])
            occ = np.zeros(H.shape, bool)
            seen = set()
            for t in steps:
                oi, oj = int(round(d[1] * t / CELL)), int(round(d[0] * t / CELL))
                if (oi, oj) in seen or (oi == 0 and oj == 0) or abs(oi) >= nz or abs(oj) >= nx:
                    continue
                seen.add((oi, oj))
                di, si = self._pair(nz, oi)
                dj, sj = self._pair(nx, oj)
                occ[di, dj] |= H[si, sj] > H[di, dj] + (t * tan + 0.08)
            lit += ~occ
        lit /= len(jitter)
        # sky occlusion: horizon angle in 8 directions within 4 m
        ao = np.zeros(H.shape, np.float32)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for a, b in dirs:
            ln = np.hypot(a, b)
            top = np.zeros(H.shape, np.float32)
            for k in (1, 2, 4, 8, 16):
                if k >= min(nz, nx):
                    continue
                di, si = self._pair(nz, a * k)
                dj, sj = self._pair(nx, b * k)
                r = (H[si, sj] - H[di, dj]) / (k * ln * CELL)
                np.maximum(top[di, dj], r, out=top[di, dj])
            ao += top / np.sqrt(1 + top * top)       # sin of the horizon angle
        ao = 1.0 - ao / len(dirs)
        # slope shading from the height field, with cliffs and wall edges clipped
        from scipy.ndimage import gaussian_filter
        Hs = gaussian_filter(H, 1.0)
        gz, gx = np.gradient(Hs, CELL)
        np.clip(gx, -1.0, 1.0, out=gx)
        np.clip(gz, -1.0, 1.0, out=gz)
        s = self.sun
        nd = (s[1] - gx * s[0] - gz * s[2]) / np.sqrt(1 + gx * gx + gz * gz) / s[1]
        del gx, gz, Hs
        np.clip(nd, 0.0, 1.25, out=nd)
        sunf = gaussian_filter(lit, 0.8) * (0.3 + 0.7 * nd)
        amb = 0.5 + 0.5 * gaussian_filter(ao, 0.7)
        self.sunf = sunf.astype(np.float32)
        self.amb = amb.astype(np.float32)
        log(f"lightmap: {100 * (lit > 0.5).mean():.0f}% of the map in sun, mean sky occlusion "
            f"{float(ao.mean()):.2f}, sun from azimuth {az0:.0f} at {el0:.0f} deg")

    def preview(self, path, step=4):
        v = AMBIENT.mean() * self.amb[::step, ::step] + SUN.mean() * self.sunf[::step, ::step]
        Image.fromarray(np.clip(v * 255, 0, 255).astype(np.uint8)).save(path)

    # ---- per tile -------------------------------------------------------------------------
    def tile_frame(self, lo, hi):
        """Bounding box (x, z) of one tile's drawn geometry -> the frame its planar UVs use."""
        lo = np.asarray(lo, np.float64) - CELL
        hi = np.asarray(hi, np.float64) + CELL
        return (lo[0], lo[1], (hi[0] - lo[0]) / (PX - 1), (hi[1] - lo[1]) / (MAP_ROWS - 1))

    def tile_png(self, frame):
        from scipy.ndimage import map_coordinates
        x0, z0, sx, sz = frame
        gx = (x0 + np.arange(PX) * sx - self.x0) / CELL - 0.5
        gz = (z0 + np.arange(MAP_ROWS) * sz - self.z0) / CELL - 0.5
        zz, xx = np.meshgrid(gz, gx, indexing='ij')
        co = np.stack([zz, xx])
        sunf = map_coordinates(self.sunf, co, order=1, mode='nearest')
        amb = map_coordinates(self.amb, co, order=1, mode='nearest')
        img = np.empty((PX, PX, 3), np.float32)
        img[:MAP_ROWS] = amb[:, :, None] * AMBIENT + sunf[:, :, None] * SUN
        lv = LEVEL_LO + np.clip((np.arange(PX) - 0.5) / (PX - 3), 0, 1) * (LEVEL_HI - LEVEL_LO)
        f = ((lv - LEVEL_LO) / (LEVEL_HI - LEVEL_LO))[:, None]
        tint = (1 - f) * (AMBIENT / AMBIENT.mean()) + f * ((AMBIENT + SUN) / (AMBIENT + SUN).mean())
        img[MAP_ROWS:] = (lv[:, None] * tint)[None, :, :]
        buf = io.BytesIO()
        Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), 'RGB').save(buf, format='PNG')
        return buf.getvalue()

    def _h(self, x, z):
        return self.H.ravel()[self._cell(x, z)]

    def map_prim(self, P, T, frame, N=None):
        """-> (vidx, T2, uv1, planar triangles): new vertex i is old vertex vidx[i].

        Shades are per VERTEX (its smooth normal against the sun, a shadow march from its own
        height) and the ramp is continuous, so a vertex is only split where it is used both by a
        planar triangle and by a shaded one. One shade per triangle split nearly every vertex of
        a rock (+25% vertices on Summer Camp, 1.4 MB on the heaviest tile).
        """
        P = np.asarray(P, np.float64)
        T = np.asarray(T, np.int64)
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        cen = (a + b + c) / 3.0
        clear = np.zeros(len(T), np.int32)
        for q in (cen, 0.8 * a + 0.2 * cen, 0.8 * b + 0.2 * cen, 0.8 * c + 0.2 * cen):
            clear += self._h(q[:, 0], q[:, 2]) <= q[:, 1] + COVER_GAP
        # A face that points DOWN yet is the highest thing at its spot is an up-facing surface with
        # reversed winding (the game draws both sides): light it as the top surface it is.
        planar = (np.abs(n[:, 1]) > 0.5) & (clear >= 3)
        key = T * 2 + (~planar)[:, None]
        uniq, inv = np.unique(key.ravel(), return_inverse=True)
        vidx = uniq // 2
        sh = (uniq % 2) == 1
        x0, z0, sx, sz = frame
        uv = np.empty((len(uniq), 2), np.float32)
        Pn = P[vidx]
        uv[:, 0] = (np.clip((Pn[:, 0] - x0) / sx, 0, PX - 1) + 0.5) / PX
        uv[:, 1] = (np.clip((Pn[:, 2] - z0) / sz, 0, MAP_ROWS - 1) + 0.5) / PX
        if sh.any():
            vs = vidx[sh]
            if N is None:
                vn = np.zeros((len(P), 3))
                for k in range(3):
                    np.add.at(vn, T[~planar][:, k], n[~planar])
                nr = vn[vs]
            else:
                nr = np.asarray(N, np.float64)[vs]
            nr = nr / np.maximum(np.linalg.norm(nr, axis=1), 1e-12)[:, None]
            s = self.sun
            d = s[[0, 2]] / np.hypot(s[0], s[2])
            tan = s[1] / np.hypot(s[0], s[2])
            nd = np.clip(nr @ s, 0.0, None) / s[1]
            start = P[vs] + nr * 0.3
            start[:, 1] += 0.1
            blocked = np.zeros(len(vs), bool)
            for t in [0.5, 1, 1.5, 2, 3, 4, 5, 6.5, 8, 10, 12, 15, 18, 22, 26, 30, 36, 42, 50, 60, 72, 86, 100]:
                blocked |= self._h(start[:, 0] + d[0] * t, start[:, 2] + d[1] * t) > start[:, 1] + t * tan + 0.1
            covered = self._h(start[:, 0], start[:, 2]) > start[:, 1] + COVER_GAP + 0.3
            # Ground under a rock, a tree or an overlapping shell is "covered" too, and it is seen:
            # 0.6 x ambient there (0.26) read as black textures in game (2026-09-30). Shade, not night.
            val = AMBIENT.mean() * np.where(nr[:, 1] < -0.5, 0.75, 0.9) + SUN.mean() * nd * (~blocked)
            val = np.where((nr[:, 1] > 0.5) & covered, AMBIENT.mean() * 0.85, val)
            f = np.clip((val - LEVEL_LO) / (LEVEL_HI - LEVEL_LO), 0.0, 1.0)
            uv[sh, 0] = (1.0 + f * (PX - 3)) / PX
            uv[sh, 1] = (MAP_ROWS + STRIP / 2.0) / PX
        return vidx, inv.reshape(-1, 3), uv, int(planar.sum())
