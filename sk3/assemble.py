"""Exported GLB blocks -> Skate 3 tile arenas (render, texture, collision + grind splines).

    python -m sk3.assemble <glb dir> <out dir> [--tile 100]

Input is what sk4/export_glb.py writes: block_*.glb (one mesh, one primitive per material; colour in
baseColorTexture, specular in metallicRoughnessTexture, normal in normalTexture, lightmap in
emissiveTexture on TEXCOORD_1), blenrose_materials.json (per material: exclude_collision /
exclude_pres, shader name, collision surface), splines.json (grind polylines).

Output: <out>/tiles.json and, per 100 m tile `<cx>_<cz>` (tile centres, as retail names them):
    cPres_<cx>_<cz>.arena     render arena (sk3/render_write)
    cSim_<cx>_<cz>.arena      collision arena (sk3/collision_write) with the tile's splines
    tex/<name_id>.arena       one texture arena per distinct image (sk3/texture_write)
Which textures go to cPres_Global and which stream per tile is the packer's decision (tiles.json
lists, per tile, the texture ids it uses).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import struct

import numpy as np
from PIL import Image

from . import collision_write as cw
from . import render_write as rwr
from . import splines as sp
from . import texture_write as tw

COMPONENTS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
DEFAULT_SURFACE = 0xB700          # most common retail surface id (meaning not yet measured)
MAX_PART_VERTS = 8000            # retail: largest vertex block 349 KB (~12k verts at 28 B)
ARENA_MAX_MESHES = 48            # retail cPres arenas: <= 60 meshes (p99 57)
ARENA_MAX_BYTES = 750_000        # retail cPres arenas: <= 921 KB (p99 706 KB); one arena hung the loader at 7 MB


def h64(*parts) -> int:
    """stable 62-bit id (bit 62 is the texture low-res flag, bit 63 kept clear)"""
    d = hashlib.blake2b('|'.join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(d, 'big') & 0x3FFFFFFFFFFFFFFF


def h32(*parts) -> int:
    d = hashlib.blake2b('|'.join(map(str, parts)).encode(), digest_size=4).digest()
    return int.from_bytes(d, 'big') or 1


# ---------------------------------------------------------------------------------------------- GLB
class Glb:
    def __init__(self, path):
        d = open(path, 'rb').read()
        n = struct.unpack_from('<I', d, 12)[0]
        self.j = json.loads(d[20:20 + n])
        p = 20 + n
        bl = struct.unpack_from('<I', d, p)[0]
        self.bin = d[p + 8:p + 8 + bl]

    def accessor(self, i):
        a = self.j['accessors'][i]
        v = self.j['bufferViews'][a['bufferView']]
        dt = DTYPES[a['componentType']]
        k = COMPONENTS[a['type']]
        off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        arr = np.frombuffer(self.bin, dt, a['count'] * k, off)
        return arr.reshape(a['count'], k) if k > 1 else arr

    def image(self, i):
        im = self.j['images'][i]
        v = self.j['bufferViews'][im['bufferView']]
        o = v.get('byteOffset', 0)
        return self.bin[o:o + v['byteLength']]

    def texture_image(self, ref):
        if not ref:
            return None
        return self.image(self.j['textures'][ref['index']]['source'])

    def prims(self):
        for mesh in self.j['meshes']:
            for pr in mesh['primitives']:
                at = pr['attributes']
                m = self.j['materials'][pr['material']]
                yield {
                    'material': m['name'],
                    'P': self.accessor(at['POSITION']).astype(np.float64),
                    'N': self.accessor(at['NORMAL']).astype(np.float64) if 'NORMAL' in at else None,
                    'UV0': self.accessor(at['TEXCOORD_0']).astype(np.float64) if 'TEXCOORD_0' in at else None,
                    'UV1': self.accessor(at['TEXCOORD_1']).astype(np.float64) if 'TEXCOORD_1' in at else None,
                    'I': self.accessor(pr['indices']).astype(np.int64).reshape(-1, 3),
                    'images': {
                        'diffuse': self.texture_image(m.get('pbrMetallicRoughness', {}).get('baseColorTexture')),
                        'specular': self.texture_image(m.get('pbrMetallicRoughness', {}).get('metallicRoughnessTexture')),
                        'normal': self.texture_image(m.get('normalTexture')),
                        'lightmap': self.texture_image(m.get('emissiveTexture')),
                    },
                }


# ---------------------------------------------------------------------------------------- textures
def pow2_rgba(png: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(png)).convert('RGBA')
    w, h = im.size
    W, H = 1 << max(2, (w - 1).bit_length()), 1 << max(2, (h - 1).bit_length())
    W, H = min(W, 2048), min(H, 2048)
    if (W, H) != (w, h):
        im = im.resize((W, H), Image.LANCZOS)
    return np.asarray(im, np.uint8)


# default flat images for slots a material leaves empty (our own pixels, not game data)
def flat(rgba, size=16):
    return np.full((size, size, 4), rgba, np.uint8)


DEFAULTS = {'normal': flat((128, 128, 255, 255)), 'specular': flat((10, 10, 10, 255)),
            'lightmap': flat((255, 255, 255, 255)), 'diffuse': flat((128, 128, 128, 255))}


class TextureSet:
    def __init__(self, out_dir):
        self.dir = os.path.join(out_dir, 'tex')
        os.makedirs(self.dir, exist_ok=True)
        self.ids = {}            # content digest -> name id
        self.meta = {}           # name id -> {w, h, fmt bytes}

    def add(self, slot, png=None):
        if png is None:
            rgba = DEFAULTS[slot]
            key = 'default:' + slot
        else:
            key = hashlib.blake2b(png, digest_size=16).hexdigest()
        if key in self.ids:
            return self.ids[key]
        if png is not None:
            rgba = pow2_rgba(png)
        nid = h64('tex', key)
        alpha = slot == 'diffuse' and bool((rgba[..., 3] < 250).any())
        spec = tw.from_image(rgba, nid, h32('texarena', key), alpha=alpha)
        data = tw.build(spec)
        open(os.path.join(self.dir, f'{nid:016x}.arena'), 'wb').write(data)
        Image.fromarray(rgba, 'RGBA').save(os.path.join(self.dir, f'{nid:016x}.png'))
        self.ids[key] = nid
        self.meta[nid] = {'bytes': len(data), 'w': int(rgba.shape[1]), 'h': int(rgba.shape[0]), 'slot': slot,
                          'alpha': alpha, 'default': png is None}
        return nid


# ------------------------------------------------------------------------------------------- tiling
def tile_key(x, z, size):
    cx = np.floor(np.asarray(x) / size) * size + size / 2
    cz = np.floor(np.asarray(z) / size) * size + size / 2
    return cx.astype(np.int64), cz.astype(np.int64)


def split_by_tile(P, I, size):
    c = P[I].mean(1)
    kx, kz = tile_key(c[:, 0], c[:, 2], size)
    out = {}
    for key in set(zip(kx.tolist(), kz.tolist())):
        sel = (kx == key[0]) & (kz == key[1])
        out[key] = I[sel]
    return out


def compact(P, I, *attrs):
    used, inv = np.unique(I.reshape(-1), return_inverse=True)
    return (P[used], inv.reshape(-1, 3)) + tuple(None if a is None else a[used] for a in attrs)


def clip_polyline(points, size):
    """split a polyline into runs whose segments' midpoints share one tile -> {tile: [runs]}"""
    pts = np.asarray(points, np.float64)
    out = collections.defaultdict(list)
    run, cur = [pts[0]], None
    for a, b in zip(pts[:-1], pts[1:]):
        m = (a + b) / 2
        key = tuple(int(v) for v in tile_key(m[0], m[2], size))
        if cur is not None and key != cur:
            out[cur].append(run)
            run = [a]
        cur = key
        run.append(b)
    if cur is not None and len(run) > 1:
        out[cur].append(run)
    return out


# --------------------------------------------------------------------------------------------- main
def assemble(glb_dir, out_dir, size=100.0):
    os.makedirs(out_dir, exist_ok=True)
    mats = json.load(open(os.path.join(glb_dir, 'blenrose_materials.json')))
    tex = TextureSet(out_dir)
    render = collections.defaultdict(list)      # tile -> parts
    coll = collections.defaultdict(list)        # tile -> (V, T, surface)
    tile_tex = collections.defaultdict(set)
    blocks = sorted(f for f in os.listdir(glb_dir) if f.startswith('block_') and f.endswith('.glb'))
    for bf in blocks:
        g = Glb(os.path.join(glb_dir, bf))
        for pr in g.prims():
            m = mats.get(pr['material'], {})
            pres = not m.get('exclude_pres', False)
            collide = not m.get('exclude_collision', False)
            surf = int(m.get('collision', {}).get('physics_surface', '0') or 0) or DEFAULT_SURFACE
            texids = None
            if pres:
                texids = {s: tex.add(s, pr['images'].get(s)) for s in ('diffuse', 'specular', 'normal', 'lightmap')}
            for key, I in split_by_tile(pr['P'], pr['I'], size).items():
                P, T, N, UV0, UV1 = compact(pr['P'], I, pr['N'], pr['UV0'], pr['UV1'])
                if collide:
                    coll[key].append((P, T, surf))
                if not pres:
                    continue
                tile_tex[key].update(texids.values())
                step = max(1, int(len(T) * MAX_PART_VERTS / max(len(P), 1)))
                s = 0
                while s < len(T):                            # parts under MAX_PART_VERTS vertices
                    P2, T2, N2, U0, U1 = compact(P, T[s:s + step], N, UV0, UV1)
                    if len(P2) > MAX_PART_VERTS and step > 1:
                        step //= 2
                        continue
                    s += step
                    render[key].append({
                        'positions': P2, 'indices': T2,
                        'normals': N2 if N2 is not None else np.tile([0.0, 1.0, 0.0], (len(P2), 1)),
                        'uv0': U0 if U0 is not None else np.zeros((len(P2), 2)),
                        'uv1': U1 if U1 is not None else np.full((len(P2), 2), 0.5),
                        'material': {'name': pr['material'][:60],
                                     'shader': 'environmentsimple.default',
                                     'textures': texids}})
        print(f'  {bf}: tiles so far {len(set(render) | set(coll))}, textures {len(tex.ids)}', flush=True)

    spl = collections.defaultdict(list)
    sj = os.path.join(glb_dir, 'splines.json')
    if os.path.exists(sj):
        for s in json.load(open(sj)).get('splines', []):
            for key, runs in clip_polyline(s['points'], size).items():
                spl[key].extend(runs)

    tiles = {}
    for key in sorted(set(render) | set(coll)):
        name = f'{key[0]}_{key[1]}'
        rec = {'tile': list(key), 'textures': sorted(f'{t:016x}' for t in tile_tex[key])}
        if render[key]:
            groups, cur, est = [], [], 0
            for p in render[key]:
                b = len(p['positions']) * 28 + len(p['indices']) * 6 + 600
                if cur and (len(cur) >= ARENA_MAX_MESHES or est + b > ARENA_MAX_BYTES):
                    groups.append(cur); cur, est = [], 0
                cur.append(p); est += b
            groups.append(cur)
            rec['pres_arenas'], rec['pres_bytes'] = [], 0
            for gi, g in enumerate(groups):
                data = rwr.build_render_arena(g, h32('cPres', name, gi), name=f'cPres_{name}_{gi}')
                fn = f'cPres_{name}_{gi:02d}.arena'
                open(os.path.join(out_dir, fn), 'wb').write(data)
                rec['pres_arenas'].append(fn)
                rec['pres_bytes'] += len(data)
            rec['pres_tris'] = int(sum(len(p['indices']) for p in render[key]))
        if coll[key]:
            Vs, Ts, Ss, base = [], [], [], 0
            for P, T, s in coll[key]:
                Vs.append(P); Ts.append(T + base); Ss.append(np.full(len(T), s)); base += len(P)
            mesh, _rep = cw.build_mesh(np.concatenate(Vs), np.concatenate(Ts), np.concatenate(Ss))
            ss = None
            if spl[key]:
                ss = sp.SplineSet(0, [sp.Spline(h64('spline', name, i), 0x2C70170700000000,
                                                [sp.straight_segment(a, b) for a, b in zip(r[:-1], r[1:])])
                                      for i, r in enumerate(spl[key])])
            data, _spec = cw.build_tile_arena(mesh, h32('cSim', name), f'cSim_{name}', h64('cSimInst', name),
                                              spline_set=ss, splines_id=h64('splines', name))
            open(os.path.join(out_dir, f'cSim_{name}.arena'), 'wb').write(data)
            rec['sim_bytes'] = len(data)
            rec['sim_tris'] = int(sum(len(t) for t in Ts))
            rec['splines'] = len(spl[key])
        tiles[name] = rec
    json.dump({'tile_size': size, 'tiles': tiles,
               'textures': {f'{k:016x}': v for k, v in tex.meta.items()}},
              open(os.path.join(out_dir, 'tiles.json'), 'w'), indent=1)
    print(f'  tiles {len(tiles)}, render parts {sum(len(v) for v in render.values())}, '
          f'textures {len(tex.ids)} ({sum(v["bytes"] for v in tex.meta.values()) / 1e6:.1f} MB), '
          f'splines {sum(len(v) for v in spl.values())}')
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glb_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--tile', type=float, default=100.0)
    a = ap.parse_args()
    assemble(a.glb_dir, a.out_dir, a.tile)


if __name__ == '__main__':
    main()
