"""Skate 3 render arenas (cPres_<x>_<z>_high.xsf, cPres_Global.xsf): decode meshes, materials and
their texture references from a retail RW4 arena. Facts and how they were measured:
specs/sk3_render.md. Encoders (byte exact) live in sk3/render_write.py.

Object types in a render arena (all big-endian, Xbox 360):
    0x10031   base resource: vertex bytes, 16-bit indices, or texture texels
    0x200e9   vertex declaration           0x200ea  vertex buffer header (GPU fetch constant)
    0x200eb   index buffer header          0x20081  buffer binding {1, 1, index buffer, vertex buffer}
    0xeb0023  mesh: bbox, material ref, decl/binding/ib/vb, primitive list
    0xeb0001  model: bbox, matrix, mesh list, name, per-primitive bounds + vec4
    0xeb000d  model instance: hash, matrix, bbox, guid, model, name strings
    0xeb0005  material table: materials = lists of named params (texture guids, floats, shader name)
    0xeb000b  export table: guid -> (type, object or subref), plus a per-type index
    0xeb0008  arena version words (0x19, 0x0d)
    0xeb0024  light probe records (n x 160 B of floats)
    0x200e8   texture header (owned by the texture code; kept raw here)

A "ref" word is either a dictionary index or 0x00800000 | subref index (the subref record gives
(object index, byte offset)).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from . import rw4

T_BASE, T_TEX, T_DECL, T_VB, T_IB, T_BIND = 0x10031, 0x200e8, 0x200e9, 0x200ea, 0x200eb, 0x20081
T_MODEL, T_MATS, T_VER, T_EXPORTS, T_INST, T_MESH, T_PROBES = (
    0xeb0001, 0xeb0005, 0xeb0008, 0xeb000b, 0xeb000d, 0xeb0023, 0xeb0024)
T_QUADS, T_POLYS, T_GUIDS = 0xeb0016, 0xeb0025, 0xeb0028     # big worlds only
SUBREF = 0x00800000

# vertex element formats seen (D3DDECLTYPE words of the Xbox 360 declaration) -> (numpy dtype, n)
FMT = {
    0x002A23B9: ('>f4', 3),     # FLOAT3
    0x002C235F: ('>f2', 2),     # FLOAT16_2
    0x001A2360: ('>f2', 4),     # FLOAT16_4
    0x001A215A: ('>i2', 4),     # SHORT4N
    0x002C2159: ('>i2', 2),     # SHORT2N
    0x002A2190: ('>u4', 1),     # 11:11:10 signed normalized, x in the low bits
    0x001A2286: ('u1', 4),      # UBYTE4 (blend indices; one skinned mesh in University)
    0x002C2359: ('>i2', 2),     # SHORT2 (usage 13; same mesh)
}
# element ids (byte 3 of the usage word); the declaration's mask word = OR of 1 << id
EL_POSITION, EL_TEX0, EL_TEX1, EL_TEX3, EL_TANGENT = 1, 6, 7, 9, 0x15
UV_NAMES = {EL_TEX0: 'uv0', EL_TEX1: 'uv1', EL_TEX3: 'uv3'}     # element id -> semantic name
UV_EXTRA = {EL_TEX0: 'uv2'}                                     # 2nd pair of a 4-wide TEX0
PRIM_TRIANGLELIST = 4


def u32(d, o):
    return struct.unpack_from('>I', d, o)[0]


def words(d, o=0, n=None):
    n = (len(d) - o) // 4 if n is None else n
    return list(struct.unpack_from(f'>{n}I', d, o))


def floats(d, o, n):
    return list(struct.unpack_from(f'>{n}f', d, o))


def cstr(d, o):
    return d[o:d.index(b'\0', o)].decode('latin-1')


# ---------------------------------------------------------------- simple objects

def dec_decl(b):
    w = words(b, 0, 4)
    n = w[2] >> 16
    els = []
    for k in range(n):
        so, ty, use, last = struct.unpack_from('>IIII', b, 16 + 16 * k)
        els.append(dict(stream=so >> 16, offset=so & 0xFFFF, type=ty, method=use >> 24,
                        usage=(use >> 16) & 0xFF, usage_index=(use >> 8) & 0xFF, id=use & 0xFF,
                        w3=last))
    strides = list(b[16 + 16 * n:16 + 17 * n])
    return dict(w0=w[0], w2lo=w[2] & 0xFFFF, w3=w[3], elements=els, mask=w[1], strides=strides)


def decl_stride(decl):
    e = decl['elements'][-1]
    dt, n = FMT[e['type']]
    return e['offset'] + np.dtype(dt).itemsize * n


def dec_vb(b):
    w = words(b)
    return dict(head=w[0:6], resource=w[6] >> 2, fetch_type=w[6] & 3, size=w[8], word7=w[7], w9=w[9])


def dec_ib(b):
    w = words(b)
    return dict(head=w[0:6], resource=w[6], size=w[7], count=w[8])


def dec_bind(b):
    w = words(b)
    return dict(w0=w[0], w1=w[1], ib=w[2], vb=w[3])


def dec_mesh(b):
    w = words(b)
    n = w[14]
    return dict(bbox=floats(b, 0, 8), w8=w[8], material=w[9], decl=w[10], bind=w[11], ib=w[12],
                vb=w[13], extra_ref=w[15], bounds_ref=w[16], prim_off=w[17], w18=w[18],
                pad=w[20:24],
                prims=[dict(type=w[24 + 4 * k], base_vertex=w[25 + 4 * k], start=w[26 + 4 * k],
                            count=w[27 + 4 * k]) for k in range(n)],
                tail=w[24 + 4 * n])


def dec_model(b):
    w = words(b, 0, 16)
    nm, nm2 = w[12] >> 16, w[12] & 0xFFFF
    one, nb = w[13] >> 16, w[13] & 0xFFFF
    nbp = (nb + 3) // 4 * 4
    return dict(bbox=floats(b, 0, 8), matrix=floats(b, w[8], 16),
                meshes=[u32(b, w[9] + 8 * k) for k in range(nm)],
                mesh_pad=[u32(b, w[9] + 8 * k + 4) for k in range(nm)],
                name=cstr(b, w[11]), n_meshes2=nm2, w13hi=one,
                bounds=np.frombuffer(b, '>f4', 8 * nbp, w[14]).reshape(nbp, 8).copy(),
                extra=np.frombuffer(b, '>f4', 4 * nbp, w[15]).reshape(nbp, 4).copy(),
                n_bounds=nb, offsets=w[8:12] + w[14:16])


def dec_inst(b):
    """eb000d: {hash 0xacb31c9a, count, w2, 0x20 = records offset, string pool offset, 0 x3},
    count x 0xA0-byte records, then the string pool (deduplicated, first use first)."""
    w = words(b, 0, 8)
    recs = []
    for k in range(w[1]):
        r = w[3] + 0xA0 * k
        t = words(b, r + 0x80, 8)
        recs.append(dict(matrix=floats(b, r, 16), bbox=floats(b, r + 0x40, 8),
                         guid=struct.unpack_from('>Q', b, r + 0x60)[0], links=words(b, r + 0x68, 6),
                         model=t[0], t1=t[1], t2=t[2], strings=[cstr(b, o) for o in t[3:7]], t7=t[7]))
    return dict(hash=w[0], w2=w[2], head_pad=w[5:8], records=recs)


def dec_version(b):
    return dict(words=words(b))


def dec_probes(b):
    """eb0024: {count, 0x10 = records offset, 0xdededede x2}, count x 40 floats."""
    w = words(b, 0, 4)
    n = w[0]
    return dict(rec_off=w[1], fill=w[2:4], records=np.frombuffer(b, '>f4', 40 * n, w[1]).reshape(n, 40).copy())


def dec_quads(b):
    """eb0016 (big worlds): {count, 0x10, 0, 0}, count x 112 B: 5 points (x, y, z, 1) - four corners
    and the centre -, a direction vec4, a 64-bit guid (0x2c70170a... like a texture guid), u32 0,
    u32 0xefbeadde."""
    w = words(b, 0, 4)
    n = w[0]
    rec = np.frombuffer(b, np.dtype([('points', '>f4', (5, 4)), ('axis', '>f4', 4), ('guid', '>u8'),
                                     ('w', '>u4'), ('tag', '>u4')]), n, w[1]).copy()
    return dict(rec_off=w[1], fill=w[2:4], records=rec)


def dec_guid_quads(b):
    """eb0028 (big worlds): {count, 0x10, 0, 0}, count x 4 u64 (first and third look like guids
    shared across records, the third often 0x2c70170a...)."""
    w = words(b, 0, 4)
    n = w[0]
    return dict(rec_off=w[1], fill=w[2:4], records=np.frombuffer(b, '>u8', 4 * n, w[1]).reshape(n, 4).copy())


def dec_polys(b):
    """eb0025 (big worlds): bbox (8 floats), {count, 0x30 = points offset, 0, 0}, count x 4 points
    (x, y, z, 0) - quads."""
    w = words(b, 0x20, 4)
    n = w[0]
    return dict(bbox=floats(b, 0, 8), rec_off=w[1], fill=w[2:4],
                points=np.frombuffer(b, '>f4', 16 * n, w[1]).reshape(n, 4, 4).copy())


def dec_exports(b):
    cnt, tab, tab_end, npairs, pairs = words(b, 0, 5)
    ents, end = [], tab_end
    for k in range(cnt):
        no, h, g, ty, val = struct.unpack_from('>IIQII', b, tab + 24 * k)
        ents.append(dict(name=cstr(b, no) if no else None, hash=h, guid=g, type=ty, value=val))
        if no:
            end = max(end, b.index(b'\0', no) + 1)
    # bytes between the string pool and the index are not zeroed by the retail tool (garbage)
    return dict(entries=ents, index=[tuple(words(b, pairs + 8 * k, 2)) for k in range(npairs)],
                offsets=(tab, tab_end, pairs), string_pad=b[end:pairs])


def dec_materials(b):
    cnt, total, tab, par, strs = words(b, 0, 5)
    mats, end = [], strs
    for k in range(cnt):
        n, z, off = words(b, tab + 12 * k, 3)
        params = []
        for j in range(n):
            p = off + 32 * j
            no, ty, f1, f2 = words(b, p, 4)
            q = dict(name=cstr(b, no), type=ty, fill=(f1, f2))
            end = max(end, no + len(q['name']) + 1)
            if ty == 0xFFFF:
                g, so, z2 = struct.unpack_from('>QII', b, p + 16)
                q.update(guid=g, value=cstr(b, so), z=z2)
                end = max(end, so + len(q['value']) + 1)
            else:
                q.update(raw=words(b, p + 16, 4))
            params.append(q)
        mats.append(dict(params=params, z=z))
    # most retail tables end with 16 zero bytes after the string pool (rule unknown)
    return dict(materials=mats, offsets=(tab, par, strs), tail=len(b) - end)


DECODERS = {T_DECL: dec_decl, T_VB: dec_vb, T_IB: dec_ib, T_BIND: dec_bind, T_MESH: dec_mesh,
            T_MODEL: dec_model, T_INST: dec_inst, T_VER: dec_version, T_PROBES: dec_probes,
            T_EXPORTS: dec_exports, T_MATS: dec_materials, T_QUADS: dec_quads,
            T_GUIDS: dec_guid_quads, T_POLYS: dec_polys}


# ---------------------------------------------------------------- vertex data

def split_elements(data, decl):
    """Raw per-element numpy arrays (exact on-disk values)."""
    stride = decl_stride(decl)
    raw = np.frombuffer(data, 'u1').reshape(-1, stride)
    out = {}
    for e in decl['elements']:
        dt, n = FMT[e['type']]
        sz = np.dtype(dt).itemsize * n
        out[e['id']] = raw[:, e['offset']:e['offset'] + sz].copy().view(dt).reshape(len(raw), n)
    return out


def _snorm(v, bits):
    v = v.astype(np.int64)
    v = np.where(v >= 1 << (bits - 1), v - (1 << bits), v)
    return v / ((1 << (bits - 1)) - 1)


def unpack_111110(w):
    w = w.reshape(-1).astype(np.int64)
    return np.stack([_snorm(w & 0x7FF, 11), _snorm((w >> 11) & 0x7FF, 11), _snorm((w >> 22) & 0x3FF, 10)], 1)


def vertex_semantics(el):
    """Semantic vertex attributes from split_elements() output.
    TEX1 SHORT4N = (lightmap u * binormal sign, lightmap v * sign(normal z), normal x, normal y);
    TEX1 SHORT2N = (lightmap u, lightmap v); TEX0 FLOAT16_4 = (uv0, uv2) - uv2 is used by decal
    materials. Tangent = dP/du of uv0; bitangent = -binormal_sign * cross(normal, tangent)."""
    s = {}
    for i, a in el.items():
        if i == EL_POSITION:
            s['position'] = a.astype(np.float64)
        elif i == EL_TANGENT:
            s['tangent'] = unpack_111110(a)
        elif i == EL_TEX1 and a.dtype == np.dtype('>i2') and a.shape[1] == 4:
            q = a.astype(np.int64)
            s['uv1'] = np.abs(q[:, :2]) / 32767.0
            nx, ny = q[:, 2] / 32767.0, q[:, 3] / 32767.0
            nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1)) * np.where(q[:, 1] < 0, -1, 1)
            s['normal'] = np.stack([nx, ny, nz], 1)
            s['binormal_sign'] = np.where(q[:, 0] < 0, -1, 1)
        elif i in UV_NAMES:
            v = a.astype(np.int64) / 32767.0 if a.dtype == np.dtype('>i2') else a.astype(np.float64)
            s[UV_NAMES[i]] = v[:, :2]
            if v.shape[1] == 4:
                s[UV_EXTRA[i]] = v[:, 2:]
        else:
            s[f'el{i:x}'] = a.copy()        # no semantic name yet: on-disk values, kept exact
    return s


# ---------------------------------------------------------------- arena-level view

@dataclass
class Part:
    """One primitive of one mesh, world space (instance matrices are identity in retail)."""
    mesh_index: int
    prim: int
    positions: np.ndarray
    indices: np.ndarray        # (n, 3) into this part's own vertex arrays
    attrs: dict
    material: dict | None
    material_index: int | None


@dataclass
class RenderArena:
    arena: rw4.Arena
    decoded: dict = field(default_factory=dict)     # object index -> decoded dict

    def resolve(self, ref):
        """ref word -> (object index, byte offset)."""
        if ref & SUBREF:
            return tuple(self.arena.subrefs[ref & ~SUBREF])
        return ref, 0

    def materials(self):
        """All materials of the arena as (object index, entry byte offset, material dict)."""
        out = []
        for i, d in self.decoded.items():
            if self.arena.objects[i].type_id == T_MATS:
                tab = d['offsets'][0]
                out += [(i, tab + 12 * k, m) for k, m in enumerate(d['materials'])]
        return out

    def material_at(self, ref):
        """A mesh's material (one per mesh; every primitive of the mesh uses it)."""
        i, off = self.resolve(ref)
        tab = self.decoded[i]['offsets'][0]
        j = (off - tab) // 12
        return j, self.decoded[i]['materials'][j]

    def parts(self):
        O = self.arena.objects
        for i, o in enumerate(O):
            if o.type_id != T_MESH:
                continue
            m = self.decoded[i]
            decl = self.decoded[m['decl']]
            vb, ib = self.decoded[m['vb']], self.decoded[m['ib']]
            el = split_elements(O[vb['resource']].data, decl)
            sem = vertex_semantics(el)
            idx = np.frombuffer(O[ib['resource']].data, '>u2', ib['count']).astype(np.int64)
            for k, p in enumerate(m['prims']):
                tri = idx[p['start']:p['start'] + p['count']] + p['base_vertex']
                used, inv = np.unique(tri, return_inverse=True)
                try:
                    mi, mat = self.material_at(m['material'])
                except (KeyError, IndexError):
                    mi, mat = None, None
                yield Part(i, k, sem['position'][used],
                           inv.reshape(-1, 3), {a: v[used] for a, v in sem.items()}, mat, mi)


def decode(arena_bytes) -> RenderArena:
    a = rw4.parse(arena_bytes)
    ra = RenderArena(a)
    for o in a.objects:
        f = DECODERS.get(o.type_id)
        if f:
            ra.decoded[o.index] = f(o.data)
    return ra


def texture_refs(material):
    """{param name: guid} for every texture-valued param (type 0xffff with a guid)."""
    return {p['name']: p['guid'] for p in material['params']
            if p['type'] == 0xFFFF and p['guid'] and p['name'] not in ('Name',)}


def shader_name(material):
    for p in material['params']:
        if p['name'] == 'AttribulatorMaterialName':
            return p['value']
    return None


# ---------------------------------------------------------------- export

def export_obj(parts, path):
    with open(path, 'w') as f:
        base = 1
        for p in parts:
            f.write(f'o mesh{p.mesh_index}_{p.prim}\n')
            mat = p.material
            if mat:
                f.write(f'# shader {shader_name(mat)}\n')
            for v in p.positions:
                f.write(f'v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n')
            n = len(p.positions)
            uv = p.attrs.get('uv0', np.zeros((n, 2)))
            nr = p.attrs.get('normal', np.zeros((n, 3)))
            for t in uv:
                f.write(f'vt {t[0]:.5f} {1 - t[1]:.5f}\n')
            for v in nr:
                f.write(f'vn {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
            for t in p.indices + base:
                f.write('f ' + ' '.join(f'{x}/{x}/{x}' for x in t) + '\n')
            base += len(p.positions)
