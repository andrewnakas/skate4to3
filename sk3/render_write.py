"""Encoders for Skate 3 render-arena objects (inverse of sk3/render.py), byte exact against every
retail cPres arena tested (tests/test_render_roundtrip.py; facts in specs/sk3_render.md), plus a
builder that makes a whole render arena from plain mesh parts.

Every encoder takes the dict its decoder returns and derives all counts, offsets, masks, strides,
string pools and padding itself; only values a generator has to choose are read from the dict.
"""
from __future__ import annotations

import struct

import numpy as np

from . import render as R
from . import rw4_write

P = struct.pack


def _align(x, a):
    return (x + a - 1) // a * a


def _w(*v):
    return P(f'>{len(v)}I', *v)


def _f(*v):
    return P(f'>{len(v)}f', *v)


class _Pool:
    """NUL-terminated strings, deduplicated in first-use order."""
    def __init__(self, base):
        self.base, self.buf, self.at = base, bytearray(), {}

    def add(self, s):
        if s not in self.at:
            self.at[s] = self.base + len(self.buf)
            self.buf += s.encode('latin-1') + b'\0'
        return self.at[s]


# ---------------------------------------------------------------- small objects

def enc_decl(d):
    els = d['elements']
    n = len(els)
    mask = 0
    for e in els:
        mask |= 1 << e['id']
    stride = R.decl_stride(d)
    out = _w(d['w0'], mask, n << 16 | d['w2lo'], d['w3'])
    for e in els:
        out += _w(e['stream'] << 16 | e['offset'], e['type'],
                  e['method'] << 24 | e['usage'] << 16 | e['usage_index'] << 8 | e['id'], e['w3'])
    return out + bytes([stride]) * n


VB_HEAD = [1, 1, 0, 0, 0, 0xFFFF0000]
IB_HEAD = [0x20000002, 1, 0, 0, 0, 0xFFFF0000]


def enc_vb(d):
    # fetch constant: dword0 = resource index << 2 | 3 (vertex), dword1 = 0x10000000 | size | endian 2
    return _w(*d['head'], d['resource'] << 2 | d['fetch_type'], 0x10000000 | d['size'] | 2, d['size'], d['w9'])


def enc_ib(d):
    return _w(*d['head'], d['resource'], d['size'], d['count'])


def enc_bind(d):
    return _w(d['w0'], d['w1'], d['ib'], d['vb'])


def enc_mesh(d):
    n = len(d['prims'])
    out = _f(*d['bbox']) + _w(d['w8'], d['material'], d['decl'], d['bind'], d['ib'], d['vb'], n,
                              d['extra_ref'], d['bounds_ref'], d['prim_off'], d['w18'],
                              d['prim_off'] + 16 * n, *d['pad'])
    for p in d['prims']:
        out += _w(p['type'], p['base_vertex'], p['start'], p['count'])
    return out + _w(d['tail'])


def enc_model(d):
    n = len(d['meshes'])
    nb = d['n_bounds']
    nbp = (nb + 3) // 4 * 4
    mat_at, list_at = 0x40, 0x80
    ptr_at = list_at + 8 * n
    name_at = ptr_at + 4
    name = d['name'].encode('latin-1') + b'\0'
    bounds_at = _align(name_at + len(name), 16)
    extra_at = bounds_at + 32 * nbp
    out = bytearray(extra_at + 16 * nbp)
    out[0:0x20] = _f(*d['bbox'])
    out[0x20:0x40] = _w(mat_at, list_at, ptr_at, name_at, n << 16 | d['n_meshes2'],
                        d['w13hi'] << 16 | nb, bounds_at, extra_at)
    out[mat_at:mat_at + 64] = _f(*d['matrix'])
    for k, (m, z) in enumerate(zip(d['meshes'], d['mesh_pad'])):
        out[list_at + 8 * k:list_at + 8 * k + 8] = _w(m, z)
    out[ptr_at:ptr_at + 4] = _w(name_at)
    out[name_at:name_at + len(name)] = name
    out[bounds_at:extra_at] = np.asarray(d['bounds'], '>f4').tobytes()
    out[extra_at:] = np.asarray(d['extra'], '>f4').tobytes()
    return bytes(out)


def enc_inst(d):
    recs = d['records']
    pool = _Pool(0x20 + 0xA0 * len(recs))
    out = _w(d['hash'], len(recs), d['w2'], 0x20, pool.base, *d['head_pad'])
    for r in recs:
        offs = [pool.add(s) for s in r['strings']]
        out += _f(*r['matrix']) + _f(*r['bbox']) + P('>Q', r['guid']) + _w(*r['links'])
        out += _w(r['model'], r['t1'], r['t2'], *offs, r['t7'])
    return out + bytes(pool.buf)


def enc_version(d):
    return _w(*d['words'])


def _enc_array(d, rec):
    head = _w(len(rec), d['rec_off'], *d['fill'])
    return head + b'\0' * (d['rec_off'] - len(head)) + rec.tobytes()


def enc_probes(d):
    return _enc_array(d, np.asarray(d['records'], '>f4'))


def enc_quads(d):
    return _enc_array(d, np.asarray(d['records']))


def enc_guid_quads(d):
    return _enc_array(d, np.asarray(d['records'], '>u8'))


def enc_polys(d):
    pts = np.asarray(d['points'], '>f4')
    return _f(*d['bbox']) + _w(len(pts), d['rec_off'], *d['fill']) + pts.tobytes()


# the per-type index after an export table: these 25 types in this order; the value is the first
# entry of the first two types (materials 0xeb0066, material table 0xeb0005) and the entry count
# for everything else (fits every retail cPres arena with an index)
INDEX_TYPES = (0xeb0066, 0xeb0005, 0xeb0067, 0xeb0006, 0xeb0001, 0xeb000a, 0xeb0065, 0xeb0007,
               0xeb0069, 0xeb000d, 0xeb006b, 0xeb0019, 0xeb0064, 0xeb0004, 0xeb0068, 0xeb0009,
               0xeb0016, 0xeb0013, 0xeb0014, 0xeb0018, 0xeb0017, 0xeb0020, 0xeb0024, 0xeb0026,
               0xeb0027)


def export_index(entries):
    n = len(entries)
    n66 = sum(1 for e in entries if e['type'] == 0xeb0066)
    has05 = any(e['type'] == 0xeb0005 for e in entries)
    first = 0 if n66 else n
    second = n66 if has05 else n
    return [(t, v) for t, v in zip(INDEX_TYPES, [first, second] + [n] * (len(INDEX_TYPES) - 2))]


def enc_exports(d):
    ents = d['entries']
    tab = 0x14
    tab_end = tab + 24 * len(ents)
    pool = _Pool(tab_end)
    rows = b''.join(P('>IIQII', pool.add(e['name']) if e['name'] is not None else 0, e['hash'],
                      e['guid'], e['type'], e['value']) for e in ents)
    strings = bytes(pool.buf)
    has_index = bool(d['index'])
    pairs_at = _align(tab_end + len(strings), 4)
    index = export_index(ents) if has_index else []
    head = _w(len(ents), tab, tab_end, len(index), pairs_at)
    body = head + rows + strings
    pad = d.get('string_pad', b'')
    body += pad + b'\0' * (pairs_at - len(body) - len(pad))
    return body + b''.join(_w(t, v) for t, v in index)


def enc_materials(d):
    mats = d['materials']
    cnt = len(mats)
    total = sum(len(m['params']) for m in mats)
    tab = 0x14
    par = _align(tab + 12 * cnt, 16)
    strs = par + 32 * total
    pool = _Pool(strs)
    head, rows, off = bytearray(), bytearray(), par
    for m in mats:
        head += _w(len(m['params']), m['z'], off)
        off += 32 * len(m['params'])
        for p in m['params']:
            no = pool.add(p['name'])
            if p['type'] == 0xFFFF:
                val = P('>QII', p['guid'], pool.add(p['value']), p['z'])
            else:
                val = _w(*p['raw'])
            rows += _w(no, p['type'], *p['fill']) + val
    out = _w(cnt, total, tab, par, strs) + head
    return out + b'\0' * (par - len(out)) + rows + bytes(pool.buf) + b'\0' * d.get('tail', 0)


ENCODERS = {R.T_DECL: enc_decl, R.T_VB: enc_vb, R.T_IB: enc_ib, R.T_BIND: enc_bind,
            R.T_MESH: enc_mesh, R.T_MODEL: enc_model, R.T_INST: enc_inst, R.T_VER: enc_version,
            R.T_PROBES: enc_probes, R.T_EXPORTS: enc_exports, R.T_MATS: enc_materials,
            R.T_QUADS: enc_quads, R.T_GUIDS: enc_guid_quads, R.T_POLYS: enc_polys}


# ---------------------------------------------------------------- vertex / index data

def pack_111110(v):
    v = np.asarray(v, np.float64)
    out = np.zeros(len(v), np.int64)
    for k, (sh, bits) in enumerate(((0, 11), (11, 11), (22, 10))):
        q = np.rint(v[:, k] * ((1 << (bits - 1)) - 1)).astype(np.int64)
        out |= (q & ((1 << bits) - 1)) << sh
    return out.astype('>u4').reshape(-1, 1)


def _sn16(v):
    return np.rint(np.asarray(v, np.float64) * 32767).astype(np.int64)


def semantics_to_elements(sem, decl):
    """Inverse of render.vertex_semantics: per element id the on-disk value array."""
    out = {}
    for e in decl['elements']:
        dt, n = R.FMT[e['type']]
        i = e['id']
        if i == R.EL_POSITION:
            a = sem['position']
        elif i == R.EL_TANGENT:
            out[i] = pack_111110(sem['tangent'])
            continue
        elif i == R.EL_TEX1 and dt == '>i2' and n == 4:
            nz_neg = np.signbit(sem['normal'][:, 2])
            a = np.stack([_sn16(sem['uv1'][:, 0]) * sem['binormal_sign'],
                          _sn16(sem['uv1'][:, 1]) * np.where(nz_neg, -1, 1),
                          _sn16(sem['normal'][:, 0]), _sn16(sem['normal'][:, 1])], 1)
        elif i in R.UV_NAMES:
            a = sem[R.UV_NAMES[i]]
            if n == 4:
                a = np.concatenate([a, sem[R.UV_EXTRA[i]]], 1)
            if dt == '>i2':
                a = _sn16(a)
        else:
            a = sem[f'el{i:x}']
        out[i] = np.asarray(a).astype(dt)
    return out


def pack_vertices(sem, decl):
    el = semantics_to_elements(sem, decl)
    n = len(sem['position'])
    stride = R.decl_stride(decl)
    raw = np.zeros((n, stride), 'u1')
    for e in decl['elements']:
        b = np.ascontiguousarray(el[e['id']]).view('u1').reshape(n, -1)
        raw[:, e['offset']:e['offset'] + b.shape[1]] = b
    return raw.tobytes()


def pack_indices(idx):
    b = np.asarray(idx, '>u2').tobytes()
    return b + b'\0' * (_align(len(b), 16) - len(b))


# ---------------------------------------------------------------- whole arena

def objects_from_render(ra: R.RenderArena):
    """Semantic description of every object: (type_id, align, kind, payload). Base resources that a
    vertex/index buffer points at become ('vertices', semantics, decl index) / ('indices', array);
    texture headers and their texels are carried raw (texture code owns them)."""
    O = ra.arena.objects
    role = {}
    for i, d in ra.decoded.items():
        t = O[i].type_id
        if t == R.T_MESH:
            role[ra.decoded[d['vb']]['resource']] = ('vertices', d['decl'])
            role[ra.decoded[d['ib']]['resource']] = ('indices', d['ib'])
    out = []
    for o in O:
        if o.index in ra.decoded:
            out.append((o.type_id, o.align, 'obj', ra.decoded[o.index]))
        elif role.get(o.index, ('',))[0] == 'vertices':
            decl = ra.decoded[role[o.index][1]]
            out.append((o.type_id, o.align, 'vertices',
                        (R.vertex_semantics(R.split_elements(o.data, decl)), role[o.index][1])))
        elif role.get(o.index, ('',))[0] == 'indices':
            ib = ra.decoded[role[o.index][1]]
            out.append((o.type_id, o.align, 'indices',
                        np.frombuffer(o.data, '>u2', ib['count']).astype(np.int64)))
        else:
            out.append((o.type_id, o.align, 'raw', o.data))
    return out


def encode_objects(objs):
    """[(type_id, align, kind, payload)] -> [rw4_write.SpecObject]."""
    decls = {}
    for k, (t, a, kind, p) in enumerate(objs):
        if t == R.T_DECL:
            decls[k] = p
    spec = []
    for t, a, kind, p in objs:
        if kind == 'obj':
            data = ENCODERS[t](p)
        elif kind == 'vertices':
            data = pack_vertices(p[0], decls[p[1]])
        elif kind == 'indices':
            data = pack_indices(p)
        else:
            data = p
        spec.append(rw4_write.SpecObject(t, a, data))
    return spec


def rebuild(arena_bytes):
    """Decode -> semantic -> encode -> arena bytes (the round trip the test checks)."""
    ra = R.decode(arena_bytes)
    arena_id = struct.unpack_from('>I', arena_bytes, 0x1C)[0]
    return rw4_write.build(rw4_write.Spec(arena_id, encode_objects(objects_from_render(ra)),
                                          [tuple(s) for s in ra.arena.subrefs]))


# ---------------------------------------------------------------- builder

# vertex declarations retail uses per shader (elements: type, usage, usage index, id), measured over
# the small worlds (specs/sk3_render.md "Shader -> declaration")
_POS = (0x002A23B9, 0, 0, R.EL_POSITION)
_UV0 = (0x002C235F, 5, 0, R.EL_TEX0)
_UV0_UV2 = (0x001A2360, 5, 0, R.EL_TEX0)
_LM_NRM = (0x001A215A, 5, 1, R.EL_TEX1)
_LM = (0x002C2159, 5, 1, R.EL_TEX1)
_TAN = (0x002A2190, 6, 0, R.EL_TANGENT)
DECL_BY_SHADER = {
    'environmentsimple.default': (_POS, _UV0, _LM_NRM, _TAN),
    'environment.default': (_POS, _UV0, _LM_NRM, _TAN),
    'environment.decal': (_POS, _UV0_UV2, _LM_NRM, _TAN),
    'environment.decal_tileable': (_POS, _UV0_UV2, _LM_NRM, _TAN),
    'environment.transparent': (_POS, _UV0, _LM_NRM),
    'advertisement.default': (_POS, _UV0, _LM_NRM),
    'environmentsimple.diffuse': (_POS, _UV0, _LM),
    'incandescent.default': (_POS, _UV0, _LM),
    'incandescent.backlituvscroll': (_POS, _UV0, _LM),
    'incandescent.videoscreen': (_POS, _UV0, _LM),
}
# parameter order retail writes per shader (after Name, AttribulatorMaterialName)
PARAMS_BY_SHADER = {
    'environmentsimple.default': ('lightmap', 'specular', 'diffuse', 'normal'),
    'environmentsimple.diffuse': ('lightmap', 'diffuse'),
    'environment.default': ('lightmap', 'detailNormalUVScale', 'specular', 'diffuse', 'macroOverlayOpacity',
                            'detail', 'macrooverlay', 'normal', 'macroOverlayUVScale'),
    'environment.transparent': ('lightmap', 'transparent', 'specular', 'diffuse'),
    'advertisement.default': ('lightmap', 'diffuse'),
    'incandescent.default': ('exposure', 'diffuse'),
    'incandescent.backlituvscroll': ('uAnimationSpeed', 'lightmap', 'vAnimationSpeed', 'diffuse'),
    'incandescent.videoscreen': ('diffuse',),
    'environment.decal': ('environment', 'lightmap', 'detailNormalUVScale', 'decal', 'specular', 'diffuse',
                          'macroOverlayOpacity', 'detail', 'embeddedDecal', 'macrooverlay', 'normal',
                          'macroOverlayUVScale'),
    'dynamicobject.default': ('detailNormalUVScale', 'gamePlayDesignate', 'specular', 'diffuse', 'detail', 'normal'),
}
PARAMS_BY_SHADER['environment.decal_tileable'] = PARAMS_BY_SHADER['environment.decal']
# float params: the most common retail value of each
FLOAT_DEFAULTS = {'detailNormalUVScale': 8.0, 'macroOverlayOpacity': 1.0, 'macroOverlayUVScale': 0.3,
                  'exposure': 10.0, 'embeddedDecal': 1.0, 'gamePlayDesignate': 0.0,
                  'uAnimationSpeed': -0.25, 'vAnimationSpeed': 0.0}
INST_HASH = 0xACB31C9A          # word 0 of every retail eb000d
NO_HASH = 0xFEFFFFFF            # export entries without a name
FILL = (0xDEADBEEF, 0xDEADC0DE)
MODEL_MATRIX = (1.0, -0.0, 0.0, -0.0, -0.0, 1.0, -0.0, 0.0, 0.0, -0.0, 1.0, -0.0, -0.0, 0.0, -0.0, 1.0)
IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def guid64(*parts):
    """Deterministic 64-bit id (FNV-1a) for things the builder has to name."""
    h = 0xCBF29CE484222325
    for b in '|'.join(str(p) for p in parts).encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def make_decl(shader):
    els, off = [], 0
    for ty, usage, ui, i in DECL_BY_SHADER[shader]:
        els.append(dict(stream=0, offset=off, type=ty, method=0, usage=usage, usage_index=ui, id=i, w3=1))
        dt, n = R.FMT[ty]
        off += np.dtype(dt).itemsize * n
    return dict(w0=0, w2lo=1, w3=0, elements=els)


def tangent_frame(pos, nrm, uv, tri):
    """Per-vertex tangent (dP/du made orthogonal to the normal) and binormal sign, matching what
    retail vertices measure: tangent . dP/du ~ 1 and sign(TEX1.x) = -sign(dot(cross(n, t), dP/dv))."""
    e1, e2 = pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]]
    d1, d2 = uv[tri[:, 1]] - uv[tri[:, 0]], uv[tri[:, 2]] - uv[tri[:, 0]]
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    T = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    B = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    t, b = np.zeros_like(pos), np.zeros_like(pos)
    for k in range(3):
        np.add.at(t, tri[:, k], T)
        np.add.at(b, tri[:, k], B)
    t -= nrm * (t * nrm).sum(1, keepdims=True)
    ln = np.linalg.norm(t, axis=1)
    bad = ln < 1e-9
    if bad.any():           # no uv gradient: any vector perpendicular to the normal
        alt = np.where(np.abs(nrm[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 0, 1.0]])
        alt = alt - nrm * (alt * nrm).sum(1, keepdims=True)
        t[bad] = alt[bad]
        ln = np.linalg.norm(t, axis=1)
    t = t / ln[:, None]
    hand = np.sign((np.cross(nrm, t) * b).sum(1))
    return t, np.where(hand > 0, -1, 1)


def _material(m, mi):
    shader = m.get('shader', 'environmentsimple.default')
    name = m.get('name', f'mat{mi}')
    tex = m.get('textures', {})
    fl = {**FLOAT_DEFAULTS, **m.get('floats', {})}
    tex_names = m.get('texture_names', {})
    params = [dict(name='Name', type=0xFFFF, fill=FILL, guid=m.get('guid', guid64('mat', name)), value=name, z=0),
              dict(name='AttribulatorMaterialName', type=0xFFFF, fill=FILL, guid=0, value=shader, z=0)]
    for pn in m['params'] if 'params' in m else PARAMS_BY_SHADER[shader]:
        if pn in fl:
            v = struct.unpack('>I', struct.pack('>f', fl[pn]))[0]
            params.append(dict(name=pn, type=0x20000, fill=FILL, raw=[v, 0, 0, 0]))
        else:
            g = tex[pn]
            params.append(dict(name=pn, type=0xFFFF, fill=FILL, guid=g,
                               value=f'{tex_names.get(pn, pn)}_0x{g:016x}', z=0))
    return dict(params=params, z=0), shader


def _mesh_semantics(p, decl):
    pos = np.asarray(p['positions'], np.float64)
    tri = np.asarray(p['indices'], np.int64).reshape(-1, 3)
    ids = {e['id']: R.FMT[e['type']] for e in decl['elements']}
    uv0 = np.asarray(p['uv0'], np.float64)
    sem = {'position': pos, 'uv0': uv0}
    if ids[R.EL_TEX0][1] == 4:
        sem['uv2'] = np.asarray(p.get('uv2', uv0), np.float64)
    if R.EL_TEX1 in ids:
        sem['uv1'] = np.clip(np.asarray(p.get('uv1', np.zeros((len(pos), 2))), np.float64), 0, 1)
    if R.EL_TANGENT in ids or (R.EL_TEX1 in ids and ids[R.EL_TEX1][1] == 4):
        nrm = np.asarray(p['normals'], np.float64)
        nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        t, bs = tangent_frame(pos, nrm, uv0, tri)
        sem.update(normal=nrm, binormal_sign=bs, tangent=t)
    return sem, tri


def build_render_arena(parts, arena_id, name='Model'):
    """parts: [{positions (n,3) world space, normals (n,3), uv0 (n,2), uv1 (n,2) lightmap uv in [0,1],
    indices (m,3), material: {name, shader, textures: {param: guid}, floats, texture_names}}]
    -> one cPres render arena laid out like retail: version, material table, instance, then per part
    (vertex data, vb, index data, ib, mesh, declaration, binding), model, export table."""
    mats, shaders = zip(*[_material(p['material'], k) for k, p in enumerate(parts)])
    n = len(parts)
    model_index = 3 + 7 * n
    objs = [(R.T_VER, 0x10, 'obj', dict(words=[0x19, 0x0D, 0, 0])),
            (R.T_MATS, 0x10, 'obj', dict(materials=list(mats), tail=0)),
            None]                                       # instance, filled in below
    mesh_idx, refs, bounds, extra = [], [], [], []
    lo_all, hi_all = np.full(3, np.inf), np.full(3, -np.inf)
    for k, p in enumerate(parts):
        decl = make_decl(shaders[k])
        sem, tri = _mesh_semantics(p, decl)
        pos = sem['position']
        if len(pos) > 0xFFFF:
            raise ValueError('a part needs <= 65535 vertices (16-bit indices)')
        vdata = pack_vertices(sem, decl)
        idx = tri.reshape(-1)
        lo, hi = pos.min(0), pos.max(0)
        lo_all, hi_all = np.minimum(lo_all, lo), np.maximum(hi_all, hi)
        half = np.linalg.norm(hi - lo) / 2
        bounds.append([*lo, half, *hi, half])
        c = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]]) / 2
        extra.append([*np.abs(c).sum(0), 0.0])
        sr = len(refs)
        refs += [('mat', k), ('extra', k), ('bounds', k)]
        base = len(objs)
        objs += [(R.T_BASE, 4, 'vertices', (sem, base + 5)),
                 (R.T_VB, 4, 'obj', dict(head=VB_HEAD, resource=base, fetch_type=3, size=len(vdata), w9=0)),
                 (R.T_BASE, 4, 'indices', idx),
                 (R.T_IB, 4, 'obj', dict(head=IB_HEAD, resource=base + 2, size=_align(2 * len(idx), 16),
                                         count=len(idx))),
                 (R.T_MESH, 0x10, 'obj', dict(
                     bbox=[*lo, 0.0, *hi, 0.0], w8=0, material=R.SUBREF | sr, decl=base + 5, bind=base + 6,
                     ib=base + 3, vb=base + 1, extra_ref=R.SUBREF | (sr + 1), bounds_ref=R.SUBREF | (sr + 2),
                     prim_off=0x60, w18=1, pad=[0, 0, 0, 0], tail=0,
                     prims=[dict(type=R.PRIM_TRIANGLELIST, base_vertex=0, start=0, count=len(idx))])),
                 (R.T_DECL, 4, 'obj', decl),
                 (R.T_BIND, 4, 'obj', dict(w0=1, w1=1, ib=base + 3, vb=base + 1))]
        mesh_idx.append(base + 4)
    nbp = (n + 3) // 4 * 4
    bb, ex = np.zeros((nbp, 8), '>f4'), np.zeros((nbp, 4), '>f4')
    bb[:n], ex[:n] = bounds, extra
    model = dict(bbox=[*lo_all, 0.0, *hi_all, 0.0], matrix=list(MODEL_MATRIX), meshes=mesh_idx,
                 mesh_pad=[0] * n, name=name, n_meshes2=n, w13hi=1, bounds=bb, extra=ex, n_bounds=n)
    w = R.words(enc_model(model), 0x20, 8)
    at = {'mat': (1, 0x14, 12), 'extra': (model_index, w[7], 16), 'bounds': (model_index, w[6], 32)}
    subrefs = [(at[r][0], at[r][1] + at[r][2] * k) for r, k in refs]
    inst_sr = len(subrefs)
    subrefs.append((2, 0x20))                           # the instance record's matrix
    inst_guid = guid64('inst', arena_id, name)
    objs[2] = (R.T_INST, 0x10, 'obj', dict(hash=INST_HASH, w2=2, head_pad=[0, 0, 0], records=[dict(
        matrix=list(IDENTITY), bbox=model['bbox'], guid=inst_guid, links=[0xFFFFFFFF] * 6,
        model=model_index, t1=0, t2=0, strings=[name, 'undefined', 'undefined', 'undefined'], t7=0)]))
    objs.append((R.T_MODEL, 0x10, 'obj', model))
    ents = [dict(name=None, hash=NO_HASH, guid=m['params'][0]['guid'], type=0xeb0066, value=R.SUBREF | 3 * k)
            for k, m in enumerate(mats)]
    ents += [dict(name=None, hash=NO_HASH, guid=guid64('mats', arena_id, name), type=R.T_MATS, value=1),
             dict(name=None, hash=NO_HASH, guid=inst_guid, type=0xeb0069, value=R.SUBREF | inst_sr),
             dict(name=None, hash=NO_HASH, guid=guid64('instobj', arena_id, name), type=R.T_INST, value=2)]
    objs.append((R.T_EXPORTS, 0x10, 'obj', dict(entries=ents, index=[True], string_pad=b'')))
    out = rw4_write.build(rw4_write.Spec(arena_id, encode_objects(objs), subrefs))
    R.decode(out)                       # sanity: our own reader takes it
    return out
