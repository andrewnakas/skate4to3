"""Encoders for Skate 3 collision objects (inverse of sk3/collision.py) and a builder that turns
triangles + surface ids (+ grind splines) into a cSim arena.

encode_mesh(parse_mesh(x)) == x for every retail 0x80006 (tests/test_collision_roundtrip.py); the
same holds for the other cSim objects' encoders. The builder (build_mesh / build_tile_arena) uses only
the layout rules measured on retail data (specs/sk3_collision.md) with its own clustering and
KD-tree construction.
"""
from __future__ import annotations

import math
import struct

import numpy as np

from . import collision as co
from . import rw4_write

MESH_FLAGS = 0x00100202
GRANULARITY = np.float32(0.001)


def _al(x, a):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------------------------
# 0x80006 clustered mesh
def encode_cluster(c: co.Cluster) -> bytes:
    vc = len(c.qverts)
    if c.vertex_mode == 1:
        vb = struct.pack('>3i', *c.offset) + c.qverts.astype('>u2').tobytes()
    elif c.vertex_mode == 2:
        vb = c.qverts.astype('>i4').tobytes()
    else:
        raise ValueError(c.vertex_mode)
    ust = _al(len(vb), 16) // 16
    ub = bytearray()
    for f, v, e, s in c.units:
        ub += bytes([f, *v, *e])
        if f & co.UNIT_SURFACE:
            ub += struct.pack('>H', s)
    total = 16 + 16 * ust + len(ub)
    hdr = struct.pack('>5HBBB3x', len(c.units), len(ub), ust, ust, total, vc, c.normal_count, c.vertex_mode)
    return hdr + vb + bytes(16 * ust - len(vb)) + bytes(ub)


def encode_mesh(m: co.ClusteredMesh) -> bytes:
    nunits = sum(len(c.units) for c in m.clusters)
    nb, ncl = len(m.nodes), len(m.clusters)
    ctab = 0x90 + 32 * nb
    cl = [encode_cluster(c) for c in m.clusters]
    ptrs, p = [], _al(ctab + 4 * ncl, 16)
    for b in cl:
        ptrs.append(p)
        p = _al(p + len(b), 16)
    size = p
    out = bytearray(size)
    P = lambda off, fmt, *v: struct.pack_into('>' + fmt, out, off, *v)
    (bmin, bmax), (tmin, tmax) = m.bbox, m.tree_bbox
    P(0x00, '3f', *bmin)
    P(0x10, '3f', *bmax)
    P(0x24, 'II', (nunits + 1).bit_length() if m.max_leaf is None else m.max_leaf, nunits)
    P(0x30, 'IIfI', 0x60, ctab, m.granularity, m.flags)
    P(0x40, 'IIII', ncl, ncl, nunits, nunits)
    P(0x50, 'I', size)
    P(0x58, 'I', 0x80000000)
    P(0x60, 'III', 0x90 if nb else m.node_ptr, nb, nunits)
    P(0x70, '3f', *tmin)
    P(0x80, '3f', *tmax)
    for k, n in enumerate(m.nodes):
        (c0, i0), (c1, i1) = n.children
        P(0x90 + 32 * k, '6I2f', n.parent, n.axis, c0, i0, c1, i1, *n.extents)
    P(ctab, f'{ncl}I', *ptrs)
    for q, b in zip(ptrs, cl):
        out[q:q + len(b)] = b
    return bytes(out)


# ---------------------------------------------------------------------------------------------
# 0x80001 volume
def encode_volume(v: co.Volume) -> bytes:
    return struct.pack('>16f', *v.matrix) + struct.pack('>II', v.vtype, v.geometry) + struct.pack('>6I', *v.tail)


def encode_aggregate(g: co.BoxAggregate) -> bytes:
    n, nb = len(g.volumes), len(g.nodes)
    toff = 0x50 + 0x60 * n
    out = bytearray(toff + 0x30 + 32 * nb)
    P = lambda off, fmt, *v: struct.pack_into('>' + fmt, out, off, *v)
    P(0, '3f', *g.bbox[0]); P(0x10, '3f', *g.bbox[1])
    P(0x20, '4I', 0, g.w24, n, 0); P(0x30, 'I', 0x50); P(0x40, 'I', toff)
    for k, v in enumerate(g.volumes):
        out[0x50 + 0x60 * k:0xB0 + 0x60 * k] = encode_volume(v)
    P(toff, '3I', toff + 0x30 if nb else g.node_ptr, nb, g.n_entries)
    P(toff + 0x10, '3f', *g.tree_bbox[0]); P(toff + 0x20, '3f', *g.tree_bbox[1]); P(toff + 0x2C, 'I', g.tree_pad)
    for k, nd in enumerate(g.nodes):
        (c0, i0), (c1, i1) = nd.children
        P(toff + 0x30 + 32 * k, '6I2f', nd.parent, nd.axis, c0, i0, c1, i1, *nd.extents)
    return bytes(out)


def encode_model(m: co.CollisionModel) -> bytes:
    return struct.pack(f'>3I{len(m.volumes)}II', m.bounding, len(m.volumes), 0x0C, *m.volumes, 0)


def encode_link(volume: int, instance: int) -> bytes:          # eb0020
    return struct.pack('>4I', volume, instance, 0, 0)


ARENA_WORDS = struct.pack('>4I', 0x19, 0x0D, 0, 0)              # eb0008


def encode_instances(il: co.InstanceList) -> bytes:
    n = len(il.instances)
    soff = 0x20 + 0xA0 * n
    pool, at = bytearray(), {}
    for ins in il.instances:
        for s in ins.strings:
            if s not in at:
                at[s] = soff + len(pool)
                pool += s.encode('latin-1') + b'\0'
    out = bytearray(soff) + pool
    struct.pack_into('>5I', out, 0, il.kind, n, len(at), 0x20, soff)
    for k, ins in enumerate(il.instances):
        e = 0x20 + 0xA0 * k
        struct.pack_into('>16f', out, e, *ins.matrix)
        struct.pack_into('>3f', out, e + 0x40, *ins.bbox[0]); struct.pack_into('>3f', out, e + 0x50, *ins.bbox[1])
        struct.pack_into('>4Q', out, e + 0x60, *ins.ids)
        struct.pack_into('>8I', out, e + 0x80, 0, ins.flags, 0, *(at[s] for s in ins.strings), 0)
    return bytes(out)


def encode_imports(t: co.ImportTable) -> bytes:
    n, m = len(t.entries), len(t.types)
    t2 = 0x14 + 24 * n
    out = bytearray(struct.pack('>5I', n, 0x14, t2, m, t2))
    for i, ty, v in t.entries:
        out += struct.pack('>IIQII', 0, 0xFEFFFFFF, i, ty, v)
    for ty in t.types:
        out += struct.pack('>II', ty, n)
    return bytes(out)


# ---------------------------------------------------------------------------------------------
# type id -> (decode, encode). Types not listed are carried as bytes (see specs/sk3_collision.md).
def _mesh_dec(d):
    m = co.parse_mesh(d)
    m.max_leaf = None                 # derived on encode
    return m


CODECS = {
    co.T_MESH: (_mesh_dec, encode_mesh),
    co.T_VOLUME: (co.parse_volume, encode_volume),
    0x80004: (co.parse_aggregate, encode_aggregate),
    co.T_MESHREF: (co.parse_model, encode_model),
    0xEB0020: (co.parse_words, lambda w: struct.pack('>4I', *w)),
    co.T_HDR: (lambda d: None, lambda _x: ARENA_WORDS),
    co.T_INSTANCE: (co.parse_instances, encode_instances),
    co.T_IMPORTS: (co.parse_imports, encode_imports),
}


def _splines():
    from . import splines as sp
    return sp.parse, sp.encode


CODECS[co.T_SPLINES] = _splines()


def reencode(type_id, data):
    """decode + encode one object; None when the type has no codec"""
    c = CODECS.get(type_id)
    return None if c is None else c[1](c[0](data))


# =============================================================================================
# Builder: triangles -> 0x80006 (own clustering + KD tree, retail layout rules)
EDGE_FLAT = 26
EDGE_CONVEX = 0x20
EDGE_VERTEX_OFF = 0x40          # bit 0x40 of edge byte k: vertex k of the triangle disabled


def edge_code(theta):
    """5-bit angle code measured on retail edges: floor(-2 log2(theta / 90deg)) + 2, clamped 0..26
    (theta = angle between the two face normals; 26 = flat)"""
    if theta <= 1e-9:
        return EDGE_FLAT
    return int(max(0, min(EDGE_FLAT, math.floor(-2 * math.log2(theta / (math.pi / 2))) + 2)))


def dequantise(q, granularity=GRANULARITY):
    """the decoder's arithmetic: (int coordinate) * f32 granularity, rounded to f32"""
    return (np.asarray(q, np.int64) * np.float64(np.float32(granularity))).astype(np.float32)


def edge_bytes(Q, T, enable_vertices=True):
    """retail-style edge bytes for triangles T over integer positions Q (welded by position)"""
    V = Q.astype(np.float64)
    N = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-30)[:, None]
    key = {}
    W = np.array([key.setdefault(tuple(q), len(key)) for q in Q.tolist()], np.int64)
    TT = W[T]
    emap = {}
    for t in range(len(TT)):
        for k in range(3):
            a, b = int(TT[t, k]), int(TT[t, (k + 1) % 3])
            emap.setdefault((min(a, b), max(a, b)), []).append((t, k))
    E = np.zeros((len(T), 3), np.uint8)
    conv = np.zeros(len(key), np.int64)
    conc = np.zeros(len(key), bool)
    bound = np.zeros(len(key), bool)
    for t in range(len(TT)):
        for k in range(3):
            a, b = int(TT[t, k]), int(TT[t, (k + 1) % 3])
            nb = [x for x in emap[(min(a, b), max(a, b))] if x[0] != t]
            if len(nb) != 1:
                E[t, k] = EDGE_FLAT
                bound[a] = bound[b] = True
                continue
            u = nb[0][0]
            th = math.acos(float(np.clip(np.dot(N[t], N[u]), -1, 1)))
            code = edge_code(th)
            opp = [int(T[u, j]) for j in range(3) if TT[u, j] not in (a, b)]
            side = float(np.dot(V[opp[0]] - V[T[t, 0]], N[t])) if opp else 0.0
            convex = code < EDGE_FLAT and side < 0
            E[t, k] = code | (EDGE_CONVEX if convex else 0)
            if convex:
                conv[a] += 1
                conv[b] += 1
            elif code < EDGE_FLAT:
                conc[a] = conc[b] = True
    # vertex k enabled only for closed, purely convex corners (>= 3 sharp convex edges); retail enables
    # ~4% of vertices with a rule not pinned down (specs/sk3_collision.md), never on open borders.
    corner = (conv // 2 >= 3) & ~conc & ~bound if enable_vertices else np.zeros(len(key), bool)
    for t in range(len(TT)):
        for k in range(3):
            if not corner[TT[t, k]]:
                E[t, k] |= EDGE_VERTEX_OFF
    return E


def _kd(Pmin, Pmax, C, idx, leaf_max, nodes, leaves, parent):
    """recursive median split on centroids; appends branch nodes in preorder"""
    if len(idx) <= leaf_max or np.ptp(C[idx], 0).max() <= 0:
        return None
    ext = np.ptp(C[idx], 0)
    axis = int(np.argmax(ext))
    order = idx[np.argsort(C[idx, axis], kind='stable')]
    h = len(order) // 2
    L, R = order[:h], order[h:]
    me = len(nodes)
    nodes.append([parent, axis, None, None, float(Pmax[L, axis].max()), float(Pmin[R, axis].min())])
    for side, part in ((2, L), (3, R)):
        sub = _kd(Pmin, Pmax, C, part, leaf_max, nodes, leaves, me)
        if sub is None:
            leaves.append(part)
            nodes[me][side] = ('leaf', len(leaves) - 1)
        else:
            nodes[me][side] = ('node', sub)
    return me


def build_mesh(V, T, surface=None, granularity=0.001, leaf_max=8, max_cluster_verts=255,
               edges=None, enable_vertices=False):
    """triangles (V float (n,3), T int (m,3)), per-triangle surface ids (u16, or -1 / None = no field)
    -> (ClusteredMesh, report). Degenerate triangles after quantisation are dropped."""
    g = np.float32(granularity)
    Q = np.round(np.asarray(V, np.float64) / np.float64(g)).astype(np.int64)
    T = np.asarray(T, np.int64).reshape(-1, 3)
    S = np.full(len(T), -1, np.int64) if surface is None else np.asarray(surface, np.int64)
    Vq = dequantise(Q, g).astype(np.float64)
    area = np.linalg.norm(np.cross(Vq[T[:, 1]] - Vq[T[:, 0]], Vq[T[:, 2]] - Vq[T[:, 0]]), axis=1)
    same = (Q[T[:, 0]] == Q[T[:, 1]]).all(1) | (Q[T[:, 1]] == Q[T[:, 2]]).all(1) | (Q[T[:, 0]] == Q[T[:, 2]]).all(1)
    keep = (area > 0) & ~same
    T, S = T[keep], S[keep]
    E = edge_bytes(Q, T, enable_vertices) if edges is None else np.asarray(edges, np.uint8)[keep]
    P = Vq[T]                                  # (m, 3, 3) dequantised (f32 values)
    Pmin, Pmax, C = P.min(1), P.max(1), P.mean(1)
    nodes, leaves = [], []
    root = _kd(Pmin, Pmax, C, np.arange(len(T)), leaf_max, nodes, leaves, 0)
    if root is None:
        leaves = [np.arange(len(T))]
    # leaves in DFS left-first order == append order of _kd (it recurses left before right)
    clusters, leaf_ids = [], []
    cur_units, cur_vmap, cur_bytes = [], {}, 0
    cur_lo = cur_hi = None

    def flush():
        if cur_units:
            clusters.append((list(cur_units), dict(cur_vmap)))

    for lf in leaves:
        need = {int(v) for t in lf for v in T[t]} - set(cur_vmap)
        lbytes = sum(9 if S[t] >= 0 else 7 for t in lf)
        lq = Q[list(need)] if need else None
        span = False
        if cur_units and lq is not None:                 # keep clusters in 16-bit vertex mode when possible
            lo_ = np.minimum(cur_lo, lq.min(0)); hi_ = np.maximum(cur_hi, lq.max(0))
            span = (hi_ - lo_).max() > 0xFFFE and (cur_hi - cur_lo).max() <= 0xFFFE
        if cur_units and (len(cur_vmap) + len(need) > max_cluster_verts or cur_bytes + lbytes > 0xFFFF or span):
            flush()
            cur_units, cur_vmap, cur_bytes = [], {}, 0
        if lq is not None:
            if not cur_vmap:
                cur_lo, cur_hi = lq.min(0), lq.max(0)
            cur_lo, cur_hi = np.minimum(cur_lo, lq.min(0)), np.maximum(cur_hi, lq.max(0))
        leaf_ids.append((len(lf), len(clusters) << 16 | cur_bytes))
        for t in lf:
            for v in T[t]:
                cur_vmap.setdefault(int(v), len(cur_vmap))
            cur_units.append(int(t))
            cur_bytes += 9 if S[t] >= 0 else 7
        if len(cur_vmap) > max_cluster_verts:
            raise ValueError('a single leaf needs more than 255 vertices')
    flush()
    out_clusters = []
    for units, vmap in clusters:
        verts = np.array([Q[v] for v in sorted(vmap, key=vmap.get)], np.int64)
        lo = verts.min(0) - 1
        if (verts.max(0) - lo).max() <= 0xFFFF:
            mode, off, q = 1, tuple(int(x) for x in lo), verts - lo
        else:
            mode, off, q = 2, None, verts
        us = []
        for t in units:
            s = int(S[t])
            f = 0x21 | (co.UNIT_SURFACE if s >= 0 else 0)
            us.append((f, tuple(vmap[int(v)] for v in T[t]), tuple(int(x) for x in E[t]), s if s >= 0 else None))
        out_clusters.append(co.Cluster(mode, off, q, us, 0))
    out_nodes = []
    for p, ax, a, b, e0, e1 in nodes:
        ch = []
        for kind, k in (a, b):
            ch.append((co.BRANCH, k) if kind == 'node' else leaf_ids[k])
        out_nodes.append(co.Node(p, ax, tuple(ch), (e0, e1)))
    allv = Vq[T].reshape(-1, 3)
    bb = (tuple(float(x) for x in allv.min(0)), tuple(float(x) for x in allv.max(0)))
    m = co.ClusteredMesh(bb, None, float(g), MESH_FLAGS, bb, out_nodes, out_clusters,
                         node_ptr=0x90, n_entries=len(T))
    rep = dict(triangles=len(T), dropped=int((~keep).sum()), clusters=len(out_clusters),
               branch_nodes=len(out_nodes), leaves=len(leaves), mode2=sum(c.vertex_mode == 2 for c in out_clusters))
    return m, rep


# =============================================================================================
# Tile arena builder: the object order and links of a retail tile collision arena (cSim_<x>_<z>
# arena 0): eb0008, eb000d (one instance, identity), 0x80001 (aggregate volume -> mesh), 0x80006,
# eb000a (model -> volume), [eb0004 splines], eb000b. Subrefs: the instance matrix, each spline entry.
INSTANCE_KIND = 0xACB31C9A
IMPORT_TYPES = (0xeb0066, 0xeb0005, 0xeb0067, 0xeb0006, 0xeb0001, 0xeb000a, 0xeb0065, 0xeb0007,
                0xeb0069, 0xeb000d, 0xeb006b, 0xeb0019, 0xeb0064, 0xeb0004, 0xeb0068, 0xeb0009,
                0xeb0016, 0xeb0013, 0xeb0014, 0xeb0018, 0xeb0017, 0xeb0020, 0xeb0024, 0xeb0026, 0xeb0027)
IDENTITY = (1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0)


def build_tile_arena(mesh: co.ClusteredMesh, arena_id: int, name: str, export_id: int,
                     spline_set=None, splines_id: int = 0, imports=(), instance_ids=None):
    """-> (arena bytes, rw4_write.Spec). export_id: the eb000b row naming the instance (retail: a
    different u64 from the instance's own id, instance_ids[0]). `imports`: extra (id, type, value) rows for eb000b (retail
    tiles list external eb0069 / eb0064 references there - their targets are an open question)."""
    from . import splines as sp
    ids = instance_ids or (export_id, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
    inst = co.InstanceList(INSTANCE_KIND, [co.Instance(IDENTITY, mesh.bbox, ids, 4,
                                                        (name, 'undefined', 'undefined', 'undefined'))])
    objs = [rw4_write.SpecObject(co.T_HDR, 16, ARENA_WORDS),
            rw4_write.SpecObject(co.T_INSTANCE, 16, encode_instances(inst)),
            rw4_write.SpecObject(co.T_VOLUME, 16, encode_volume(co.Volume(
                (1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0), 6, 3, (0, 0, 0, 0, 0xFFFFFFFF, 1)))),
            rw4_write.SpecObject(co.T_MESH, 16, encode_mesh(mesh)),
            rw4_write.SpecObject(co.T_MESHREF, 16, encode_model(co.CollisionModel(0, [2])))]
    subrefs = [(1, 0x20)]
    rows = list(imports) + [(export_id, co.T_INSTANCE, 1)]
    if spline_set is not None and spline_set.splines:
        k = len(objs)
        objs.append(rw4_write.SpecObject(sp.T_SPLINES, 16, sp.encode(spline_set)))
        subrefs += sp.subrefs(k, spline_set)
        rows.append((splines_id, sp.T_SPLINES, k))
    objs.append(rw4_write.SpecObject(co.T_IMPORTS, 16, encode_imports(co.ImportTable(rows, list(IMPORT_TYPES)))))
    spec = rw4_write.Spec(arena_id, objs, subrefs)
    return rw4_write.build(spec), spec
