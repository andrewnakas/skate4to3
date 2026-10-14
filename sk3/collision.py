"""Skate 3 collision (cSim_* streams): the clustered triangle mesh (type 0x80006), its volume wrapper
(0x80001) and the small objects that ride along in cSim arenas. Every layout fact is measured on
retail arenas and written up in specs/sk3_collision.md; the encoders in sk3/collision_write.py
rebuild every retail object byte for byte from what is decoded here.

Clustered mesh 0x80006 (big-endian, offsets from the object start):
    0x00 bbox min xyz, pad   0x10 bbox max xyz, pad
    0x20 u32 0   0x24 u32 unit-index bits = bit_length(unit count + 1)   0x28 u32 unit count   0x2C 0
    0x30 u32 tree offset (0x60)   0x34 u32 cluster-table offset   0x38 f32 vertex granularity (0.001)
    0x3C u32 0x00100202 (unit-id layout: 16 offset bits; 2-byte group / surface fields)
    0x40 u32 cluster count  0x44 u32 cluster capacity (= count)  0x48 u32 unit count  0x4C u32 unit capacity
    0x50 u32 object size    0x54 0   0x58 0x80000000   0x5C 0
    0x60 KD tree: u32 node offset (0x90; stale pointer when there are no branch nodes), u32 branch count,
         u32 entry count (= unit count), u32 0, bbox min xyz pad, bbox max xyz pad
    0x90 branch nodes, 32 B: u32 parent, u32 axis, (u32 content, u32 index) x2, f32 left max, f32 right min
         content 0xFFFFFFFF = child branch node `index`; else a leaf of `content` units starting at
         unit id `index` = cluster << 16 | byte offset into that cluster's unit data
    cluster table: u32 offset per cluster, zero padded to 16
    clusters, each 16-aligned, zero padded (object size is 16-aligned):
        u16 unit count, u16 unit bytes, u16 unit start (x16 from +0x10), u16 normal start (x16, = unit start),
        u16 cluster size, u8 vertex count, u8 normal count (0), u8 vertex mode, 3 zero bytes
        +0x10 vertices: mode 1: i32 x3 offset then u16 x3 per vertex; mode 2: i32 x3 per vertex;
              world = (offset + v) * granularity; zero padded to the unit start
        units: u8 flags (0xA1 / 0x21), u8 v0 v1 v2, u8 edge0 edge1 edge2, [u16 surface id when flags & 0x80]
        edge byte k (edge v_k -> v_k+1): low 5 bits angle code floor(-2 log2(theta/90deg)) + 2 (0..26, 26 flat
        or unmatched), 0x20 convex edge, 0x40 vertex v_k disabled (all corners of a vertex agree)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

T_VOLUME, T_MESH = 0x80001, 0x80006
T_IMPORTS, T_HDR, T_INSTANCE, T_MESHREF = 0xeb000b, 0xeb0008, 0xeb000d, 0xeb000a
T_SPLINES, T_PROPS = 0xeb0004, 0xeb001d

UNIT_TRI = 0x01
UNIT_EDGES = 0x20
UNIT_GROUP = 0x40
UNIT_SURFACE = 0x80
BRANCH = 0xFFFFFFFF


def _u32(d, o):
    return struct.unpack_from('>I', d, o)[0]


@dataclass
class Cluster:
    vertex_mode: int                 # 1: i32 offset + u16 per axis, 2: i32 per axis
    offset: tuple                    # mode 1: i32 x3; mode 2: None
    qverts: np.ndarray               # (n, 3) int64 quantised coordinates (before the offset)
    units: list                      # (flags, (v0, v1, v2), (e0, e1, e2), surface or None)
    normal_count: int = 0

    def vertices(self, granularity):
        off = np.array(self.offset or (0, 0, 0), dtype=np.int64)
        return ((self.qverts + off) * np.float64(np.float32(granularity))).astype(np.float32)


@dataclass
class Node:
    parent: int
    axis: int
    children: tuple                  # ((content, index), (content, index))
    extents: tuple                   # (left max, right min) as f32 bit patterns kept exact via floats


@dataclass
class ClusteredMesh:
    bbox: tuple                      # (min xyz, max xyz) floats
    max_leaf: int                    # 0x24 unit-index bits (None = derive on encode)
    granularity: float
    flags: int                       # 0x3C
    tree_bbox: tuple
    nodes: list
    clusters: list
    node_ptr: int = 0x90             # tree +0x00 (a stale pointer when there are no branch nodes)
    n_entries: int = 0               # tree +0x08 (= unit count)

    # --- convenience -------------------------------------------------------------------------
    def unit_offsets(self, ci):
        """byte offset of each unit of cluster ci inside its unit data"""
        out, o = [], 0
        for f, _v, _e, _s in self.clusters[ci].units:
            out.append(o)
            o += unit_size(f)
        return out

    def triangles(self):
        """(V (n,3) f32, T (m,3) int, surface (m,) int (-1 none), edges (m,3) u8, unit ids (m,))"""
        V, T, S, E, U, base = [], [], [], [], [], 0
        for ci, c in enumerate(self.clusters):
            V.append(c.vertices(self.granularity))
            offs = self.unit_offsets(ci)
            for (f, v, e, s), o in zip(c.units, offs):
                T.append([base + x for x in v])
                S.append(-1 if s is None else s)
                E.append(e)
                U.append(ci << 16 | o)
            base += len(c.qverts)
        V = np.concatenate(V) if V else np.zeros((0, 3), np.float32)
        return (V, np.array(T, np.int64).reshape(-1, 3), np.array(S, np.int64),
                np.array(E, np.uint8).reshape(-1, 3), np.array(U, np.int64))

    def leaves(self):
        """[(count, first unit id, (bbmin, bbmax))] walking the KD tree from the root"""
        out = []
        if not self.nodes:
            return [(self.n_entries, 0, tuple(np.array(b, np.float64) for b in self.tree_bbox))]
        stack = [(0, np.array(self.tree_bbox[0], np.float64), np.array(self.tree_bbox[1], np.float64))]
        while stack:
            k, lo, hi = stack.pop()
            n = self.nodes[k]
            for side, (content, index) in enumerate(n.children):
                clo, chi = lo.copy(), hi.copy()
                if side == 0:
                    chi[n.axis] = n.extents[0]
                else:
                    clo[n.axis] = n.extents[1]
                if content == BRANCH:
                    stack.append((index, clo, chi))
                else:
                    out.append((content, index, (clo, chi)))
        return out


def unit_size(flags):
    nv = 3 if flags & 0x0F == UNIT_TRI else 4
    return 1 + nv + (nv if flags & UNIT_EDGES else 0) + (2 if flags & UNIT_GROUP else 0) + \
        (2 if flags & UNIT_SURFACE else 0)


def parse_mesh(d: bytes) -> ClusteredMesh:
    f = lambda o, n: struct.unpack_from(f'>{n}f', d, o)
    bmin, bmax = f(0x00, 3), f(0x10, 3)
    max_leaf = _u32(d, 0x24)
    tree_at, ctab = _u32(d, 0x30), _u32(d, 0x34)
    gran = f(0x38, 1)[0]
    flags = _u32(d, 0x3C)
    ncl = _u32(d, 0x40)
    node_ptr, nbranch, nent = struct.unpack_from('>III', d, tree_at)
    tmin, tmax = f(tree_at + 0x10, 3), f(tree_at + 0x20, 3)
    nodes = []
    for k in range(nbranch):
        p, ax, c0, i0, c1, i1 = struct.unpack_from('>6I', d, node_ptr + 32 * k)
        e0, e1 = f(node_ptr + 32 * k + 24, 2)
        nodes.append(Node(p, ax, ((c0, i0), (c1, i1)), (e0, e1)))
    clusters = []
    for p in struct.unpack_from(f'>{ncl}I', d, ctab):
        uc, ub, ust, nst, tsz, vc, nc, mode = struct.unpack_from('>5HBBB', d, p)
        v = p + 0x10
        if mode == 1:
            off = struct.unpack_from('>3i', d, v)
            q = np.frombuffer(d, '>u2', vc * 3, v + 12).reshape(vc, 3).astype(np.int64)
        elif mode == 2:
            off = None
            q = np.frombuffer(d, '>i4', vc * 3, v).reshape(vc, 3).astype(np.int64)
        else:
            raise ValueError(f'vertex mode {mode}')
        units, u = [], v + 16 * ust
        for _ in range(uc):
            fl = d[u]
            nv = 3 if fl & 0x0F == UNIT_TRI else 4
            vi = tuple(d[u + 1:u + 1 + nv])
            o = u + 1 + nv
            ed = tuple(d[o:o + nv]) if fl & UNIT_EDGES else ()
            o += len(ed)
            if fl & UNIT_GROUP:
                raise ValueError('group ids never seen in retail')
            s = struct.unpack_from('>H', d, o)[0] if fl & UNIT_SURFACE else None
            units.append((fl, vi, ed, s))
            u += unit_size(fl)
        clusters.append(Cluster(mode, off, q, units, nc))
    return ClusteredMesh((bmin, bmax), max_leaf, gran, flags, (tmin, tmax), nodes, clusters,
                         node_ptr=node_ptr, n_entries=nent)


# ---------------------------------------------------------------------------------------------
# 0x80001 volume: 3x4 transform (rows x, y, z, translation as 4 x 16 B), u32 volume type (6 = mesh),
# u32 dictionary index of the geometry object, then fixed words.
@dataclass
class Volume:
    matrix: tuple            # 16 floats (rows), last column always 0
    vtype: int
    geometry: int            # dictionary index of the 0x80006 object
    tail: tuple              # words 0x48..0x5F


def parse_volume(d: bytes) -> Volume:
    m = struct.unpack_from('>16f', d, 0)
    vt, gi = struct.unpack_from('>II', d, 0x40)
    return Volume(m, vt, gi, struct.unpack_from('>6I', d, 0x48))


def parse_box_or_aggregate(d: bytes):
    """0x80001: (matrix 16 f32, type, payload words)"""
    return parse_volume(d)


# ---------------------------------------------------------------------------------------------
# 0x80004 box aggregate: a list of inline volumes (oriented boxes) with a small KD tree.
#   0x00 bbox min/max   0x20 u32 0, u32 2, u32 volume count, u32 0
#   0x30 u32 volume array offset (0x50), 0 x3   0x40 u32 tree offset, 0 x3
#   0x50 volumes, 0x60 B each = a 0x80001 body (matrix, u32 type 4 = box, f32 half extents xyz, tail)
#   tree: u32 node ptr (stale when no branch), u32 branch count, u32 entry count, u32 0,
#         bbox min (pad 0), bbox max (its pad word is not cleared in retail: kept raw), branch nodes
@dataclass
class BoxAggregate:
    bbox: tuple
    w24: int
    volumes: list               # Volume
    node_ptr: int
    nodes: list                 # Node
    n_entries: int
    tree_bbox: tuple
    tree_pad: int               # raw pad word after tree bbox max


def parse_aggregate(d: bytes) -> BoxAggregate:
    f = lambda o, n: struct.unpack_from(f'>{n}f', d, o)
    _z, w24, n, _z2 = struct.unpack_from('>4I', d, 0x20)
    voff, toff = _u32(d, 0x30), _u32(d, 0x40)
    vols = [parse_volume(d[voff + 0x60 * k:voff + 0x60 * (k + 1)]) for k in range(n)]
    node_ptr, nb, ne = struct.unpack_from('>III', d, toff)
    nodes = []
    for k in range(nb):
        p, ax, c0, i0, c1, i1 = struct.unpack_from('>6I', d, toff + 0x30 + 32 * k)
        nodes.append(Node(p, ax, ((c0, i0), (c1, i1)), f(toff + 0x30 + 32 * k + 24, 2)))
    return BoxAggregate((f(0, 3), f(0x10, 3)), w24, vols, node_ptr, nodes, ne,
                        (f(toff + 0x10, 3), f(toff + 0x20, 3)), _u32(d, toff + 0x2C))


# ---------------------------------------------------------------------------------------------
# small link objects
#   eb0008 (16 B): u32 0x19, u32 0x0D, 0, 0 in every retail arena (arena-format words)
#   eb000a (20 B): u32 bounding-volume dict index (0 = none), u32 count (1), u32 0x0C, u32 volume index[count], u32 0
#   eb0020 (8+ B): u32 volume dict index, u32 instance (eb000d) dict index, 0, 0
def parse_words(d: bytes):
    return struct.unpack_from(f'>{len(d) // 4}I', d, 0)


@dataclass
class CollisionModel:            # eb000a
    bounding: int
    volumes: list


def parse_model(d: bytes) -> CollisionModel:
    b, n, off = struct.unpack_from('>III', d, 0)
    return CollisionModel(b, list(struct.unpack_from(f'>{n}I', d, off)))


# ---------------------------------------------------------------------------------------------
# eb000d instance list: header {u32 kind hash (0xacb31c9a / 0x7369d9b5), u32 count, u32 unique strings,
#   u32 0x20 (entries), u32 string pool offset, 0 x3}; entries 0xA0 B: matrix 4x4 (subref target),
#   bbox min/max, u64 id, 3 x u64 (ids or 0xFF..), u32 0, u32 flags, u32 0, u32 name ptr,
#   u32 str1, str2, str3 ptrs, u32 0; pool: NUL-terminated strings, unique, in order of first reference.
@dataclass
class Instance:
    matrix: tuple
    bbox: tuple
    ids: tuple                  # 4 x u64
    flags: int
    strings: tuple              # (name, s1, s2, s3)


@dataclass
class InstanceList:
    kind: int
    instances: list


def _cstr(d, o):
    return d[o:d.index(b'\0', o)].decode('latin-1')


def parse_instances(d: bytes) -> InstanceList:
    kind, n, _ns, eoff, _soff = struct.unpack_from('>5I', d, 0)
    out = []
    for k in range(n):
        e = eoff + 0xA0 * k
        m = struct.unpack_from('>16f', d, e)
        bb = (struct.unpack_from('>3f', d, e + 0x40), struct.unpack_from('>3f', d, e + 0x50))
        ids = struct.unpack_from('>4Q', d, e + 0x60)
        _a, fl, _b, p0, p1, p2, p3, _c = struct.unpack_from('>8I', d, e + 0x80)
        out.append(Instance(m, bb, ids, fl, tuple(_cstr(d, p) for p in (p0, p1, p2, p3))))
    return InstanceList(kind, out)


# ---------------------------------------------------------------------------------------------
# eb000b arena import/export table (also in cPres arenas):
#   {u32 n, u32 0x14, u32 table2 offset, u32 m, u32 table2 offset}, n x 24 B
#   {u32 0, u32 0xFEFFFFFF, u64 id, u32 type id, u32 value (dict index, or 0x00800000 | import k)},
#   then m x {u32 type id, u32 n} (the same 25 type ids in every retail cSim arena)
@dataclass
class ImportTable:
    entries: list               # (id, type, value)
    types: list                 # type ids (the same 25 in every retail cSim arena); each written with word n


def parse_imports(d: bytes) -> ImportTable:
    n, eo, _t2, m, t2 = struct.unpack_from('>5I', d, 0)
    ents = [struct.unpack_from('>QII', d, eo + 24 * k + 8) for k in range(n)]
    return ImportTable(ents, [struct.unpack_from('>I', d, t2 + 8 * k)[0] for k in range(m)])


# =============================================================================================
# Query harness: walks the bytes of a 0x80006 the way the layout implies (KD nodes -> leaf (count,
# unit id) -> cluster table -> unit bytes at that offset -> cluster vertices), no decoded lists.
class MeshQuery:
    def __init__(self, d: bytes):
        self.d = d
        self.g = np.float64(struct.unpack_from('>f', d, 0x38)[0])
        tree = _u32(d, 0x30)
        self.node_ptr, self.nb, self.ne = struct.unpack_from('>III', d, tree)
        self.bmin = np.array(struct.unpack_from('>3f', d, tree + 0x10), np.float64)
        self.bmax = np.array(struct.unpack_from('>3f', d, tree + 0x20), np.float64)
        self.ctab = _u32(d, 0x34)
        self._vcache = {}

    def _cluster(self, ci):
        if ci not in self._vcache:
            p = _u32(self.d, self.ctab + 4 * ci)
            _uc, _ub, ust, _ns, _ts, vc, _nc, mode = struct.unpack_from('>5HBBB', self.d, p)
            if mode == 1:
                off = np.array(struct.unpack_from('>3i', self.d, p + 16), np.int64)
                q = np.frombuffer(self.d, '>u2', vc * 3, p + 28).reshape(vc, 3).astype(np.int64) + off
            else:
                q = np.frombuffer(self.d, '>i4', vc * 3, p + 16).reshape(vc, 3).astype(np.int64)
            V = (q * np.float64(np.float32(self.g))).astype(np.float32).astype(np.float64)
            self._vcache[ci] = (p + 16 + 16 * ust, V)
        return self._vcache[ci]

    def leaf_triangles(self, count, uid):
        """[(unit id, (3,3) vertices, surface or None)] for a leaf"""
        ci, o = uid >> 16, uid & 0xFFFF
        base, V = self._cluster(ci)
        out = []
        for _ in range(count):
            u = base + o
            f = self.d[u]
            v = list(self.d[u + 1:u + 4])
            s = struct.unpack_from('>H', self.d, u + 7)[0] if f & UNIT_SURFACE else None
            out.append((ci << 16 | o, V[v], s))
            o += unit_size(f)
        return out

    def leaves_hit(self, test):
        """walk the tree; test(lo, hi) -> bool decides whether a cell is visited"""
        out = []
        if self.nb == 0:
            return [(self.ne, 0, self.bmin, self.bmax)] if test(self.bmin, self.bmax) else []
        stack = [(0, self.bmin.copy(), self.bmax.copy())]
        ar = np.arange(3)
        while stack:
            k, lo, hi = stack.pop()
            p, ax, c0, i0, c1, i1 = struct.unpack_from('>6I', self.d, self.node_ptr + 32 * k)
            e0, e1 = struct.unpack_from('>2f', self.d, self.node_ptr + 32 * k + 24)
            for (c, i), (clo, chi) in (((c0, i0), (lo, np.where(ar == ax, e0, hi))),
                                       ((c1, i1), (np.where(ar == ax, e1, lo), hi))):
                if not test(clo, chi):
                    continue
                if c == BRANCH:
                    stack.append((i, clo, chi))
                elif c:
                    out.append((c, i, clo, chi))
        return out

    def ray_down(self, x, z, y0=1e9):
        """highest triangle hit at or below y0 on the vertical line (x, z): (y, unit id, surface) or None"""
        eps = 1e-6
        test = lambda lo, hi: lo[0] - eps <= x <= hi[0] + eps and lo[2] - eps <= z <= hi[2] + eps and lo[1] <= y0
        best = None
        for c, uid, _lo, _hi in self.leaves_hit(test):
            for u, P, s in self.leaf_triangles(c, uid):
                y = vertical_hit(P, x, z)
                if y is not None and y <= y0 and (best is None or y > best[0]):
                    best = (y, u, s)
        return best

    def cells_containing(self, p, eps=0.0):
        return self.leaves_hit(lambda lo, hi: bool(np.all(lo - eps <= p) and np.all(p <= hi + eps)))


def vertical_hit(P, x, z):
    """height of triangle P (3,3) at (x, z), or None (barycentric test in the xz projection)"""
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = P
    d = (x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)
    if abs(d) < 1e-14:
        return None
    l1 = ((x - x0) * (z2 - z0) - (x2 - x0) * (z - z0)) / d
    l2 = ((x1 - x0) * (z - z0) - (x - x0) * (z1 - z0)) / d
    l0 = 1 - l1 - l2
    t = -1e-9
    if l0 < t or l1 < t or l2 < t:
        return None
    return l0 * y0 + l1 * y1 + l2 * y2
