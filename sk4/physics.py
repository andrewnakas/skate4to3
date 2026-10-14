"""skate. collision: the PhysicsAsset resource (res type 0x41759364, `<asset>.physics`).

skate. ships its own collision for 962 of the 1,125 meshes placed on the Isle of Grom, at
roughly LOD0 detail, while the render side often has only a far LOD resident. This reads it.
Not Havok: there are no hk* classes, and the EBX side is Frostbite `PhysicsAsset`
(`Resource` = this res's resId, `SourcePartIndexToMaterialMap`).

Layout, reverse-engineered 2026-09-13 on the bowl, a sidewalk, a bank and a convex climber.
Little-endian. Every offset field is SELF-RELATIVE: its target is the field's own position
plus its value.

    +0x00 u32 version (2 for a mesh's own asset, 0 for a sublevel's staticmodelgroup)
    +0x04 u32 magic 0x69AF7015   +0x18 u32 nDescriptors   +0x1C u32 nGroups
    +0x24 -> body frame (quaternion, inertia, centre of mass; unused for static geometry)
    +0x28 -> shape descriptors, 84 B each, shared: many parts may instance one
    +0x2C -> groups, 44 B each (nGroups of them; the part records follow)
    +0x30 -> part records, 72 B each, up to the +0x34 target
    +0x34 -> shape data (v2) / per-part world AABBs then shape data (v0)

A mesh's own asset is in that mesh's frame. A sublevel's `staticmodelgroup.physics` holds
every static-model-group instance of the sublevel with its placement already applied, in
Frostbite WORLD space (the root sublevel's: 10,289 parts over 2,041 shared descriptors).

    part record   +0 quaternion xyzw   +16 translation xyz   +28 scale   +36 u32 kind
                  (2 convex hull, 5 mesh)   +40 u64 shape tag   +48 f32 convex radius
                  +56 u32 descriptor index (0xFFFFFFFF for a primitive with no polytope)
                  +64 u32 packed material (== the EBX `Packed` value on a single-part asset)

    descriptor    a polytope: vertices, half-edges, polygonal faces
                  +0 AABB min   +12 AABB max (shape-local, centred on the part)
                  +24 u16 nVerts   +26 u16 nHalfEdges (saturates at 0x0FFF; use faces)
                  +28 u16 nFaces   +36 -> vertices f32[nVerts][3]
                  +40 -> per-vertex half-edge offsets u16[nVerts+1]
                  +44 -> twin half-edge u16[]   +48 -> half-edge origin vertex u16[]
                  +52 -> per-half-edge f32 (welding cosines; meshes only)
                  +64 -> face -> first half-edge u16[nFaces+1]   +76 -> face planes (hulls)

A face is the polygon origin[faceStart[f]:faceStart[f+1]]. Placed in the mesh's own frame by
R(quaternion) * (scale * v) + translation, which is the frame MeshSet positions use, so the
same instance transform places both.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x69AF7015
KIND_CONVEX, KIND_MESH = 2, 5
DESC_SIZE, PART_SIZE = 84, 72


@dataclass
class Part:
    kind: int
    material: int
    positions: np.ndarray      # (n, 3) float64, mesh-local
    triangles: np.ndarray      # (m, 3) int64
    faces: int
    polygon_sizes: dict


def _rel(d, o):
    v = struct.unpack_from('<I', d, o)[0]
    return o + v if v else 0


def _quat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse(d: bytes) -> list[Part]:
    magic = struct.unpack_from('<I', d, 4)[0]
    if magic != MAGIC:
        raise ValueError(f"not a skate. physics resource (magic {magic:#x})")
    ndesc = struct.unpack_from('<I', d, 0x18)[0]
    desc0, part0 = _rel(d, 0x28), _rel(d, 0x30)
    nparts = (_rel(d, 0x34) - part0) // PART_SIZE
    parts, cache = [], {}
    for k in range(nparts):
        P = part0 + k * PART_SIZE
        q = struct.unpack_from('<4f', d, P)
        t = np.array(struct.unpack_from('<3f', d, P + 16))
        s = struct.unpack_from('<f', d, P + 28)[0] or 1.0
        kind, = struct.unpack_from('<I', d, P + 36)
        di, = struct.unpack_from('<I', d, P + 56)
        material, = struct.unpack_from('<I', d, P + 64)
        if di >= ndesc:
            parts.append(Part(kind, material, np.zeros((0, 3)), np.zeros((0, 3), np.int64), 0, {}))
            continue
        if di not in cache:
            cache[di] = polytope(d, desc0 + di * DESC_SIZE)
        V, T, nf, sizes = cache[di]
        W = (s * V) @ _quat(*q).T + t
        parts.append(Part(kind, material, W, T, nf, sizes))
    return parts


def polytope(d, D):
    """-> (vertices (n,3) shape-local, triangles (m,3), faces, polygon-size histogram)."""
    nv, _nh, nf = struct.unpack_from('<3H', d, D + 24)
    vo, fo, oo = _rel(d, D + 36), _rel(d, D + 64), _rel(d, D + 48)
    V = np.frombuffer(d, '<f4', nv * 3, vo).reshape(-1, 3).astype(np.float64)
    first = np.frombuffer(d, '<u2', nf + 1, fo).astype(np.int64)
    origin = np.frombuffer(d, '<u2', int(first[-1]), oo).astype(np.int64)
    tris, sizes = [], {}
    for f in range(nf):
        poly = origin[first[f]:first[f + 1]]
        sizes[len(poly)] = sizes.get(len(poly), 0) + 1
        for i in range(1, len(poly) - 1):           # physics faces are convex: fan them
            tris.append((poly[0], poly[i], poly[i + 1]))
    return V, np.array(tris, dtype=np.int64).reshape(-1, 3), nf, sizes
