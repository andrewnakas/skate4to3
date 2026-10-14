"""Frostbite MeshSet res (type 0x49B156D4) for skate.

Relocatable blob: every u64 pointer is relative to +0x10, and resMeta's first two
u32s are (relocTableOffset, relocTableSize) listing the pointer fields.

    +0x10 AABB (min vec3+pad, max vec3+pad)
    +0x30 LOD pointers, u64 each, zero-terminated
    LOD record (0xB0):  +0x08 u32 sectionCount   +0x0C u64 sections
                        +0x58 u32 indexBufferSize  +0x5C u32 vertexBufferSize
                        +0x74 guid chunk (VB then IB, in one chunk)
                        +0x8C u64 name
    section (0x180):    +0x08 u64 materialName
                        +0x138 u64 shaderBlockDepotKey - the material's key into its bundle's
                               ShaderBlockDepot, which is where skate. actually stores the
                               texture binding. MeshMaterial.Shader in the mesh EBX is null on
                               every asset, so this key is the only route to the real textures.
                        +0x20 u32 primitiveCount  +0x24 u32 startIndex
                        +0x28 u32 vertexOffset (bytes into VB)  +0x2C u32 vertexCount
                        +0x70 16 x {u8 usage, u8 format, u8 offset, u8 stream}
                        +0xB0 8 x {u8 stride, u8 _}   +0xD0 u8 elementCount, u8 streamCount
                        +0x150 AABB

Vertex data is multi-stream: for a section, stream k occupies vertexCount*stride[k]
bytes, streams back to back from vertexOffset. The index buffer starts at
vertexBufferSize. Verified on cmn_prp_dirtpile_01_mesh LOD1 byte-exact: every
index in range, zero degenerate triangles, positions inside the section AABB.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

BASE = 0x10
USAGE_POS, USAGE_UV0 = 0x01, 0x21
FORMATS = {  # format -> (numpy dtype, components)
    0x02: ('<f4', 2), 0x03: ('<f4', 3), 0x04: ('<f4', 4),
    0x06: ('<f2', 2), 0x07: ('<f2', 3), 0x08: ('<f2', 4),
}


@dataclass
class Section:
    material: str
    primitives: int
    start_index: int
    vertex_offset: int
    vertices: int
    elements: list      # [(usage, format, offset, stream)]
    strides: list
    aabb: tuple
    depot_key: int = 0

    @property
    def is_shadow(self) -> bool:
        return self.material.endswith('_Shadow') or all(e[0] == USAGE_POS for e in self.elements)


@dataclass
class Lod:
    name: str
    chunk: str
    index_buffer_size: int
    vertex_buffer_size: int
    sections: list


@dataclass
class MeshSet:
    name: str
    aabb: tuple
    lods: list


def _cstr(d, o):
    return d[o:d.index(b'\0', o)].decode('utf-8', 'replace')


def parse(d: bytes) -> MeshSet:
    u32 = lambda o: struct.unpack_from('<I', d, o)[0]
    u64 = lambda o: struct.unpack_from('<Q', d, o)[0]
    aabb = struct.unpack_from('<3f4x3f', d, BASE)
    lods = []
    for k in range(8):
        v = u64(BASE + 0x20 + 8 * k)
        if v == 0 or v + BASE >= len(d):
            break
        L = v + BASE
        nsec, sptr = u32(L + 0x08), u64(L + 0x0C) + BASE
        ibs, vbs = struct.unpack_from('<II', d, L + 0x58)
        secs = []
        for s in range(nsec):
            S = sptr + s * 0x180
            prim, start, voff, vcnt = struct.unpack_from('<4I', d, S + 0x20)
            ne, ns = d[S + 0xD0], d[S + 0xD1]
            secs.append(Section(
                _cstr(d, u64(S + 0x08) + BASE), prim, start, voff, vcnt,
                [struct.unpack_from('<4B', d, S + 0x70 + 4 * e) for e in range(ne)],
                [d[S + 0xB0 + 2 * j] for j in range(ns)],
                struct.unpack_from('<3f4x3f', d, S + 0x150),
                u64(S + 0x138) if S + 0x140 <= len(d) else 0))
        lods.append(Lod(_cstr(d, u64(L + 0x8C) + BASE), d[L + 0x74:L + 0x84].hex(), ibs, vbs, secs))
    name = lods[0].name.rsplit('_lod', 1)[0].removeprefix('Mesh:') if lods else ''
    return MeshSet(name, aabb, lods)


def index_width(lod: Lod) -> int:
    total = max((s.start_index + 3 * s.primitives for s in lod.sections), default=0)
    return 4 if total * 4 <= lod.index_buffer_size else 2


def decode(chunk: bytes, lod: Lod, sec: Section):
    """-> (positions (n,3) float32, uv0 (n,2) float32 | None, triangles (m,3) uint32)."""
    streams, pos = [], sec.vertex_offset
    for st in sec.strides:
        streams.append((pos, st))
        pos += sec.vertices * st
    out = {}
    for usage, fmt, off, stream in sec.elements:
        if usage not in (USAGE_POS, USAGE_UV0) or usage in out:
            continue
        if fmt not in FORMATS:
            if usage == USAGE_POS:
                raise ValueError(f"unsupported position format {fmt:#x}")
            continue
        dt, n = FORMATS[fmt]
        so, stride = streams[stream]
        block = np.frombuffer(chunk, dtype=np.uint8, count=sec.vertices * stride, offset=so)
        width = np.dtype(dt).itemsize * n
        vals = np.frombuffer(block.reshape(sec.vertices, stride)[:, off:off + width].tobytes(), dtype=dt)
        out[usage] = vals.reshape(sec.vertices, n).astype(np.float32)
    w = index_width(lod)
    ib = np.frombuffer(chunk, dtype='<u4' if w == 4 else '<u2', count=sec.primitives * 3,
                       offset=lod.vertex_buffer_size + sec.start_index * w)
    return out[USAGE_POS][:, :3], out.get(USAGE_UV0), ib.astype(np.uint32).reshape(-1, 3)
