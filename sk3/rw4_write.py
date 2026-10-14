"""RW4 arena writer (Xbox 360, 'xb2'): build(spec) -> bytes, byte-exact against every retail world
arena measured so far (tests/test_rw4_roundtrip.py; layout facts in specs/SK3_FORMATS.md "RW4 arena").

A spec holds only what a generator knows: the arena id, the objects in dictionary order as
(type_id, align, data), and the subref records (object index, byte offset). Everything else - the
type registry, every header word, the section block, object/dictionary/subref/graphics placement -
is derived here. `header_overrides` exists for experiments only; no retail arena needs it.

Layout written (all big-endian):
    0x00  header, 0xAC bytes
    0xAC  sections: directory 0x10004, types 0x10005, arena guid 0x10006, subrefs 0x10007, atoms 0x10008
          main objects in dictionary order, each at its own alignment, zero padding
          dictionary, 24 B per object {ptr, 0, size, align, typeIndex, typeId}, right after the last object
          subref records {u32 object index, u32 offset}, then 24 zero bytes per record
    h44   graphics region: base resources (types 0x10030-0x1003F) in dictionary order, ptr relative
          to h44, each at its own alignment, zero padding, no tail padding
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

HEADER_SIZE = 0xAC
MAGIC = b'\x89RW4'
# 0x08-0x17: identical in every retail arena
FIXED_08 = b'\r\n\x1a\n\x01\x20\x04\x00454\x00000\x00'

# The two type registries retail arenas carry. Each arena uses the first (shortest) one that holds
# every type its objects use; each comes with its own header 0x6C word.
REGISTRY_SMALL = (0x0, 0x10030, 0x10031, 0x10032, 0x10033, 0x10010, 0x200e8, 0xeb0008, 0xeb000b)
REGISTRY_FULL = (
    0x0, 0x10030, 0x10031, 0x10032, 0x10033, 0x10010,
    0xeb0000, 0xeb0001, 0xeb0003, 0xeb0004, 0xeb0005, 0xeb0006, 0xeb000a, 0xeb000d, 0xeb0019,
    0xeb0007, 0xeb0008, 0xeb000c, 0xeb0009, 0xeb000b, 0xeb000e, 0xeb0011, 0xeb000f, 0xeb0010,
    0xeb0012, 0xeb0022, 0xeb0013, 0xeb0014, 0xeb0015, 0xeb0016, 0xeb001a, 0xeb001c, 0xeb001d,
    0xeb001b, 0xeb001e, 0xeb001f, 0xeb0021, 0xeb0017, 0xeb0020, 0xeb0024, 0xeb0023, 0xeb0025,
    0xeb0026, 0xeb0027, 0xeb0028, 0xeb0029, 0xeb0018, 0xec0010, 0x10000, 0x10002, 0x200eb,
    0x200ea, 0x200e9, 0x20081, 0x200e8, 0x80002, 0x80001, 0x80006, 0x80003, 0x80004, 0x40006,
    0x40007, 0x1000f)
REGISTRIES = ((REGISTRY_SMALL, 0x304), (REGISTRY_FULL, 0xAC))   # (types, header 0x6C)


def is_base_resource(type_id):
    return 0x10030 <= type_id <= 0x1003F


def _align(x, a):
    return (x + a - 1) // a * a


@dataclass
class SpecObject:
    type_id: int
    align: int
    data: bytes


@dataclass
class Spec:
    arena_id: int
    objects: list                      # SpecObject, dictionary order
    subrefs: list = field(default_factory=list)   # (object index, byte offset)
    types: tuple | None = None         # None: derived (shortest known registry covering the objects)
    header_overrides: dict = field(default_factory=dict)   # {offset: u32}; empty for retail


def pick_registry(type_ids):
    used = set(type_ids)
    for reg, h6c in REGISTRIES:
        if used <= set(reg):
            return reg, h6c
    raise ValueError(f'no known registry holds types {sorted(hex(t) for t in used - set(REGISTRY_FULL))}')


def build(spec: Spec) -> bytes:
    objs = spec.objects
    if spec.types is None:
        types, h6c = pick_registry(o.type_id for o in objs)
    else:
        types = tuple(spec.types)
        h6c = dict((r, v) for r, v in REGISTRIES).get(types, 0xAC)
    tindex = {t: i for i, t in enumerate(types)}
    n, ns = len(objs), len(spec.subrefs)

    # section block
    sec = HEADER_SIZE
    sizes = [0x0C + 4 * 4, 0x0C + 4 * len(types), 0x24, 0x1C, 0x0C]
    offs = [sum(sizes[:i]) for i in range(5)]           # relative to the block
    atoms_end = sec + sum(sizes)

    # main objects
    ptrs, p = [0] * n, atoms_end
    for i, o in enumerate(objs):
        if not is_base_resource(o.type_id):
            p = _align(p, o.align)
            ptrs[i] = p
            p += len(o.data)
    dict_at = _align(p, 4)          # always == p in retail (last main object ends 4-aligned)
    rec_at = dict_at + 24 * n
    rec_end = rec_at + 8 * ns
    main_end = rec_end + 24 * ns

    # graphics region
    g = 0
    for i, o in enumerate(objs):
        if is_base_resource(o.type_id):
            g = _align(g, o.align)
            ptrs[i] = g
            g += len(o.data)
    br_aligns = [o.align for o in objs if is_base_resource(o.type_id)]
    main_aligns = [o.align for o in objs if not is_base_resource(o.type_id)]

    out = bytearray(main_end + g)
    W = lambda off, *v: struct.pack_into(f'>{len(v)}I', out, off, *v)

    # header
    out[0:4] = MAGIC
    out[4:8] = b'xb2\0'
    out[8:0x18] = FIXED_08
    H = {0x1C: spec.arena_id, 0x20: n, 0x24: n, 0x28: 0x10, 0x30: dict_at, 0x34: sec,
         0x44: main_end, 0x48: max(main_aligns, default=0x10), 0x50: 1,
         0x54: g, 0x58: max(br_aligns, default=1), 0x60: 1, 0x68: 1, 0x6C: h6c, 0x70: 4,
         0x78: 1, 0x80: 1, 0x88: 1, 0x90: 1}
    H.update(spec.header_overrides)
    for off, v in H.items():
        W(off, v)

    # sections
    W(sec, 0x10004, 4, 0x0C, *offs[1:])
    W(sec + offs[1], 0x10005, len(types), 0x0C, *types)
    W(sec + offs[2], 0x10006, 3, 0x18, spec.arena_id, 0xFFB00000, spec.arena_id, 0, 0, 0)
    W(sec + offs[3], 0x10007, ns, 0, 0, rec_end, rec_at, ns)
    W(sec + offs[4], 0x10008, 0, 0)

    # objects + dictionary
    for i, o in enumerate(objs):
        at = (main_end if is_base_resource(o.type_id) else 0) + ptrs[i]
        out[at:at + len(o.data)] = o.data
        W(dict_at + 24 * i, ptrs[i], 0, len(o.data), o.align, tindex[o.type_id], o.type_id)
    for k, (idx, off) in enumerate(spec.subrefs):
        W(rec_at + 8 * k, idx, off)
    return bytes(out)


def spec_from_arena(a) -> Spec:
    """Everything a generator would supply, read back from a parsed arena (sk3.rw4.Arena)."""
    return Spec(arena_id=struct.unpack_from('>I', a.data, 0x1C)[0],
                objects=[SpecObject(o.type_id, o.align, o.data) for o in a.objects],
                subrefs=[tuple(s) for s in a.subrefs])
