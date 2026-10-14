"""RW4 arenas ('\x89RW4' + 'xb2' on Xbox 360): the object container Skate 3 render, collision
and texture streams are made of. Every field here is checked against retail arenas
(specs/SK3_FORMATS.md "RW4 arena"; tests/test_rw4.py).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

DICT_ENTRY = 24
SEC_DIRECTORY, SEC_TYPES, SEC_GUID, SEC_SUBREFS, SEC_ATOMS = 0x10004, 0x10005, 0x10006, 0x10007, 0x10008


def u32(d, o):
    return struct.unpack_from('>I', d, o)[0]


@dataclass
class Obj:
    index: int
    ptr: int
    size: int
    align: int          # dictionary +0x0C: 0x1000 base resources, 4 / 0x10 / ... objects
    reloc: int          # dictionary +0x04 (0 in every retail arena so far)
    type_index: int
    type_id: int
    data: bytes

    @property
    def is_base_resource(self):
        return 0x10030 <= self.type_id <= 0x1003F


@dataclass
class Arena:
    data: bytes
    platform: str
    header: dict
    types: list
    objects: list = field(default_factory=list)
    subrefs: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)

    def of_type(self, t):
        return [o for o in self.objects if o.type_id == t]


def parse(d: bytes) -> Arena:
    if d[:4] != b'\x89RW4':
        raise ValueError('not an RW4 arena')
    plat = d[4:7].decode('ascii')
    H = {f'h{o:02x}': u32(d, o) for o in range(0x08, 0xAC, 4)}
    dict_start, sections = u32(d, 0x30), u32(d, 0x34)
    n_used = u32(d, 0x24)
    gfx_size = u32(d, 0x6C if plat == 'ps3' else 0x54)
    gbase = len(d) if gfx_size == 0 else u32(d, 0x44)
    secs = {}
    if u32(d, sections) == SEC_DIRECTORY:
        n, first = u32(d, sections + 4), sections + u32(d, sections + 8)
        for i in range(n):
            c = sections + u32(d, first + 4 * i)
            if c + 4 <= len(d):
                secs[u32(d, c)] = c
    types = []
    if SEC_TYPES in secs:
        t = secs[SEC_TYPES]
        cnt, arr = u32(d, t + 4), t + u32(d, t + 8)
        types = [u32(d, arr + 4 * i) for i in range(cnt)]
    a = Arena(d, plat, H, types, sections=secs)
    for i in range(n_used):
        b = dict_start + i * DICT_ENTRY
        ptr, reloc, size, align, ti, tid = struct.unpack_from('>IIIIII', d, b)
        if ti < len(types):
            tid = types[ti]
        start = (gbase if 0x10030 <= tid <= 0x1003F else 0) + ptr
        a.objects.append(Obj(i, ptr, size, align, reloc, ti, tid, d[start:start + size]))
    if SEC_SUBREFS in secs:
        s = secs[SEC_SUBREFS]
        cnt, at = u32(d, s + 4), u32(d, s + 0x14)
        a.subrefs = [struct.unpack_from('>II', d, at + 8 * i) for i in range(cnt)] if cnt else []
    return a
