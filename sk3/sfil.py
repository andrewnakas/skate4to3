"""SFIL stream files (.xsf): a header, then one aligned chunk per arena (specs/SK3_FORMATS.md "SFIL").

    0x00 'SFIL'  0x04 version 3  0x0C group hash  0x10 align  0x14 group count  0x18 platform  0x1C tail
    chunk: u64 asset id, u32 payload size, u32 payload offset (= align), u32 aligned chunk size
    payload: a raw RW4 arena ('\x89RW4'), or the Xbox two-section form (0x50-byte descriptor:
    main {u32 raw size, u32 stored, u32 compressed} at 0x00, graphics at 0x20; RefPack when compressed).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from . import refpack

MAGIC = b'SFIL'
XBOX_DESC = 0x50


@dataclass
class Asset:
    id: int
    arena: bytes
    form: str          # 'raw' | 'xbox2'


def payload_to_arena(p: bytes):
    if p[:4] == b'\x89RW4':
        return p, 'raw'
    main_raw, main_st, main_c = struct.unpack_from('>III', p, 0x00)
    gfx_raw, gfx_st, gfx_c = struct.unpack_from('>III', p, 0x20)
    m = p[XBOX_DESC:XBOX_DESC + main_st]
    g = p[XBOX_DESC + main_st:XBOX_DESC + main_st + gfx_st]
    m = refpack.decompress(m) if main_c else m
    g = (refpack.decompress(g) if gfx_c else g) if gfx_st else b''
    return m + g, 'xbox2'


def read(data: bytes):
    if data[:4] != MAGIC:
        raise ValueError('not an SFIL stream')
    version, _z, group, align, groups, platform, tail = struct.unpack_from('>IIIIIII', data, 4)
    hdr = {'version': version, 'group': group, 'align': align, 'groups': groups,
           'platform': platform, 'tail': tail}
    out, off = [], align
    while off + 0x20 <= len(data):
        aid = struct.unpack_from('>Q', data, off)[0]
        size, doff, csize = struct.unpack_from('>III', data, off + 8)
        if not size or not doff or off + doff + size > len(data):
            break
        arena, form = payload_to_arena(data[off + doff:off + doff + size])
        out.append(Asset(aid, arena, form))
        off += csize or ((doff + size + align - 1) // align * align)
    return hdr, out
