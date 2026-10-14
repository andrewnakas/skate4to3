"""SFIL stream files (.xsf): a header, then one aligned chunk per arena (specs/sk3_packaging.md "SFIL").

    0x00 'SFIL'  0x04 version 3  0x08 0  0x0C group hash  0x10 align  0x14 group count (1)
    0x18 platform (2 = EA Xbox form, 0 = raw form)  0x1C tail word; zero padding to `align`
    chunk: u64 asset id, u32 payload size, u32 payload offset (= align), u32 chunk size
           (EA: payload offset + size rounded up to align; community raw packs: one more align
           block when size is already a multiple); zeros to the payload; zeros after it
    payload: a raw RW4 arena ('\\x89RW4'; platform 0, align 0x100 in every community pack), or EA's
    Xbox two-section form (platform 2, align 0x80): a 0x50-byte descriptor, main section
    {u32 raw size = arena h44, u32 stored size, u32 RefPack flag} at 0x00, graphics section
    {raw = arena length - h44, stored, flag} at 0x20, the rest zero; then the stored main and
    graphics bytes back to back. EA compresses a section exactly when its raw size is >= 2048.

Byte-exact rewrite: `read` keeps each section's stored bytes beside the arena, and `write` reuses
them while the arena is unchanged, because our RefPack parse matches EA's only for some sections.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from . import refpack

MAGIC = b'SFIL'
XBOX_DESC = 0x50
EA_ALIGN = 0x80         # EA's Xbox form
RAW_ALIGN = 0x100       # the raw form community packs use
PLATFORM_XBOX = 2
PLATFORM_RAW = 0
COMPRESS_MIN = 2048     # EA: sections of 2044 bytes and less stored, 2056 and more RefPacked


@dataclass
class Asset:
    id: int
    arena: bytes
    form: str = 'raw'                 # 'raw' | 'xbox2'
    compressed: tuple = (None, None)  # (main, graphics) RefPack flags as read; None = decide
    stored: tuple | None = None       # (main, graphics) stored bytes as read, for exact rewrite
    _source: bytes | None = field(default=None, repr=False)


@dataclass
class Header:
    group: int
    align: int = EA_ALIGN
    platform: int = PLATFORM_XBOX
    tail: int = 0
    version: int = 3
    groups: int = 1
    pad_over: bool = False   # chunk size = offset + (size // align + 1) * align (community raw packs)

    @classmethod
    def from_dict(cls, h: dict) -> 'Header':
        return cls(h['group'], h['align'], h['platform'], h['tail'], h['version'], h['groups'],
                   h.get('pad_over', False))


def payload_to_arena(p: bytes):
    if p[:4] == b'\x89RW4':
        return p, 'raw'
    a = payload_split(p)
    return a.arena, 'xbox2'


def payload_split(p: bytes) -> Asset:
    main_raw, main_st, main_c = struct.unpack_from('>III', p, 0x00)
    gfx_raw, gfx_st, gfx_c = struct.unpack_from('>III', p, 0x20)
    m = p[XBOX_DESC:XBOX_DESC + main_st]
    g = p[XBOX_DESC + main_st:XBOX_DESC + main_st + gfx_st]
    mr = refpack.decompress(m) if main_c else m
    gr = (refpack.decompress(g) if gfx_c else g) if gfx_st else b''
    arena = mr + gr
    return Asset(0, arena, 'xbox2', (bool(main_c), bool(gfx_c)), (m, g), arena)


def read(data: bytes):
    if data[:4] != MAGIC:
        raise ValueError('not an SFIL stream')
    version, _z, group, align, groups, platform, tail = struct.unpack_from('>IIIIIII', data, 4)
    hdr = {'version': version, 'group': group, 'align': align, 'groups': groups,
           'platform': platform, 'tail': tail, 'pad_over': False}
    out, off = [], align
    while off + 0x20 <= len(data):
        aid = struct.unpack_from('>Q', data, off)[0]
        size, doff, csize = struct.unpack_from('>III', data, off + 8)
        if not size or not doff or off + doff + size > len(data):
            break
        p = data[off + doff:off + doff + size]
        if p[:4] == b'\x89RW4':
            out.append(Asset(aid, p, 'raw'))
        else:
            a = payload_split(p)
            a.id = aid
            out.append(a)
        if size % align == 0 and csize == doff + size + align:
            hdr['pad_over'] = True
        off += csize or ((doff + size + align - 1) // align * align)
    return hdr, out


def _section(raw: bytes, flag, stored, compress: bool, depth: int):
    if stored is not None:
        return stored, flag
    use = (len(raw) >= COMPRESS_MIN) if flag is None else flag
    if use and compress:
        return refpack.compress(raw, depth), True
    return raw, False


def xbox_payload(a: Asset, compress: bool = True, depth: int = 0) -> bytes:
    arena = a.arena
    h44 = struct.unpack_from('>I', arena, 0x44)[0]
    main, gfx = arena[:h44], arena[h44:]
    keep = a.stored is not None and a._source is not None and a._source == arena
    ms, mc = _section(main, a.compressed[0], a.stored[0] if keep else None, compress, depth)
    if gfx:
        gs, gc = _section(gfx, a.compressed[1], a.stored[1] if keep else None, compress, depth)
    else:
        gs, gc = b'', False
    desc = bytearray(XBOX_DESC)
    struct.pack_into('>III', desc, 0x00, len(main), len(ms), int(mc))
    struct.pack_into('>III', desc, 0x20, len(gfx), len(gs), int(gc))
    return bytes(desc) + ms + gs


def write(hdr, assets, form: str | None = None, compress: bool = True, depth: int = 0) -> bytes:
    """Serialise a stream. `hdr` is a Header or the dict `read` returns; `form` forces every
    payload to 'raw' or 'xbox2' (default: each asset's own form)."""
    h = hdr if isinstance(hdr, Header) else Header.from_dict(hdr)
    al = h.align
    out = bytearray(al)
    out[0:4] = MAGIC
    struct.pack_into('>IIIIIII', out, 4, h.version, 0, h.group, al, h.groups, h.platform, h.tail)
    for a in assets:
        f = form or a.form
        p = a.arena if f == 'raw' else xbox_payload(a, compress, depth)
        csize = (al + len(p) + (al if h.pad_over else al - 1)) // al * al
        chunk = bytearray(al)
        struct.pack_into('>QIII', chunk, 0, a.id, len(p), al, csize)
        out += chunk
        out += p
        out += bytes(csize - al - len(p))
    return bytes(out)


def raw_stream(group: int, assets, tail: int = 0) -> bytes:
    """The raw container form: align 0x100, platform 0, arenas stored as-is."""
    return write(Header(group, RAW_ALIGN, PLATFORM_RAW, tail, pad_over=True), assets, form='raw')


def ea_stream(group: int, assets, tail: int = 0, compress: bool = True, depth: int = 0) -> bytes:
    """EA's own Xbox form: align 0x80, platform 2, two-section payloads, RefPack >= 2048 bytes."""
    return write(Header(group, EA_ALIGN, PLATFORM_XBOX, tail), assets, form='xbox2',
                 compress=compress, depth=depth)
