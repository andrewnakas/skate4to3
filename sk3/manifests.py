"""DIST stream manifests: MMAP (.xmm), CMAP (.xsm), CSPA (.xss), ATOC (.xst) and the location table
(.loc). Readers and writers are byte-exact over every retail world and DLC pack measured
(tests/test_manifests.py); `generate` builds a stream group's four manifests from a tile list.
Facts and how they were measured: specs/sk3_packaging.md "Manifests".

All binary manifests are big-endian and start {magic, u32 version 3, u32 0, u32 group hash}.
    MMAP  24 B: then u32 0, u32 0.
    CMAP  0x20 header: +0x10 entry count, +0x14 platform (2 EA form, 0 raw form), +0x18 0xDEADC0DE,
          +0x1C 0; then 0x68-byte entries {u64 asset id, u64 group id, u32 alignment,
          u32 stream size, u32 chunk count, char name[0x40], u32 0xFFFFFFFF, u32 0, u32 0}.
    CSPA  0x18 header: +0x10 cell count, +0x14 0; then 80-byte cells {u64 asset id, 8 bytes
          (zero, or stale ASCII in EA's files), 4 x float4: tile box min, tile box max,
          content box min, content box max (w = 1)}; one cell per CMAP entry except *_Global,
          in CMAP order.
    ATOC  0x18 header: +0x10 record count, +0x14 0; then 64-byte records {u64 asset id,
          u32 0xAB329A6A, u32 arena length, the arena's ten header words at +0x44..+0x68,
          u32 0, 4 bytes (zero, or stale bytes such as CDCDCDCD in EA's files)};
          one record per distinct asset id, in first-appearance order over the CMAP tiles'
          stream chunks.
"""
from __future__ import annotations

import re
import os
import struct
from dataclasses import dataclass, field

VERSION = 3
ROOT_GROUP = 0xE171E1ABFEDB69ED      # group id of every *_Global entry
# True reproduces the community packs' bug (tiles of a group with no *_Global point at a missing
# Global id) so tests can regenerate those packs byte for byte. Never set it for real builds.
LEGACY_MISSING_GLOBAL_GROUP = False
ATOC_TAG = 0xAB329A6A
SENTINEL = 0xDEADC0DE


def fnv1_64(text: str) -> int:
    """FNV-1 (multiply, then xor) 64-bit: CMAP asset ids are this over the tile name."""
    h = 0xCBF29CE484222325
    for b in text.encode('ascii'):
        h = ((h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF) ^ b
    return h


def _head(magic: bytes, group: int, w10: int, w14: int) -> bytearray:
    return bytearray(magic + struct.pack('>IIIII', VERSION, 0, group, w10, w14))


def _check(d: bytes, magic: bytes):
    if d[:4] != magic:
        raise ValueError(f'not a {magic.decode()} manifest')
    ver, z, group = struct.unpack_from('>III', d, 4)
    if ver != VERSION or z:
        raise ValueError(f'{magic.decode()}: unexpected version/zero word {ver}/{z}')
    return group


# ---- MMAP -------------------------------------------------------------------------------------
def read_mmap(d: bytes) -> int:
    g = _check(d, b'MMAP')
    if len(d) != 24 or any(d[16:]):
        raise ValueError('MMAP: unexpected size or tail')
    return g


def write_mmap(group: int) -> bytes:
    return bytes(_head(b'MMAP', group, 0, 0))


# ---- CMAP -------------------------------------------------------------------------------------
@dataclass
class CmapEntry:
    asset_id: int
    group_id: int
    alignment: int
    size: int
    chunks: int
    name: str


@dataclass
class Cmap:
    group: int
    platform: int
    entries: list = field(default_factory=list)


def read_cmap(d: bytes) -> Cmap:
    g = _check(d, b'CMAP')
    count, platform, sent, z = struct.unpack_from('>IIII', d, 0x10)
    if sent != SENTINEL or z or len(d) != 0x20 + 0x68 * count:
        raise ValueError('CMAP: unexpected header')
    out = Cmap(g, platform)
    for i in range(count):
        b = 0x20 + 0x68 * i
        aid, gid, al, size, chunks = struct.unpack_from('>QQIII', d, b)
        raw = d[b + 0x1C:b + 0x5C]
        name = raw.split(b'\0', 1)[0].decode('ascii')
        if raw[len(name):].strip(b'\0') or d[b + 0x5C:b + 0x68] != b'\xff\xff\xff\xff' + bytes(8):
            raise ValueError(f'CMAP entry {i}: unexpected padding')
        out.entries.append(CmapEntry(aid, gid, al, size, chunks, name))
    return out


def write_cmap(m: Cmap) -> bytes:
    out = _head(b'CMAP', m.group, len(m.entries), m.platform)
    out += struct.pack('>II', SENTINEL, 0)
    for e in m.entries:
        nm = e.name.encode('ascii')
        if len(nm) >= 0x40:
            raise ValueError(f'tile name too long: {e.name}')
        out += struct.pack('>QQIII', e.asset_id, e.group_id, e.alignment, e.size, e.chunks)
        out += nm + bytes(0x40 - len(nm)) + b'\xff\xff\xff\xff' + bytes(8)
    return bytes(out)


# ---- CSPA -------------------------------------------------------------------------------------
@dataclass
class Cell:
    asset_id: int
    box: tuple            # (minx, miny, minz, maxx, maxy, maxz): the tile's box
    content: tuple        # same, the content's box
    pad: bytes = bytes(8)
    w: tuple = (1.0, 1.0, 1.0, 1.0)


@dataclass
class Cspa:
    group: int
    cells: list = field(default_factory=list)


def read_cspa(d: bytes) -> Cspa:
    g = _check(d, b'CSPA')
    count, z = struct.unpack_from('>II', d, 0x10)
    if z or len(d) != 0x18 + 80 * count:
        raise ValueError('CSPA: unexpected header')
    out = Cspa(g)
    for i in range(count):
        b = 0x18 + 80 * i
        aid = struct.unpack_from('>Q', d, b)[0]
        f = struct.unpack_from('>16f', d, b + 16)
        out.cells.append(Cell(aid, f[0:3] + f[4:7], f[8:11] + f[12:15], d[b + 8:b + 16],
                              (f[3], f[7], f[11], f[15])))
    return out


def write_cspa(m: Cspa) -> bytes:
    out = _head(b'CSPA', m.group, len(m.cells), 0)
    for c in m.cells:
        b, k, w = c.box, c.content, c.w
        out += struct.pack('>Q', c.asset_id) + c.pad
        out += struct.pack('>16f', b[0], b[1], b[2], w[0], b[3], b[4], b[5], w[1],
                           k[0], k[1], k[2], w[2], k[3], k[4], k[5], w[3])
    return bytes(out)


# ---- ATOC -------------------------------------------------------------------------------------
@dataclass
class AtocRecord:
    asset_id: int
    size: int
    words: tuple          # arena header words +0x44..+0x68
    tag: int = ATOC_TAG
    pad: bytes = bytes(8) # zero, or zero + 4 stale bytes in EA's files

    @classmethod
    def of_arena(cls, asset_id: int, arena: bytes) -> 'AtocRecord':
        return cls(asset_id, len(arena), struct.unpack_from('>10I', arena, 0x44))


@dataclass
class Atoc:
    group: int
    records: list = field(default_factory=list)


def read_atoc(d: bytes) -> Atoc:
    g = _check(d, b'ATOC')
    count, z = struct.unpack_from('>II', d, 0x10)
    if z or len(d) != 0x18 + 64 * count:
        raise ValueError('ATOC: unexpected header')
    out = Atoc(g)
    for i in range(count):
        b = 0x18 + 64 * i
        r = struct.unpack_from('>QII10I', d, b)
        out.records.append(AtocRecord(r[0], r[2], tuple(r[3:]), r[1], d[b + 56:b + 64]))
    return out


def write_atoc(m: Atoc) -> bytes:
    out = _head(b'ATOC', m.group, len(m.records), 0)
    for r in m.records:
        out += struct.pack('>QII10I', r.asset_id, r.tag, r.size, *r.words) + r.pad
    return bytes(out)


# ---- location table (.loc, text) --------------------------------------------------------------
@dataclass
class Location:
    name: str
    transform: tuple      # 16 floats, row-major 4x4, translation in the last row
    category: str = ''
    empty_tag: bool = True    # '<Category />' (community packs) vs '<Category></Category>' (EA)


_LOC = re.compile(r'    <Location>\r\n        <Name>(.*?)</Name>\r\n        '
                  r'(?:<Category />|<Category>(.*?)</Category>)\r\n        '
                  r'<Transform>(.*?)</Transform>\r\n    </Location>\r\n', re.S)


def read_loc(d: bytes) -> list:
    t = d.decode('utf-8')
    if not t.startswith('<LocationTable>\r\n') or not t.endswith('</LocationTable>\r\n'):
        raise ValueError('not a location table')
    body = t[len('<LocationTable>\r\n'):-len('</LocationTable>\r\n')]
    out, pos = [], 0
    for m in _LOC.finditer(body):
        if m.start() != pos:
            raise ValueError('location table: unexpected text')
        pos = m.end()
        vals = tuple(float(x) for x in m.group(3).split(', '))
        out.append(Location(m.group(1), vals, m.group(2) or '', m.group(2) is None))
    if pos != len(body):
        raise ValueError('location table: trailing text')
    return out


def write_loc(locs) -> bytes:
    s = ['<LocationTable>\r\n']
    for L in locs:
        cat = '<Category />' if L.empty_tag and not L.category else f'<Category>{L.category}</Category>'
        s.append(f'    <Location>\r\n        <Name>{L.name}</Name>\r\n        {cat}\r\n        '
                 f'<Transform>{", ".join("%.6f" % v for v in L.transform)}</Transform>\r\n'
                 f'    </Location>\r\n')
    s.append('</LocationTable>\r\n')
    return ''.join(s).encode('utf-8')


def spawn_transform(x: float, y: float, z: float, yaw_deg: float = 0.0) -> tuple:
    """Row-major 4x4 with a yaw about +Y and the position in the last row."""
    import math
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, x, y, z, 1.0)


# ---- generator --------------------------------------------------------------------------------
KINDS = (('Pres', 'cPres'), ('Sim', 'cSim'), ('Tex', 'cTex'))
_TILE = re.compile(r'^c(?:Pres|Sim|Tex)_(-?\d+)_(-?\d+)(?:_\w+)?$')


@dataclass
class Tile:
    """One stream file of a group: its name (cPres_<x>_<z>_high / cPres_Global), the stream's
    chunk list [(asset id, arena)], the stream's length on disk, and optional y/content bounds."""
    name: str
    arenas: list
    stream_size: int
    content: tuple | None = None   # (minx, miny, minz, maxx, maxy, maxz)
    yrange: tuple | None = None    # (miny, maxy) for the tile box; default from content


def tile_center(name: str):
    m = _TILE.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def group_hash(dist: str, kind: str) -> int:
    """Our stream group word. Retail values are not a function of any name we know (a stream's
    files and manifests only have to agree), so we derive a stable one."""
    h = fnv1_64(f'{dist}_{kind}')
    return (h ^ (h >> 32)) & 0xFFFFFFFF


def generate(dist: str, kind: str, tiles: list, cell: float = 100.0, platform: int = 0,
             alignment: int = 0x100, size_includes_header: bool = True, group: int | None = None):
    """The four manifests of one stream group ('Pres' / 'Sim' / 'Tex') from its tiles.

    Asset ids: FNV-1 64 of the tile name; *_Global: FNV-1 64 of "<name>_<DIST>" in group ROOT_GROUP;
    every other tile's group id is the *_Global id of its kind (whether or not a Global exists:
    community packs that load do both). CMAP size: the stream length (community raw packs) or the
    length minus the alignment (EA). Returns {'.xmm','.xsm','.xss','.xst': bytes}."""
    prefix = dict(KINDS)[kind]
    g = group if group is not None else group_hash(dist, kind)
    glob_id = fnv1_64(f'{prefix}_Global_{dist}')
    order = sorted(tiles, key=lambda t: (not t.name.endswith('_Global'), t.name))
    cm = Cmap(g, platform)
    cs = Cspa(g)
    at = Atoc(g)
    seen = set()
    half = cell / 2
    # Tiles hang off the group's *_Global entry when there is one; with none (retail
    # BlackBoxPark's Sim group) every tile is in ROOT_GROUP. A tile group naming a Global the
    # list does not contain makes the game's stream tile-list loader (sub_82C98B08) write N+1 ids
    # into an N-id vector - the heap overrun that crashed Isle of Grom / DM Jumpline.
    tile_group = (glob_id if LEGACY_MISSING_GLOBAL_GROUP or any(t.name.endswith('_Global') for t in order)
                  else ROOT_GROUP)
    for t in order:
        is_global = t.name.endswith('_Global')
        aid = glob_id if is_global else (fnv1_64(f'{t.name}_{dist}') if os.environ.get('SK3_UNIQUE_TILE_IDS')
                                          else fnv1_64(t.name))
        size = t.stream_size if size_includes_header else t.stream_size - alignment
        cm.entries.append(CmapEntry(aid, ROOT_GROUP if is_global else tile_group, alignment, size,
                                    len(t.arenas), t.name))
        for a_id, arena in t.arenas:
            if a_id not in seen:
                seen.add(a_id)
                at.records.append(AtocRecord.of_arena(a_id, arena))
        if not is_global:
            c = tile_center(t.name)
            if c is None:
                raise ValueError(f'{t.name}: not a <prefix>_<x>_<z>[_suffix] tile name')
            k = t.content or (c[0] - half, 0.0, c[1] - half, c[0] + half, 0.0, c[1] + half)
            y0, y1 = t.yrange or (k[1], k[4])
            cs.cells.append(Cell(aid, (c[0] - half, y0, c[1] - half, c[0] + half, y1, c[1] + half), k))
    return {'.xmm': write_mmap(g), '.xsm': write_cmap(cm), '.xss': write_cspa(cs), '.xst': write_atoc(at)}
