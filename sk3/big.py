"""EA BIG archives ("EB\\0\\3"), the container Skate 3 ships worlds and DLC packs in.

Layout (big-endian; specs/SK3_FORMATS.md "BIG"):

    0x00 'EB' 00 03        0x04 u32 entry count      0x08 u32 flags (0x00100400 world, 0x00500400 DLC)
    0x0C u32 name table offset                       0x10 u32 name table length
    0x14 u32 name layout = nameSlot << 24 | rootSlot << 16 | rootCount
    0x1C u32 total size     0x20 u32 path-directory length (0 in world archives)
    0x30 records, 16 B each, ascending by id: u32 offset/16, u32 offset high, u32 size, u32 id
    id = djb2("<root>/<name>")
    name table: u8 0, then one fixed-width slot per record IN RECORD ORDER, each
                [u8 root index][name, NUL padded]; then, 32-aligned, the root paths in fixed slots.
    path directory (DLC only) directly after the name table.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = b'EB\x00\x03'
RECORDS = 0x30
NUL = b'\x00'


def djb2(text: str) -> int:
    h = 5381
    for b in text.encode('utf-8'):
        h = (h * 33 + b) & 0xFFFFFFFF
    return h


@dataclass
class Entry:
    root: str
    name: str
    offset: int
    size: int
    id: int

    @property
    def path(self) -> str:
        return f'{self.root}/{self.name}'


@dataclass
class Archive:
    path: str
    flags: int
    name_layout: int
    roots: list
    entries: list = field(default_factory=list)
    directory: bytes = b''
    name_table_offset: int = 0
    stride: int = 0

    def read(self, e: Entry) -> bytes:
        with open(self.path, 'rb') as f:
            f.seek(e.offset)
            return f.read(e.size)

    def by_name(self) -> dict:
        return {e.name: e for e in self.entries}


def read(path: str) -> Archive:
    with open(path, 'rb') as f:
        h = f.read(RECORDS)
        if h[:4] != MAGIC:
            raise ValueError(f'{path}: not an EB v3 archive')
        count, flags, nt, nl, layout = struct.unpack_from('>IIIII', h, 4)
        dlen = struct.unpack_from('>I', h, 0x20)[0]
        f.seek(0)
        head = f.read(max(RECORDS + 16 * count, nt + nl + dlen))
    recs = [struct.unpack_from('>IIII', head, RECORDS + 16 * i) for i in range(count)]
    table = head[nt:nt + nl]
    nroots, rslot = layout & 0xFFFF, (layout >> 16) & 0xFF
    rstart = (1 + (layout >> 24) * count + 31) & ~31
    roots = [table[rstart + i * rslot:rstart + (i + 1) * rslot].split(NUL, 1)[0].decode('ascii', 'replace')
             for i in range(nroots)]
    if not all(r.startswith('data/') for r in roots):           # fall back to scanning the table
        roots = []
        for raw in table.split(NUL):
            if raw.startswith(b'data/'):
                r = raw.decode('ascii')
                if r not in roots:
                    roots.append(r)

    def slot_at(k, stride):
        """(root, name) for slot k. The root index is one byte: with more than 256 roots it wraps,
        so the root is the one among index, index+256, ... whose path hashes to record k's id."""
        s = table[1 + k * stride:1 + (k + 1) * stride]
        if len(s) < 2 or s[0] >= len(roots):
            return None
        name = s[1:].split(NUL, 1)[0].decode('ascii', 'replace')
        if not name or name.startswith('data/'):
            return None
        cands = roots[s[0]::256]
        if len(cands) > 1 and k < len(recs):
            for r in cands:
                if djb2(f'{r}/{name}') == recs[k][3]:
                    return r, name
        return cands[0], name

    def hits(st):
        n = 0
        for k in range(len(recs)):
            r = slot_at(k, st)
            if r and djb2(f'{r[0]}/{r[1]}') == recs[k][3]:
                n += 1
        return n

    # Slot stride: the header byte when every slot verifies under it, else the stride under which
    # the most slots hash to their record's id.
    stride = layout >> 24
    if hits(stride) != len(recs):
        best = -1
        for st in range(4, 321):
            n = hits(st)
            if n > best:
                best, stride = n, st
            if n == len(recs):
                break
    a = Archive(path, flags, layout, roots, directory=head[nt + nl:nt + nl + dlen],
                name_table_offset=nt, stride=stride)
    for k, (off16, offhi, size, rid) in enumerate(recs):
        r = slot_at(k, stride)
        if not r or djb2(f'{r[0]}/{r[1]}') != rid:
            r = ('', f'~unnamed_{rid:08X}')          # audio archives carry no names at all
        a.entries.append(Entry(r[0], r[1], off16 * 16 + (offhi << 36), size, rid))
    return a
