"""Binary bundle manifest: entry 0 of every bundle. Stored raw, not block-compressed.

    u32 BE size (of what follows)     u32 BE magic 0xD68E799D
    -- little-endian from here; offsets are relative to +4 --
    u32 totalCount  u32 ebxCount  u32 resCount  u32 chunkCount
    u32 stringsOffset  u32 chunkMetaOffset  u32 chunkMetaSize
    sha1      totalCount x 20
    ebx       ebxCount   x {u32 nameOffset, u32 originalSize}
    res       resCount   x {u32 nameOffset, u32 originalSize}
    resType   resCount   x u32
    resMeta   resCount   x 16
    resRid    resCount   x u64
    chunks    chunkCount x {guid 16, u32 logicalOffset, u32 logicalSize}
    strings   at stringsOffset
    chunkMeta at chunkMetaOffset (DbObject)

Bundle entries 1.. follow the manifest order: ebx, then res, then chunks.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = 0xD68E799D


@dataclass
class Manifest:
    ebx: list = field(default_factory=list)      # [(name, originalSize, sha1)]
    res: list = field(default_factory=list)      # [(name, originalSize, type, meta16, rid, sha1)]
    chunks: list = field(default_factory=list)   # [(guidhex, logicalOffset, logicalSize, sha1)]


def parse(m: bytes) -> Manifest:
    size, magic = struct.unpack_from('>II', m, 0)
    if magic != MAGIC:
        raise ValueError(f"bad manifest magic {magic:#x}")
    if size + 4 != len(m):
        raise ValueError(f"size {size}+4 != {len(m)}")
    total, nebx, nres, nchk, soff, cmoff, cmsize = struct.unpack_from('<7I', m, 8)
    if nebx + nres + nchk != total:
        raise ValueError(f"counts {nebx}+{nres}+{nchk} != {total}")
    p = 36
    sha1 = [m[p + i * 20:p + i * 20 + 20].hex() for i in range(total)]
    p += total * 20
    ebx_raw = struct.unpack_from(f'<{nebx * 2}I', m, p); p += nebx * 8
    res_raw = struct.unpack_from(f'<{nres * 2}I', m, p); p += nres * 8
    rtype = struct.unpack_from(f'<{nres}I', m, p); p += nres * 4
    rmeta = [m[p + i * 16:p + i * 16 + 16] for i in range(nres)]; p += nres * 16
    rrid = struct.unpack_from(f'<{nres}Q', m, p); p += nres * 8
    chunks = []
    for i in range(nchk):
        g = m[p:p + 16].hex()
        lo, ls = struct.unpack_from('<II', m, p + 16)
        chunks.append((g, lo, ls))
        p += 24
    if p != soff + 4:
        raise ValueError(f"tables end at {p}, strings at {soff + 4}")
    if cmoff + 4 + cmsize != len(m):
        raise ValueError(f"chunkMeta {cmoff}+4+{cmsize} != {len(m)}")
    sbase = soff + 4

    def name(off):
        a = sbase + off
        return m[a:m.index(b'\x00', a)].decode('utf-8', 'replace')

    out = Manifest()
    for i in range(nebx):
        out.ebx.append((name(ebx_raw[2 * i]), ebx_raw[2 * i + 1], sha1[i]))
    for i in range(nres):
        out.res.append((name(res_raw[2 * i]), res_raw[2 * i + 1], rtype[i], rmeta[i],
                        rrid[i], sha1[nebx + i]))
    for i, (g, lo, ls) in enumerate(chunks):
        out.chunks.append((g, lo, ls, sha1[nebx + nres + i]))
    return out
