"""Block-level decoding of CAS ranges.

Every CAS range (a TOC chunk or a bundle entry) is a run of blocks, each with an
8-byte big-endian header:

    u32 decompressedSize   u8 type   u8 flags   u16 compressedSizeLow

The flags' low nibble is bits 16-19 of compressedSize: a stored block of exactly
0x10000 bytes cannot fit in the u16, so it is written as flags 0x71, size 0.
Missing that made 11.5% of chunks look like an unknown container; with it, every
sampled chunk walks to its exact end.
"""
from __future__ import annotations

import struct

from oodle import decompress_blocks

STORED = 0x00
KRAKEN = 0x11


def parse_blocks(data: bytes) -> list[tuple[int, int, bytes]]:
    """[(type, decompressedSize, payload)], or ValueError if the walk does not land."""
    p, out = 0, []
    while p < len(data):
        if p + 8 > len(data):
            raise ValueError(f"truncated block header at {p}/{len(data)}")
        dec, typ, flags, c16 = struct.unpack_from('>IBBH', data, p)
        comp = ((flags & 0x0F) << 16) | c16
        if comp == 0:
            raise ValueError(f"zero-size block at {p}")
        out.append((typ, dec, data[p + 8:p + 8 + comp]))
        p += 8 + comp
    if p != len(data):
        raise ValueError(f"block walk overran: {p} != {len(data)}")
    return out


def decompress_many(ranges: list[bytes]) -> list[bytes]:
    """Decompress many on-disk ranges; every Kraken block goes in one wine call."""
    parsed = [parse_blocks(r) for r in ranges]
    jobs = [(pl, dec) for blocks in parsed for typ, dec, pl in blocks if typ == KRAKEN]
    kraken = iter(decompress_blocks(jobs))
    res = []
    for blocks in parsed:
        parts = []
        for typ, dec, pl in blocks:
            if typ == KRAKEN:
                parts.append(next(kraken))
            elif typ == STORED:
                if len(pl) != dec:
                    raise ValueError(f"stored block size {len(pl)} != {dec}")
                parts.append(pl)
            else:
                raise ValueError(f"unknown block type 0x{typ:02X}")
        res.append(b''.join(parts))
    return res
