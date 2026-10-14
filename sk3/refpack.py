"""RefPack (EA "QFS") LZ decompression, the compression inside Xbox 360 .xsf section payloads."""
from __future__ import annotations


def looks_like(b: bytes) -> bool:
    return len(b) >= 5 and b[1] == 0xFB and (b[0] & 0x3E) == 0x10


def decompress(src: bytes) -> bytes:
    flags = src[0]
    if src[1] != 0xFB:
        raise ValueError('not a RefPack stream')
    p = 2
    width = 4 if flags & 0x01 else 3
    if flags & 0x80:
        p += width
    size = int.from_bytes(src[p:p + width], 'big')
    p += width
    out = bytearray(size)
    o = 0
    while True:
        b0 = src[p]
        p += 1
        if b0 < 0x80:
            b1 = src[p]; p += 1
            run = b0 & 3
            dist = ((b0 & 0x60) << 3) + b1 + 1
            ln = ((b0 >> 2) & 7) + 3
        elif b0 < 0xC0:
            b1, b2 = src[p], src[p + 1]; p += 2
            run = b1 >> 6
            dist = ((b1 & 0x3F) << 8) + b2 + 1
            ln = (b0 & 0x3F) + 4
        elif b0 < 0xE0:
            b1, b2, b3 = src[p], src[p + 1], src[p + 2]; p += 3
            run = b0 & 3
            dist = ((b0 & 0x10) << 12) + (b1 << 8) + b2 + 1
            ln = ((b0 & 0x0C) << 6) + b3 + 5
        elif b0 < 0xFC:
            run, dist, ln = ((b0 & 0x1F) + 1) << 2, 0, 0
        else:
            run, dist, ln = b0 & 3, 0, 0
        if run:
            out[o:o + run] = src[p:p + run]
            p += run
            o += run
        if ln:
            s = o - dist
            if dist >= ln:
                out[o:o + ln] = out[s:s + ln]
            else:
                for i in range(ln):
                    out[o + i] = out[s + i]
            o += ln
        if b0 >= 0xFC:
            break
    if o != size:
        raise ValueError(f'RefPack produced {o} bytes, header said {size}')
    return bytes(out)
