"""RefPack (EA "QFS") LZ: the compression inside Xbox 360 .xsf section payloads.

Stream: u8 flags (0x10; bit 0 = 4-byte sizes, bit 7 = a compressed-size field first), 0xFB,
the uncompressed size (3 or 4 bytes, big-endian), then opcodes (specs/sk3_packaging.md "RefPack"):

    0xxxxxxx b1            run 0-3, copy 3-10   from 1-1024 back
    10xxxxxx b1 b2         run 0-3, copy 4-67   from 1-16384 back
    110xxxxx b1 b2 b3      run 0-3, copy 5-1028 from 1-131072 back
    111xxxxx (< 0xFC)      run 4-112 literals (multiple of 4)
    111111xx               run 0-3 literals, end of stream
"""
from __future__ import annotations

MAX_DIST = 0x20000
MAX_LEN = 1028
TAIL = 4          # EA never copies into the last four bytes (they always go out as literals)


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


def ops(src: bytes):
    """The opcode list of a stream: [(kind, out pos, literal run, copy len, copy dist)]."""
    flags = src[0]
    p = 2 + (4 if flags & 1 else 3) * (2 if flags & 0x80 else 1)
    o = 0
    res = []
    while True:
        b0 = src[p]
        if b0 < 0x80:
            run, ln, dist, k, w = b0 & 3, ((b0 >> 2) & 7) + 3, ((b0 & 0x60) << 3) + src[p + 1] + 1, 'A', 2
        elif b0 < 0xC0:
            run, ln = src[p + 1] >> 6, (b0 & 0x3F) + 4
            dist, k, w = ((src[p + 1] & 0x3F) << 8) + src[p + 2] + 1, 'B', 3
        elif b0 < 0xE0:
            run, ln = b0 & 3, ((b0 & 0x0C) << 6) + src[p + 3] + 5
            dist, k, w = ((b0 & 0x10) << 12) + (src[p + 1] << 8) + src[p + 2] + 1, 'C', 4
        elif b0 < 0xFC:
            run, ln, dist, k, w = ((b0 & 0x1F) + 1) << 2, 0, 0, 'L', 1
        else:
            run, ln, dist, k, w = b0 & 3, 0, 0, 'E', 1
        res.append((k, o, run, ln, dist))
        p += w + run
        o += run + ln
        if k == 'E':
            return res


def _cost(ln: int, dist: int) -> int:
    """Opcode bytes needed for a copy, or 0 when no opcode can encode it."""
    if ln <= 10 and dist <= 1024:
        return 2
    if 4 <= ln <= 67 and dist <= 16384:
        return 3
    if 5 <= ln and dist <= MAX_DIST:
        return 4
    return 0


def parse(src: bytes, depth: int = 0):
    """Greedy match list [(pos, len, dist)]. At each position every earlier occurrence of the next
    three bytes within 128 KiB is tried, nearest first; the copy that saves the most bytes
    (len - opcode bytes) wins, a tie going to the longer copy. `depth` > 0 caps the candidates
    tried per position (faster). This rule reproduces EA's own parse for long stretches of retail
    streams but not all of it (specs/sk3_packaging.md "RefPack")."""
    n = len(src)
    head: dict = {}
    prev = [-1] * n
    out = []
    i = 0

    def insert(j):
        if j + 3 <= n:
            k = src[j:j + 3]
            prev[j] = head.get(k, -1)
            head[k] = j

    while i < n:
        best_len = best_dist = best_score = 0
        if i + 3 <= n - TAIL:
            c = head.get(src[i:i + 3], -1)
            mx = min(MAX_LEN, n - TAIL - i)
            tried = 0
            while c >= 0:
                dist = i - c
                if dist > MAX_DIST:
                    break
                # Best score any copy from this far back could still reach; candidates only get
                # farther, so once the best so far cannot be beaten the search stops.
                if dist <= 1024:
                    ub = max(min(mx, 10) - 2, min(mx, 67) - 3, mx - 4)
                elif dist <= 16384:
                    ub = max(min(mx, 67) - 3, mx - 4)
                else:
                    ub = mx - 4
                if best_score > ub or (best_score == ub and best_len >= mx):
                    break
                ln = 3
                while ln < mx and src[c + ln] == src[i + ln]:
                    ln += 1
                cost = _cost(ln, dist)
                if cost:
                    s = ln - cost
                    if s > best_score or (s == best_score and ln > best_len):
                        best_len, best_dist, best_score = ln, dist, s
                tried += 1
                if depth and tried >= depth:
                    break
                c = prev[c]
        if best_score > 0:
            out.append((i, best_len, best_dist))
            for k in range(best_len):
                insert(i + k)
            i += best_len
        else:
            insert(i)
            i += 1
    return out


def encode(src: bytes, matches) -> bytes:
    """Serialise a match list. Literals go out in 112-byte runs, with the last 0-3 riding on the
    next copy (or on the end-of-stream opcode)."""
    n = len(src)
    out = bytearray()
    if n >= 1 << 24:
        out += b'\x11\xfb' + n.to_bytes(4, 'big')
    else:
        out += b'\x10\xfb' + n.to_bytes(3, 'big')
    lit = 0

    def literals(frm, to):
        cnt = to - frm
        bulk = cnt - (cnt & 3)
        p = frm
        while bulk:
            take = min(112, bulk)
            out.append(0xE0 + (take >> 2) - 1)
            out.extend(src[p:p + take])
            p += take
            bulk -= take
        return p, cnt & 3

    for pos, ln, dist in matches:
        p, run = literals(lit, pos)
        d = dist - 1
        if ln <= 10 and dist <= 1024:
            out.append(((d >> 3) & 0x60) | ((ln - 3) << 2) | run)
            out.append(d & 0xFF)
        elif 4 <= ln <= 67 and dist <= 16384:
            out.append(0x80 | (ln - 4))
            out.append((run << 6) | (d >> 8))
            out.append(d & 0xFF)
        else:
            l = ln - 5
            out.append(0xC0 | ((d >> 12) & 0x10) | ((l >> 6) & 0x0C) | run)
            out.append((d >> 8) & 0xFF)
            out.append(d & 0xFF)
            out.append(l & 0xFF)
        out += src[p:p + run]
        lit = pos + ln
    p, run = literals(lit, n)
    out.append(0xFC + run)
    out += src[p:p + run]
    return bytes(out)


def compress(src: bytes, depth: int = 0) -> bytes:
    return encode(src, parse(src, depth))
