"""Skate 3 grind splines: RW4 object type 0xEB0004 in cSim_* arenas (specs/sk3_collision.md
"Grind splines"). Measured on every retail eb0004 of the small worlds, re-checked on the big ones.

    0x00 u16 kind (0 in tiles; 1 in one SkateSchool cSim_Global set of flat, curved splines)
    0x02 u16 spline count   0x04 u32 segment count (all splines)   0x08 u32 table offset (0x10)
    0x0C u32 segment data offset (= 0x10 + 32 * count)
    table, 32 B per spline: u64 id, u64 template id (0x2c701707 << 32 | ...), u32 0,
          u32 offset of the first segment, u32 offset of the last segment, u32 0
          (each table entry is a subref target: arena subrefs point at 0x10 + 32 k)
    segments, 144 B (9 rows of 16 B), contiguous, spline after spline:
          row0 A (t^3), row1 B (t^2), row2 C (t), row3 D (start point, w = 1.0)   -> P(t) = A t^3 + B t^2 + C t + D
          row4 0, row5 bbox min, row6 bbox max (of the segment), row7 (0, 0, u32 flags, 0), row8 0
          flags: bit 0x10 always; bits 0x20 / 0x40 vary per segment; the other bits are constant per spline.
Straight segments in retail use C = 0, B = 3 (E - D), A = -2 (E - D) (eased Hermite with zero end tangents).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

T_SPLINES = 0xEB0004
SEG = 0x90


@dataclass
class Segment:
    a: tuple
    b: tuple
    c: tuple
    d: tuple                  # start point
    bmin: tuple
    bmax: tuple
    flags: int

    def eval(self, t):
        t = np.asarray(t, np.float64)[..., None]
        A, B, C, D = (np.array(x, np.float64) for x in (self.a, self.b, self.c, self.d))
        return ((A * t + B) * t + C) * t + D

    @property
    def end(self):
        return self.eval(1.0)


@dataclass
class Spline:
    id: int
    template: int
    segments: list = field(default_factory=list)

    def points(self, per_segment=8):
        if not self.segments:
            return np.zeros((0, 3))
        ts = np.linspace(0, 1, per_segment + 1)[:-1]
        pts = [s.eval(ts) for s in self.segments] + [self.segments[-1].end[None]]
        return np.concatenate(pts)


@dataclass
class SplineSet:
    kind: int
    splines: list


def parse(d: bytes) -> SplineSet:
    w0, nseg, toff, doff = struct.unpack_from('>4I', d, 0)
    n = w0 & 0xFFFF
    out = []
    for k in range(n):
        sid, tmpl, _z0, first, last, _z1 = struct.unpack_from('>QQIIII', d, toff + 32 * k)
        s = Spline(sid, tmpl)
        for q in range(first, last + 1, SEG):
            r = struct.unpack_from('>36f', d, q)
            flags = struct.unpack_from('>I', d, q + 0x78)[0]
            s.segments.append(Segment(r[0:3], r[4:7], r[8:11], r[12:15], r[20:23], r[24:27], flags))
        out.append(s)
    return SplineSet(w0 >> 16, out)


def encode(ss: SplineSet) -> bytes:
    n = len(ss.splines)
    nseg = sum(len(s.segments) for s in ss.splines)
    toff, doff = 0x10, 0x10 + 32 * n
    out = bytearray(doff + SEG * nseg)
    struct.pack_into('>4I', out, 0, ss.kind << 16 | n, nseg, toff, doff)
    q = doff
    for k, s in enumerate(ss.splines):
        first = q
        for g in s.segments:
            struct.pack_into('>3f', out, q, *g.a)
            struct.pack_into('>3f', out, q + 0x10, *g.b)
            struct.pack_into('>3f', out, q + 0x20, *g.c)
            struct.pack_into('>4f', out, q + 0x30, *g.d, 1.0)
            struct.pack_into('>3f', out, q + 0x50, *g.bmin)
            struct.pack_into('>3f', out, q + 0x60, *g.bmax)
            struct.pack_into('>I', out, q + 0x78, g.flags)
            q += SEG
        struct.pack_into('>QQIIII', out, toff + 32 * k, s.id, s.template, 0, first, max(first, q - SEG), 0)
    return bytes(out)


def subrefs(obj_index: int, ss: SplineSet):
    """the arena subref records retail emits for a spline object: one per table entry"""
    return [(obj_index, 0x10 + 32 * k) for k in range(len(ss.splines))]


def segment_bbox(a, b, c, d):
    """retail segment bbox: min / max of the Bezier control points of P(t)
    (D, D + C/3, D + 2C/3 + B/3, D + C + B + A) - measured on the curved retail segments"""
    A, B, C, D = (np.array(x, np.float64) for x in (a, b, c, d))
    cps = np.array([D, D + C / 3, D + 2 * C / 3 + B / 3, D + C + B + A])
    return tuple(np.float32(cps.min(0))), tuple(np.float32(cps.max(0)))


def straight_segment(p0, p1, flags=0x10):
    """a segment from p0 to p1 in the retail straight-rail form (A = -2 dP, B = 3 dP, C = 0)"""
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    dp = (p1.astype(np.float64) - p0)
    a, b, c = tuple(np.float32(-2 * dp)), tuple(np.float32(3 * dp)), (0.0, 0.0, 0.0)
    return Segment(a, b, c, tuple(p0), tuple(np.minimum(p0, p1)), tuple(np.maximum(p0, p1)), flags)
