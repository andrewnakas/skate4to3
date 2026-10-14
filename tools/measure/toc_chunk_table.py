#!/usr/bin/env python3
"""Measure the TOC chunk-table layout from bytes, with a Studio oracle chunk.

    python tools/measure/toc_chunk_table.py <toc> <descriptor-order guid hex> <encoded size>

Finds the guid in the region between the chunk-guid and chunk-record sections, reports its
position modulo the candidate entry strides, then tries index->record mappings and prints which
one lands on a record whose size equals the oracle's encoded size.
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sk4'))
import superbundle as sb

toc, ghex, enc = sys.argv[1], sys.argv[2], int(sys.argv[3])
S = sb.SuperBundle(toc, '')
d, base = S.d, sb.BASE
lo, hi = base + S.chunk_guids, base + S.chunk_records
n = S.chunks_count
print(f'chunks {n:,}  region {hi - lo:,} B = {(hi - lo) / n:.3f} B/chunk')
g = bytes.fromhex(ghex)
for name, pat in (('descriptor order', g), ('byte-reversed', g[::-1])):
    p = d.find(pat, lo, hi)
    print(f'{name:17s}: ' + ('absent' if p < 0 else f'at +{p - lo} (mod16 {(p - lo) % 16}, mod20 {(p - lo) % 20})'))
    if p < 0:
        continue
    rel = p - lo
    cands = {}
    if rel % 20 == 0:
        cands['interleaved20'] = struct.unpack_from('>I', d, p + 16)[0]
    if rel % 16 == 0 and rel < n * 16:
        cands['split16+4'] = struct.unpack_from('>I', d, lo + n * 16 + (rel // 16) * 4)[0]
    for lay, idx in cands.items():
        for mname, rec in (('idx', idx), ('idx>>2', idx >> 2), ('(idx&0x7fffffff)>>2', (idx & 0x7FFFFFFF) >> 2),
                           ('idx&0x7fffffff', idx & 0x7FFFFFFF)):
            o = hi + rec * 16
            if o + 16 > len(d):
                continue
            a, b, off, size = struct.unpack_from('>IIII', d, o)
            hit = '  <== oracle size' if size == enc else ''
            print(f'  {lay:13s} idx={idx:#010x} {mname:20s} rec {rec:7d}: a={a:#x} b={b:#x} off={off} size={size}{hit}')
