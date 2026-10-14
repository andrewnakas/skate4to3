"""Frostbite superbundle TOC + CAS resolution for skate.

Level TOCs (Data/Win32/levels/game/<lvl>/<lvl>.toc) are a big-endian binary index,
not DbObject. See FORMAT-NOTES.md for how the layout was derived; every size
relation below is checked by `validate()`.
"""
from __future__ import annotations

import glob
import os
import struct
from dataclasses import dataclass

BASE = 0x22C          # payload start, common to every TOC in this build
HDR_FIELDS = 15       # 60-byte header

# (field0, field1>>16) -> install package holding that chunk's CAS files.
PACKAGES = {
    (0xBC80, 0xE699): 'initialinstallpackage',
    (0x7837, 0x152C): 'defaultinstallpackage',
}


def package_name(a: int, hi: int) -> str:
    try:
        return PACKAGES[(a, hi)]
    except KeyError:
        raise KeyError(f"unknown install package id ({a:#x}, {hi:#x})") from None


@dataclass(frozen=True)
class CasRange:
    package: str
    cas_index: int
    offset: int
    size: int


@dataclass
class Bundle:
    index: int
    name_offset: int
    entries: list          # [CasRange]; entries[0] is the bundle's own metadata


@dataclass(frozen=True)
class ChunkRef:
    guid: str
    package: str
    cas_index: int
    offset: int
    size: int


class SuperBundle:
    def __init__(self, toc_path: str, game_root: str):
        self.path = toc_path
        self.root = game_root
        self.d = open(toc_path, 'rb').read()
        h = [self.u32(i * 4) for i in range(HDR_FIELDS)]
        (self.bundle_hashmap, self.bundle_data, self.bundles_count,
         self.chunk_hashmap, self.chunk_guids, self.chunks_count) = h[:6]
        self.hdr = h
        self.chunk_records = h[6]      # 16 B per chunk: pkg id, cas index, offset, size
        self.validate()

    def u32(self, off: int) -> int:
        return struct.unpack('>I', self.d[BASE + off:BASE + off + 4])[0]

    def validate(self) -> None:
        """The arithmetic that proves the layout; cheap, so always on."""
        assert self.bundle_hashmap == HDR_FIELDS * 4, self.bundle_hashmap
        # The bundle hashmap is sized for a perfect hash: nb slots in some TOCs, nb+1 in others.
        assert self.bundle_data >= self.bundle_hashmap + self.bundles_count * 4
        assert self.chunk_hashmap == self.bundle_data + self.bundles_count * 16
        assert self.chunk_guids == self.chunk_hashmap + self.chunks_count * 4
        idx = self.chunk_guids + self.chunks_count * 16
        assert self.chunk_records == idx + self.chunks_count * 4, (self.chunk_records, idx)

    def chunk(self, i: int) -> ChunkRef | None:
        """Chunk-table entry i. Layout pending re-measurement (specs/SKATE_FORMATS.md, P1)."""
        raise NotImplementedError('chunk table layout: re-measure in P1')

    def chunks(self):
        return (c for c in (self.chunk(i) for i in range(self.chunks_count)) if c is not None)

    def bundle(self, i: int) -> Bundle:
        """Bundle payloads live in this TOC's tail and are CAS range lists.

        Record (16 B): u32 nameOffset, u32 0x40000000|size, u32 0, u32 offset.
        Payload: 9 x u32 header (flagsOffset at +8, count at +12, entriesOffset
        at +16), then entries, then one flag byte per entry. Flag bit 0x80 means
        an 8-byte (package, cas) id precedes that entry's u32 offset, u32 size;
        otherwise it continues in the previous entry's CAS file. Entry 0 always
        carries flag 0x84. Verified: all 251 bundles walk exactly onto
        flagsOffset, 114,685 entries, none out of bounds.
        """
        o = self.bundle_data + i * 16
        name_off, sizef, _, off = (self.u32(o + j * 4) for j in range(4))
        p = self.d[BASE + off:BASE + off + (sizef & 0xFFFFFF)]
        h = struct.unpack_from('>9I', p, 0)
        flags_off, count, pos = h[2], h[3], h[4]
        flags = p[flags_off:flags_off + count]
        entries, cur = [], None
        for e in range(count):
            if flags[e] & 0x80:
                a, b = struct.unpack_from('>II', p, pos)
                pos += 8
                cur = (package_name(a, b >> 16), b & 0xFFFF)
            eo, es = struct.unpack_from('>II', p, pos)
            pos += 8
            entries.append(CasRange(cur[0], cur[1], eo, es))
        if pos != flags_off:
            raise ValueError(f"bundle {i}: entry walk ended at {pos}, flags at {flags_off}")
        return Bundle(i, name_off, entries)

    def bundles(self):
        return (self.bundle(i) for i in range(self.bundles_count))

    def read_range(self, r) -> bytes:
        with open(self.cas_path(r), 'rb') as f:
            f.seek(r.offset)
            return f.read(r.size)

    def cas_path(self, ref) -> str:
        # Language subdirectories only ever hold cas_01/cas_02, which no level
        # chunk references, so the root-level file is unambiguous.
        return os.path.join(self.root, 'Data/Win32/configurations/layout',
                            ref.package, f'cas_{ref.cas_index:02d}.cas')

    def read(self, ref: ChunkRef) -> bytes:
        with open(self.cas_path(ref), 'rb') as f:
            f.seek(ref.offset)
            return f.read(ref.size)
