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


# A ReSkate mod (SKATE_MOD_ROOT = Mods/<name>) ships its own CAS files under Win32/configurations/
# layout/<package>/; its TOCs flag those with bit 16 of the package word (0x17837 = the mod's
# defaultinstallpackage; measured on DM_Jumpline and zeex64-Full_Skate_3_Map). Such a package
# resolves to an ABSOLUTE directory, which os.path.join(LAYOUT, pkg, ...) then follows unchanged.
MOD_ROOT = os.environ.get('SKATE_MOD_ROOT', '')
MOD_FLAG = 0x10000


def package_name(a: int, hi: int) -> str:
    try:
        if a & MOD_FLAG:
            if not MOD_ROOT:
                raise KeyError
            return os.path.join(MOD_ROOT, 'Win32', 'configurations', 'layout', PACKAGES[(a & 0xFFFF, hi)])
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
        # 15 header words in the game's own TOCs, 12 in ReSkate Studio mod TOCs: word 0 (the
        # bundle hash map offset) is the header length either way.
        self.hdr_fields = self.u32(0) // 4
        h = [self.u32(i * 4) for i in range(self.hdr_fields)]
        (self.bundle_hashmap, self.bundle_data, self.bundles_count,
         self.chunk_hashmap, self.chunk_guids, self.chunks_count) = h[:6]
        self.hdr = h
        self.chunk_records = h[6]      # 16 B per chunk: pkg id, cas index, offset, size
        self.validate()

    def u32(self, off: int) -> int:
        return struct.unpack('>I', self.d[BASE + off:BASE + off + 4])[0]

    def validate(self) -> None:
        """The arithmetic that proves the layout; cheap, so always on."""
        assert self.hdr_fields in (12, 15), self.bundle_hashmap
        # The bundle hashmap is sized for a perfect hash: nb slots in some TOCs, nb+1 in others.
        assert self.bundle_data >= self.bundle_hashmap + self.bundles_count * 4
        if self.chunks_count == 0:           # Studio writes 0 for every chunk-table offset then
            return
        assert self.chunk_hashmap == self.bundle_data + self.bundles_count * 16
        assert self.chunk_guids == self.chunk_hashmap + self.chunks_count * 4
        # chunk table: 20 B per chunk (16 B guid + u32 index)
        assert self.chunk_records == self.chunk_guids + self.chunks_count * 20, self.chunk_records

    def chunk(self, i: int) -> ChunkRef | None:
        """Chunk-table entry i (layout measured in specs/SKATE_FORMATS.md, "TOC chunk table").

        20 B per entry: 16 B guid stored byte-reversed, then u32 BE idx. idx 0xFFFFFFFF = no
        record; otherwise the high bit is set and the low bits are a byte offset into a table of
        4-byte words, so the 16-byte location record is number (idx & 0x7FFFFFFF) >> 2.
        Returned guid is in descriptor order (as DxTexture / MeshSet reference it).
        """
        g = BASE + self.chunk_guids + i * 20
        guid = self.d[g:g + 16][::-1].hex()
        idx = struct.unpack_from('>I', self.d, g + 16)[0]
        if idx == 0xFFFFFFFF:
            return None
        o = BASE + self.chunk_records + ((idx & 0x7FFFFFFF) >> 2) * 16
        a, b, off, size = struct.unpack_from('>IIII', self.d, o)
        return ChunkRef(guid, package_name(a, b >> 16), b & 0xFFFF, off, size)

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
