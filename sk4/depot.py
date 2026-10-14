"""skate. ShaderBlockDepot resources (res type 0x73312045): material key -> typed parameters.

Layout measured in specs/SKATE_FORMATS.md ("ShaderBlockDepot"); every one of 64,330 blocks in
2,475 depots walks exactly to its end with the element sizes below.

    resMeta  u32 relocOffset, u32 relocSize, u16 ?, u16 ?, u32 keyCount
    +0x00    u64 -> key table (always 0x10)
    key      {u64 materialKey, u64 -> descriptor}    (keys share descriptors)
    desc     {u64 -> block, u64 blockHash, u64 blockSize}
    block    u32 recordCount, then records:
             u64 paramHash, u32 type, u16 count, u16 id, u8 count&0xFF, payload
             texture: count x {16 B instance guid, 16 B partition guid} of the TextureAsset EBX
             others:  3 zero bytes, then count x element (size by type)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

TEXTURE = 0x14C5FC30
VEC3 = 0x885EFF59
FLOAT = 0x2BB65C2D
ELEMENT = {          # type -> element size in bytes (measured)
    0x566E2DD0: 1,
    0x57F970CA: 4, FLOAT: 4, 0x36ED3444: 4, 0x8E4A817E: 4, 0x7826E998: 4,
    0x2A36E9D5: 8,
    VEC3: 12,
    0x7F999829: 16,
}


@dataclass
class Param:
    hash: int
    type: int
    count: int
    id: int
    raw: bytes                     # payload after the 3 pad bytes (values), or the guid pairs

    @property
    def textures(self) -> list[tuple[str, str]]:
        """[(instance guid hex, partition guid hex)] for a texture parameter."""
        return [(self.raw[i:i + 16].hex(), self.raw[i + 16:i + 32].hex())
                for i in range(0, len(self.raw), 32)] if self.type == TEXTURE else []

    @property
    def floats(self) -> tuple:
        if self.type in (VEC3, FLOAT, 0x7F999829):
            return struct.unpack(f'<{len(self.raw) // 4}f', self.raw)
        return ()


def parse_block(b: bytes) -> list[Param]:
    n = struct.unpack_from('<I', b, 0)[0]
    p, out = 4, []
    for _ in range(n):
        h, t, c, ident, _c8 = struct.unpack_from('<QIHHB', b, p)
        p += 17
        if t == TEXTURE:
            ln, skip = 32 * c, 0
        else:
            ln, skip = c * ELEMENT[t], 3
        out.append(Param(h, t, c, ident, b[p + skip:p + skip + ln]))
        p += skip + ln
    if any(b[p:]):
        raise ValueError(f'block walk ended at {p} of {len(b)} with data left')
    return out


def parse(d: bytes, meta: bytes) -> dict[int, list[Param]]:
    """material key -> parameters of its block."""
    nkeys = struct.unpack_from('<I', meta, 12)[0]
    kt = struct.unpack_from('<Q', d, 0)[0]
    blocks, out = {}, {}
    for k in range(nkeys):
        key, dp = struct.unpack_from('<QQ', d, kt + 16 * k)
        if dp not in blocks:
            bp, _bh, bs = struct.unpack_from('<QQQ', d, dp)
            blocks[dp] = parse_block(d[bp:bp + bs])
        out[key] = blocks[dp]
    return out
