"""Skate 3 (Xbox 360) texture arenas: decode (specs/sk3_textures.md).

A texture arena holds exactly four objects, in this dictionary order:
    [0] 0x10031 base resource  - the GPU pixel data (tiled, 0x1000 aligned in the graphics region)
    [1] 0x200e8 texture        - 52 B: 7 header words + the 24-byte Xenos texture fetch constant
    [2] 0xeb000b name table    - 72 B: one entry {name, 64-bit id, type hash, dictionary index}
    [3] 0xeb0008               - 8 B, always 00000019 0000000d

Pixel layout (all measured, see the spec): every level is stored Xenos-2D-tiled; the base level at
the base resource start, mip levels from (fetch dword5 >> 12) << 12; each level's surface is
padded to 32x32 blocks; levels whose larger side is <= 16 texels share one packed 32x32-block tile.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from . import rw4

T_BASE, T_TEXTURE, T_NAMES, T_EB0008 = 0x10031, 0x200e8, 0xeb000b, 0xeb0008
HDR_WORDS = (3, 1, 0, 0, 0, 0xFFFF0000, 0xFFFF0000)   # identical in every retail texture
FNV_TEXTURE = 0xAC462E4A          # FNV-1 32 of b'Texture'
LOWRES_FLAG = 0x40000000_00000000  # set in the name-table id of the low-res stand-in copy

# Xenos texture formats seen (fetch dword1 & 0x3F): name, block w/h, bytes per block
FORMATS = {
    4: ('R5G6B5', 1, 2),
    6: ('A8R8G8B8', 1, 4),
    18: ('DXT1', 4, 8),
    19: ('DXT3', 4, 16),
    20: ('DXT5', 4, 16),
}


def _bits(v, lo, n):
    return (v >> lo) & ((1 << n) - 1)


@dataclass
class FetchConstant:
    """The 6-dword GPUTEXTURE_FETCH_CONSTANT (public Xenos layout), decoded to the fields we use;
    `words` keeps all 192 bits."""
    words: tuple

    tiled = property(lambda s: _bits(s.words[0], 31, 1))
    pitch = property(lambda s: _bits(s.words[0], 22, 9) * 32)          # texels
    clamp = property(lambda s: tuple(_bits(s.words[0], 10 + 3 * i, 3) for i in range(3)))
    format = property(lambda s: _bits(s.words[1], 0, 6))
    endian = property(lambda s: _bits(s.words[1], 6, 2))
    base_page = property(lambda s: _bits(s.words[1], 12, 20))
    width = property(lambda s: _bits(s.words[2], 0, 13) + 1)
    height = property(lambda s: _bits(s.words[2], 13, 13) + 1)
    num_format = property(lambda s: _bits(s.words[3], 0, 1))
    swizzle = property(lambda s: tuple(_bits(s.words[3], 1 + 3 * i, 3) for i in range(4)))
    mag_filter = property(lambda s: _bits(s.words[3], 19, 2))
    min_filter = property(lambda s: _bits(s.words[3], 21, 2))
    mip_filter = property(lambda s: _bits(s.words[3], 23, 2))
    aniso = property(lambda s: _bits(s.words[3], 25, 3))
    mip_min = property(lambda s: _bits(s.words[4], 2, 4))
    mip_max = property(lambda s: _bits(s.words[4], 6, 4))
    dimension = property(lambda s: _bits(s.words[5], 9, 2))
    packed_mips = property(lambda s: _bits(s.words[5], 11, 1))
    mip_page = property(lambda s: _bits(s.words[5], 12, 20))

    @property
    def levels(self):
        return self.mip_max + 1


@dataclass
class Texture:
    name: str                 # '0x<16 hex>.Texture'
    id: int                   # 64-bit id from the name table (LOWRES_FLAG may be set)
    fc: FetchConstant
    header: tuple             # the 7 words before the fetch constant
    pixels: bytes             # base resource bytes (tiled, all levels)
    arena_id: int = 0
    names_raw: bytes = b''    # the 0xeb000b object as stored (its pad byte is not derivable)
    extra: dict = field(default_factory=dict)

    @property
    def format_name(self):
        return FORMATS.get(self.fc.format, (f'fmt{self.fc.format}',))[0]

    @property
    def name_id(self):
        return int(self.name[2:18], 16)


def parse_names(b: bytes):
    """0xeb000b: {count, entries at, strings at, 0, size} then per entry
    {name offset, u32 ?, u64 id, u32 type hash, u32 dictionary index}, then the NUL-ended names."""
    count, ent, _strs, _z, size = struct.unpack_from('>5I', b, 0)
    out = []
    for k in range(count):
        noff, h, gid, thash, idx = struct.unpack_from('>IIQII', b, ent + 0x18 * k)
        name = b[noff:b.index(b'\0', noff)].decode('ascii')
        out.append(dict(name=name, hash=h, id=gid, type_hash=thash, index=idx))
    return out


def from_arena(data: bytes) -> Texture:
    a = rw4.parse(data)
    t = a.of_type(T_TEXTURE)
    if len(t) != 1:
        raise ValueError(f'{len(t)} texture objects')
    w = struct.unpack('>13I', t[0].data)
    nb = a.of_type(T_NAMES)[0].data
    n = parse_names(nb)[0]
    return Texture(n['name'], n['id'], FetchConstant(w[7:]), w[:7], a.of_type(T_BASE)[0].data,
                   struct.unpack_from('>I', data, 0x1C)[0], nb, dict(name_hash=n['hash']))


# ---------------------------------------------------------------- Xenos tiling (public address math)

def _log_bpb(bpb):
    return {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[bpb]


def tiled_offsets(wb, hb, bpb):
    """Block index (in units of bpb) of every (y, x) block of a tiled surface wb x hb blocks
    (wb = row pitch in blocks, already 32-aligned). Returns an int array [hb, wb]."""
    lb = _log_bpb(bpb)
    y, x = np.mgrid[0:hb, 0:wb].astype(np.int64)
    aw = (wb + 31) & ~31
    macro = ((x >> 5) + (y >> 5) * (aw >> 5)) << (lb + 7)
    micro = ((x & 7) + ((y & 6) << 2)) << lb
    off = macro + ((micro & ~15) << 1) + (micro & 15) + ((y & 8) << (3 + lb)) + ((y & 1) << 4)
    return (((off & ~511) << 3) + ((off & 448) << 2) + (off & 63) + ((y & 16) << 7)
            + (((((y & 8) >> 2) + (x >> 3)) & 3) << 6)) >> lb


def _next_pow2(v):
    return 1 << max(0, (v - 1).bit_length())


def _a32(v):
    return (v + 31) & ~31


def _a4k(v):
    return (v + 0xFFF) & ~0xFFF


def faces(fc: FetchConstant):
    return 6 if fc.dimension == 3 else 1


def level_dims(fc: FetchConstant, lv: int):
    if lv == 0:
        return fc.width, fc.height
    return max(1, _next_pow2(fc.width) >> lv), max(1, _next_pow2(fc.height) >> lv)


def packed_start(fc: FetchConstant):
    """First level stored in the shared packed tile (min side <= 16 texels), or None."""
    if not fc.packed_mips:
        return None
    for lv in range(fc.levels):
        if min(level_dims(fc, lv)) <= 16:
            return lv
    return None


def level_layout(fc: FetchConstant):
    """[(level, face, width, height, surface byte offset, block x, block y, surface wb, surface hb)].
    Facts in specs/sk3_textures.md 'Mip layout'."""
    _n, bw, bpb = FORMATS[fc.format]
    nf = faces(fc)
    ps = packed_start(fc)
    out = []
    off = 0
    for lv in range(fc.levels):
        if lv == 1:
            off = fc.mip_page << 12
        if ps is not None and lv > ps:
            continue
        w, h = level_dims(fc, lv)
        sw, sh = _a32((w + bw - 1) // bw), _a32((h + bw - 1) // bw)
        size = _a4k(sw * sh * bpb)
        for f in range(nf):
            if ps is not None and lv == ps:
                for k in range(lv, fc.levels):
                    kw, kh = level_dims(fc, k)
                    px, py = packed_tile_offset(k - ps, w, h, bw)
                    out.append((k, f, kw, kh, off + f * size, px, py, sw, sh))
            else:
                out.append((lv, f, w, h, off + f * size, 0, 0, sw, sh))
        off += nf * size
    return out


def packed_tile_offset(t, w, h, bw):
    """Block (x, y) of the t-th packed level inside the packed surface whose first level is w x h.
    Measured rule, see specs/sk3_textures.md 'Packed mip tail': wide (w > h) levels 0-2 step down
    in y (16, 8, 4 texels), later ones step along x (4 x their own width); square/tall the same with
    x and y swapped."""
    wide = w > h
    major = w if wide else h
    along_minor, along_major = (16 >> t, 0) if t < 3 else (0, 4 * max(1, major >> t))
    # t < 3: step across the short axis (y for wide, x for square/tall); later: along the long axis
    x, y = (along_major, along_minor) if wide else (along_minor, along_major)
    return x // bw, y // bw


# ---------------------------------------------------------------- block decoders (public formats)

def _swap16(b: bytes) -> bytes:
    a = np.frombuffer(b, np.uint8)
    if len(a) & 1:
        a = np.concatenate([a, [0]]).astype(np.uint8)
    return a.reshape(-1, 2)[:, ::-1].tobytes()


def _swap32(b: bytes) -> bytes:
    a = np.frombuffer(b, np.uint8)
    return a.reshape(-1, 4)[:, ::-1].tobytes()


def endian_fix(b: bytes, endian: int) -> bytes:
    """Xenos endian field: 0 none, 1 8-in-16, 2 8-in-32, 3 16-in-32."""
    if endian == 1:
        return _swap16(b)
    if endian == 2:
        return _swap32(b)
    if endian == 3:
        a = np.frombuffer(b, '<u2').reshape(-1, 2)[:, ::-1]
        return a.tobytes()
    return b


def _565(c):
    c = c.astype(np.uint32)
    r = (c >> 11) & 31
    g = (c >> 5) & 63
    b = c & 31
    return np.stack([(r * 255 + 15) // 31, (g * 255 + 31) // 63, (b * 255 + 15) // 31], -1)


def _bc1_colors(c0, c1, force4=False):
    p0, p1 = _565(c0).astype(np.int32), _565(c1).astype(np.int32)
    four = (c0 > c1) | force4
    p2 = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four[:, None], (p0 + 2 * p1) // 3, 0)
    pal = np.stack([p0, p1, p2, p3], 1)                     # n,4,3
    alpha = np.where(four[:, None], 255, np.array([255, 255, 255, 0]))
    return pal, alpha


def decode_bc1(blocks: bytes, force4=False):
    """little-endian BC1 blocks -> uint8 [n, 16, 4] (texel order row-major in the 4x4 block)."""
    a = np.frombuffer(blocks, np.uint8).reshape(-1, 8)
    c0 = a[:, 0].astype(np.uint16) | (a[:, 1].astype(np.uint16) << 8)
    c1 = a[:, 2].astype(np.uint16) | (a[:, 3].astype(np.uint16) << 8)
    idx = a[:, 4:8].astype(np.uint32)
    bits = idx[:, 0] | (idx[:, 1] << 8) | (idx[:, 2] << 16) | (idx[:, 3] << 24)
    sel = (bits[:, None] >> (2 * np.arange(16))) & 3
    pal, alpha = _bc1_colors(c0, c1, force4)
    n = len(a)
    rgb = pal[np.arange(n)[:, None], sel]
    al = np.broadcast_to(alpha, (n, 4))[np.arange(n)[:, None], sel]
    return np.concatenate([rgb, al[..., None]], -1).astype(np.uint8)


def decode_bc3_alpha(a8):
    """[n,8] BC3 alpha blocks -> [n,16] alpha."""
    a0 = a8[:, 0].astype(np.int32)
    a1 = a8[:, 1].astype(np.int32)
    bits = np.zeros(len(a8), np.uint64)
    for k in range(6):
        bits |= a8[:, 2 + k].astype(np.uint64) << np.uint64(8 * k)
    sel = ((bits[:, None] >> (3 * np.arange(16, dtype=np.uint64))) & np.uint64(7)).astype(np.int32)
    eight = a0 > a1
    p = np.zeros((len(a8), 8), np.int32)
    p[:, 0], p[:, 1] = a0, a1
    for k in range(1, 7):
        p[:, k + 1] = ((7 - k) * a0 + k * a1) // 7
    for k in range(1, 5):
        p[:, k + 1] = np.where(eight, p[:, k + 1], ((5 - k) * a0 + k * a1) // 5)
    p[:, 6] = np.where(eight, p[:, 6], 0)
    p[:, 7] = np.where(eight, p[:, 7], 255)
    return p[np.arange(len(a8))[:, None], sel].astype(np.uint8)


def decode_bc3(blocks: bytes):
    a = np.frombuffer(blocks, np.uint8).reshape(-1, 16)
    out = decode_bc1(a[:, 8:].tobytes(), force4=True)
    out[..., 3] = decode_bc3_alpha(a[:, :8])
    return out


def decode_bc2(blocks: bytes):
    a = np.frombuffer(blocks, np.uint8).reshape(-1, 16)
    out = decode_bc1(a[:, 8:].tobytes(), force4=True)
    nib = np.stack([a[:, :8] & 15, a[:, :8] >> 4], -1).reshape(-1, 16)
    out[..., 3] = nib * 17
    return out


def blocks_to_rgba(lin: bytes, fmt: int, wb: int, hb: int):
    """Linear (row-major) little-endian blocks of a wb x hb block surface -> RGBA uint8 [h, w, 4]."""
    name, bw, bpb = FORMATS[fmt]
    if bw == 4:
        dec = {18: decode_bc1, 19: decode_bc2, 20: decode_bc3}[fmt](lin)
        return dec.reshape(hb, wb, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(hb * 4, wb * 4, 4)
    if fmt == 4:
        c = np.frombuffer(lin, '<u2').reshape(hb, wb)
        rgb = _565(c).astype(np.uint8)
        return np.concatenate([rgb, np.full((hb, wb, 1), 255, np.uint8)], -1)
    if fmt == 6:   # A8R8G8B8 as a little-endian u32 -> bytes B G R A
        c = np.frombuffer(lin, np.uint8).reshape(hb, wb, 4)
        return c[..., [2, 1, 0, 3]].copy()
    raise ValueError(f'format {fmt}')


def untile(surface: bytes, wb: int, hb: int, bpb: int, x0=0, y0=0, w=None, h=None, surf_wb=None):
    """Read a w x h block rectangle at block (x0, y0) of a tiled surface (row pitch surf_wb blocks)
    -> linear bytes."""
    surf_wb = surf_wb or wb
    w, h = w or wb, h or hb
    offs = tiled_offsets(surf_wb, y0 + h, bpb)[y0:y0 + h, x0:x0 + w]
    src = np.frombuffer(surface, np.uint8)
    idx = offs[..., None] * bpb + np.arange(bpb)
    return src[idx.reshape(-1)].tobytes()


def decode_level(tex: Texture, lv: int, face: int = 0):
    """RGBA uint8 [h, w, 4] of one level."""
    fc = tex.fc
    name, bw, bpb = FORMATS[fc.format]
    for (l, f, w, h, off, px, py, sw, sh) in level_layout(fc):
        if l == lv and f == face:
            wb, hb = (w + bw - 1) // bw, (h + bw - 1) // bw
            lin = untile(tex.pixels[off:], wb, hb, bpb, px, py, wb, hb, surf_wb=sw)
            img = blocks_to_rgba(endian_fix(lin, fc.endian), fc.format, wb, hb)
            return img[:h, :w]
    raise IndexError(lv)


def to_png(tex: Texture, path: str, lv: int = 0, face: int = 0):
    from PIL import Image
    Image.fromarray(decode_level(tex, lv, face), 'RGBA').save(path)


def contact_sheet(textures, path, cell=128, cols=8):
    """Level 0 of each texture on a magenta background (alpha shows), one cell each."""
    from PIL import Image
    textures = list(textures)
    S = Image.new('RGB', (cell * cols, cell * max(1, (len(textures) + cols - 1) // cols)), (30, 30, 30))
    for j, t in enumerate(textures):
        a = Image.fromarray(decode_level(t, 0), 'RGBA')
        bg = Image.new('RGBA', a.size, (255, 0, 255, 255))
        bg.alpha_composite(a)
        S.paste(bg.convert('RGB').resize((cell, cell), Image.NEAREST), ((j % cols) * cell, (j // cols) * cell))
    S.save(path)


if __name__ == '__main__':
    # python -m sk3.texture <worldDIST_X.big> <out dir> [max count]: PNGs + contact sheet
    import os
    import sys
    from . import big, sfil
    src, out = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    os.makedirs(out, exist_ok=True)
    arc, done = big.read(src), []
    for e in arc.entries:
        if e.name.endswith('.xsf') and len(done) < cap:
            for x in sfil.read(arc.read(e))[1]:
                if len(done) < cap and rw4.parse(x.arena).of_type(T_TEXTURE):
                    t = from_arena(x.arena)
                    to_png(t, os.path.join(out, f'{t.name_id:016x}{"_lo" if t.id & LOWRES_FLAG else ""}.png'))
                    done.append(t)
    contact_sheet(done, os.path.join(out, 'sheet.png'))
    print(f'{len(done)} textures -> {out}')
