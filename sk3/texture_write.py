"""Skate 3 (Xbox 360) texture arena writer (specs/sk3_textures.md).

Two entry points:
  * build(TexSpec) -> arena bytes. A TexSpec holds only decoded fields (format, size, levels, the
    fetch-constant words, the name-table id) and the linear (untiled) block data of every level/face
    in stored byte order. Tiling, base-resource size, mip placement, the name table and the RW4
    arena layout are derived. extract(arena) is the inverse; build(extract(x)) == x for every retail
    world texture arena (tests/test_texture_roundtrip.py).
  * from_image(rgba, name_id, ...) -> TexSpec: mip chain (box filter) -> BC1 (opaque) or BC3 (alpha)
    with our own numpy block encoder -> build().
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from . import rw4_write
from . import texture as T

OBJ_ALIGN = {T.T_BASE: 0x1000, T.T_TEXTURE: 4, T.T_NAMES: 0x10, T.T_EB0008: 0x10}
EB0008 = struct.pack('>II', 0x19, 0x0D)
NAME_HASH = 0x9B0F1678          # name-table word +0x18 in 255/260 small-world textures (else 0x9B0F2374)


@dataclass
class TexSpec:
    arena_id: int
    name_id: int                     # the 64-bit id in the name string '0x%016x.Texture'
    fc_words: tuple                  # 6 fetch-constant words (format/size/levels/... all in here)
    levels: dict                     # {(level, face): linear block bytes, stored byte order}
    table_id: int | None = None      # id in the name-table entry; None = name_id
    name_hash: int = NAME_HASH
    pad: int = 0                     # the name table's last byte (not derivable; retail: varies)
    header: tuple = T.HDR_WORDS

    @property
    def fc(self):
        return T.FetchConstant(tuple(self.fc_words))


# ---------------------------------------------------------------- tiling

def surface_size(fc):
    _n, bw, bpb = T.FORMATS[fc.format]
    return max(e[4] + T._a4k(e[7] * e[8] * bpb) for e in T.level_layout(fc))


def tile(fc, levels: dict) -> bytes:
    """Linear level blocks -> the base resource (all levels, tiled, zero elsewhere)."""
    _n, bw, bpb = T.FORMATS[fc.format]
    out = np.zeros(surface_size(fc), np.uint8)
    cache = {}
    for (lv, f, w, h, off, px, py, sw, sh) in T.level_layout(fc):
        wb, hb = (w + bw - 1) // bw, (h + bw - 1) // bw
        key = (sw, sh)
        if key not in cache:
            cache[key] = T.tiled_offsets(sw, sh, bpb)
        offs = cache[key][py:py + hb, px:px + wb]
        src = np.frombuffer(levels[(lv, f)], np.uint8).reshape(hb * wb, bpb)
        dst = (off + offs.reshape(-1)[:, None] * bpb + np.arange(bpb)).reshape(-1)
        out[dst] = src.reshape(-1)
    return out.tobytes()


def untile_levels(fc, pixels: bytes) -> dict:
    _n, bw, bpb = T.FORMATS[fc.format]
    out = {}
    for (lv, f, w, h, off, px, py, sw, sh) in T.level_layout(fc):
        wb, hb = (w + bw - 1) // bw, (h + bw - 1) // bw
        out[(lv, f)] = T.untile(pixels[off:], wb, hb, bpb, px, py, wb, hb, surf_wb=sw)
    return out


# ---------------------------------------------------------------- objects

def names_object(name_id, table_id, name_hash, pad, dict_index=1):
    name = f'0x{name_id:016x}.Texture'.encode('ascii') + b'\0'
    body = struct.pack('>5I', 1, 0x14, 0x2C, 0, 0) + struct.pack(
        '>IIQII', 0x2C, name_hash, table_id, T.FNV_TEXTURE, dict_index) + name
    size = (len(body) + 3) & ~3
    body += bytes([pad]) + b'\0' * (size - len(body) - 1) if size > len(body) else b''
    b = bytearray(body)
    struct.pack_into('>I', b, 0x10, len(b))
    return bytes(b)


def build(s: TexSpec) -> bytes:
    fc = s.fc
    tex = struct.pack('>13I', *s.header, *s.fc_words)
    nt = names_object(s.name_id, s.name_id if s.table_id is None else s.table_id, s.name_hash, s.pad)
    objs = [rw4_write.SpecObject(T.T_BASE, 0x1000, tile(fc, s.levels)),
            rw4_write.SpecObject(T.T_TEXTURE, 4, tex),
            rw4_write.SpecObject(T.T_NAMES, 0x10, nt),
            rw4_write.SpecObject(T.T_EB0008, 0x10, EB0008)]
    return rw4_write.build(rw4_write.Spec(s.arena_id, objs))


def extract(arena: bytes) -> TexSpec:
    t = T.from_arena(arena)
    n = T.parse_names(t.names_raw)[0]
    name_end = 0x2C + len(n['name']) + 1
    pad = t.names_raw[name_end] if name_end < len(t.names_raw) else 0
    return TexSpec(t.arena_id, t.name_id, t.fc.words, untile_levels(t.fc, t.pixels),
                   None if n['id'] == t.name_id else n['id'], n['hash'], pad, t.header)


# ---------------------------------------------------------------- fetch constant from fields

# every retail world texture (51,316) uses these per format
RETAIL_SWIZZLE = {4: (2, 1, 0, 5), 6: (2, 1, 0, 3), 18: (0, 1, 2, 3), 19: (0, 1, 2, 3), 20: (0, 1, 2, 3)}
RETAIL_ENDIAN = {4: 1, 6: 2, 18: 1, 19: 1, 20: 1}

def fetch_words(fmt, width, height, levels, swizzle=None, endian=None, clamp=(0, 0, 0),
                filters=(0, 0, 0, 0), cube=False, mip_page=None):
    """Fetch-constant words for a new texture, matching the retail field patterns
    (specs/sk3_textures.md 'Fetch constant'). mip_page is derived from the base surface size."""
    if swizzle is None:
        swizzle = RETAIL_SWIZZLE[fmt]
    if endian is None:
        endian = RETAIL_ENDIAN[fmt]
    sw = sum(v << (3 * i) for i, v in enumerate(swizzle))
    _n, bw, _bpb = T.FORMATS[fmt]
    pitch = T._a32((width + bw - 1) // bw) * bw // 32      # field unit: 32 texels
    w0 = 2 | (clamp[0] << 10) | (clamp[1] << 13) | (clamp[2] << 16) | (pitch << 22) | (1 << 31)
    w1 = fmt | (endian << 6)
    w2 = (width - 1) | ((height - 1) << 13) | ((5 if cube else 0) << 26)
    mag, mn, mip, an = filters
    w3 = (sw << 1) | (mag << 19) | (mn << 21) | (mip << 23) | (an << 25)
    w4 = (levels - 1) << 6
    w5 = ((3 if cube else 1) << 9) | ((1 if levels > 1 else 0) << 11)
    fc = T.FetchConstant((w0, w1, w2, w3, w4, w5))
    if mip_page is None:
        mip_page = 0
        if levels > 1:
            ps = T.packed_start(fc)
            if ps != 0:
                _n, bw, bpb = T.FORMATS[fmt]
                l0 = [e for e in T.level_layout(fc) if e[0] == 0]
                mip_page = (T.faces(fc) * T._a4k(l0[0][7] * l0[0][8] * bpb)) >> 12
    w5 |= mip_page << 12
    return (w0, w1, w2, w3, w4, w5)


# ---------------------------------------------------------------- BC1 / BC3 encoder (ours, numpy)

def _to565(rgb):
    r = (rgb[..., 0].astype(np.int32) * 31 + 127) // 255
    g = (rgb[..., 1].astype(np.int32) * 63 + 127) // 255
    b = (rgb[..., 2].astype(np.int32) * 31 + 127) // 255
    return (r << 11) | (g << 5) | b


def _blocks(img, bw=4):
    h, w = img.shape[:2]
    H, W = (h + 3) // 4 * 4, (w + 3) // 4 * 4
    if (H, W) != (h, w):
        img = np.pad(img, ((0, H - h), (0, W - w), (0, 0)), mode='edge')
    return img.reshape(H // 4, 4, W // 4, 4, -1).transpose(0, 2, 1, 3, 4).reshape(-1, 16, img.shape[2])


def encode_bc1_colors(px):
    """px: [n,16,3] uint8 -> [n,8] uint8 little-endian BC1 (4-colour mode, c0 > c1 forced).
    Endpoints: extremes along the block's principal axis, refined by one least-squares pass."""
    p = px.astype(np.float32)
    mean = p.mean(1, keepdims=True)
    d = p - mean
    cov = np.einsum('nki,nkj->nij', d, d)
    v = np.ones((len(p), 3), np.float32)
    for _ in range(8):
        v = np.einsum('nij,nj->ni', cov, v)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
    t = np.einsum('nki,ni->nk', d, v)
    lo = mean[:, 0] + v * t.min(1, keepdims=True)
    hi = mean[:, 0] + v * t.max(1, keepdims=True)
    for _ in range(2):   # least-squares refinement on the 4-level ramp
        span = np.einsum('ni,ni->n', hi - lo, hi - lo)
        tt = np.einsum('nki,ni->nk', p - lo[:, None], hi - lo) / np.maximum(span, 1e-6)[:, None]
        q = np.clip(np.rint(tt * 3), 0, 3) / 3.0
        a, b = 1 - q, q
        aa, bb, ab = (a * a).sum(1), (b * b).sum(1), (a * b).sum(1)
        det = aa * bb - ab * ab
        ok = np.abs(det) > 1e-6
        ax = np.einsum('nk,nki->ni', a, p)
        bx = np.einsum('nk,nki->ni', b, p)
        nlo = (bb[:, None] * ax - ab[:, None] * bx) / np.where(ok, det, 1)[:, None]
        nhi = (aa[:, None] * bx - ab[:, None] * ax) / np.where(ok, det, 1)[:, None]
        lo = np.where(ok[:, None], np.clip(nlo, 0, 255), lo)
        hi = np.where(ok[:, None], np.clip(nhi, 0, 255), hi)
    c0 = _to565(np.clip(np.rint(hi), 0, 255))
    c1 = _to565(np.clip(np.rint(lo), 0, 255))
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    eq = c0 == c1
    # palette as the decoder will see it, pick the nearest entry per texel
    pal = np.stack([T._565(c0), T._565(c1)], 1).astype(np.float32)
    pal = np.concatenate([pal, ((2 * pal[:, :1] + pal[:, 1:2]) // 3), ((pal[:, :1] + 2 * pal[:, 1:2]) // 3)], 1)
    dist = ((p[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)
    sel = dist.argmin(-1)
    sel = np.where(eq[:, None], 0, sel)
    bits = (sel.astype(np.uint32) << (2 * np.arange(16, dtype=np.uint32))).sum(1).astype(np.uint32)
    out = np.zeros((len(p), 8), np.uint8)
    out[:, 0], out[:, 1] = c0 & 0xFF, c0 >> 8
    out[:, 2], out[:, 3] = c1 & 0xFF, c1 >> 8
    for k in range(4):
        out[:, 4 + k] = (bits >> (8 * k)) & 0xFF
    return out


def encode_bc3_alpha(a):
    """a: [n,16] uint8 -> [n,8] BC3 alpha blocks (8-level mode, a0 > a1)."""
    a = a.astype(np.int32)
    hi, lo = a.max(1), a.min(1)
    hi = np.where(hi == lo, np.minimum(hi + 1, 255), hi)
    lo = np.where(hi == lo, lo - 1, lo)
    k = np.arange(8)
    pal = np.zeros((len(a), 8), np.int32)
    pal[:, 0], pal[:, 1] = hi, lo
    for j in range(1, 7):
        pal[:, j + 1] = ((7 - j) * hi + j * lo) // 7
    sel = np.abs(a[:, :, None] - pal[:, None, :]).argmin(-1).astype(np.uint64)
    bits = (sel << (3 * np.arange(16, dtype=np.uint64))).sum(1)
    out = np.zeros((len(a), 8), np.uint8)
    out[:, 0], out[:, 1] = hi, lo
    for j in range(6):
        out[:, 2 + j] = ((bits >> np.uint64(8 * j)) & np.uint64(0xFF)).astype(np.uint8)
    return out


def encode_level(img, fmt):
    """RGBA uint8 [h,w,4] -> linear little-endian blocks."""
    b = _blocks(img)
    if fmt == 18:
        return encode_bc1_colors(b[..., :3]).tobytes()
    if fmt == 20:
        return np.concatenate([encode_bc3_alpha(b[..., 3]), encode_bc1_colors(b[..., :3])], 1).tobytes()
    raise ValueError(fmt)


def mip_chain(img, levels):
    """Box-filtered chain, level k = max(1, dim >> k) (dims must be powers of two)."""
    out = [img]
    cur = img.astype(np.float32)
    for _ in range(1, levels):
        h, w = cur.shape[:2]
        nh, nw = max(1, h // 2), max(1, w // 2)
        cur = cur[:nh * (h // nh), :nw * (w // nw)].reshape(nh, h // nh, nw, w // nw, 4).mean((1, 3))
        out.append(np.clip(np.rint(cur), 0, 255).astype(np.uint8))
    return out


def from_image(rgba, name_id, arena_id, alpha=None, levels=None, lowres=False, clamp=(0, 0, 0)):
    """RGBA uint8 [h,w,4] (power-of-two sides, >= 4) -> TexSpec. BC1 when every alpha is 255 (or
    alpha=False), else BC3. levels: full chain by default."""
    rgba = np.asarray(rgba, np.uint8)
    h, w = rgba