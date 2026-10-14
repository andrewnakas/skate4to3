"""skate. texture extraction: res descriptor -> best available mip -> RGB image.

Frostbite streams top mips: the installed data holds a resident TAIL (the small
mips) in bundle records at a logicalOffset equal to the sum of the mips above it,
plus - for many textures - ONE catalog-held mip whose decompressed size matches a
mip size exactly. Mip 0 of streamed textures is NOT installed (verified: 5,457
bundles across 24 TOCs, every record for a sampled texture is the identical tail).
So "extract" means: take the highest-resolution mip actually present.

Descriptor layout (180 B, Oodle-compressed in the res payload):
    +0x0C u32 format enum      +0x16 u16 width    +0x18 u16 height
    +0x28 16 B chunk GUID      +0x38 u32[] mip sizes, descending
Catalog GUIDs are stored BYTE-REVERSED relative to the descriptor's.
"""
from __future__ import annotations
import io, os, struct, collections

BC1_FORMATS = {0x36, 0x37, 0x3E}
ONE_BYTE_FORMATS = {0x43, 0x3D, 0x3C, 0x40, 0x06}


def parse_descriptor(blob: bytes) -> dict | None:
    if len(blob) < 0x40:
        return None
    fmt = struct.unpack_from('<I', blob, 0x0C)[0]
    w, h = struct.unpack_from('<HH', blob, 0x16)
    if not w or not h:
        return None
    guid = blob[0x28:0x38]
    mips, off, acc = [], [], 0
    for i in range(12):
        p = 0x38 + 4 * i
        if p + 4 > len(blob):
            break
        m = struct.unpack_from('<I', blob, p)[0]
        if not m:
            break
        mips.append(m); off.append(acc); acc += m
    return {'format': fmt, 'w': w, 'h': h, 'guid': guid.hex(),
            'guid_rev': guid[::-1].hex(), 'mips': mips, 'mip_offsets': off,
            'total': acc}


def best_mip(desc, bundle_ranges, catalog_blobs):
    """Pick the highest-resolution mip fully covered by data we hold.

    bundle_ranges: [(logicalOffset, logicalSize, bytes)]
    catalog_blobs: [bytes] - each a standalone decompressed run
    """
    mips, offs = desc['mips'], desc['mip_offsets']
    for i, size in enumerate(mips):
        lo, hi = offs[i], offs[i] + size
        for (o, l, data) in bundle_ranges:
            if o <= lo and hi <= o + l and len(data) >= (hi - o):
                return i, data[lo - o: hi - o]
        for data in catalog_blobs:
            if len(data) == size:
                return i, data
            if len(data) >= hi:
                return i, data[lo:hi]
    return None, None


# DxTexture pixelFormat -> block codec, MEASURED on this install. Choosing by
# bytes-per-pixel cannot separate these: BC3, BC5, BC6H and BC7 are all 1 B/px, and
# `decode_bc7` does not raise on data that is not BC7 - it returns garbage. Format 61 is
# BC3, and running it through BC7 is what rendered all 2,618 `projectlod` baked atlases -
# about a quarter of the island's surface - as flat magenta, which was then mistaken twice
# for placeholder art.
BC_BY_FORMAT = {61: 'bc3', 67: 'bc7'}
UNKNOWN_FORMATS = set()


def decode(pixels: bytes, w: int, h: int, fmt: int):
    """BC blocks -> PIL RGB image, by the DECLARED pixel format where it is known."""
    from PIL import Image
    bpp = len(pixels) / max(1, w * h)
    def dds(fourcc, pitch):
        hdr = bytearray(128); hdr[0:4] = b'DDS '
        struct.pack_into('<I', hdr, 4, 124)
        struct.pack_into('<I', hdr, 8, 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000)
        struct.pack_into('<I', hdr, 12, h); struct.pack_into('<I', hdr, 16, w)
        struct.pack_into('<I', hdr, 20, max(1, (w + 3) // 4) * pitch)
        struct.pack_into('<I', hdr, 28, 1); struct.pack_into('<I', hdr, 76, 32)
        struct.pack_into('<I', hdr, 80, 0x4); hdr[84:88] = fourcc
        struct.pack_into('<I', hdr, 108, 0x1000)
        return bytes(hdr) + pixels
    if abs(bpp - 0.5) < 0.06:
        return Image.open(io.BytesIO(dds(b'DXT1', 8))).convert('RGB')
    codec = BC_BY_FORMAT.get(fmt)
    if codec == 'bc3':
        return Image.open(io.BytesIO(dds(b'DXT5', 16))).convert('RGB')
    if codec == 'bc7':
        import texture2ddecoder as t2d
        import numpy as np
        raw = t2d.decode_bc7(pixels, w, h)
        a = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0]]
        return Image.fromarray(a, 'RGB')
    UNKNOWN_FORMATS.add(fmt)
    # 1 B/px family, format not in the table: BC7 first, DXT5 as fallback
    try:
        import texture2ddecoder as t2d
        import numpy as np
        raw = t2d.decode_bc7(pixels, w, h)
        a = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0]]
        return Image.fromarray(a, 'RGB')
    except Exception:
        return Image.open(io.BytesIO(dds(b'DXT5', 16))).convert('RGB')


def verified_mip(desc, bundle_ranges, catalog_blobs, tol=24.0):
    """Highest-resolution mip whose bytes are provably THIS texture.

    Bundle ranges are listed under the chunk's own GUID in a bundle manifest, so they are
    trustworthy. Catalog blobs are reached through a byte-reversed GUID and have decoded as
    a DIFFERENT image: oak came back as a normal map, a signage texture as a bouquet of
    flowers, felt as a storefront. So a catalog mip is kept only when, box-downsampled, it
    matches a bundle-resident mip of the same texture; otherwise the resident mip wins.
    Returns (mip, pixels) or (None, None).
    """
    import numpy as np
    from PIL import Image
    mips, offs, fmt = desc['mips'], desc['mip_offsets'], desc['format']
    W, H = desc['w'], desc['h']

    def from_bundle(i):
        lo, hi = offs[i], offs[i] + mips[i]
        for o, l, data in bundle_ranges:
            if o <= lo and hi <= o + l and len(data) >= hi - o:
                return data[lo - o:hi - o]
        return None

    def from_catalog(i):
        for data in catalog_blobs:
            if len(data) == mips[i]:
                return data
            if len(data) == desc['total']:
                return data[offs[i]:offs[i] + mips[i]]
        return None

    ref_i = next((i for i in range(len(mips)) if from_bundle(i) is not None), None)
    if ref_i is None:
        return None, None
    ref_px = from_bundle(ref_i)
    rw, rh = W >> ref_i, H >> ref_i
    ref = None
    if rw >= 8 and rh >= 8:
        try:
            ref = np.asarray(decode(ref_px, rw, rh, fmt), dtype=np.float32)
        except Exception:
            ref = None
    if ref is not None:
        for i in range(ref_i):
            px = from_catalog(i)
            if px is None:
                continue
            try:
                img = decode(px, W >> i, H >> i, fmt).resize((rw, rh), Image.BOX)
            except Exception:
                continue
            if float(np.abs(np.asarray(img, dtype=np.float32) - ref).mean()) < tol:
                return i, px
    return ref_i, ref_px


def looks_like_normal_map(im):
    import numpy as np
    m = np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(0)
    return abs(m[0] - 128) < 14 and abs(m[1] - 128) < 14 and m[2] > 165


def valid_tangent_normal(im):
    """A usable tangent-space normal map points out of the surface: mean blue well above 128."""
    import numpy as np
    m = np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(0)
    return m[2] > 150 and abs(m[0] - 128) < 30 and abs(m[1] - 128) < 30
