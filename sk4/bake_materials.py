"""skate.'s layered materials -> textures the exporter can draw.

The ShaderBlockDepot block of a material (sk4/depot.py parses it) is a list of typed
parameter records:

    u32 N; N x { u64 paramHash, u32 typeId, u16 count, u16 tag, payload }
    typeId 0x14C5FC30 texture: u8 flag + instance guid(16) + partition guid(16)
    typeId 0x885EFF59 float3[count]   0x2A36E9D5 float2   0x2BB65C2D float
    typeId 0x8E4A817E int[count]      0x36ED3444 bool32   (others: see PLAN.md)
    every record after the fixed header carries a u32 count before its payload.

The tag is constant per parameter across materials. What this bakes:

* ROCK / TERRAIN-ROCK family (the cliff, scree and boulder rocks - about half the drawn ground).
  Colour is a 3-slice texture ARRAY at tag 0x739C (terrain_rock_02: 0 rock, 1 grass, 2 grass
  with twigs), stored mip-major. The rock slice is the base; the slice at tag 0xAAEF is laid on
  faces flatter than the angle at tag 0xC89D (25 deg on the cliffs; 345-360 = never). The
  converter used to draw slice 0 only - grey rock where skate. shows grass tops.
  -> work/textures/ARRAY_<name>_s<i>.png per slice, and work/rock_materials.json
     {depot key: {base: png, top: png, slope_deg}}.

* TERRAINMESH presets (`TerrainMesh_FetchVirtualTextureForMesh_M_<Surface>`): the block's only
  colour is a float3 at tag 0x8C54 - the surface's own base colour (Grass 0.053,0.065,0.018;
  Asphalt_Dried 0.063,0.055,0.050 ...), identical across the preset. The texture the converter
  guessed by name is re-baked so its linear mean equals that colour.
  -> work/textures/TERR_<key>.png, and the depot_key_textures.json entry repointed at it
     (original kept as `orig_png`).

    SKATE_WORK=work/mpr python tools/bake_materials.py
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bind_keys import load_depots  # noqa: E402
import depot  # noqa: E402
import textures  # noqa: E402
from cas import decompress_many  # noqa: E402
from texcache import TexCache, WORK  # noqa: E402

T_TEX, T_F3, T_F2, T_F, T_I = 0x14C5FC30, 0x885EFF59, 0x2A36E9D5, 0x2BB65C2D, 0x8E4A817E
ELEM = {T_F3: 12, T_F2: 8, 0x566E2DD0: 1, 0x7F999829: 16}
TAG_ARRAY, TAG_TOP, TAG_SLOPE, TAG_TERRAIN_COLOUR = 0x739C, 0xAAEF, 0xC89D, 0x8C54
# SEGMENT family (slab/panel sheets: concrete slabs, plywood panels): detail colour at 0xEA1D
# tiled 0.5/m, a segment sheet at 0xF7A9 on uv0 (`_uvw`: R,G = per-slab random offset, B = wear),
# per-slab palette float3[4] at 0xD220 (else 0x1632 / 0xF1E9).
TAG_SEG_DETAIL, TAG_SEG_SHEET, TAG_SEG_PAL = 0xEA1D, 0xF7A9, (0xD220, 0x1632, 0xF1E9)


def floats(rec):
    return struct.unpack(f'<{len(rec[2]) // 4}f', rec[2])


def ints(rec):
    return struct.unpack(f'<{len(rec[2]) // 4}I', rec[2])


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def array_slices(tc, name, px=1024):
    """RGB slices of a texture array at the largest mip <= px (mip-major layout)."""
    v = tc.res.get(name)
    raw_d = decompress_many([tc.rd(*v)])[0]
    d = textures.parse_descriptor(raw_d)
    nsl = struct.unpack_from('<H', raw_d, 0x1C)[0] or 1
    c = tc.cat.get(d['guid']) or tc.cat.get(d['guid_rev'])
    if not c:
        return []
    raw = decompress_many([tc.rd(*c)])[0]
    mips = d['mips']
    # the chunk holds mips from some first level: find the level whose total size matches
    need = sum(mips) * nsl
    first = 0
    while first < len(mips) and sum(mips[first:]) * nsl > len(raw):
        first += 1
    mip = first
    while (d['w'] >> mip) > px and mip + 1 < len(mips):
        mip += 1
    w, h = d['w'] >> mip, d['h'] >> mip
    base = sum(mips[first:mip]) * nsl
    m = mips[mip]
    out = []
    for s in range(nsl):
        px_ = raw[base + s * m: base + (s + 1) * m]
        im = textures.decode(px_, w, h, d['format'])
        out.append(np.asarray(im.convert('RGB')))
    _ = need
    return out


def tex_rgba(tc, name):
    got = tc.fetch(name)
    return None if got is None else np.asarray(got[3].convert('RGB'), np.float64) / 255.0


def bake_segment(detail, sheet, pal, K, N=1024):
    """One sheet repeat: detail offset per slab by the sheet's R,G, tinted by pal[R*4], worn by B."""
    g = (np.arange(N) + 0.5) / N
    U, V = np.meshgrid(g, g)

    def samp(img, u, v):
        h, w = img.shape[:2]
        return img[((v % 1) * h).astype(int) % h, ((u % 1) * w).astype(int) % w]
    seg = samp(sheet, U, V)
    R, G, B = seg[..., 0], seg[..., 1], seg[..., 2]
    det = samp(detail, U * K + R, V * K + G)
    dl = srgb_to_linear(det) / np.maximum(srgb_to_linear(detail).reshape(-1, 3).mean(axis=0), 1e-4)
    idx = np.clip(np.floor(R * len(pal)), 0, len(pal) - 1).astype(int)
    out = dl * np.asarray(pal)[idx]
    out = out * (0.75 + 0.5 * (B - B.mean()))[..., None]
    return (linear_to_srgb(out) * 255 + 0.5).astype(np.uint8)


def main():
    RES = json.load(open(os.path.join(WORK, 'global_assets.json')))['res']
    g2t = {bytes.fromhex(k): v for k, v in
           json.load(open(os.path.join(WORK, 'guid_to_texture.json'))).items()}
    del RES
    mats, _ = load_depots()
    dt = json.load(open(os.path.join(WORK, 'depot_textures.json')))
    kt_path = os.path.join(WORK, 'depot_key_textures.json')
    kt = json.load(open(kt_path))
    tc = TexCache()
    os.makedirs(os.path.join(WORK, 'textures'), exist_ok=True)

    rock, slices_done, terr = {}, {}, 0
    seg, tex_cache = {}, {}
    dens = {}
    dp_ = os.path.join(WORK, 'uv_density.json')
    if os.path.exists(dp_):
        for r in json.load(open(dp_)):
            if r.get('measured'):
                dens.setdefault(r['material'], r['measured'])
    done_presets = set()
    for k, rec in dt.items():
        if not k.isdigit():
            continue
        params = mats.get(int(k))
        if params is None:
            continue
        P = {p.tag: ((p.type, 1, b'' + p.raw[:32]) if p.type == depot.TEXTURE else (p.type, p.count, p.raw))
             for p in params}
        if TAG_SEG_DETAIL in P and TAG_SEG_SHEET in P and any(t in P for t in TAG_SEG_PAL):
            dn = g2t.get(P[TAG_SEG_DETAIL][2][17:33])
            sn = g2t.get(P[TAG_SEG_SHEET][2][17:33])
            pt = next(t for t in TAG_SEG_PAL if t in P)
            f3 = floats(P[pt])
            pal = [f3[i:i + 3] for i in range(0, len(f3), 3)]
            mats = rec.get('materials', [])
            m_dens = next((dens[m] for m in mats if m in dens), 0.1)
            if not dn or not sn or not pal:
                continue
            for nm in (dn, sn):
                if nm not in tex_cache:
                    tex_cache[nm] = tex_rgba(tc, nm)
            if tex_cache[dn] is None or tex_cache[sn] is None:
                continue
            K = 0.5 / max(m_dens, 1e-3)
            out = bake_segment(tex_cache[dn], tex_cache[sn], pal, K)
            fn = f'textures/SEG_{k}.png'
            Image.fromarray(out).save(os.path.join(WORK, fn), optimize=True)
            seg[k] = {'png': fn, 'detail': dn, 'sheet': sn, 'K': round(K, 3), 'palette': [list(map(float, c)) for c in pal],
                      'materials': mats[:3]}
            continue
        if TAG_ARRAY in P:
            name = g2t.get(P[TAG_ARRAY][2][17:33])
            if not name:
                continue
            if name not in slices_done:
                sl = array_slices(tc, name)
                files = []
                for i, a in enumerate(sl):
                    fn = f"textures/ARRAY_{name.rsplit('/', 1)[-1]}_s{i}.png"
                    Image.fromarray(a).save(os.path.join(WORK, fn), optimize=True)
                    files.append((fn, a.shape[1], a.shape[0]))
                slices_done[name] = files
            files = slices_done[name]
            if not files:
                continue
            top = ints(P[TAG_TOP])[0] if TAG_TOP in P else 0
            slope = floats(P[TAG_SLOPE])[0] if TAG_SLOPE in P else 360.0
            top = min(top, len(files) - 1)
            rock[k] = {'array': name, 'base': files[0][0], 'top': files[top][0],
                       'w': files[0][1], 'h': files[0][2], 'top_slice': top, 'slope_deg': slope,
                       'materials': rec.get('materials', [])[:3]}
        elif TAG_TERRAIN_COLOUR in P and any('TerrainMesh_FetchVirtualTexture' in m for m in rec.get('materials', [])):
            # the preset binding is keyed by surface (sk4/bind_keys.py: preset:<surface>)
            mat = next(m for m in rec['materials'] if 'TerrainMesh_FetchVirtualTexture' in m)
            surf = mat.rsplit('_M_', 1)[-1].lower()
            if surf.endswith('_shadow'):
                continue
            pk = f'preset:{surf}'
            if pk in done_presets:
                continue
            e = kt.get(pk) or kt.get(k)
            done_presets.add(pk)
            if not e or not e.get('png'):
                continue
            src = os.path.join(WORK, e.get('orig_png') or e['png'])
            if not os.path.exists(src):
                continue
            target = np.array(floats(P[TAG_TERRAIN_COLOUR])[:3], np.float64)
            a = np.asarray(Image.open(src).convert('RGB'), np.float64) / 255.0
            lin = srgb_to_linear(a)
            mean = lin.reshape(-1, 3).mean(axis=0)
            lin = lin * (target / np.maximum(mean, 1e-4))[None, None, :]
            out = (linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8)
            fn = f"textures/TERR_{pk.split(':')[-1]}.png"
            Image.fromarray(out).save(os.path.join(WORK, fn), optimize=True)
            e.setdefault('orig_png', e['png'])
            e['png'] = fn
            e.pop('tint', None)
            e['terrain_colour'] = [round(float(x), 4) for x in target]
            terr += 1
    json.dump(rock, open(os.path.join(WORK, 'rock_materials.json'), 'w'), indent=1)
    json.dump(seg, open(os.path.join(WORK, 'segment_materials.json'), 'w'), indent=1)
    print(f"  segment (slab/panel sheet) materials baked: {len(seg)}")
    # terrain fills (tools/collect.py -> fills.json): extract each fill's colour map (`_c` beside
    # its `_dt` decal asset) at full resolution
    fp_ = os.path.join(WORK, 'fills.json')
    if os.path.exists(fp_):
        fills = json.load(open(fp_))
        done = {}
        for f in fills:
            a_ = f['asset']
            if a_ not in done:
                name = a_[:-3] + '_c' if a_.endswith('_dt') else a_ + '_c'
                got = tc.fetch(name)
                if got is None:
                    done[a_] = None
                else:
                    im = got[3].convert('RGB')
                    if max(im.size) > 1024:
                        im.thumbnail((1024, 1024), Image.LANCZOS)
                    fn = f"textures/FILL_{name.rsplit('/', 1)[-1]}.png"
                    im.save(os.path.join(WORK, fn), optimize=True)
                    done[a_] = fn
            f['png'] = done[a_]
        json.dump(fills, open(fp_, 'w'))
        print(f"  terrain fill textures: " + ', '.join(f"{k.rsplit('/', 1)[-1]} -> {v}" for k, v in done.items()))
    json.dump(kt, open(kt_path, 'w'), indent=1)
    tops = {}
    for v in rock.values():
        tops[(v['array'].rsplit('/', 1)[-1], v['top_slice'], v['slope_deg'])] = tops.get(
            (v['array'].rsplit('/', 1)[-1], v['top_slice'], v['slope_deg']), 0) + 1
    print(f"  rock-array materials: {len(rock)} ({len(slices_done)} arrays): "
          + ', '.join(f"{a} top slice {s} under {g:g} deg x{n}" for (a, s, g), n in sorted(tops.items())))
    print(f"  terrain presets re-baked to skate.'s base colour: {terr}")


if __name__ == '__main__':
    main()
