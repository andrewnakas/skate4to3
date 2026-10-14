#!/usr/bin/env python3
"""Decal templates -> RGBA sheets the exporter draws (work/<level>/decal_textures.json).

    SKATE_WORK=work/mpr python sk4/decal_sheets.py

For every template key the level's decals.json uses, its depot block gives (measured in
specs/SKATE_FORMATS.md "Decal templates"):
  * the colour sheet (`_c`) and the opacity sheet (`_osk` / `_osm`, opacity in RED: about half
    the texels are exactly 0 and the cell gutters are 0);
  * the sheet grid: tag 0x406E = columns, tag 0xF7A1 = rows (u32 each);
  * a colour vector at tag 0x8C54 (0.5 = unchanged, so the multiplier is 2x the value).
The baked PNG is the colour sheet times that multiplier (in linear light) with the opacity in
alpha, resampled to the colour sheet's size. `mode` is MASK when the opacity is essentially
two-level (under 8% of covered texels between 0.1 and 0.9), else BLEND.
"""
from __future__ import annotations

import collections
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import depot  # noqa: E402
from bind_keys import load_depots, role_of  # noqa: E402
from texcache import TexCache, WORK  # noqa: E402

TAG_COLS, TAG_ROWS, TAG_COLOUR = 0x406E, 0xF7A1, 0x8C54
MAX_PX = int(os.environ.get('DECAL_MAX_PX', '2048'))


def lin(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def srgb(a):
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def main():
    G = json.load(open(os.path.join(WORK, 'guid_to_texture.json')))
    decals = json.load(open(os.path.join(WORK, 'decals.json')))
    keys = collections.Counter(d['tkey'] for d in decals if d.get('tkey'))
    mats, _ = load_depots()
    tc = TexCache()
    out_dir = os.path.join(WORK, 'textures')
    os.makedirs(out_dir, exist_ok=True)
    out, skipped = {}, collections.Counter()
    for key, uses in keys.items():
        P = mats.get(key)
        if not P:
            skipped['no depot block'] += 1
            continue
        by_tag = {p.tag: p for p in P}
        colour = opacity = None
        for p in P:
            n = G.get(p.textures[0][1]) if p.textures else None
            if not n:
                continue
            r = role_of(n)
            if r == 'c' and colour is None:
                colour = n
            elif r in ('osk', 'osm') and opacity is None:
                opacity = n
        if not colour:
            skipped['no colour sheet'] += 1
            continue
        cols = struct.unpack('<I', by_tag[TAG_COLS].raw[:4])[0] if TAG_COLS in by_tag else 1
        rows = struct.unpack('<I', by_tag[TAG_ROWS].raw[:4])[0] if TAG_ROWS in by_tag else 1
        mult = 2.0 * np.array(by_tag[TAG_COLOUR].floats[:3]) if TAG_COLOUR in by_tag else np.ones(3)
        gc = tc.fetch(colour, min_px=16)
        if not gc:
            skipped['colour sheet not on disk'] += 1
            continue
        rgb = np.asarray(gc[3].convert('RGB'), np.float64) / 255.0
        h, w, _ = rgb.shape
        if max(w, h) > MAX_PX:
            s = MAX_PX / max(w, h)
            w, h = max(1, int(w * s)), max(1, int(h * s))
            rgb = np.asarray(gc[3].convert('RGB').resize((w, h), Image.BILINEAR), np.float64) / 255.0
        rgb = srgb(lin(rgb) * mult[None, None, :])
        alpha = np.ones((h, w))
        if opacity:
            go = tc.fetch(opacity, min_px=16)
            if go:
                alpha = np.asarray(go[3].convert('RGB').resize((w, h), Image.BILINEAR), np.float64)[:, :, 0] / 255.0
            else:
                skipped['opacity sheet missing (opaque)'] += 1
        covered = alpha[alpha > 0.02]
        soft = float(((covered > 0.1) & (covered < 0.9)).mean()) if covered.size else 0.0
        mode = 'MASK' if soft < 0.08 else 'BLEND'
        img = np.dstack([rgb, alpha])
        fn = f'textures/DECAL_{key}.png'
        Image.fromarray((img * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(WORK, fn), optimize=True)
        out[str(key)] = {'png': fn, 'w': w, 'h': h, 'grid': [rows, cols], 'mode': mode,
                         'colour': colour, 'opacity': opacity, 'uses': uses}
    json.dump(out, open(os.path.join(WORK, 'decal_textures.json'), 'w'), indent=1)
    modes = collections.Counter(e['mode'] for e in out.values())
    print(f'  decal sheets {len(out):,} of {len(keys):,} templates ({dict(modes)}); skipped {dict(skipped)}')


if __name__ == '__main__':
    main()
