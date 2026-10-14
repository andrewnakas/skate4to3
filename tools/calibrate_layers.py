#!/usr/bin/env python3
"""Learn which colour layer and which tint vector a material really shows, from skate.'s own
baked far-LOD proxy atlases.

    SKATE_WORK=work python tools/calibrate_layers.py [max_meshes=1500]

A world mesh whose coarse LOD is a proxy shell (`M_Proxy_Atlas` material, bound to a
`*_proxy_c` texture) carries the game's own bake of the full material: layers, tints and
all. For every LOD0 section we sample that bake on the proxy triangles nearest to the
section, then score every (colour layer x tint vec3 x mode) prediction the material's depot
block allows. Wins are tallied per (shader, layer colour hash) and (shader, tint hash, mode)
and written to work/layer_rules.json, which sk4/bind_keys.py applies.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sk4'))
import depot  # noqa: E402
import meshset  # noqa: E402
from bind_keys import load_depots, role_of, family_of, shader_id  # noqa: E402
from cas import decompress_many  # noqa: E402
from texcache import TexCache, LAYOUT  # noqa: E402

WORK = os.environ.get('SKATE_WORK', os.path.join(ROOT, 'work'))
ATLAS = re.compile(r'proxy|atlas', re.I)


def lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class Means:
    def __init__(self, tc):
        self.tc, self.m, self.img = tc, {}, {}

    def mean(self, name):
        if name not in self.m:
            g = self.tc.fetch(name, min_px=4)
            self.m[name] = None if not g else lin(np.asarray(g[3].convert('RGB'), np.float64)).reshape(-1, 3).mean(0)
        return self.m[name]

    def image(self, name):
        if name not in self.img:
            g = self.tc.fetch(name, min_px=32)
            self.img[name] = None if not g else lin(np.asarray(g[3].convert('RGB'), np.float64))
        return self.img[name]


def sample(img, uv):
    h, w, _ = img.shape
    u = np.mod(uv[:, 0], 1.0) * (w - 1)
    v = np.mod(uv[:, 1], 1.0) * (h - 1)
    return img[v.astype(int), u.astype(int)]


def err(pred, want):
    """log-ratio distance: brightness and hue both count, scale-free."""
    eps = 1e-3
    return float(np.abs(np.log((pred + eps) / (want + eps))).mean())


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    G = json.load(open(os.path.join(WORK, 'guid_to_texture.json')))
    A = json.load(open(os.path.join(WORK, 'global_assets.json')))
    RES = A['res']
    CH = json.load(open(os.path.join(WORK, 'global_chunks.json')))
    del A
    mats, _ = load_depots()
    tc = TexCache()
    M = Means(tc)
    handles = {}

    def rd(pkg, ci, off, size):
        h = handles.get((pkg, ci))
        if h is None:
            h = handles[(pkg, ci)] = open(os.path.join(LAYOUT, pkg, f'cas_{ci:02d}.cas'), 'rb')
        h.seek(off)
        return h.read(size)

    def tex(p):
        return G.get(p.textures[0][1]) if p.textures else None

    names = sorted(n for n, v in RES.items() if v[6] == 0x49B156D4 and n.startswith('world/'))
    random.Random(3).shuffle(names)
    shader_of = {key: shader_id(P) for key, P in mats.items()}

    layer_win = collections.Counter()   # (shader, colour hash) -> wins
    layer_seen = collections.Counter()
    tint_win = collections.Counter()    # (shader, tint hash or 0, mode) -> wins
    tint_seen = collections.Counter()
    gain = []
    samples = []
    done = 0
    for s in range(0, len(names), 100):
        if done >= limit:
            break
        part = names[s:s + 100]
        sets = {n: meshset.parse(b) for n, b in zip(part, decompress_many([rd(*RES[n][1:5]) for n in part]))}
        for name, ms in sets.items():
            if done >= limit:
                break
            prox = None
            for lod in ms.lods:
                for sec in lod.sections:
                    if sec.primitives and not sec.is_shadow and ATLAS.search(sec.material):
                        P = mats.get(sec.depot_key)
                        cand = [tex(p) for p in (P or []) if tex(p) and role_of(tex(p)) == 'c']
                        cand = [c for c in cand if 'proxy' in c]
                        if cand:
                            prox = (lod, sec, cand[0])
                            break
                if prox:
                    break
            if not prox or prox[0] is ms.lods[0] or ms.lods[0].chunk not in CH or prox[0].chunk not in CH:
                continue
            img = M.image(prox[2])
            if img is None:
                continue
            ch = dict(zip([ms.lods[0].chunk, prox[0].chunk],
                          decompress_many([rd(*CH[ms.lods[0].chunk][2:6]), rd(*CH[prox[0].chunk][2:6])])))
            try:
                PP, UVp, Tp = meshset.decode(ch[prox[0].chunk], prox[0], prox[1])
            except Exception:  # noqa: BLE001
                continue
            if UVp is None or len(Tp) == 0:
                continue
            pc = PP[Tp].mean(1)
            pcol = sample(img, UVp[Tp].mean(1))
            done += 1
            for sec in ms.lods[0].sections:
                if sec.is_shadow or not sec.primitives or ATLAS.search(sec.material):
                    continue
                P = mats.get(sec.depot_key)
                if not P:
                    continue
                try:
                    P0, _uv, T0 = meshset.decode(ch[ms.lods[0].chunk], ms.lods[0], sec)
                except Exception:  # noqa: BLE001
                    continue
                c0 = P0[T0].mean(1)
                if len(c0) > 300:
                    c0 = c0[np.random.default_rng(0).choice(len(c0), 300, replace=False)]
                d2 = ((c0[:, None, :] - pc[None, :, :]) ** 2).sum(-1)
                near = d2.argmin(1)
                if np.sqrt(d2[np.arange(len(near)), near]).mean() > 0.25:
                    continue                                   # proxy shell too far from this part
                want = pcol[near].mean(0)
                layers = {}
                for p in P:
                    n = tex(p)
                    if n and role_of(n) == 'c' and M.mean(n) is not None:
                        layers.setdefault(family_of(n), (p.hash, M.mean(n)))
                if not layers:
                    continue
                tints = [(0, 'none', np.ones(3))]
                for p in P:
                    if p.type == depot.VEC3:
                        v = np.array(p.floats[:3])
                        if np.allclose(v, 0.5) or v.max() <= 0:
                            continue
                        tints.append((p.hash, 'mul', v))
                        tints.append((p.hash, 'mul2', 2 * v))
                        tints.append((p.hash, 'replace', None))
                sh = shader_of[sec.depot_key]
                samples.append({'shader': sh, 'key': sec.depot_key, 'mesh': name, 'material': sec.material,
                                'want': want.tolist(),
                                'layers': {f'{lh:#x}': [fam, m.tolist()] for fam, (lh, m) in layers.items()},
                                'vec3': {f'{q.hash:#x}': list(q.floats[:3]) for q in P if q.type == depot.VEC3}})
                best = None
                for fam, (lh, mean) in layers.items():
                    for th, mode, tv in tints:
                        if mode == 'replace':
                            v = np.array([q.floats[:3] for q in P if q.hash == th][0])
                            pred = v ** 2.2
                        else:
                            pred = mean * tv
                        e = err(pred, want)
                        if best is None or e < best[0]:
                            best = (e, lh, th, mode)
                for fam, (lh, _m) in layers.items():
                    layer_seen[(sh, lh)] += 1
                for th, mode, _tv in tints:
                    tint_seen[(sh, th, mode)] += 1
                layer_win[(sh, best[1])] += 1
                tint_win[(sh, best[2], best[3])] += 1
                gain.append(best[0])
        print(f'  meshes with proxies used: {done:,}  sections scored: {len(gain):,}', flush=True)
    rules = {'layer': {f'{sh}|{h:#x}': [layer_win[(sh, h)], layer_seen[(sh, h)]] for (sh, h) in layer_seen},
             'tint': {f'{sh}|{h:#x}|{m}': [tint_win[(sh, h, m)], tint_seen[(sh, h, m)]] for (sh, h, m) in tint_seen}}
    json.dump(rules, open(os.path.join(WORK, 'layer_rules.json'), 'w'), indent=0)
    json.dump(samples, open(os.path.join(WORK, 'layer_samples.json'), 'w'))
    modes = collections.Counter()
    for (sh, h, m), w in tint_win.items():
        modes[m] += w
    print(f'  best-fit mode totals: {dict(modes)}; median log error {np.median(gain):.3f}')


if __name__ == '__main__':
    main()
