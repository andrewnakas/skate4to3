#!/usr/bin/env python3
"""Assembled tile arenas (sk3/assemble.py output) -> tools/pack.py tiles dir -> installable pack.

    python tools/build_pack.py <arenas dir> <glb dir> <name> [--tex-budget-mb 7] [--out DIR]

Per tile: cPres stream = the render arena + a 64 px stand-in of every texture the tile uses (retail
keeps one per cTex texture in cPres); cTex stream = the full textures, halved largest-first until
the stream fits the budget; cSim stream = the collision arena. cPres_Global holds the flat default
textures every tile shares.
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import texture_write as tw
from sk3.assemble import h64, h32

STANDIN_PX = 16          # retail-style cPres stand-in; every level pads to 4 KB, so keep it tiny


def tex_arena(png, nid, size_cap=None, lowres=False):
    im = Image.open(png).convert('RGBA')
    if size_cap and max(im.size) > size_cap:
        s = size_cap / max(im.size)
        im = im.resize((max(4, int(im.size[0] * s)), max(4, int(im.size[1] * s))), Image.LANCZOS)
    rgba = np.asarray(im, np.uint8)
    if lowres:
        return tw.build(tw.lowres_standin(rgba, nid, h32('lowres', nid), size=STANDIN_PX))
    return tw.build(tw.from_image(rgba, nid, h32('tex', nid, rgba.shape[:2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arenas'); ap.add_argument('glb'); ap.add_argument('name')
    ap.add_argument('--tex-budget-mb', type=float, default=7.0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    T = json.load(open(os.path.join(a.arenas, 'tiles.json')))
    meta = T['textures']
    stage = os.path.join(a.arenas, 'pack_tiles')
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)

    def put(stream, k, data, aid=None):
        d = os.path.join(stage, stream)
        os.makedirs(d, exist_ok=True)
        aid = aid or h64('chunk', stream, k)
        open(os.path.join(d, f'{k:03d}_{aid:016x}.rx2'), 'wb').write(data)

    defaults = [t for t, m in meta.items() if m.get('default')]
    for k, t in enumerate(defaults):
        put('cPres_Global', k, open(os.path.join(a.arenas, 'tex', f'{t}.arena'), 'rb').read())
    budget = a.tex_budget_mb * 1e6
    report = []
    for name, rec in T['tiles'].items():
        x, z = rec['tile']
        texs = [t for t in rec['textures'] if not meta[t].get('default')]
        if 'pres_bytes' in rec:
            k = 0
            put(f'cPres_{x}_{z}_high', k, open(os.path.join(a.arenas, f'cPres_{name}.arena'), 'rb').read())
            for t in texs:
                k += 1
                put(f'cPres_{x}_{z}_high', k, tex_arena(os.path.join(a.arenas, 'tex', f'{t}.png'), int(t, 16), lowres=True))
        if 'sim_bytes' in rec:
            put(f'cSim_{x}_{z}_high', 0, open(os.path.join(a.arenas, f'cSim_{name}.arena'), 'rb').read())
        if texs:
            cap = {t: max(meta[t]['w'], meta[t]['h']) for t in texs}
            size = {t: meta[t]['bytes'] for t in texs}
            while sum(size.values()) > budget:
                t = max(texs, key=lambda q: (size[q], q))
                if cap[t] <= 64:
                    break
                cap[t] //= 2
                size[t] = int(size[t] / 4)
            for k, t in enumerate(texs):
                full = cap[t] >= max(meta[t]['w'], meta[t]['h'])
                data = (open(os.path.join(a.arenas, 'tex', f'{t}.arena'), 'rb').read() if full else
                        tex_arena(os.path.join(a.arenas, 'tex', f'{t}.png'), int(t, 16), size_cap=cap[t]))
                put(f'cTex_{x}_{z}_high', k, data)
            report.append(sum(size.values()))
    spawns = json.load(open(os.path.join(a.glb, 'd2s3_spawns.json')))
    spots = [{'title': s['name'], 'pos': [s['x'], s['y'], s['z']], 'yaw': s.get('yaw', 0.0)} for s in spawns]
    json.dump({'cell': int(T['tile_size']), 'spots': spots}, open(os.path.join(stage, 'map.json'), 'w'), indent=1)
    print(f'staged {len(T["tiles"])} tiles; cTex per tile median {np.median(report) / 1e6:.1f} MB max {max(report) / 1e6:.1f} MB')
    cmd = [sys.executable, os.path.join(ROOT, 'tools', 'pack.py'), stage, a.name] + (['--out', a.out] if a.out else [])
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
