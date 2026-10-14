"""Build an installable Skate 3 DLC pack folder from per-tile arenas.

    python tools/pack.py <tiles dir> <display name> [--out DIR] [--template PACK.big]
                         [--content-id ID] [--ea-streams] [--community-centres]
    python tools/pack.py --export <pack.big> <tiles dir>      # tiles dir from a working pack

Tiles dir:
    map.json        {"cell": 100,
                     "spots": [{"title": "Main", "pos": [x, y, z], "yaw": 0}, ...],
                     "content": {"cSim_50_50_high": [minx, miny, minz, maxx, maxy, maxz], ...}}
                    spots[0] is the main location; "content" (CSPA content boxes) is optional.
    c<Kind>_<x>_<z>_high/<NNN>_<asset id hex>.rx2   (Kind = Pres | Sim | Tex; NNN orders the
    c<Kind>_Global/<NNN>_<asset id hex>.rx2           chunks inside the stream)

Output: <out>/<CONTENTID>/<slug>_00000000.big and .header (content id = folder name). Copy the
folder into the recomp's dlc/ directory. The pack keeps the template's internal names and vaults
(specs/sk3_packaging.md "Generator"); it is for personal use only - never commit or publish it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import dlc  # noqa: E402

PACK_DIRS = [os.path.join(os.environ.get('APPDATA', ''), 'skate3'),
             r'D:\skate3-engine-0.1.5-windows-x86_64\dlc']


def find_template(name: str = 'summercamp_00000000.big') -> str | None:
    for root in PACK_DIRS:
        hits = glob.glob(os.path.join(root, '**', name), recursive=True)
        if hits:
            return hits[0]
    return None


def load_tiles(d: str):
    tiles = {'Pres': [], 'Sim': [], 'Tex': []}
    for sub in sorted(os.listdir(d)):
        p = os.path.join(d, sub)
        if not os.path.isdir(p):
            continue
        kind = {'cPres': 'Pres', 'cSim': 'Sim', 'cTex': 'Tex'}.get(sub.split('_')[0])
        if not kind:
            continue
        arenas = []
        for f in sorted(os.listdir(p)):
            if f.endswith('.rx2'):
                aid = int(os.path.splitext(f)[0].rsplit('_', 1)[-1], 16)
                arenas.append((aid, open(os.path.join(p, f), 'rb').read()))
        if arenas:
            tiles[kind].append((sub, arenas))
    return tiles


def export(pack: str, out: str):
    """Write a working pack's tiles + map.json (the generator's inputs) to a tiles dir."""
    tpl = dlc.read_template(pack)
    display, tiles, spots, cell, _group, _tail, content = dlc.inputs_of_pack(pack, tpl)
    for kind, tl in tiles.items():
        for name, arenas in tl:
            p = os.path.join(out, name)
            os.makedirs(p, exist_ok=True)
            for i, (aid, ar) in enumerate(arenas):
                with open(os.path.join(p, f'{i:03d}_{aid:016x}.rx2'), 'wb') as f:
                    f.write(ar)
    js = {'display': display, 'cell': cell,
          'spots': [{'title': s.title, 'pos': list(s.pos), 'yaw': s.yaw, 'matrix': list(s.matrix)}
                    for s in spots],
          'content': {k: list(v) for k, v in content.items()}}
    with open(os.path.join(out, 'map.json'), 'w') as f:
        json.dump(js, f, indent=1)
    print(f'exported {sum(len(t) for t in tiles.values())} tiles, {len(spots)} spots -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tiles', nargs='?')
    ap.add_argument('name', nargs='?')
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'packs'))
    ap.add_argument('--template')
    ap.add_argument('--content-id')
    ap.add_argument('--ea-streams', action='store_true', help="EA's two-section RefPack streams")
    ap.add_argument('--community-centres', action='store_true',
                    help='preload-list centre = tile whose far corner is nearest (community packs)')
    ap.add_argument('--no-dlc-tables', action='store_true')
    ap.add_argument('--export', nargs=2, metavar=('PACK', 'DIR'))
    a = ap.parse_args()
    if a.export:
        return export(*a.export)
    if not a.tiles or not a.name:
        ap.error('tiles dir and display name required')
    tpl_path = a.template or find_template()
    if not tpl_path:
        sys.exit('no template pack found; pass --template <pack.big>')
    tpl = dlc.read_template(tpl_path)
    mj = json.load(open(os.path.join(a.tiles, 'map.json')))
    spots = [dlc.Spot(s['title'], tuple(s['pos']), s.get('yaw', 0.0),
                      tuple(s['matrix']) if s.get('matrix') else None) for s in mj['spots']]
    if len(spots) > len(tpl.keys):
        sys.exit(f'{len(spots)} spots but the template has {len(tpl.keys)} locations')
    tiles = load_tiles(a.tiles)
    files = dlc.build_files(a.name, tiles, spots, tpl, cell=int(mj.get('cell', 100)),
                            content={k: tuple(v) for k, v in mj.get('content', {}).items()},
                            center_rule='community' if a.community_centres else 'containing',
                            stream_form='xbox2' if a.ea_streams else 'raw',
                            dlc_tables=not a.no_dlc_tables)
    path = dlc.write_pack(files, a.out, tpl.slug, a.name, a.content_id)
    n = {k: len(v) for k, v in tiles.items()}
    print(f'{path}  {os.path.getsize(path) >> 20} MB  {len(files)} files  tiles {n}  '
          f'template {os.path.basename(tpl_path)} ({tpl.slug}, {len(tpl.keys)} locations)')


if __name__ == '__main__':
    main()
