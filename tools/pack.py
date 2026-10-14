"""Build an installable Skate 3 DLC pack folder from per-tile arenas.

    python tools/pack.py <tiles dir> <display name> [--out DIR] [--slug S] [--dist DIST_Name]
                         [--content-id ID] [--ea-streams] [--community-centres] [--template [PACK.big]]
    python tools/pack.py --export <pack.big> <tiles dir>      # tiles dir from a working pack

Tiles dir:
    map.json        {"cell": 100,
                     "spots": [{"title": "Main", "pos": [x, y, z], "yaw": 0, "key": "..."}, ...],
                     "content": {"cSim_50_50_high": [minx, miny, minz, maxx, maxy, maxz], ...},
                     "slug": "...", "dist": "DIST_..."}
                    spots[0] is the main location, one location per spot (no limit); "key",
                    "content", "slug" and "dist" are optional (defaults: slug = the display
                    name's letters/digits lower-cased, dist = DIST_<CamelCaseName>, keys = slug,
                    spot1, spot2, ...). --slug / --dist override map.json.
    c<Kind>_<x>_<z>_high/<NNN>_<asset id hex>.rx2   (Kind = Pres | Sim | Tex; NNN orders the
    c<Kind>_Global/<NNN>_<asset id hex>.rx2           chunks inside the stream)

Output: <out>/<CONTENTID>/<slug>_00000000.big and .header (content id = folder name). Copy the
folder into the recomp's dlc/ directory. Every file is generated from the inputs and names,
including the AttribSys vaults and locator ids (specs/sk3_packaging.md "Generator"). --template
(off by default) instead takes the internal names, vault files and locator arenas from a working
pack (SUMMERCAMP when no path is given). The pack is for personal use only - never commit or
publish it.
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
    """Write a working pack's tiles + map.json (the generator's inputs, names included)."""
    names = dlc.names_of_pack(pack)
    display, tiles, spots, cell, _group, _tail, content = dlc.inputs_of_pack(pack, names)
    for kind, tl in tiles.items():
        for name, arenas in tl:
            p = os.path.join(out, name)
            os.makedirs(p, exist_ok=True)
            for i, (aid, ar) in enumerate(arenas):
                with open(os.path.join(p, f'{i:03d}_{aid:016x}.rx2'), 'wb') as f:
                    f.write(ar)
    js = {'display': display, 'cell': cell, 'slug': names.slug, 'dist': names.dist,
          'spots': [{'title': s.title, 'pos': list(s.pos), 'yaw': s.yaw, 'matrix': list(s.matrix), 'key': s.key}
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
    ap.add_argument('--template', nargs='?', const='', default=None,
                    help='take names, vaults and locator arenas from a working pack (default off)')
    ap.add_argument('--slug', help='pack slug (default: map.json "slug", else from the name)')
    ap.add_argument('--dist', help='DIST name (default: map.json "dist", else DIST_<Name>)')
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
    mj = json.load(open(os.path.join(a.tiles, 'map.json')))
    spots = [dlc.Spot(s['title'], tuple(s['pos']), s.get('yaw', 0.0),
                      tuple(s['matrix']) if s.get('matrix') else None, s.get('key')) for s in mj['spots']]
    if a.template is not None:
        tpl_path = a.template or find_template()
        if not tpl_path:
            sys.exit('no template pack found; pass --template <pack.big>')
        names = dlc.read_template(tpl_path)
        if len(spots) > len(names.keys):
            sys.exit(f'{len(spots)} spots but the template has {len(names.keys)} locations')
        how = f'template {os.path.basename(tpl_path)}'
    else:
        names = dlc.names_for(a.name, spots, a.slug or mj.get('slug'), a.dist or mj.get('dist'))
        how = 'template-free'
    tiles = load_tiles(a.tiles)
    files = dlc.build_files(a.name, tiles, spots, names, cell=int(mj.get('cell', 100)),
                            content={k: tuple(v) for k, v in mj.get('content', {}).items()},
                            center_rule='community' if a.community_centres else 'containing',
                            stream_form='xbox2' if a.ea_streams else 'raw',
                            dlc_tables=not a.no_dlc_tables)
    path = dlc.write_pack(files, a.out, names.slug, a.name, a.content_id)
    n = {k: len(v) for k, v in tiles.items()}
    print(f'{path}  {os.path.getsize(path) >> 20} MB  {len(files)} files  tiles {n}  '
          f'{how}: slug {names.slug}, {names.dist}, {len(names.keys)} locations')


if __name__ == '__main__':
    main()
