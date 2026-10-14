#!/usr/bin/env python3
"""ShaderBlockDepot material keys -> the textures a section is drawn with (work/depot_textures.json).

    SKATE_WORK=work python sk4/bind_keys.py

Every depot in the game is parsed with sk4/depot.py. A bound texture's role comes from its own
asset name suffix (`_c` colour, `_ny` normal, `_msk` mask, ...). A parameter hash keeps one role
inside a shader (measured: 97% of bindings across the 15 most common shaders agree with the
hash's majority role), so each hash's role is learned as the majority over all materials and
then applied to every binding, including ones whose texture is a neutral default.

A material can carry several surface layers (e.g. lumber + steel + paint). Layers are grouped by
texture family (asset name without the role suffix); the primary layer is the one bound to the
colour hash that is most often a material's ONLY colour texture (the shader's base slot), ties
broken by record order. All layers are kept in `layers` for later per-section choice.

Output, per decimal material key:
    {"c": name, "ny": name, "msk": name, "layers": [{"c","ny","msk"}...], "shader": n}
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import depot  # noqa: E402
from cas import decompress_many  # noqa: E402
from guid_index import LAYOUT  # noqa: E402

WORK = os.environ.get('SKATE_WORK', os.path.join(os.path.dirname(HERE), 'work'))
DEPOT_RES = 0x73312045
ROLE_RX = re.compile(r'_(c|co|ny|n|msk|osk|osm|ogk|rgb|packed|cu|smao|ta|isk|isv|utk)$')


def role_of(name: str) -> str:
    leaf = name.rsplit('/', 1)[-1]
    if leaf.startswith('neutral') or '/default/' in name:
        return 'neutral'
    if leaf.startswith('t_fx') or '/panoramas/' in name or '/mask/global' in name:
        return 'util'
    m = ROLE_RX.search(leaf)
    return m.group(1) if m else 'other'


def shader_id(params) -> str:
    """Stable id of a shader: its parameter-hash set."""
    import hashlib
    return hashlib.md5(b''.join(h.to_bytes(8, 'little') for h in sorted({p.hash for p in params}))).hexdigest()[:12]


def family_of(name: str) -> str:
    return ROLE_RX.sub('', name)


def load_depots():
    res = json.load(open(os.path.join(WORK, 'global_assets.json')))['res']
    rows = [(n, v) for n, v in res.items() if v[6] == DEPOT_RES]
    handles, mats = {}, {}
    for s in range(0, len(rows), 200):
        part = rows[s:s + 200]
        raw = []
        for _, v in part:
            pkg, ci, off, size = v[1:5]
            h = handles.get((pkg, ci))
            if h is None:
                h = handles[(pkg, ci)] = open(os.path.join(LAYOUT, pkg, f'cas_{ci:02d}.cas'), 'rb')
            h.seek(off)
            raw.append(h.read(size))
        for (_, v), blob in zip(part, decompress_many(raw)):
            mats.update(depot.parse(blob, bytes.fromhex(v[7])))
    return mats, len(rows)


def key_materials():
    """depot key -> Counter(material name) over every MeshSet section in the game."""
    import meshset
    res = json.load(open(os.path.join(WORK, 'global_assets.json')))['res']
    rows = [v for v in res.values() if v[6] == 0x49B156D4]
    del res
    handles, out = {}, collections.defaultdict(collections.Counter)
    for s in range(0, len(rows), 400):
        raw = []
        for v in rows[s:s + 400]:
            pkg, ci, off, size = v[1:5]
            h = handles.get((pkg, ci))
            if h is None:
                h = handles[(pkg, ci)] = open(os.path.join(LAYOUT, pkg, f'cas_{ci:02d}.cas'), 'rb')
            h.seek(off)
            raw.append(h.read(size))
        for b in decompress_many(raw):
            try:
                ms = meshset.parse(b)
            except Exception:  # noqa: BLE001
                continue
            for lod in ms.lods:
                for sec in lod.sections:
                    if sec.depot_key:
                        out[sec.depot_key][sec.material] += sec.primitives
    return out


TERRAIN_PRESET = re.compile(r'TerrainMesh_FetchVirtualTexture\w*?_M_(\w+)$')


def tokens(s: str) -> set:
    return {t for t in re.split(r'[^a-z]+', s.lower()) if t and t not in ('m', 'c', 'x', 'lg', 'dt')}


def preset_entry(surface: str, families: dict):
    """A terrain virtual-texture surface (e.g. `Grass`, `Asphalt_Dried`) -> the terrain library
    family whose leaf shares the most name tokens with it; terrain/ families first."""
    want = tokens(surface)
    best = None
    for fam, roles in families.items():
        leaf = tokens(fam.rsplit('/', 1)[-1])
        common = len(want & leaf)
        if not common:
            continue
        score = (common, '/materials/terrain/' in fam, -len(leaf - want), -len(fam))
        if best is None or score > best[0]:
            best = (score, roles)
    return best[1] if best else None


def main():
    G = json.load(open(os.path.join(WORK, 'guid_to_texture.json')))
    mats, ndepots = load_depots()
    print(f'  depots {ndepots:,}  material keys {len(mats):,}')

    def tex(p):
        return G.get(p.textures[0][1]) if p.textures else None

    votes = collections.defaultdict(collections.Counter)
    for P in mats.values():
        for p in P:
            n = tex(p)
            if n:
                r = role_of(n)
                if r not in ('neutral', 'util', 'other'):
                    votes[p.hash][r] += 1
    hash_role = {h: c.most_common(1)[0][0] for h, c in votes.items()}

    only_colour = collections.Counter()
    for P in mats.values():
        cols = [p.hash for p in P if hash_role.get(p.hash) == 'c' and role_of(tex(p) or '') == 'c']
        if len(cols) == 1:
            only_colour[cols[0]] += 1

    shaders = {}
    out, stats = {}, collections.Counter()
    for key, P in mats.items():
        sig = shader_id(P)
        shaders[sig] = shaders.get(sig, 0) + 1
        layers = collections.OrderedDict()           # family -> {role: name}
        for p in P:
            n = tex(p)
            if not n:
                continue
            r = role_of(n)
            if r in ('neutral', 'util', 'other'):
                continue
            fam = layers.setdefault(family_of(n), {'_hash': None})
            fam.setdefault(r, n)
            if r == 'c':
                fam['_hash'] = p.hash
        colour = [f for f in layers.values() if 'c' in f]
        rec = {'shader': sig}
        if colour:
            colour.sort(key=lambda f: -only_colour.get(f['_hash'], 0))
            best = colour[0]
            rec.update({k: best[k] for k in ('c', 'ny', 'msk') if k in best})
            rec['layers'] = [{k: f[k] for k in ('c', 'ny', 'msk') if k in f} for f in colour]
            stats['bound'] += 1
            stats[f'{len(colour)} colour layer(s)'] += 1
        else:
            stats['no colour texture'] += 1
        out[str(key)] = rec
    km = key_materials()
    families = collections.defaultdict(dict)        # library family -> {role: texture}
    for n in G.values():
        if n.startswith('world/materials/'):
            r = role_of(n)
            if r in ('c', 'ny', 'msk'):
                families[family_of(n)][r] = n
    families = {f: r for f, r in families.items() if 'c' in r}
    presets = 0
    for key, names in km.items():
        rec = out.get(str(key))
        if rec is None:
            continue
        rec['materials'] = [m for m, _ in names.most_common(3)]
        for m in rec['materials']:
            mt = TERRAIN_PRESET.search(m)
            if not mt or mt.group(1).lower().endswith('_shadow'):
                continue
            pk = f'preset:{mt.group(1).lower()}'
            if pk not in out:
                e = preset_entry(mt.group(1), families)
                if e:
                    out[pk] = dict(e)
                    presets += 1
    stats['terrain presets bound'] = presets
    json.dump(out, open(os.path.join(WORK, 'depot_textures_all.json'), 'w'), indent=0)
    json.dump({f'{h:#018x}': r for h, r in hash_role.items()},
              open(os.path.join(WORK, 'param_roles.json'), 'w'), indent=0)
    print(f'  param hashes with a learned role {len(hash_role):,}; shaders {len(shaders):,}')
    for k, v in sorted(stats.items()):
        print(f'  {k:24s} {v:,}')


if __name__ == '__main__':
    main()
