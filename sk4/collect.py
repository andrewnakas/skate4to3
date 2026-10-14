#!/usr/bin/env python3
"""Rebuild the analysis products in work/ from the skate. install, for any level.

The exporter and the texture chain read indices that were first built interactively on the
Linux machine and never committed. This is that analysis as one script:

    collect.py index               global_assets.json, global_chunks.json, chunk_ranges.json,
                                   chunk_catalog.json, tex_res.json
    collect.py partitions          partition_all.json  (EBX partition guid -> asset name)
    collect.py level <level>       instances.json, placements.json, blueprint_meshes.json,
                                   ecs_mesh_instances.json, material_meshes.json,
                                   level.json  (which level work/ currently describes)

Schemas (what the consumers index, see export_glb.py / bind_keys.py / slim_index.py):

    global_assets.json   {'ebx': {name: [src, pkg, cas, off, size, originalSize]},
                          'res': {name: [src, pkg, cas, off, size, originalSize, resType,
                                         resMetaHex, rid, [other srcs]]}}
                         src is '<toc stem>#<bundle index>', so a level's own assets carry
                         the level name.
    global_chunks.json   {guid: [src, kind, pkg, cas, off, size]}; kind 'toc' = a TOC chunk
                         record (whole chunk), 'bundle' = a bundle chunk starting at logical 0
    chunk_ranges.json    {guid: [[pkg, cas, off, size, logicalOffset, logicalSize], ...]}
                         every bundle-carried range of a chunk (texture mip tails)
    chunk_catalog.json   {guid: [pkg, cas, off, size]} TOC chunk records only
    tex_res.json         {name: [pkg, cas, off, size]} texture res only

Everything is keyed by asset name, so the products are tied to the install's build: rebuild
after every skate. update (the CAS offsets move).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import binarybundle  # noqa: E402
import ebx  # noqa: E402
import superbundle as sb  # noqa: E402
from cas import decompress_many  # noqa: E402

GAME = os.environ.get('SKATE_ROOT',
                      ('C:/Program Files (x86)/Steam/steamapps/common/Skate' if os.name == 'nt' else
                       '/run/media/nakas/Windows/Program Files (x86)/Steam/steamapps/common/Skate'))
LAYOUT = os.path.join(GAME, 'Data', 'Win32', 'configurations', 'layout')
WORK = os.environ.get('SKATE_WORK', os.path.join(os.path.dirname(HERE), 'work'))
TEXTURE_TYPE = 0x6BDE20BA
MESHSET_TYPE = 0x49B156D4


def _tolerant_packages():
    """Language TOCs use per-language package ids; name them instead of failing."""
    orig = sb.package_name
    sb.package_name = lambda a, hi: (orig(a, hi) if (a, hi) in sb.PACKAGES
                                     else f"pkg_{a:04x}_{hi:04x}")


def tocs():
    """Every binary superbundle TOC; level TOCs first so their assets are tagged by level."""
    out = []
    for p in glob.glob(os.path.join(GAME, 'Data', '**', '*.toc'), recursive=True):
        d = open(p, 'rb').read(0x22D)
        if len(d) > 0x22C and not d[0x22C] & 0x80:
            out.append(p)
    return sorted(out, key=lambda p: (0 if os.sep + 'levels' + os.sep in os.path.normpath(p) else 1, p))


class Reader:
    def __init__(self):
        self.h = {}

    def rd(self, pkg, ci, off, size):
        h = self.h.get((pkg, ci))
        if h is None:
            h = self.h[(pkg, ci)] = open(os.path.join(LAYOUT, pkg, f'cas_{ci:02d}.cas'), 'rb')
        h.seek(off)
        return h.read(size)


def dump(name, obj, indent=None):
    p = os.path.join(WORK, name)
    tmp = p + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=indent)
    os.replace(tmp, p)
    print(f"  wrote {name} ({os.path.getsize(p) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------------------------
def cmd_index(_args):
    _tolerant_packages()
    os.makedirs(WORK, exist_ok=True)
    r = Reader()
    EBX, RES = {}, {}
    CH, CAT = {}, {}
    RANGES = collections.defaultdict(list)
    t0 = time.time()
    for path in tocs():
        stem = os.path.splitext(os.path.basename(path))[0]
        S = sb.SuperBundle(path, GAME)
        for c in S.chunks():
            CAT.setdefault(c.guid, [c.package, c.cas_index, c.offset, c.size])
            CH.setdefault(c.guid, [stem, 'toc', c.package, c.cas_index, c.offset, c.size])
        nb = bad = 0
        for i in range(S.bundles_count):
            try:
                B = S.bundle(i)
                e0 = B.entries[0]
                m = binarybundle.parse(r.rd(e0.package, e0.cas_index, e0.offset, e0.size))
            except Exception:  # noqa: BLE001
                bad += 1
                continue
            nb += 1
            src = f'{stem}#{i}'
            ents = B.entries[1:]
            ne, nr = len(m.ebx), len(m.res)
            if len(ents) != ne + nr + len(m.chunks):
                bad += 1
                continue
            for k, (name, osz, _sha) in enumerate(m.ebx):
                e = ents[k]
                EBX.setdefault(name, [src, e.package, e.cas_index, e.offset, e.size, osz])
            for k, (name, osz, rtype, meta, rid, _sha) in enumerate(m.res):
                e = ents[ne + k]
                if name in RES:
                    RES[name][9].append(src)
                    continue
                RES[name] = [src, e.package, e.cas_index, e.offset, e.size, osz, rtype,
                             meta.hex(), rid, []]
            for k, (g, lo, ls, _sha) in enumerate(m.chunks):
                e = ents[ne + nr + k]
                RANGES[g].append([e.package, e.cas_index, e.offset, e.size, lo, ls])
                if lo == 0:
                    CH.setdefault(g, [src, 'bundle', e.package, e.cas_index, e.offset, e.size])
        print(f"  {stem:<28} bundles {nb:>5} (bad {bad})  chunks {S.chunks_count:>6}  "
              f"-> ebx {len(EBX):,} res {len(RES):,}  [{time.time() - t0:.0f}s]", flush=True)
    dump('global_assets.json', {'ebx': EBX, 'res': RES})
    dump('global_chunks.json', CH)
    dump('chunk_ranges.json', RANGES)
    dump('chunk_catalog.json', CAT)
    dump('tex_res.json', {n: v[1:5] for n, v in RES.items() if v[6] == TEXTURE_TYPE})
    types = collections.Counter(v[6] for v in RES.values())
    print(f"  res types: {', '.join(f'{t:#010x} x{n}' for t, n in types.most_common(12))}")


# ---------------------------------------------------------------------------------------------
def efix_guid(blob):
    if blob[:4] != b'RIFF':
        return None
    size = struct.unpack_from('<I', blob, 4)[0]
    p = 12
    while p + 8 <= min(len(blob), 8 + size):
        cid = blob[p:p + 4]
        csz = struct.unpack_from('<I', blob, p + 4)[0]
        if cid == b'EFIX':
            return blob[p + 8:p + 24].hex()
        p += 8 + csz
        p = (p + 1) & ~1
    return None


def cmd_partitions(args):
    EBX = json.load(open(os.path.join(WORK, 'global_assets.json')))['ebx']
    out_p = os.path.join(WORK, 'partition_all.json')
    out = json.load(open(out_p)) if os.path.exists(out_p) and not args.fresh else {}
    have = set(out.values())
    names = sorted(n for n in EBX if n not in have)
    print(f"  EBX to index: {len(names):,} (cached {len(have):,})")
    r = Reader()
    B, bad, t0 = 2000, 0, time.time()
    for s in range(0, len(names), B):
        chunk = names[s:s + B]
        try:
            blobs = decompress_many([r.rd(*EBX[n][1:5]) for n in chunk])
        except Exception:  # noqa: BLE001
            blobs = []
            for n in chunk:
                try:
                    blobs.append(decompress_many([r.rd(*EBX[n][1:5])])[0])
                except Exception:  # noqa: BLE001
                    blobs.append(b'')
        for n, b in zip(chunk, blobs):
            g = efix_guid(b)
            if g:
                out[g] = n
            else:
                bad += 1
        print(f"    {min(s + B, len(names)):,}/{len(names):,}  [{time.time() - t0:.0f}s]", flush=True)
        if (s // B) % 10 == 9:
            dump('partition_all.json', out)
    dump('partition_all.json', out)
    print(f"  {len(out):,} partitions ({bad} unreadable)")


# ---------------------------------------------------------------------------------------------
def lt_rows(v):
    """LinearTransform dict -> [[right], [up], [forward], [trans]]."""
    return [[float(v[k]['x']), float(v[k]['y']), float(v[k]['z'])]
            for k in ('right', 'up', 'forward', 'trans')]


def compose(child, parent):
    """Row-vector transforms: world = local @ child @ parent (4x3 row form, as rows)."""
    import numpy as np
    C = np.eye(4)
    P = np.eye(4)
    C[:3, :3], C[3, :3] = np.array(child[:3]), np.array(child[3])
    P[:3, :3], P[3, :3] = np.array(parent[:3]), np.array(parent[3])
    W = C @ P
    return [W[0, :3].tolist(), W[1, :3].tolist(), W[2, :3].tolist(), W[3, :3].tolist()]


class Level:
    def __init__(self, level):
        self.level = level
        G = json.load(open(os.path.join(WORK, 'global_assets.json')))
        self.EBX, self.RES = G['ebx'], G['res']
        del G
        self.P = json.load(open(os.path.join(WORK, 'partition_all.json')))
        self.r = Reader()
        self.cache = {}

    def load(self, names):
        need = [n for n in dict.fromkeys(names) if n not in self.cache and n in self.EBX]
        B = 400
        for s in range(0, len(need), B):
            chunk = need[s:s + B]
            blobs = decompress_many([self.r.rd(*self.EBX[n][1:5]) for n in chunk])
            for n, b in zip(chunk, blobs):
                try:
                    self.cache[n] = ebx.Partition(b)
                except Exception as e:  # noqa: BLE001
                    print(f"    {n}: {e}")
                    self.cache[n] = None
        return [self.cache.get(n) for n in names]

    def name_of(self, imp):
        return self.P.get(imp.partition) if isinstance(imp, ebx.Import) else None


def level_parts(L, level):
    pfx = f'levels/game/{level}/'
    names = sorted(n for n in L.EBX if n.lower().startswith(pfx.lower()))
    print(f"  {len(names):,} EBX partitions under {pfx}")
    return dict(zip(names, L.load(names)))


def extract_decals(L, parts):
    """EnvironmentDecalVolumeData -> decals.json (specs/SKATE_FORMATS.md "Decal volumes").

    Transform rows are right/up/forward/trans; `up` is the thin projection axis (641 of 658 Summer
    Camp volumes) and the decal face spans right x forward. A volume's own ShaderInstanceKey is a
    per-instance key with no depot entry; its EnvironmentDecalVolumeTemplateData's key is the one
    the depots bind (64/64 templates on Summer Camp), so rows carry both: `key`, `tkey`.
    Row/Column pick a cell of the template's sheet.
    """
    tpl_names = {}
    found = []
    for n, p in parts.items():
        if not p:
            continue
        for i in p.instances_of('EnvironmentDecalVolumeData'):
            d = p.read(i)
            if not d.get('IsEnabled', True):
                continue
            imp = d.get('Template')
            tn = L.name_of(imp) if imp else None
            found.append((n, d, tn))
            if tn:
                tpl_names[tn] = None
    names = sorted(tpl_names)
    for tn, tp in zip(names, L.load(names)):
        if tp:
            for i in tp.instances_of('EnvironmentDecalVolumeTemplateData'):
                tpl_names[tn] = tp.read(i).get('ShaderInstanceKey')
    rows = []
    for n, d, tn in found:
        rows.append({'key': d.get('ShaderInstanceKey'), 'tkey': tpl_names.get(tn), 'template': tn,
                     'T': lt_rows(d['Transform']), 'row': int(d.get('Row') or 0),
                     'col': int(d.get('Column') or 0), 'alpha': float(d.get('Alpha', 1.0)), 'src': n})
    dump('decals.json', rows)
    print(f"  decal volumes {len(rows):,} (enabled), {len(names):,} templates, "
          f"{sum(1 for v in tpl_names.values() if v):,} with a template key")
    return rows


def extract_fills(L, parts):
    """TerrainFillDecalData -> fills.json: the asphalt/concrete patches skate. paints onto its
    ground (roads, paths, plazas). Each is a closed polygon in world space whose LayerDecalAsset
    names a material folder (`.../asphalt_fine_02_2x2m_dt`); its colour map is the `_c` sibling."""
    rows = []
    for n, p in parts.items():
        if not p:
            continue
        for i in p.instances_of('TerrainFillDecalData'):
            d = p.read(i)
            asset = L.name_of((d.get('LayerDecalAsset') or {}).get('DecalAsset'))
            pts = [[float(q['x']), float(q['y']), float(q['z'])] for q in d.get('Points', [])]
            if not asset or len(pts) < 3:
                continue
            rows.append({'points': pts, 'asset': asset, 'order': int(d.get('DrawOrderIndex') or 0),
                         'closed': bool(d.get('IsClosed', True)), 'src': n})
    dump('fills.json', rows)
    print(f"  terrain fills {len(rows):,}: " + ', '.join(
        f"{a.rsplit('/', 1)[-1]} x{c}" for a, c in collections.Counter(r['asset'] for r in rows).most_common(6)))
    return rows


def cmd_decals(args):
    L = Level(args.level)
    parts = level_parts(L, args.level)
    extract_decals(L, parts)
    extract_fills(L, parts)


def cmd_level(args):
    L = Level(args.level)
    parts = level_parts(L, args.level)
    types = collections.Counter()
    for n, p in parts.items():
        if p:
            for _, t in p.instances:
                types[p.type_name(t)] += 1
    if args.types:
        for t, c in types.most_common(200):
            print(f"    {c:>7}  {t}")
        return

    instances, placements, ecs = [], [], []
    bps = set()
    unresolved = collections.Counter()
    for n, p in parts.items():
        if not p:
            continue
        for i in p.instances_of('StaticModelGroupEntityData'):
            d = p.read(i)
            for m in d.get('MemberDatas', []):
                mesh = L.name_of(m.get('MeshAsset'))
                if not mesh:
                    unresolved['smg member'] += 1
                    continue
                for T in m.get('InstanceTransforms', []):
                    instances.append([mesh, lt_rows(T), n])
        for kind in ('ObjectReferenceObjectData', 'SpatialReferenceObjectData',
                     'PrefabReferenceObjectData', 'EffectReferenceObjectData'):
            for i in p.instances_of(kind):
                d = p.read(i)
                bp = L.name_of(d.get('Blueprint'))
                if not bp:
                    unresolved[kind] += 1
                    continue
                bps.add(bp)
                placements.append([kind, bp, lt_rows(d['BlueprintTransform']), n,
                                   bool(d.get('Excluded', False))])
    print(f"  static-model-group instances {len(instances):,}; blueprint placements "
          f"{len(placements):,} over {len(bps):,} blueprints; unresolved {dict(unresolved)}")

    # blueprint -> meshes, one level of nesting (ObjectBlueprint -> StaticModelEntityData)
    bm = {}
    bparts = dict(zip(sorted(bps), L.load(sorted(bps))))
    for bp, p in bparts.items():
        meshes = []
        if p:
            for i in p.instances_of('StaticModelEntityData'):
                mesh = L.name_of(p.read(i).get('Mesh'))
                if mesh:
                    meshes.append(mesh)
        bm[bp] = meshes
    print(f"  blueprints with a static mesh: {sum(1 for v in bm.values() if v):,} of {len(bm):,}")

    # which materials the level's meshes use (texture chain input)
    placed = {m for m, _, _ in instances} | {m for v in bm.values() for m in v}
    import meshset
    mats = collections.defaultdict(set)
    have = sorted(m for m in placed if m in L.RES)
    B = 400
    for s in range(0, len(have), B):
        chunk = have[s:s + B]
        for n, b in zip(chunk, decompress_many([L.r.rd(*L.RES[x][1:5]) for x in chunk])):
            try:
                ms = meshset.parse(b)
            except Exception:  # noqa: BLE001
                continue
            for lod in ms.lods:
                for sec in lod.sections:
                    if not sec.is_shadow:
                        mats[sec.material].add(n)
    print(f"  placed meshes {len(placed):,} ({len(have):,} with a MeshSet), materials {len(mats):,}")

    dump('instances.json', instances)
    dump('placements.json', placements)
    dump('blueprint_meshes.json', {'blueprint_meshes': bm})
    dump('ecs_mesh_instances.json', ecs)
    dump('material_meshes.json', {k: sorted(v) for k, v in sorted(mats.items())}, indent=0)
    dump('level.json', {'level': args.level, 'types': dict(types.most_common())}, indent=1)
    extract_decals(L, parts)
    extract_fills(L, parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('index')
    p = sub.add_parser('partitions')
    p.add_argument('--fresh', action='store_true')
    p = sub.add_parser('level')
    p.add_argument('level')
    p.add_argument('--types', action='store_true', help='list the EBX types in the level and stop')
    p = sub.add_parser('decals', help='only the decal volumes -> decals.json')
    p.add_argument('level')
    args = ap.parse_args()
    {'index': cmd_index, 'partitions': cmd_partitions, 'level': cmd_level,
     'decals': cmd_decals}[args.cmd](args)


if __name__ == '__main__':
    main()
