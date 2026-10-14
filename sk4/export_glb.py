#!/usr/bin/env python3
"""Isle of Grom (skate.) -> d2s3 stage-1 output.

Writes what d2s3's own exporter leaves in work/<name>/glb, so stages 2-7 run unchanged
with `--skip-export`:

    block_+UU_+VV.glb        world-space GLBs, one per 400 m block, by instance centre
    blenrose_materials.json  the arena builder material routing
    splines.json             grind rails
    d2s3_tiles.json          per-100 m tile Y extents (the packer's CSPA bounds)
    d2s3_spawns.json         spawn markers
    d2s3_export.json         the stats the build report reads

Coordinates: skate.'s Frostbite space is written straight through as glTF/Skate 3 world space.
Negating Z (the pairing d2s3 applies to Unity) made the island play MIRRORED in Skate 3, so it is
now opt-in (--mirror-z). Winding is only reversed for an instance whose own matrix is mirrored
(det < 0), or for every instance when --mirror-z flips the space.

Sources (all produced by the analysis in this repo's work/):
    instances.json            StaticModelGroup members, every level partition
    placements.json           Object/Spatial blueprint references (plots, generated roads)
    blueprint_meshes.json     blueprint -> mesh
    ecs_mesh_instances.json   ECS mesh components (vehicles), parent chain composed
    physics_placements.json   objects only skate.'s baked sublevel collision places
                              (tools/recover_placements.py), optional
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import io
import json
import math
import os
import re
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE]
sys.path.append(os.path.join(ROOT, 'external', 'pylib'))

from PIL import Image  # noqa: E402
import grind_edges as d2s3_splines  # noqa: E402
from glb_writer import GlbBuilder, Material, Primitive  # noqa: E402

import find_spawns  # noqa: E402
import meshset  # noqa: E402
import physics  # noqa: E402
from cas import decompress_many  # noqa: E402

GAME = os.environ.get('SKATE_ROOT', ('C:/Program Files (x86)/Steam/steamapps/common/Skate' if os.name == 'nt' else
                       '/run/media/nakas/Windows/Program Files (x86)/Steam/steamapps/common/Skate'))
LAYOUT = os.path.join(GAME, 'Data', 'Win32', 'configurations', 'layout')
WORK = os.environ.get('SKATE_WORK', os.path.join(ROOT, 'work'))

# Frostbite footprint of the island (x0, z0, x1, z1). The level also holds marketing sets
# near the origin and a water vista kilometres out.
CROP = (-1600.0, 2600.0, -800.0, 3700.0)
# Per-level footprints (tools/collect.py writes work/level.json naming the level). Summer Camp
# (dingolevel_mpr): 99% of placements fall in x -290..410, z -800..-200; the ring out to this box
# is trees and scree rock.
LEVEL_CROPS = {
    'dingolevel_isle_of_grom': CROP,
    'dingolevel_mpr': (-500.0, -1050.0, 600.0, 50.0),
}
LEVEL_TITLES = {'dingolevel_isle_of_grom': 'Isle of Grom', 'dingolevel_mpr': 'Summer Camp'}
_LEVEL_JSON = os.path.join(WORK, 'level.json')
LEVEL = json.load(open(_LEVEL_JSON))['level'] if os.path.exists(_LEVEL_JSON) else 'dingolevel_isle_of_grom'
CROP = LEVEL_CROPS.get(LEVEL, CROP)
# Distant backdrop sublevels (iog_watervista, bam_vista): a vista mesh kilometres out, in its
# own frame. They wreck the recentre and nobody can skate them.
VISTA_SRC = re.compile(r'/[^/]*vista$')
# ...but a vista sublevel also carries ordinary art that sits INSIDE the playable crop: Summer
# Camp's bam_vista places 41 instances there (10 grass hill shells, 20 rocks, fences), and dropping
# them lost both their render and their collision. Only the true backdrop meshes stay dropped.
VISTA_BACKDROP = re.compile(r'vista|terrainmesh|oceanwaterplane|_island_')
# Water sheets whose collision is a different tessellation of the same slope as their render
# (Summer Camp's river: 144 triangles over 172,000 m2, up to 3 m apart). skate. rides the
# collision, so the sheet is drawn from it.
DRAW_FROM_COLLISION = re.compile(r'_water_[a-z]_\d+(?:_bed)?_mesh$')
# A collision triangle whose area-samples lie farther than this from every drawn surface is
# drawn itself (the "fill" pass), with its instance's own material, so nothing you can hit is
# invisible. Measured tool: tools/invisible_collision.py.
FILL_DIST = 0.5
FLAT_PROP_MAX = 0.08
STUB_HULLS = {}
FLAT_DROPPED = collections.Counter()
# ...unless the sheet is not lying on anything: a plywood board laid as a bridge over a gap (Summer
# Camp's lower gondola station: an 8x8 ft sheet spans a 0.9 m slot 1.4 m deep, under 5% of its area) is the ride
# surface. A sheet keeps its collision when more than FLAT_UNSUPPORTED of its top has no
# collision within FLAT_SUPPORT m below it.
FLAT_SUPPORT, FLAT_UNSUPPORTED = 0.12, 0.01
FLAT_PENDING = []
# Loose sheets stacked on one another (Summer Camp's pier kicker: three plywood sheets and a
# skatelite sheet, each tilted 11 deg, 2-5 cm apart): every sheet's upper edge was a lip on the
# way up. Only the top sheet of a stack collides.
SHEET_PROP = re.compile(r'plywoodsheet|skatelitesheet|constructionplywood', re.I)
SHEET_STACK_XZ = 0.5
FLAT_KEPT = collections.Counter()
# A tile under --min-tile-triangles is only a sliver if it also draws less than this (m2).
SLIVER_M2 = 300.0

# Material classes, one flat swatch each. Textures shared between tiles are promoted into
# the always-resident cPres_Global (11 MB cap), so ~500 per-material swatches cannot fit.
# (name, pattern on the lower-cased material, rgb, collide). First match wins.
CLASSES = [
    ('Rail',     r'(?<!none)grindable$', (205, 170, 40), True),
    ('Decal',    r'decal|trash_?scatter|puddle|leak|stain', (95, 90, 85), False),
    ('Glass',    r'glass|window', (120, 150, 170), True),
    ('Emissive', r'emissive', (235, 225, 190), False),
    # Summer Camp's river sheet is `River_M` and its bed `RiverBed_M` (classed Misc until 2026-09-23).
    ('Water',    r'water|ocean|river(?!bed)', (40, 90, 120), False),
    ('Foliage',  r'leaf|leaves|foliage|ivy|bush|tree|veg_|plant|flower|grass_skirt', (70, 110, 50), False),
    ('Grass',    r'grass|lawn|moss', (95, 130, 60), True),
    ('Sand',     r'sand|beach', (190, 170, 130), True),
    ('Dirt',     r'dirt|soil|mud|gravel|ground', (120, 95, 70), True),
    ('Rock',     r'rock|cliff|boulder|stone|granite|pebble|riverbed', (125, 120, 112), True),
    ('Asphalt',  r'asphalt|road|tarmac', (62, 62, 64), True),
    ('Concrete', r'concrete|cement|pavement|sidewalk|curb|skatepark|cinder|vtar_.*atlas|_proxy_atlas',
     (165, 162, 154), True),
    ('Brick',    r'brick', (150, 82, 62), True),
    ('Tile',     r'tile|ceramic|marble', (205, 205, 198), True),
    ('Wood',     r'wood|plank|timber|bark|pallet|plywood|cardboard', (140, 102, 64), True),
    ('Rust',     r'rust|corrod', (132, 72, 42), True),
    ('Metal',    r'metal|steel|iron|alumin|chrome|galvan|zinc|copper|grate|manhole', (112, 114, 118), True),
    ('Vehicle',  r'veh|brakelight|tyre|tire|wheel', (80, 92, 112), True),
    ('Fabric',   r'fabric|cloth|canvas|tarp|carpet|banner|vinyl', (150, 62, 62), True),
    ('Plastic',  r'plastic|rubber|pvc|black', (52, 52, 56), True),
    ('Paint',    r'paint|coat', (172, 172, 166), True),
    ('Sign',     r'sign|poster|graffiti|sticker', (200, 200, 200), True),
]
MISC = ('Misc', (145, 142, 138), True)
# Streaming tile edge in metres (the arena builder cuts, the packer declares a CSPA cell per tile).
TILE_SIZE = 100.0
_CLASS_RE = [(n, re.compile(p), rgb, c) for n, p, rgb, c in CLASSES]

# Clutter that costs triangles and adds nothing to ride. Measured: pebbles, asphalt chips
# and trash scatter alone are ~300k triangles on the island.
# `decal` used to drop all 459 decal placements including decal_ivy_01_2x2m (68x), which is
# ivy you can see on the walls. Only the pathological ones go: default_decalmesh_atlas carries
# 3.2 km bounds that wreck the recentre. Cables are thin but visible, so they stay too.
# common/utilities/ holds lighting helpers - util_shadowbox/shadowplane - which are invisible
# in skate. but rendered here as giant slabs, stretched up to 297x on one axis. That is the
# "things in the wrong place" a player sees.
DROP = re.compile(r'^world/vegetation/|pebbles|asphalt_chips|trash_?scatter|(^|/)effects/|occluder'
                  r'|leaves_|litter|default_decalmesh|^common/utilities/')
# No art asset is a sliver. Anything this anisotropic is a utility plane or box scaled to cover
# an area, and it renders as a slab through the middle of the level.
MAX_ANISOTROPY = 20.0
# What a skater comes to a spot for, by skate.'s own mesh names: the skatepark kit, DIY pieces,
# generated rails and stairs, ramps, funboxes, bowls. Spawns are placed at the densest clusters.
SKATE_FEATURE = re.compile(r'skatepark|skateparkdiy|localrail|stairslite|ramp|funbox|kicker|bowl|'
                           r'halfpipe|quarterpipe|manualpad|hubba|_rail_rollon|rainbowrail')
# Meshes where the coarsest shipped LOD is plenty. `rock` was here too, and it was the only
# entry that actually cost anything: it took cliffrock_06 from 1,252 triangles to 312 across
# 388 placements, cliffrock_02 from 1,058 to 264, and those are the island's visible rock
# formations. Measured across everything COARSE matched, 49% of the detail was thrown away
# before the budget was even consulted - and for lights, speakers, banners and barriers the
# best and coarsest LODs are identical, so the rule was a no-op there anyway. Graduated LOD
# stepping now spends the budget only where a tile is genuinely over, so the blunt cut is gone.
COARSE = re.compile(r'_lgt_|veh_|banner|speaker|constructionbarrier|palette')
# skate. ships its own collision as `<asset>.physics` (tools/physics.py) for 945 of the island's
# placed meshes, at about LOD0 detail even where the render mesh is a far-LOD shell. That is the
# surface skate. is ridden on, so it IS the collision: nothing is derived from render geometry for
# a mesh that has one, and a mesh skate. gives no collision (decals, cables, leaf piles, lights,
# wheels) gets none here either. Ride geometry among it is never thinned.
RIDE_COLLISION = re.compile(r'^world/(plots|terrain|roads)/|/generated/|localrail')
# Rock and cliff meshes are placed as props, but on this island they ARE the ground: the Rock
# class is 64% of the surface. They sit between clutter and ride geometry when collision is cut.
ROCK_COLLISION = re.compile(r'rock|cliff|boulder|landscaping|stone_?pile')


def collision_tier(mesh):
    """2 ride geometry, 1 rock ground, 0 clutter - the order collision detail is given up in reverse."""
    if RIDE_COLLISION.search(mesh) or SKATE_FEATURE.search(mesh.rsplit('/', 1)[-1]):
        return 2
    return 1 if ROCK_COLLISION.search(mesh) else 0


def physics_name(mesh):
    return mesh.removesuffix('_mesh') + '.physics'


def _column_query(P, T, up_idx, cell=2.0):
    """-> f(Q, reach) = (covered, floor): is there an up-facing surface ABOVE each point, and one
    within `reach` BELOW it. Heights are the floor triangles' exact heights at the point."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    lo = P[:, [0, 2]].min(0) - 4 * cell
    ncol = int(np.ceil((P[:, [0, 2]].max(0) - lo).max() / cell)) + 8
    tlo = np.floor((np.minimum(np.minimum(a, b), c)[:, [0, 2]] - lo) / cell).astype(np.int64)
    thi = np.floor((np.maximum(np.maximum(a, b), c)[:, [0, 2]] - lo) / cell).astype(np.int64)
    buckets = collections.defaultdict(list)
    for i in up_idx:
        for x in range(tlo[i, 0], thi[i, 0] + 1):
            for z in range(tlo[i, 1], thi[i, 1] + 1):
                buckets[x * ncol + z].append(i)

    def query(Q, reach):
        cov = np.zeros(len(Q), bool)
        flo = np.zeros(len(Q), bool)
        qk = np.floor((Q[:, [0, 2]] - lo) / cell).astype(np.int64)
        key = qk[:, 0] * ncol + qk[:, 1]
        order = np.argsort(key, kind='stable')
        keys, starts = np.unique(key[order], return_index=True)
        for j, kk in enumerate(keys):
            qall = order[starts[j]:starts[j + 1] if j + 1 < len(starts) else len(order)]
            ui = buckets.get(int(kk))
            if not ui:
                continue
            ui = np.array(ui)
            # queries x candidates arrays: a cell under a big floor or a dense rock holds
            # thousands of each, so work in chunks of at most ~200k pairs
            step = max(1, 200_000 // len(ui))
            for s0 in range(0, len(qall), step):
                qi = qall[s0:s0 + step]
                _column_chunk(a, b, c, ui, Q, qi, reach, cov, flo)
        return cov, flo
    return query


def _column_chunk(a, b, c, ui, Q, qi, reach, cov, flo):
    """One cell's queries against its candidate floors, written into cov / flo."""
    A, B, C = a[ui][None], b[ui][None], c[ui][None]
    q = Q[qi][:, None]
    v0, v1, v2 = (B - A)[..., [0, 2]], (C - A)[..., [0, 2]], (q - A)[..., [0, 2]]
    den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    wv = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
    ww = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
    ins = (wv >= -1e-6) & (ww >= -1e-6) & (wv + ww <= 1 + 1e-6)
    h = A[..., 1] + wv * (B - A)[..., 1] + ww * (C - A)[..., 1]
    y = q[..., 1]
    cov[qi] = (ins & (h > y + 0.02)).any(axis=1)
    flo[qi] = (ins & (h <= y + 0.02) & (h >= y - reach)).any(axis=1)


def unreachable_faces(P, T, reach=10.0, margin=1.0):
    """Mask of collision triangles no skater can touch. Removing them changes nothing a player
    can feel, which is the only kind of reduction that keeps skate.'s surface intact.

    * A face turned DOWN only collides with something coming up at it, so it is unreachable
      when there is no floor within `reach` below it to launch from: a rock's bottom buried in
      terrain, the underside of a slab over the sea.
    * A face turned SIDEWAYS is unreachable when the air just outside it is roofed over - by an
      up-facing surface above it there and `margin` metres to every side - and there is no floor
      within `reach` below: a rock's flank sunk into the terrain, a ledge's side below grade.

    `reach` is 10 m, not the 3 m first used: an air off a ramp clears 3 m easily, and a warehouse
    ceiling or the top of a tall wall indoors is exactly what such an air hits.
    """
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-12
    nn = np.where(ok[:, None], n / np.maximum(ln, 1e-12)[:, None], 0.0)
    ny = nn[:, 1]
    up = ok & (ny > 0.3)
    down = ok & (ny < -0.3)
    side = ok & ~up & ~down
    out = np.zeros(len(T), bool)
    if not up.any():
        return out
    query = _column_query(P, T, np.nonzero(up)[0])
    cen = (a + b + c) / 3
    di = np.nonzero(down)[0]
    if len(di):
        _, flo = query(cen[di], reach)
        out[di[~flo]] = True
    si = np.nonzero(side)[0]
    if len(si):
        q = cen[si] + 0.05 * nn[si]
        _, flo = query(q, reach)
        roofed = ~flo
        for dx, dz in ((0.0, 0.0), (margin, 0.0), (-margin, 0.0), (0.0, margin), (0.0, -margin)):
            if not roofed.any():
                break
            cov, _ = query(q + np.array([dx, 0.0, dz]), reach)
            roofed &= cov
        out[si[roofed]] = True
    return out


def surface_distance(Q, P, T, cell=2.0):
    """Exact distance from each point to the triangle soup (P, T), searching the 3x3 cells around."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    lo = np.minimum(P[:, [0, 2]].min(0), Q[:, [0, 2]].min(0)) - 2 * cell
    ncol = int(np.ceil((np.maximum(P[:, [0, 2]].max(0), Q[:, [0, 2]].max(0)) - lo).max() / cell)) + 5
    tlo = np.floor((np.minimum(np.minimum(a, b), c)[:, [0, 2]] - lo) / cell).astype(np.int64)
    thi = np.floor((np.maximum(np.maximum(a, b), c)[:, [0, 2]] - lo) / cell).astype(np.int64)
    buckets = collections.defaultdict(list)
    for i in range(len(T)):
        for x in range(tlo[i, 0], thi[i, 0] + 1):
            for z in range(tlo[i, 1], thi[i, 1] + 1):
                buckets[x * ncol + z].append(i)
    out = np.full(len(Q), np.inf)
    qk = np.floor((Q[:, [0, 2]] - lo) / cell).astype(np.int64)
    for j, p in enumerate(Q):
        cand = set()
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cand.update(buckets.get((qk[j, 0] + dx) * ncol + qk[j, 1] + dz, ()))
        if not cand:
            continue
        ii = np.fromiter(cand, np.int64)
        A, B, C = a[ii], b[ii], c[ii]
        ab, ac, ap = B - A, C - A, p - A
        d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
        bp, cp = p - B, p - C
        d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
        d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
        inside = (va >= 0) & (vb >= 0) & (vc >= 0)
        q = A + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]
        best = np.where(inside, np.linalg.norm(p - q, axis=1), np.inf)
        for s_, e_ in ((A, B), (B, C), (C, A)):
            ed = e_ - s_
            L = np.maximum((ed * ed).sum(1), 1e-30)
            t_ = np.clip(((p - s_) * ed).sum(1) / L, 0, 1)
            best = np.minimum(best, np.linalg.norm(p - (s_ + ed * t_[:, None]), axis=1))
        out[j] = best.min()
    return out


# Pieces whose ride surface is a curve (quarter pipes, banks, bowls, kickers, wedges). skate.
# installs only their LOD2 (a large quarter pipe: 84 triangles for LOD0's 388) and their
# collision has the same facets, so a small transition rides as a few flat steps.
CURVED_FEATURE = re.compile(r'ramp|quarter|qp_|bowl|bank|halfpipe|kicker|funbox|transition|wedge|'
                            r'launch|spine|pyramid|miniramp|sideramp')


def loop_subdivide(P, T, crease_deg=35.0, iters=2, weld=0.001):
    """Crease-aware Loop subdivision: smooth across soft edges, keep sharp ones exact.

    Edges whose faces meet at more than `crease_deg`, and open borders, are creases: their
    midpoints stay on the straight edge, and a vertex on two crease edges moves only along
    them (1/8, 6/8, 1/8), on more than two not at all. Everything else follows Loop's rules,
    so a coplanar region stays in its plane and a faceted transition becomes a smooth curve.
    Returns (P, T) welded and refined; triangle count x4 per iteration.
    """
    P = np.asarray(P, np.float64)
    T = np.asarray(T, np.int64)
    u, inv = np.unique(np.round(P / weld).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    P2 = np.zeros((len(u), 3))
    np.add.at(P2, inv, P)
    P = P2 / np.bincount(inv, minlength=len(u))[:, None]
    T = inv[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    cos_c = math.cos(math.radians(crease_deg))
    for _ in range(iters):
        n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        face = np.tile(np.arange(len(T)), 3)
        opp = np.concatenate([T[:, 2], T[:, 0], T[:, 1]])
        key = np.sort(E, axis=1)
        ek, einv = np.unique(key, axis=0, return_inverse=True)
        einv = einv.reshape(-1)
        cnt = np.bincount(einv, minlength=len(ek))
        # per edge: the (up to) two faces and opposite vertices
        order = np.argsort(einv, kind='stable')
        first = np.full(len(ek), -1)
        second = np.full(len(ek), -1)
        seen = np.zeros(len(ek), np.int64)
        for j in order:
            e = einv[j]
            if seen[e] == 0:
                first[e] = j
            elif seen[e] == 1:
                second[e] = j
            seen[e] += 1
        crease = cnt != 2
        two = ~crease
        f1, f2 = face[first[two]], face[second[two]]
        crease[np.nonzero(two)[0][(n[f1] * n[f2]).sum(1) < cos_c]] = True
        a, b = ek[:, 0], ek[:, 1]
        odd = 0.5 * (P[a] + P[b])
        sm = ~crease
        c_, d_ = opp[first[sm]], opp[second[sm]]
        odd[sm] = 0.375 * (P[a[sm]] + P[b[sm]]) + 0.125 * (P[c_] + P[d_])
        # even vertices
        nv = len(P)
        deg = np.bincount(np.concatenate([a, b]), minlength=nv)
        nsum = np.zeros((nv, 3))
        np.add.at(nsum, a, P[b])
        np.add.at(nsum, b, P[a])
        ck = np.bincount(np.concatenate([a[crease], b[crease]]), minlength=nv)
        csum = np.zeros((nv, 3))
        np.add.at(csum, a[crease], P[b[crease]])
        np.add.at(csum, b[crease], P[a[crease]])
        k = np.maximum(deg, 1)
        beta = np.where(k > 3, 3.0 / (8.0 * k), 3.0 / 16.0)
        even = (1 - k * beta)[:, None] * P + beta[:, None] * nsum
        on2 = ck == 2
        even[on2] = 0.75 * P[on2] + 0.125 * csum[on2]
        fixed = (ck == 1) | (ck > 2)
        even[fixed] = P[fixed]
        # new triangles
        m = nv + einv.reshape(3, -1).T      # edge-point index per (face, edge slot)
        e01, e12, e20 = m[:, 0], m[:, 1], m[:, 2]
        T = np.concatenate([np.stack([T[:, 0], e01, e20], 1), np.stack([T[:, 1], e12, e01], 1),
                            np.stack([T[:, 2], e20, e12], 1), np.stack([e01, e12, e20], 1)])
        P = np.concatenate([even, odd])
    return P, T


SIDE_REACH = 0.25   # m: how far sideways the seam pass looks for a floor when none is directly below


def seat_and_weld(parts, lip=0.12, weld=0.02, cell=2.0, flat_lip=0.05):
    """Make the ride surface one piece: no toe lips, no seam ledges. Edits `parts` in place.

    skate.'s pieces are separate objects dropped onto the ground: a ramp's or bank's toe often
    sits a few centimetres above the surface under it, and Skate 3's wheels catch on that step
    (the skater stops dead rolling onto the ramp). Two passes, collision only:

    * seat: a vertex at the LOW end of one of its own sloped up-facing faces (a ramp/bank/kicker
      toe), sitting 0-`lip` m above another piece's up-facing surface, is lowered onto that
      surface. Flat tops (curbs, ledges, plates) are never touched, nor is anything higher.
    * weld: vertices of different pieces within `weld` m of each other are merged to one point,
      so abutting slabs share their edge instead of leaving a hairline step.

    Returns (seated vertices, welded vertices).
    """
    if not parts:
        return 0, 0
    Ws = [np.asarray(W, np.float64) for _, W, _ in parts]
    offs = np.cumsum([0] + [len(W) for W in Ws])
    P = np.concatenate(Ws)
    P_orig = P.copy()
    owner_v = np.repeat(np.arange(len(parts)), [len(W) for W in Ws])
    Ts = [T + offs[j] for j, (_, _, T) in enumerate(parts) if len(T)]
    if not Ts:
        return 0, 0
    T = np.concatenate(Ts)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    ny = np.where(ln > 1e-12, n[:, 1] / np.maximum(ln, 1e-12), 0.0)
    up = ny > 0.7
    # toe vertices: lowest corner of a sloped (not flat) up-facing face
    sloped = np.nonzero(up & (ny < 0.995))[0]
    toe = np.zeros(len(P), bool)
    if len(sloped):
        ys = P[T[sloped], 1]
        rise = ys.max(axis=1) - ys.min(axis=1)
        low = (ys <= ys.min(axis=1, keepdims=True) + 0.01) & (rise > 0.02)[:, None]
        toe[T[sloped][low]] = True
    # any other vertex of an up-facing face: a plate, slab or base lying a few cm on another
    # piece leaves a lip along its edge; those are lowered too, but only up to `flat_lip`, so a
    # curb or ledge (10 cm and more) is never touched
    upv = np.zeros(len(P), bool)
    upv[T[up].ravel()] = True
    limit = np.where(toe, lip, flat_lip)
    # open (boundary) edges of each piece's up-facing surface: where one piece meets the next.
    # A seam can step UP as well as down; those vertices may also RISE to the neighbour (below).
    bnd = np.zeros(len(P), bool)
    if up.any():
        Tu = T[up]
        e = np.sort(np.concatenate([Tu[:, [0, 1]], Tu[:, [1, 2]], Tu[:, [2, 0]]]), axis=1)
        ue, ec = np.unique(e, axis=0, return_counts=True)
        bnd[ue[ec == 1].ravel()] = True
    toe = toe | upv
    seated = 0
    ti = np.nonzero(toe)[0]
    ui = np.nonzero(up)[0]
    if len(ti) and len(ui):
        A, B, C = a[ui], b[ui], c[ui]
        tri_owner = owner_v[T[ui, 0]]
        lo_xz = np.minimum(np.minimum(A, B), C)[:, [0, 2]]
        hi_xz = np.maximum(np.maximum(A, B), C)[:, [0, 2]]
        base = np.minimum(lo_xz.min(0), P[ti][:, [0, 2]].min(0)) - cell
        kl = np.floor((lo_xz - base) / cell).astype(np.int64)
        kh = np.floor((hi_xz - base) / cell).astype(np.int64)
        grid = collections.defaultdict(list)
        for i in range(len(ui)):
            for x in range(kl[i, 0], kh[i, 0] + 1):
                for z in range(kl[i, 1], kh[i, 1] + 1):
                    grid[(x, z)].append(i)
        for v in ti:
            q = P[v]
            cand = grid.get(tuple(np.floor((q[[0, 2]] - base) / cell).astype(np.int64).tolist()))
            if not cand:
                continue
            cand = np.array(cand)
            cand = cand[tri_owner[cand] != owner_v[v]]
            if not len(cand):
                continue
            A_, B_, C_ = A[cand], B[cand], C[cand]
            v0, v1, v2 = (B_ - A_)[:, [0, 2]], (C_ - A_)[:, [0, 2]], (q - A_)[:, [0, 2]]
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            den = np.where(np.abs(den) < 1e-12, 1e-12, den)
            wv = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            ww = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
            ins = (wv >= -1e-4) & (ww >= -1e-4) & (wv + ww <= 1 + 1e-4)
            if not ins.any():
                # Nothing directly under this point: a trim board or kerb laid in the joint between
                # two deck tiles (Summer Camp's pier: 31 trims 4.6 cm proud of the deck, every 8 m).
                # Use a neighbouring floor within SIDE_REACH m, extended to this point.
                if not (upv[v] or bnd[v]):
                    continue
                cx_ = (A_ + B_ + C_)[:, [0, 2]] / 3.0
                lo_ = np.minimum(np.minimum(A_, B_), C_)[:, [0, 2]] - SIDE_REACH
                hi_ = np.maximum(np.maximum(A_, B_), C_)[:, [0, 2]] + SIDE_REACH
                near_ = ((q[[0, 2]] >= lo_) & (q[[0, 2]] <= hi_)).all(axis=1)
                if not near_.any():
                    continue
                ins = near_
            h = (A_[:, 1] + wv * (B_ - A_)[:, 1] + ww * (C_ - A_)[:, 1])[ins]
            gap = q[1] - h
            ok = (gap > 0.002) & (gap <= limit[v])
            if ok.any():
                P[v, 1] = float(h[ok].max())
                seated += 1
            elif bnd[v]:
                # a seam edge sitting just BELOW its neighbour: raise it onto it, so the wheel
                # rolls up a centimetre-scale ramp instead of hitting a step
                rise = -gap
                okr = (rise > 0.002) & (rise <= flat_lip)
                if okr.any():
                    P[v, 1] = float(h[okr].min())
                    seated += 1
    # weld across pieces, by true distance (grid rounding split pairs 1 cm apart across cells)
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree
    pairs = cKDTree(P).query_pairs(weld, output_type='ndarray')
    if len(pairs):
        pairs = pairs[owner_v[pairs[:, 0]] != owner_v[pairs[:, 1]]]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else
                   (np.zeros(0), (np.zeros(0, int), np.zeros(0, int))), shape=(len(P), len(P)))
    _, inv = connected_components(g, directed=False)
    cnt = np.bincount(inv)
    multi = cnt[inv] > 1
    welded = 0
    if multi.any():
        acc = np.zeros((len(cnt), 3))
        np.add.at(acc, inv, P)
        mean = acc / cnt[:, None]
        # only merge where the group mixes pieces; within one piece skate.'s own topology stands
        first_owner = np.full(len(cnt), -1)
        first_owner[inv[::-1]] = owner_v[::-1]
        mixed = np.zeros(len(cnt), bool)
        np.logical_or.at(mixed, inv, owner_v != first_owner[inv])
        sel = mixed[inv]
        P[sel] = mean[inv[sel]]
        welded = int(sel.sum())
    # Big level plates keep their authored height. A 4x4 m patio tile has only its corners: seat
    # or weld one onto the ground under it and the whole tile tilts, its neighbours tilt the other
    # way, and every tile edge became a 3-6 cm lip (Summer Camp's chalet patio, all authored at
    # 97.765, measured -4..+4 cm after this pass - where the user clipped, 2026-09-23). Seams
    # between level plates are ramped instead (ramp_seams).
    a0, b0, c0 = P_orig[T[:, 0]], P_orig[T[:, 1]], P_orig[T[:, 2]]
    n0 = np.cross(b0 - a0, c0 - a0)
    l0 = np.linalg.norm(n0, axis=1)
    plate = (n0[:, 1] > 0.995 * np.maximum(l0, 1e-12)) & (0.5 * l0 > 0.5)
    keep = np.zeros(len(P), bool)
    keep[T[plate].ravel()] = True
    P[keep] = P_orig[keep]
    for j, (p_, W, T_) in enumerate(parts):
        parts[j] = (p_, P[offs[j]:offs[j + 1]].astype(np.asarray(W).dtype), T_)
    return seated, welded


def reduce_welded(objs, target):
    """Ride collision of one tile reduced to about `target` triangles AS ONE WELDED SURFACE.

    Ride pieces butt against each other - plot slabs, terrain patches, ledges on a sidewalk -
    so their shared edges must be welded (1 mm) or every border is pinned and the interiors
    over-collapse: reducing them object by object measured p99 18-60 cm and spikes of 6.9 m,
    against p99 at most 4.4 cm welded. Borders of the whole tile are still held, so no seam
    opens to the next tile. Returns ([(W, T)], (p99, max) metres of deviation).
    """
    import pyfqmr
    objs = [(W, T) for W, T in objs if len(T)]
    if not objs or sum(len(T) for _, T in objs) <= target:
        return objs, (0.0, 0.0)
    P = np.concatenate([W for W, _ in objs])
    off, Ts = 0, []
    for W, T in objs:
        Ts.append(T + off)
        off += len(W)
    T = np.concatenate(Ts)
    samples = P[np.unique(T)]
    u, inv = np.unique(np.round(P, 3), axis=0, return_inverse=True)
    T = inv.reshape(-1)[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    smp = pyfqmr.Simplify()
    smp.setMesh(u.astype(np.float64), T.astype(np.int32))
    smp.simplify_mesh(target_count=int(target), aggressiveness=7, preserve_border=True, verbose=False)
    P2, T2, _ = smp.getMesh()
    P2, T2 = np.asarray(P2, np.float64), np.asarray(T2, np.int64)
    d = surface_distance(samples[::max(1, len(samples) // 3000)], P2, T2)
    d = d[np.isfinite(d)]
    return [(P2, T2)], (float(np.percentile(d, 99)) if len(d) else 0.0, float(d.max()) if len(d) else 0.0)


MIN_OBJECT_TRIS = 12


def reduce_tile(parts, target, prop_floor, rock_floor):
    """One tile's collision parts [(tier, W, T)] cut to about `target` triangles.

    Clutter gives up detail first (object by object, down to `prop_floor` of itself), then rocks
    (object by object, down to `rock_floor`), then ride geometry (welded as one surface). Returns
    (parts, {group: (p99, max)}, {group: triangles before}).
    """
    names = {0: 'prop', 1: 'rock', 2: 'ride'}
    groups = {0: [], 1: [], 2: []}
    for tier, W, T in parts:
        if len(T):
            groups[min(2, int(tier))].append((W, T))
    n = {g: sum(len(T) for _, T in objs) for g, objs in groups.items()}
    need = sum(n.values()) - target
    floors = {0: prop_floor, 1: rock_floor, 2: 0.0}
    out, errs = [], {}
    for g in (0, 1, 2):
        cut = max(0, min(need, n[g] - int(n[g] * floors[g]))) if need > 0 else 0
        need -= cut
        objs = groups[g]
        if cut and objs:
            red, errs[names[g]] = (reduce_welded if g == 2 else reduce_objects)(objs, max(1, n[g] - cut))
        else:
            red = objs
        out += [(g, W, T) for W, T in red]
    return out, errs, {names[g]: n[g] for g in (0, 1, 2)}


def closest_on_mesh(Q, P, T, chunk=48):
    """Closest point on the triangle soup (P, T) to each point of Q (brute force, small meshes)."""
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    AB, AC = B - A, C - A
    out = np.empty_like(Q)
    for s0 in range(0, len(Q), chunk):
        p = Q[s0:s0 + chunk, None, :]                       # (q, 1, 3)
        ap, bp, cp = p - A, p - B, p - C                    # (q, t, 3)
        d1, d2 = (AB * ap).sum(-1), (AC * ap).sum(-1)
        d3, d4 = (AB * bp).sum(-1), (AC * bp).sum(-1)
        d5, d6 = (AB * cp).sum(-1), (AC * cp).sum(-1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        den = va + vb + vc
        den = np.where(np.abs(den) < 1e-30, 1e-30, den)
        v, w = vb / den, vc / den
        cand = [A + AB * v[..., None] + AC * w[..., None]]
        ok = [(va >= 0) & (vb >= 0) & (vc >= 0)]
        for s_, e_ in ((A, B), (B, C), (C, A)):
            ed = e_ - s_
            L = np.maximum((ed * ed).sum(-1), 1e-30)
            t_ = np.clip(((p - s_) * ed).sum(-1) / L, 0, 1)
            cand.append(s_ + ed * t_[..., None])
            ok.append(np.ones_like(ok[0]))
        best_d = np.full(p.shape[:2], np.inf)
        best_p = np.zeros(p.shape[:2] + (3,))
        for cq, okq in zip(cand, ok):
            dd = np.where(okq, np.linalg.norm(p - cq, axis=-1), np.inf)
            sel = dd < best_d
            best_d = np.where(sel, dd, best_d)
            best_p = np.where(sel[..., None], cq, best_p)
        k = best_d.argmin(axis=1)
        out[s0:s0 + chunk] = best_p[np.arange(len(k)), k]
    return out


def reduce_objects(objs, target, min_tris=MIN_OBJECT_TRIS):
    """Reduce a set of separate collision objects to about `target` triangles, EACH ON ITS OWN.

    Collapsing a tile's objects as one welded mesh let edges collapse across the air between
    them: props vanished or became collision sheets metres from any surface (measured p99 8 m,
    spikes of 66 m). Here every object keeps its own mesh, all shrink by one shared fraction,
    and none goes below `min_tris`, so a barrier stays a barrier and a rock stays a rock.
    Returns ([(W, T)], (p99, max) metres), deviation measured per object against itself.
    """
    import pyfqmr
    objs = [(W, T) for W, T in objs if len(T)]
    n = np.array([len(T) for _, T in objs])
    if not len(n) or n.sum() <= target:
        return objs, (0.0, 0.0)
    out, dev, goal = objs, [], target
    for _round in range(4):
        lo, hi = 0.0, 1.0
        for _ in range(40):
            f = (lo + hi) / 2
            if np.maximum(np.minimum(n, min_tris), n * f).sum() > goal:
                hi = f
            else:
                lo = f
        f = lo
        out, dev = [], []
        per_obj = max(1, 4000 // max(1, int((n * f < n).sum())))
        for (W, T), ni in zip(objs, n):
            want = max(min(ni, min_tris), int(ni * f))
            if want >= ni:
                out.append((W, T))
                continue
            used, inv = np.unique(T, return_inverse=True)
            Wl, Tl = W[used], inv.reshape(-1, 3)
            smp = pyfqmr.Simplify()
            smp.setMesh(Wl.astype(np.float64), Tl.astype(np.int32))
            smp.simplify_mesh(target_count=int(want), aggressiveness=7, preserve_border=True, verbose=False)
            W2, T2, _ = smp.getMesh()
            W2, T2 = np.asarray(W2, np.float64), np.asarray(T2, np.int64)
            if not len(T2):
                out.append((Wl, Tl))
                continue
            # Quadric placement puts a collapsed vertex where the error is least, which on a rock
            # cut to 12% of itself is often OUTSIDE the drawn shape: a wall you hit and cannot see.
            # Snap every kept vertex back onto the original surface. Faces between snapped vertices
            # are chords, inside a convex rock rather than outside it.
            W2 = closest_on_mesh(W2, Wl, Tl)
            out.append((W2, T2))
            d = surface_distance(Wl[::max(1, len(Wl) // per_obj)], W2, T2)
            dev.append(d[np.isfinite(d)])
        total = sum(len(T) for _, T in out)
        if total <= target * 1.02:
            break
        goal = max(1, int(goal * target / total))
    d = np.concatenate(dev) if dev else np.zeros(1)
    return out, (float(np.percentile(d, 99)) if len(d) else 0.0, float(d.max()) if len(d) else 0.0)


def qem_simplify(P, T, target):
    """Quadric edge-collapse to about `target` triangles; falls back to clustering without it.

    Clustering snaps every vertex to a grid, which is what shredded props to 5% of themselves;
    edge collapse removes the flattest detail first. Used only on non-ride geometry.
    """
    if len(T) <= target:
        return P, T
    try:
        import fast_simplification
    except ImportError:
        P2, _, T2 = cluster_simplify(P, None, T, target)
        return P2, T2
    P2, T2 = fast_simplification.simplify(P.astype(np.float32), T.astype(np.int64),
                                          target_reduction=float(1 - target / len(T)))
    return P2.astype(np.float64), T2.astype(np.int64)


# Ride geometry, grouped so a plot's pieces change LOD together (see the graduated LOD step).
RIDE_GROUP = re.compile(r'^world/plots/[^/]+/([^/]+)/|/generated/(local\w+?)_agasset')


def ride_group(mesh):
    m = RIDE_GROUP.search(mesh)
    return ('plot', m.group(1) or m.group(2)) if m else None


# Light-glow proxies (CMN_LGT_Emissive_M_proxy): pure decoration, and the largest class in
# five of the six tiles that blew the 5 MB stream limit on the first build.
DROP_CLASSES = {'Emissive'}
# What decimation must never touch. Collision has to cover the GROUND, not just the ride
# surfaces: measured against DM Jumpline, whose collision covers 84% of the cells within 12 m
# of its spawn and exactly matches its own visual coverage, the first collision copy covered
# 38% where visuals covered 99.7% - so the skater landed on the patch under the marker and
# then dropped through the first hole. This island's ground is mostly Rock and Misc, so those
# belong here; only clutter is decimated.
PROTECTED = {'Rail', 'Concrete', 'Asphalt', 'Tile', 'Brick', 'Wood',
             'Rock', 'Dirt', 'Grass', 'Sand', 'Misc', 'Paint',
             # a river sheet is ~140 triangles over 170,000 m2: thinning it only opens holes
             'Water'}
# Props (non-PROTECTED classes) are never drawn below this fraction of their triangles. It was
# 0.05, and 13 Summer Camp tiles shipped their poles, fences and signs as 5% shreds whose whole
# collision stayed: walls you hit and cannot see. The fill pass draws what is still hidden.
PROP_THIN_FLOOR = 0.25
# Graduated LOD stepping keeps a prop at a LOD with at least this share of its finest LOD's
# triangles while anything else in the tile can still give (see `nxt` in main).
PROP_LOD_FLOOR = 0.25
PROP_SMALL_TRIS = 8000
# Props whose coarsest LODs read as broken up close; these keep the floor even when the tile is
# still over (the floor was lifted in 9 of Summer Camp's 13 hot tiles, bike rack included).
DETAIL_PROP = re.compile(r'bicycle')
# Ground may still be thinned if a tile is far over budget, but never below this fraction:
# holes in a floor cost more than bytes.
GROUND_FLOOR = 0.7
# Surfaces it would be wrong to stand on, whatever their shape.
NEVER_COLLIDE = {'Water', 'Foliage', 'Emissive'}
# A triangle at least this large (m2) and facing up is treated as floor regardless of class.
FLOOR_TRI_AREA = 2.0
CLASS_SOURCES = collections.defaultdict(set)
RESCUED = collections.Counter()
TEXTURES = {}
TEXTURE_BLOBS = {}   # GLB image key -> PNG bytes (per-mesh images and 'tex:<asset>')
CLASS_BLOBS = {}
CLASS_NORMALS = {}
KEY_TEX = {}        # depot key (str) -> texture asset name
MATERIAL_TEX = {}   # material name -> texture asset name
TEX_BLOBS = {}      # texture asset name -> PNG bytes
# Where each blob key's full-resolution source lives, so per-tile copies can be cut at any size
# up to native without loading every texture at that size up front. key (blob key without its
# 'tex:'/'dec:' prefix) -> {'fp', 'tint', 'nfp', 'nat', 'nnat', 'rgba'}.
TEX_SRC = {}
TEX_CACHE_BYTES = [300 * 1024 * 1024]
_FULL = collections.OrderedDict()
_FULL_BYTES = [0]
# Real render stream bytes per triangle. The planning figure PRES_BYTES_PER_TRI (55.2) is what
# the budget passes use; the shipped Summer Camp streams came in at 0.69x of it (cPres_150_-50:
# estimate 9.2 MB, built 6.3 MB), so the texture planner uses the measured figure.
MESH_BYTES_PER_TRI = 45.0
# the stream estimate undershot the built streams by ~1.19x on Summer Camp HD (11.23 vs 13.32 MB)
TEX_EST_FUDGE = 1.19   # fitted on 36 Summer Camp render streams, 2026-09-22 (p95 resid 0.7 MB)
GLOBAL_TEX_KEYS = set()
TEX_NORMALS = {}
ATLAS_BASES = set()
KEY_TINT = {}       # depot key -> paint, for blocks with a colour but no texture
# Baked sheets: `projectlod` far-LOD proxy atlases and anything named an atlas. UVs into
# these index a packed sheet, so they are exempt from tiling correction.
ATLAS_TEX_RX = re.compile(r'projectlod|_proxy_c$|_atlas')
# skate.'s tiling library materials carry their real-world repeat in the name (`_2x2m`, `_1x1m`)
LIBRARY_TEX_RX = re.compile(r'_\d+x\d+m(_|$)')
# Far-LOD proxy plates get their own material name so downstream can tell them from real
# art: they are the plot silhouettes the engine draws from kilometres away, they carry only
# magenta placeholder colour, and standing on one means standing on scenery. The spawn
# search excludes them (tools/find_spawns.py).
# mesh name -> the material its FINEST LOD declares. The fine LODs of 283 meshes on this crop
# are not in the install (the bowl's LOD0-2 chunks are in no index, in either GUID order), so
# the only geometry we can draw is the proxy shell. But the MeshSet res IS resident, and it
# lists every LOD's section materials - so the shell can be painted with the material the
# real thing uses: `Concrete_Segmented` for iog_plt_11's bowl, granite for ftue's bridge,
# rather than one flat class concrete over all 296,000 m2 of it.
PROXY_MAT = {}
# the arena builder derives cPres and cSim from the SAME triangles, so the only way to give
# collision less detail than the render mesh is to emit a second copy routed collision-only,
# which is what d2s3 does for Descenders. Measured ceiling: across 380 stock and community
# cSim streams the largest is 0.70 MB (p90 0.38). The first Isle of Grom build routed every
# solid class into collision at full render density, shipped 1.55 MB cSim tiles, and corrupted
# the guest heap on load_thread while the stream was being parsed.
COLLISION_CLASS = 'Collision'
COLLISION_RGB = (80, 80, 90)
# Measured on the first build (DIST_IsleOfGrom, median over tiles >5k triangles). the arena builder
# streams cap at 5 MB; a fat cPres tile crashes the stream, a fat cSim hangs the load.
PRES_BYTES_PER_TRI, SIM_BYTES_PER_TRI = 55.2, 30.2

SWATCH_PX = 256          # d2s3: smaller images arrive zero-padded, near-black, served white
SPECULAR_GREY = 5        # d2s3 material_specular 0.02
# the arena builder maps glTF EMISSIVE -> the game's LIGHTMAP channel. Supply nothing and it binds
# the base game's `default_white`, so every surface is lit as if its baked lightmap were pure
# white - on `environment.default`, whose whole job is compositing the diffuse against baked
# light. That is why the island came out bleached and flat with correct textures on it.
# The swatch must NOT be uniform: the engine's near-black test on this slot is really a
# near-UNIFORM detector (<= 2 distinct block values across the probe grid), and a flat swatch
# is exactly one, so it is discarded and the slot served the 1x1 WHITE fallback - which is how
# d2s3 measured 0.5 and 0.25 as indistinguishable. A gentle gradient defeats it, and the
# amplitude has to clear DXT1's RGB565 quantisation (8 on R/B, 4 on G).
LIGHTMAP_GREY = 0.60     # 0 disables; 1.0 is the white we get for free today
LIGHTMAP_DITHER = 8.0
ATTRIBUTOR = 'environmentsimple.default'
# Real skate. textures, extracted once by tools/extract_textures.py. Frostbite streams
# top mips and the installed game ships only the resident tail plus (often) one
# catalog-held mip, so these are the best level actually present - typically 128-512 px.
# A mesh with no usable texture falls back to its flat class swatch.
# skate.'s surface materials are the `*_2x2m` library: one texture repeat per 2 m, so
# 0.50 tiles/m is what the game's own UVs are authored to deliver.
# Rather than a hand table, each material's density is MEASURED off the placed geometry
# and corrected to that target; the class table below is only a fallback for a material
# too small to measure. It was itself derived from those measurements, which is why the
# two agree: Wood 0.256 -> x1.95, Glass 0.255 -> x1.96, Tile 0.323 -> x1.55.
UV_TARGET = 0.50
UV_MEASURE_MIN_M2 = 25.0     # below this a material's density is too noisy to trust
UV_FIX_LO, UV_FIX_HI = 0.08, 12.0
UV_SCALE = {'Wood': 1.95, 'Glass': 1.95, 'Tile': 1.55, 'Plastic': 0.20, 'Rust': 1.19,
            'Foliage': 0.60}
MIN_TEXTURE_PX = 32
# cPres_Global cost of one promoted image, measured on the Summer Camp full-resolution build
# (colour and normal alike): 22.4 KB of fixed overhead + 0.667 B per top-mip texel (mips
# included) - 64 px 25.1 KB, 128 px 33.3 KB, 256 px 66.1 KB, 512 px 197 KB, 1024 px 721 KB.
# The overhead dominates: 263 shared images at 64 px are 6.6 MB before any detail.
SHARED_IMAGE_FIXED = 22_413
SHARED_BYTES_PER_PX = 0.6667
# One texture (cmn_prp_trashboxes_01_a_c, the single 1024x1024 that survived streaming)
# is 2.8 MB of the 4.9 MB total; every other texture is under 150 KB. The last build's
# worst tile was 4.9 MB against a 5.0 MB stream ceiling, so the big one gets capped.
MAX_TEXTURE_PX = 256
# 171 of the 194 textures in use are 128 px at source and cannot go higher - streaming
# keeps the top mips out of the install - so a cap above that costs nothing for them and
# only spends bytes on the 20 that do carry more.
MAX_MATERIAL_PX = 256
# Textures shared by two or more tiles are promoted into the always-resident cPres_Global,
# whose limit is 11 MB (d2s3/knobs.py: "Igloo boots with 10.9 MB; Favela hangs with 21.8"),
# which is why the cap is low. But coverage is wildly uneven: `terrain_rock_02_2x2m_c_ta`
# alone is 64% of the island's visible surface, so spending three times its bytes on it
# buys more than the same bytes spread over two hundred bolts. A texture earns the higher
# cap by measured area, read from the previous export's uv_density.json.
HOT_MATERIAL_PX = 512
HOT_TEXTURE_M2 = 40_000.0
HOT_TEX = set()
CLASS_TEXTURES = os.path.join(WORK, 'class_textures.json')
# Decal volumes (graffiti, stains, posters): tools/collect.py -> decals.json, tools/bake_decals.py
# -> decal_textures.json + RGBA PNGs. Drawn as lifted quads in their own class.
DECALS_JSON = os.path.join(WORK, 'decals.json')
DECAL_TEXTURES = os.path.join(WORK, 'decal_textures.json')
DECAL_CLASS = 'DecalVol'
DECAL_LIFT = 0.02       # m off the surface the decal is projected onto (its own quad fallback)
# Skate 3's world pass depth-tests a D32F buffer with a standard (not reversed) projection: two
# surfaces 2 cm apart flash beyond ~130 m, mm-thin layers from ~30 m (the user saw graffiti and
# ground decals flash, 2026-09-23). Wall art sits 5 cm out; ground decals 2.5 cm (a wheel is
# 2.7 cm tall - more would hide it); each decal overlapping an earlier one stacks DECAL_STACK
# higher, so no two share a depth.
DECAL_LIFT_WALL, DECAL_LIFT_FLOOR, DECAL_STACK = 0.05, 0.025, 0.006
DECAL_MODE = {}         # 'dec:<png>' -> 'MASK' | 'BLEND' | None
# Decals are cut out in GEOMETRY: the recomp drew every transparent texel black whatever the
# shading family (2026-09-23 screenshots), so only the opaque cells of a DECAL_CUT_GRID grid over
# each decal's art are drawn at all (merged by quadtree). ~4 cm steps on a 2.5 m tag.
DECAL_CUT_GRID = 32
MATERIAL_TEXTURES = os.path.join(WORK, 'material_textures.json')
DEPOT_KEY_TEXTURES = os.path.join(WORK, 'depot_key_textures.json')
DEPOT_KEY_TINTS = os.path.join(WORK, 'depot_key_tints.json')
TEXTURE_MANIFEST = os.path.join(WORK, 'texture_manifest.json')
TEXTURE_BINDINGS = os.path.join(WORK, 'texture_bindings.json')


# Colour grade measured against skate. footage (Summer Camp video, 2026-09-30): sunlit surfaces
# sampled in the video vs the same surfaces in Skate 3 screenshots of this build, converted to a
# target texture mean through the engine's measured gain (concrete texture 182 -> 155 on screen).
# skate. paints these families with palette/VT colours the converter does not reproduce: its
# concrete is warm beige-grey, its ramp plywood/skatelite light honey and even, its dirt grey-tan.
# (file-name regex, target mean sRGB, contrast scale)
GRADE = [
    (re.compile(r'concrete_poured|TERR_concrete|FILL_concrete', re.I), (181, 168, 145), 0.6),
    (re.compile(r'skatelite|plywood', re.I), (209, 175, 138), 0.6),
    (re.compile(r'TERR_dirt', re.I), (146, 138, 114), 1.0),
    (re.compile(r'ARRAY_array_terrain_rock_02_2x2m_c_s0', re.I), (160, 152, 128), 1.0),
]


def grade_array(name, a):
    """Apply GRADE to an RGB(A) uint8 array whose source file name is `name` (in place copy)."""
    if not name:
        return a
    for rx, target, contrast in GRADE:
        if rx.search(name):
            rgb = a[..., :3].astype(np.float32) / 255.0
            lin = rgb ** 2.2
            mean = lin.reshape(-1, 3).mean(axis=0)
            tgt = (np.asarray(target, np.float32) / 255.0) ** 2.2
            if contrast != 1.0:
                lin = mean + (lin - mean) * contrast
            lin = lin * (tgt / np.maximum(mean, 1e-4))
            out = a.copy()
            out[..., :3] = np.clip((np.clip(lin, 0, 1) ** (1 / 2.2)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            return out
    return a


def _graded_png(fp, data):
    """PNG bytes of `fp` with GRADE applied when its name matches a rule."""
    name = os.path.basename(fp)
    if not any(rx.search(name) for rx, _, _ in GRADE):
        return data
    im = Image.open(io.BytesIO(data))
    a = np.asarray(im.convert('RGBA' if 'A' in im.mode else 'RGB'))
    buf = io.BytesIO()
    Image.fromarray(grade_array(name, a)).save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


def _capped(fp, limit=None):
    limit = limit or MAX_TEXTURE_PX
    data = _graded_png(fp, open(fp, 'rb').read())
    im = Image.open(io.BytesIO(data))
    if max(im.size) > limit:
        im.thumbnail((limit, limit))
        buf = io.BytesIO()
        im.save(buf, format='PNG', optimize=True)
        data = buf.getvalue()
    return data


def normal_z(a):
    """Rebuild a normal map's blue channel as Z = sqrt(1 - x^2 - y^2), in place.

    skate.'s `_ny` textures keep X and Y in red and green and something that is NOT Z in blue
    (flat plastic stores a constant 0.78, asphalt averages 0.60). Skate 3's environmentsimple
    shader reads blue as Z raw, without normalising: kd = (x*0.58 + y*0.62 + z*0.39) * 2.396
    with z = 2*blue - 1, so asphalt lit at 0.18 of a flat surface and anything under 0.5 blue
    went negative - the black floors of 2026-09-30, once per-tile textures kept their own
    normal maps.
    """
    xy = a[..., :2].astype(np.float32) / 127.5 - 1.0
    z = np.sqrt(np.clip(1.0 - (xy * xy).sum(axis=-1), 0.0, 1.0))
    a[..., 2] = np.clip(127.5 + 127.5 * z + 0.5, 0, 255).astype(np.uint8)
    return a


def _capped_normal(fp, limit=None):
    """A normal map PNG: never colour graded (GRADE matches by file name and turned the
    Concrete and Wood class normals into flat concrete- and wood-coloured images), Z rebuilt."""
    limit = limit or MAX_TEXTURE_PX
    im = Image.open(fp).convert('RGB')
    if max(im.size) > limit:
        im.thumbnail((limit, limit))
    buf = io.BytesIO()
    Image.fromarray(normal_z(np.array(im))).save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


def hot_textures():
    """Texture assets covering more than HOT_TEXTURE_M2 of the island, from the last export.

    Read from work/, not from the output directory: the build stages the GLBs by emptying
    that directory first, so anything left there is gone by the time the next export runs.
    """
    dens = os.path.join(WORK, 'uv_density.json')
    binding = MATERIAL_TEXTURES
    if not (os.path.exists(dens) and os.path.exists(binding)):
        return set()
    base_of = {m: e['base'] for m, e in json.load(open(binding)).items() if e.get('base')}
    area = collections.Counter()
    for r in json.load(open(dens)):
        b = base_of.get(r['material'])
        if b:
            area[b] += r['m2']
    return {b for b, a in area.items() if a >= HOT_TEXTURE_M2}


def _tint_srgb(tint):
    """A paint factor as a multiplier for sRGB bytes.

    The paint is a LINEAR-light colour parameter and the PNG is sRGB: multiplying the bytes by
    it raw squared the darkening, so stained decks (0.053, 0.015, 0.000) and painted asphalt
    (0.126, 0.109, 0.099) came out near black - 14,600 m2 of floor under 30/255 on Summer Camp.
    """
    return np.clip(np.asarray(tint, dtype=np.float32), 0.0, None) ** (1 / 2.2)


def _tinted(data: bytes, tint) -> bytes:
    """PNG with the material's paint applied. See TINT in tools/bind_depot.py."""
    im = Image.open(io.BytesIO(data)).convert('RGB')
    a = np.asarray(im, dtype=np.float32) * _tint_srgb(tint)[None, None, :]
    buf = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB').save(
        buf, format='PNG', optimize=True)
    im.close()
    return buf.getvalue()


def _full_array(key, edge, normal=False):
    """Decoded source texture resized to a power-of-two edge <= `edge` (LRU cached)."""
    ck = (key, edge, normal)
    if ck in _FULL:
        _FULL.move_to_end(ck)
        return _FULL[ck]
    src = TEX_SRC[key]
    fp = os.path.join(WORK, src['nfp'] if normal else src['fp'])
    im = Image.open(fp)
    im = im.convert('RGBA' if (src.get('rgba') and not normal) else 'RGB')
    w, h = im.size
    while max(w, h) > edge and min(w, h) > 1:
        w, h = max(1, w // 2), max(1, h // 2)
    if (w, h) != im.size:
        im = im.resize((w, h), Image.LANCZOS)
    a = np.asarray(im, dtype=np.uint8).copy()
    if normal:
        normal_z(a)
    else:
        a = grade_array(os.path.basename(src['fp']), a)
    if src.get('tint') and not normal:
        t = _tint_srgb(src['tint'])
        a[..., :3] = np.clip(a[..., :3].astype(np.float32) * t[None, None, :], 0, 255).astype(np.uint8)
    _FULL[ck] = a
    _FULL_BYTES[0] += a.nbytes
    while _FULL_BYTES[0] > TEX_CACHE_BYTES[0] and len(_FULL) > 1:
        _k, old = _FULL.popitem(last=False)
        _FULL_BYTES[0] -= old.nbytes
    return a


def full_png(key, edge, normal=False):
    """PNG of `key` at <= `edge` px from its full-resolution source.

    A tile's copy is NOT altered to make it unique: the arena builder folds the image name
    ('<texture>@<u>_<v>') into the texture GUID (TextureGuidStrategy.ContentAddressedFullGuid),
    which keeps identical copies in their own tiles. The old way, a 4x4-texel block shifted
    40-88 levels in blue (colour) and red (normal), showed on 64 px copies as a 12.5 cm tinted,
    differently lit square every 2 m ("little squares repeating", owner, 2026-10-01).
    """
    a = _full_array(key, edge, normal)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


def _tex_key(base, tint):
    """Blobs are per (asset, paint): one master texture, several painted variants."""
    if not tint:
        return base
    return base + '|' + ','.join(f'{v:.3f}' for v in tint)


def _load_binding(path, into_albedo, into_normal, into_atlas):
    """A binding file -> {owner: blob key}, plus one PNG blob per (asset, paint).

    Blobs are keyed by the texture asset rather than by the owner, because 2,961 depot keys
    and 1,131 material names resolve to only 194 distinct textures: keying by the owner
    embeds the same image once per owner and pushes tiles past the 5 MB stream ceiling.

    The key includes the PAINT, though. skate. ships near-greyscale master textures and
    colours them per material - `_M_Grey`, `_M_BrownWarm`, `OrangePaint` all share one
    texture - so collapsing to the asset alone renders every variant identically. That is
    why so much of the island came out the same grey.
    """
    owner = {}
    if not os.path.exists(path):
        return owner
    for name, e in json.load(open(path)).items():
        base = e.get('base')
        fp = os.path.join(WORK, e['png'])
        if not base or not os.path.exists(fp):
            continue
        tint = e.get('tint')
        key = _tex_key(base, tint)
        if key not in TEX_SRC:
            nfp = e.get('normal') if e.get('normal') and os.path.exists(os.path.join(WORK, e['normal'])) else None
            TEX_SRC[key] = {'fp': e['png'], 'tint': tint, 'nfp': nfp,
                            'nat': max(e.get('w', 0), e.get('h', 0)) or None,
                            'nnat': max(e.get('nw', 0), e.get('nh', 0)) or None}
        if key not in into_albedo:
            cap = HOT_MATERIAL_PX if base in HOT_TEX else MAX_MATERIAL_PX
            png_ = _capped(fp, cap)
            into_albedo[key] = _tinted(png_, tint) if tint else png_
            if e.get('normal') and os.path.exists(os.path.join(WORK, e['normal'])):
                into_normal[key] = _capped_normal(os.path.join(WORK, e['normal']), cap)
            if ATLAS_TEX_RX.search(base.lower()):
                into_atlas.add(key)
        owner[name] = key
    return owner


def load_material_textures():
    """source material name -> texture asset name (the texture the GAME assigns it).

    Resolved by tools/bind_depot.py through the material's shader block; this is the
    fallback path for a section whose own depot key carries no colour slot.
    """
    return _load_binding(MATERIAL_TEXTURES, TEX_BLOBS, TEX_NORMALS, ATLAS_BASES)


def proxy_materials(meshsets, resident):
    """mesh -> the material its finest LOD names, for meshes whose only LOD is a proxy shell."""
    out = {}
    for name, m in meshsets.items():
        if not any(ATLAS_RX.search(sec.material.lower())
                   for lod in m.lods if lod.chunk in resident
                   for sec in lod.sections if sec.primitives):
            continue
        for lod in m.lods:                       # finest first
            best = collections.Counter()
            for sec in lod.sections:
                if sec.is_shadow or not sec.primitives:
                    continue
                if ATLAS_RX.search(sec.material.lower()):
                    continue
                if classify(sec.material)[0] in DROP_CLASSES:
                    continue
                best[sec.material] += sec.primitives
            if best:
                out[name] = best.most_common(1)[0][0]
                break
    return out


def load_key_textures():
    """depot key -> texture asset name, the texture bound to THAT shader block.

    A material name is not an identity: `m_proxy_atlas` is shared by 237 depot keys holding
    226 different baked atlases and covering 24% of the island, so binding it by name paints
    a quarter of the map with one arbitrary wall-cable atlas. The depot key is the identity
    the game itself uses, and tools/meshset.py reads it off each section at +0x138.
    """
    return _load_binding(DEPOT_KEY_TEXTURES, TEX_BLOBS, TEX_NORMALS, ATLAS_BASES)


def load_class_textures():
    """material class -> (albedo PNG, normal PNG or None) for unbound meshes."""
    if not os.path.exists(CLASS_TEXTURES):
        return {}, {}
    alb, nrm = {}, {}
    for cls, e in json.load(open(CLASS_TEXTURES)).items():
        fp = os.path.join(WORK, e['png'])
        if not os.path.exists(fp):
            continue
        alb[cls] = _capped(fp)
        np_ = e.get('normal')
        if np_:
            nfp = os.path.join(WORK, np_)
            if os.path.exists(nfp):
                nrm[cls] = _capped_normal(nfp)
    return alb, nrm


def load_textures():
    """mesh asset name -> (texture key, PNG bytes); empty if the cache is absent."""
    if not (os.path.exists(TEXTURE_MANIFEST) and os.path.exists(TEXTURE_BINDINGS)):
        return {}
    man = json.load(open(TEXTURE_MANIFEST))
    bind = json.load(open(TEXTURE_BINDINGS))
    blobs, out = {}, {}
    for mesh, b in bind.items():
        entry = man.get(b['base'])
        if not entry:
            continue
        if entry['w'] < MIN_TEXTURE_PX or entry['h'] < MIN_TEXTURE_PX:
            continue
        key = b['base'].rsplit('/', 1)[-1]
        if key not in blobs:
            fp = os.path.join(WORK, entry['png'])
            if not os.path.exists(fp):
                continue
            data = open(fp, 'rb').read()
            if max(entry['w'], entry['h']) > MAX_TEXTURE_PX:
                im = Image.open(io.BytesIO(data))
                im.thumbnail((MAX_TEXTURE_PX, MAX_TEXTURE_PX))
                buf = io.BytesIO()
                im.save(buf, format='PNG', optimize=True)
                data = buf.getvalue()
            blobs[key] = data
        out[mesh] = (key, blobs[key])
    return out


def classify(material: str):
    s = material.lower()
    for name, rx, rgb, collide in _CLASS_RE:
        if rx.search(s):
            return name, rgb, collide
    return MISC


def png(rgb) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (SWATCH_PX, SWATCH_PX), rgb).save(buf, format='PNG')
    return buf.getvalue()


class Cas:
    def __init__(self):
        self.handles = {}

    def read(self, package, cas, offset, size) -> bytes:
        key = (package, cas)
        if key not in self.handles:
            self.handles[key] = open(os.path.join(LAYOUT, package, f'cas_{cas:02d}.cas'), 'rb')
        f = self.handles[key]
        f.seek(offset)
        return f.read(size)


def load_instances(crop=CROP):
    """[(mesh, R rows 3x3, t)] inside the island crop, in Frostbite space."""
    out, vista = [], []
    for mesh, T, src in json.load(open(os.path.join(WORK, 'instances.json'))):
        if not mesh:
            continue
        if not VISTA_SRC.search(src):
            out.append((mesh, np.array(T[:3]), np.array(T[3]), 'static'))
        elif not VISTA_BACKDROP.search(mesh.rsplit('/', 1)[-1]):
            vista.append((mesh, np.array(T[:3]), np.array(T[3]), 'static'))
    if vista:
        # keep a vista instance only where no ordinary placement of the same mesh sits within 1 m
        seen = collections.defaultdict(list)
        for m_, _r, t_, _s in out:
            seen[m_].append(t_)
        kept = [v for v in vista
                if not any(np.linalg.norm(v[2] - t_) < 1.0 for t_ in seen.get(v[0], ()))]
        x0_, z0_, x1_, z1_ = crop
        inside = [v for v in kept if x0_ < v[2][0] < x1_ and z0_ < v[2][2] < z1_]
        if inside:
            print(f"vista sublevel: kept {len(inside)} instance(s) inside the crop "
                  f"({dict(collections.Counter(v[0].rsplit('/', 1)[-1] for v in inside).most_common(4))})")
        out += kept
    bm = json.load(open(os.path.join(WORK, 'blueprint_meshes.json')))['blueprint_meshes']
    for kind, bp, M, _src, excluded in json.load(open(os.path.join(WORK, 'placements.json'))):
        # Object/Spatial references point at ObjectBlueprints holding one static model at
        # identity. Prefab and effect references hold nested transforms; not handled yet.
        if excluded or kind not in ('ObjectReferenceObjectData', 'SpatialReferenceObjectData'):
            continue
        for mesh in dict.fromkeys(bm.get(bp, [])):
            out.append((mesh, np.array(M[:3]), np.array(M[3]), 'blueprint'))
    for mesh, M in json.load(open(os.path.join(WORK, 'ecs_mesh_instances.json'))):
        if mesh:
            M = np.array(M)
            out.append((mesh, M[:3, :3], M[3, :3], 'ecs'))
    # Objects skate. bakes into a sublevel's staticmodelgroup.physics that none of the sources
    # above reach (nested prefabs and the like), placed from skate.'s own baked transforms by
    # tools/recover_placements.py.
    pp = os.path.join(WORK, 'physics_placements.json')
    if os.path.exists(pp):
        for mesh, M, t, _sub in json.load(open(pp)):
            out.append((mesh, np.array(M, float), np.array(t, float), 'physics'))
    x0, z0, x1, z1 = crop
    out = [i for i in out if x0 < i[2][0] < x1 and z0 < i[2][2] < z1]
    # Drop what sits far below the island. Three traffic cones are placed at y=0 and the BAM
    # terrain shells at y -476..-79, against an island ground band of roughly 190-300 m. A few
    # hundred such triangles stretched a 200 m crop to a 256 m vertical span, skewed the
    # percentile recentre (ground came out at y=50 instead of ~1) and inflated the CSPA tile
    # Y bounds that the packer writes.
    if out:
        sc = np.array([np.linalg.norm(np.asarray(i[1], float), axis=1) for i in out])
        aniso = sc.max(axis=1) / np.maximum(sc.min(axis=1), 1e-6)
        bad = aniso > MAX_ANISOTROPY
        if bad.any():
            names = collections.Counter(i[0].rsplit('/', 1)[-1] for i, b in zip(out, bad) if b)
            print(f"dropped {int(bad.sum())} instance(s) stretched over {MAX_ANISOTROPY:.0f}x "
                  f"(utility planes, not art): {dict(names.most_common(3))}")
            out = [i for i, b in zip(out, bad) if not b]
    if out:
        ys = np.array([i[2][1] for i in out])
        floor = float(np.percentile(ys, 5)) - 30.0
        keep = [i for i in out if i[2][1] >= floor]
        if len(keep) != len(out):
            print(f"dropped {len(out) - len(keep)} instance(s) below y={floor:.1f} "
                  f"(island ground band starts at p5 {float(np.percentile(ys, 5)):.1f})")
        out = keep
    return out


def vertex_normals(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    fn = np.cross(b - a, c - a)
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    return (N / ln).astype(np.float32)


def rdp(pts, tol):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    d = np.linalg.norm(pts - a, axis=1) if L < 1e-9 else np.linalg.norm(np.cross(pts - a, ab), axis=1) / L
    i = int(np.argmax(d))
    if d[i] > tol:
        return np.vstack([rdp(pts[:i + 1], tol)[:-1], rdp(pts[i:], tol)])
    return np.array([a, b])


def length(line):
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()) if len(line) > 1 else 0.0


def top_line(P, step=0.3):
    """Fallback rail: the highest point per slice along the rail's main horizontal axis."""
    xz = P[:, [0, 2]]
    c = xz.mean(0)
    _, v = np.linalg.eigh(np.cov((xz - c).T))
    s = (xz - c) @ v[:, -1]
    b = np.floor((s - s.min()) / step).astype(int)
    pts = []
    for k in range(b.max() + 1):
        idx = np.nonzero(b == k)[0]
        if len(idx):
            pts.append(P[idx[np.argmax(P[idx, 1])]])
    return rdp(np.array(pts), 0.03) if len(pts) >= 2 else None


# Every convex right-angle edge on man-made geometry is grindable -- ledge lips, box tops,
# curbs, benches, planters, stair edges -- not only skate.'s authored _Grindable rails
# (155 of them). Natural terrain is excluded: rock and ground creases are noise, not rails.
GRIND_CLASSES = {'Concrete', 'Asphalt', 'Brick', 'Tile', 'Wood', 'Metal', 'Rust',
                 'Paint', 'Plastic', 'Misc', 'Rail'}
GRIND_MIN_ANGLE, GRIND_MAX_ANGLE, GRIND_MIN_LEN = 60.0, 120.0, 1.0
GRIND_MAX_TOTAL = 8000


# skate.'s plot shells are painted by a streamed VIRTUAL texture: materials vtar_SGN_Atlas_01..04
# and m_proxy_atlas. `atlas` sent all five into Sign, so 214,584 m2 of floors and walls - the
# second-largest surface on the island - wore a sign texture at 0.056 tiles/m, nine times too
# sparse, because their authored UVs address the atlas, not a 2 m tiling material. They are
# 60% floor, 34% wall, 6% ceiling, so a top-down projection would streak a third of them.
# Materials whose UVs index a sheet rather than tile a surface. A section here is only
# world-projected if we could NOT find the sheet: with the sheet in hand its own UVs are
# right and reprojecting them would be the error. The terrain virtual-texture presets used
# to be listed too; they no longer need it, because tools/bind_terrain_presets.py resolves
# their surface and their authored UVs already measure 0.49 tiles/m.
ATLAS_RX = re.compile(r'vtar_.*atlas|_proxy_atlas')


def lightmap_swatch(level: float) -> bytes:
    """A PNG for the lightmap channel at `level` grey, with a gradient so it survives.

    See LIGHTMAP_GREY: a flat swatch reads as one block value, trips the engine's
    near-uniform test, and is replaced by the 1x1 white fallback.
    """
    v = int(round(max(0.0, min(1.0, level)) * 255))
    ramp = np.linspace(-LIGHTMAP_DITHER, LIGHTMAP_DITHER, SWATCH_PX, dtype=np.float32)
    field = ramp[:, None] + ramp[None, :] * 0.5
    px = np.clip(v + field, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.repeat(px[:, :, None], 3, axis=2), 'RGB').save(buf, format='PNG')
    return buf.getvalue()


def area_samples(a, b, c, spacing, cap, rng):
    """Area-uniform points on triangles (a, b, c): (points, m2 each, triangle index)."""
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    ns = np.clip(np.ceil(ar / spacing).astype(np.int64), 1, cap)
    rep = np.repeat(np.arange(len(ar)), ns)
    u, v = rng.random(len(rep)), rng.random(len(rep))
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    q = a[rep] + (b[rep] - a[rep]) * u[:, None] + (c[rep] - a[rep]) * v[:, None]
    return q, (ar / ns)[rep], rep


RAMP_STATS = {}


def ramp_seams(parts, lo=0.015, hi=0.08, max_deg=2.5, seg=0.25, min_run=1.0):
    """Collision-only ramps over every small drop between rideable surfaces.

    `tools/seam_scan.py` found 24 km of 1.5-10 cm steps where one up-facing collision surface
    ends above another (chalet patio tiles 8 cm proud of the ground, rock tops, the pier's low
    deck tile, sheets on concrete): each is a lip Skate 3's wheels stop on (the user, 2026-09-23:
    "i want everything to not have any seams that could catch you and be smooth"). Moving the
    pieces cannot fix a tile whose corners rest elsewhere, so every open edge of the up-facing
    collision is cut into `seg` m pieces; where the surface just outside is `lo`-`hi` m LOWER, a
    wedge runs from the edge down to that surface at <= `max_deg` degrees. Returns [(W, T)].
    """
    import seam_scan as ss
    A, B, C, R_ = [], [], [], []
    for tier, W, T in parts:
        if not len(T):
            continue
        W = np.asarray(W, np.float64)
        a, b, c = W[T[:, 0]], W[T[:, 1]], W[T[:, 2]]
        n = np.cross(b - a, c - a)
        up = n[:, 1] > 0.7 * np.linalg.norm(n, axis=1)
        A.append(a[up]), B.append(b[up]), C.append(c[up])
        # rock ground (tier 1): rock-to-rock edges are tessellation, 60% of all seam segments and the
        # collision budget of whole 3x3 windows - a rock edge is ramped only where it drops onto a
        # NON-rock surface (dirt mounds and banks onto concrete, where the user caught)
        R_.append(np.full(int(up.sum()), (not isinstance(tier, bool)) and tier == 1))
    if not A:
        return []
    A, B, C, R_ = np.concatenate(A), np.concatenate(B), np.concatenate(C), np.concatenate(R_)
    P0, P1, P2, tri = ss.open_edges(A, B, C, with_tri=True)
    L = np.linalg.norm((P1 - P0)[:, [0, 2]], axis=1)
    k = L > 0.05
    P0, P1, P2, L, RK = P0[k], P1[k], P2[k], L[k], R_[tri][k]
    ns = np.maximum(1, np.round(L / seg).astype(np.int64))
    rep = np.repeat(np.arange(len(L)), ns)
    j = np.concatenate([np.arange(n) for n in ns.tolist()])
    f0, f1 = j / ns[rep], (j + 1) / ns[rep]
    Q0 = P0[rep] + (P1 - P0)[rep] * f0[:, None]
    Q1 = P0[rep] + (P1 - P0)[rep] * f1[:, None]
    d = (P1 - P0)[rep][:, [0, 2]]
    out = np.stack([-d[:, 1], d[:, 0]], 1) / np.linalg.norm(d, axis=1)[:, None]
    inside = P2[rep][:, [0, 2]] - (Q0[:, [0, 2]] + Q1[:, [0, 2]]) / 2
    flip = (out * inside).sum(1) > 0
    out[flip] = -out[flip]
    H = ss.Heights(A, B, C)
    H_nr = ss.Heights(A[~R_], B[~R_], C[~R_]) if (~R_).any() else H
    rks = RK[rep]

    def hq(p, ref, window):
        y = H.query(p, ref, window=window)
        if rks.any():
            y[rks] = H_nr.query(p[rks], ref[rks], window=window if np.ndim(window) == 0 else window[rks])
        return y

    def lower(Q, dist):
        p = Q.copy()
        p[:, 0] += out[:, 0] * dist
        p[:, 2] += out[:, 1] * dist
        return p, hq(p, Q[:, 1] - (lo + hi) / 2, (hi - lo) / 2 + 0.005)

    M = (Q0 + Q1) / 2
    inward = M.copy()
    inward[:, 0] -= out[:, 0] * 0.02
    inward[:, 2] -= out[:, 1] * 0.02
    vis = ~H.above(inward, M[:, 1])        # an edge under another surface meets no wheel
    eid = rep * 1_000_000 + j               # edge id, then position along it
    Q0, Q1, out, M, eid, rks = Q0[vis], Q1[vis], out[vis], M[vis], eid[vis], rks[vis]
    # probe the segment at its ends and middle: the surface below is often patchy along an edge
    drops = []
    for f_ in (0.1, 0.5, 0.9):
        Pm = Q0 + (Q1 - Q0) * f_
        _, ym = lower(Pm, 0.04)
        dm = Pm[:, 1] - ym
        drops.append(np.where(np.isfinite(dm) & (dm >= lo) & (dm <= hi), dm, np.nan))
    drops = np.stack(drops, 1)
    ok = np.isfinite(drops).any(axis=1)
    drop = np.where(ok, np.nanmax(np.where(np.isfinite(drops), drops, -1), axis=1), np.nan)
    # long and gentle: the wheel rolls over a join instead of hitting a kink (user, 2026-09-23:
    # "the seams on all these connections ... need to be super long and super smooth")
    run = np.maximum(min_run, drop / np.tan(np.radians(max_deg)))
    Q0, Q1, out, run, drop, eid, rks = Q0[ok], Q1[ok], out[ok], run[ok], drop[ok], eid[ok], rks[ok]
    # The foot sits on the lower surface where one is found under the ramp's end and lower than
    # the drop measured at the edge; otherwise 1 cm under the edge-minus-drop height. Ground that
    # rises away from the edge (a V between two pieces, an uphill slope) buries the foot, which
    # leaves no lip; ground that falls away is followed.
    R0, R1 = Q0.copy(), Q1.copy()
    for R_, Q_ in ((R0, Q0), (R1, Q1)):
        R_[:, 0] += out[:, 0] * run
        R_[:, 2] += out[:, 1] * run
        y = hq(R_, Q_[:, 1] - drop, 0.03 + 0.4 * run)
        base = Q_[:, 1] - drop
        R_[:, 1] = np.where(np.isfinite(y), np.minimum(y + 0.002, base), base - 0.01)
    good = np.ones(len(Q0), bool)
    RAMP_STATS.update(candidates=int(len(ok)), drops=int(ok.sum()), placed=int(good.sum()))
    Q0, Q1, R0, R1, eid = Q0[good], Q1[good], R0[good], R1[good], eid[good]
    if not len(Q0):
        return []
    # One quad per run of consecutive segments along an edge while a straight foot line stays
    # on or under the ground (a floating foot would be a new lip): 4x fewer triangles, which the
    # 3x3 collision planner otherwise takes out of the props and ride pieces around them.
    order = np.argsort(eid, kind='stable')
    Q0, Q1, R0, R1, eid = Q0[order], Q1[order], R0[order], R1[order], eid[order]
    gq0, gq1, gr0, gr1 = [], [], [], []
    i = 0
    N = len(eid)
    while i < N:
        k = i
        while k + 1 < N and eid[k + 1] == eid[k] + 1:
            # try to extend the run [i, k+1]
            ts = np.linspace(0.0, 1.0, k + 3 - i)[1:-1]
            feet = R1[i:k + 1, 1]                       # interior joins between segments
            line = R0[i, 1] + (R1[k + 1, 1] - R0[i, 1]) * ts
            if (line - feet > 0.005).any() or k + 1 - i >= 16:
                break
            k += 1
        gq0.append(Q0[i]), gq1.append(Q1[k]), gr0.append(R0[i]), gr1.append(R1[k])
        i = k + 1
    Q0, Q1, R0, R1 = (np.array(v) for v in (gq0, gq1, gr0, gr1))
    RAMP_STATS["quads"] = len(Q0)
    n = len(Q0)
    W = np.concatenate([Q0, Q1, R1, R0])
    i = np.arange(n)
    T = np.concatenate([np.stack([i, i + n, i + 2 * n], 1), np.stack([i, i + 2 * n, i + 3 * n], 1)])
    nrm = np.cross(W[T[:, 1]] - W[T[:, 0]], W[T[:, 2]] - W[T[:, 0]])
    T[nrm[:, 1] < 0] = T[nrm[:, 1] < 0][:, [0, 2, 1]]
    return [(W, T)]


def flush_splines(lines, parts, step=0.25, reach=0.12, flush=0.06, frac=0.6):
    """Grind lines that are NOT flush seams: drop a line when over `frac` of its samples have
    up-facing collision within `flush` m of its own height `reach` m out on both sides."""
    import seam_scan as ss
    A, B, C = [], [], []
    for _, W, T in parts:
        if not len(T):
            continue
        W = np.asarray(W, np.float64)
        a, b, c = W[T[:, 0]], W[T[:, 1]], W[T[:, 2]]
        n = np.cross(b - a, c - a)
        up = n[:, 1] > 0.7 * np.linalg.norm(n, axis=1)
        A.append(a[up]), B.append(b[up]), C.append(c[up])
    if not A:
        return lines
    H = ss.Heights(np.concatenate(A), np.concatenate(B), np.concatenate(C))
    S, N, owner = [], [], []
    for li, l in enumerate(lines):
        l = np.asarray(l, np.float64)
        for a_, b_ in zip(l[:-1], l[1:]):
            d = (b_ - a_)[[0, 2]]
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            k = max(1, int(L / step))
            t = (np.arange(k) + 0.5) / k
            S.append(a_ + (b_ - a_) * t[:, None])
            N.append(np.repeat(np.array([[-d[1], d[0]]]) / L, k, axis=0))
            owner.append(np.full(k, li))
    if not S:
        return lines
    S, N, owner = np.concatenate(S), np.concatenate(N), np.concatenate(owner)
    both = np.ones(len(S), bool)
    # buried: a rideable surface 3-25 cm above the line AND on both sides of it - an edge inside a
    # deck's slab (the chalet patio had 20 m lines 10 cm under its boards; the board snapped down
    # onto them). Only above the line itself is not enough: a curved rail (the rainbow rail) is a
    # chain of overlapping slices, each slice's top edge a few cm under the next, with open air
    # beside it - those lines are the rail (they stopped grinding, user 2026-09-23).
    buried = H.above(S, S[:, 1], eps=0.03, reach=0.25)
    for sgn in (1.0, -1.0):
        q = S.copy()
        q[:, 0] += sgn * N[:, 0] * reach
        q[:, 2] += sgn * N[:, 1] * reach
        both &= np.isfinite(H.query(q, S[:, 1], window=flush))
        buried &= H.above(q, S[:, 1], eps=0.03, reach=0.25)
    bad = both | buried
    fl = np.bincount(owner, weights=bad.astype(float), minlength=len(lines))
    tot = np.bincount(owner, minlength=len(lines))
    return [l for i, l in enumerate(lines) if tot[i] == 0 or fl[i] / tot[i] <= frac]


def support_index(coll, cell=2.0):
    """Up-facing collision triangles of `coll` = [(tier, W, T)] in a `cell` m XZ hash."""
    A, B, C = [], [], []
    for _, W, T in coll:
        if not len(T):
            continue
        a, b, c = W[T[:, 0]], W[T[:, 1]], W[T[:, 2]]
        n = np.cross(b - a, c - a)
        up = n[:, 1] > 0.5 * np.linalg.norm(n, axis=1)
        A.append(a[up]), B.append(b[up]), C.append(c[up])
    A, B, C = (np.concatenate(x) if x else np.zeros((0, 3)) for x in (A, B, C))
    lo = np.floor(np.minimum(np.minimum(A, B), C)[:, [0, 2]] / cell).astype(np.int64)
    hi = np.floor(np.maximum(np.maximum(A, B), C)[:, [0, 2]] / cell).astype(np.int64)
    grid = collections.defaultdict(list)
    for i in np.nonzero(((hi - lo + 1).prod(axis=1)) <= 4096)[0]:
        for x in range(lo[i, 0], hi[i, 0] + 1):
            for z in range(lo[i, 1], hi[i, 1] + 1):
                grid[(x, z)].append(i)
    big = np.nonzero(((hi - lo + 1).prod(axis=1)) > 4096)[0]   # a lake sheet: checked always
    return A, B, C, grid, big, cell


def flat_unsupported(pieces, index):
    """Fraction of a flat sheet's top with no up-facing collision within FLAT_SUPPORT m under it.
    `pieces` = [(W, T)] of the sheet in world space; `index` from support_index()."""
    A0, B0, C0, grid, big, cell = index
    rng = np.random.default_rng(0)
    tops = []
    for W, T in pieces:
        a, b, c = W[T[:, 0]], W[T[:, 1]], W[T[:, 2]]
        n = np.cross(b - a, c - a)
        up = n[:, 1] > 0.7 * np.linalg.norm(n, axis=1)
        if up.any():
            tops.append(area_samples(a[up], b[up], c[up], 0.02, 4000, rng)[0])
    if not tops:
        return 0.0
    q = np.concatenate(tops)
    lo = np.floor(q[:, [0, 2]].min(0) / cell).astype(np.int64)
    hi = np.floor(q[:, [0, 2]].max(0) / cell).astype(np.int64)
    cand = set(big.tolist())
    for x in range(lo[0], hi[0] + 1):
        for z in range(lo[1], hi[1] + 1):
            cand.update(grid.get((x, z), ()))
    if not cand:
        return 1.0
    cand = np.fromiter(cand, np.int64)
    A, B, C = A0[cand], B0[cand], C0[cand]
    v0, v1 = (B - A)[:, [0, 2]], (C - A)[:, [0, 2]]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    good = np.abs(den) > 1e-12
    A, B, C, v0, v1, den = A[good], B[good], C[good], v0[good], v1[good], den[good]
    ok = np.zeros(len(q), bool)
    for i in range(0, len(q), 256):
        p = q[i:i + 256]
        v2 = p[:, None, [0, 2]] - A[None, :, [0, 2]]
        w1 = (v2[..., 0] * v1[None, :, 1] - v1[None, :, 0] * v2[..., 1]) / den[None]
        w2 = (v0[None, :, 0] * v2[..., 1] - v2[..., 0] * v0[None, :, 1]) / den[None]
        ins = (w1 >= -1e-4) & (w2 >= -1e-4) & (w1 + w2 <= 1 + 1e-4)
        dy = p[:, None, 1] - (A[None, :, 1] + w1 * (B - A)[None, :, 1] + w2 * (C - A)[None, :, 1])
        ok[i:i + 256] = (ins & (dy >= -0.02) & (dy <= FLAT_SUPPORT)).any(axis=1)
    return float(1.0 - ok.mean())


def triplanar(W, T, scale=2.0):
    """Re-project each face along its dominant axis at `scale` metres per tile.

    Vertices are duplicated only where the projection axis changes, never per face, so a
    mostly-flat shell costs almost nothing extra - the worst tile is already 4.7 MB of a 5 MB
    stream ceiling and un-indexing everything would push it over.
    """
    n = np.cross(W[T[:, 1]] - W[T[:, 0]], W[T[:, 2]] - W[T[:, 0]])
    ax = np.abs(n).argmax(axis=1)
    Ws, UVs, Ts, off = [], [], [], 0
    for a, (ui, vi) in ((0, (2, 1)), (1, (0, 2)), (2, (0, 1))):
        sel = ax == a
        if not sel.any():
            continue
        used, inv = np.unique(T[sel].reshape(-1), return_inverse=True)
        Wa = W[used]
        Ws.append(Wa)
        UVs.append(np.stack([Wa[:, ui], Wa[:, vi]], axis=1) / scale)
        Ts.append(inv.reshape(-1, 3) + off)
        off += len(Wa)
    return np.concatenate(Ws), np.concatenate(UVs).astype(np.float32), np.concatenate(Ts)


def rail_polylines(P, T):
    """Grind lines for one authored _Grindable section, in its own (world) space."""
    P = P.astype(np.float64)
    xz = P[:, [0, 2]]
    extent = float(np.linalg.norm(xz.max(0) - xz.min(0)))
    try:
        lines = d2s3_splines.grind_edges(P, T.astype(np.int64), min_length=1.0)
    except Exception:  # noqa: BLE001
        lines = []
    lines = [np.asarray(l, dtype=np.float64) for l in lines if len(l) >= 2]
    # A round rail has several shallow creases along its top; keep the highest of any lines
    # running within 20 cm of each other, or the game snaps between parallel stubs.
    lines.sort(key=lambda l: -float(l[:, 1].mean()))
    kept = []
    for l in lines:
        dup = any((np.linalg.norm(l[:, None, [0, 2]] - k[None, :, [0, 2]], axis=2).min(1) < 0.2).mean() > 0.7
                  for k in kept)
        if not dup:
            kept.append(l)
    if sum(length(l) for l in kept) < 0.5 * extent:
        t = top_line(P)
        if t is not None and length(t) >= 1.0:
            kept = [t]
    return kept


def cluster_simplify(P, UV, T, target):
    """Vertex-clustering decimation to at most `target` triangles.

    The same idea as d2s3.decimate: snap to a grid, weld per cell, drop what collapses. The
    cell size is binary-searched. Only ever applied to non-rideable classes.
    """
    if len(T) <= target:
        return P, UV, T
    lo = P.min(0)
    ext = float(np.max(P.max(0) - lo)) or 1.0

    def run(cell):
        q = np.floor((P - lo) / cell).astype(np.int64)
        key = (q[:, 0] * 8193 + q[:, 1]) * 8193 + q[:, 2]
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        T2 = inv.reshape(-1)[T]
        T2 = T2[(T2[:, 0] != T2[:, 1]) & (T2[:, 1] != T2[:, 2]) & (T2[:, 0] != T2[:, 2])]
        if len(T2):
            _, keep = np.unique(np.sort(T2, axis=1), axis=0, return_index=True)
            T2 = T2[np.sort(keep)]
        return first, inv.reshape(-1), T2

    lo_c, hi_c, best = ext / 8000, ext, None
    for _ in range(16):
        mid = (lo_c * hi_c) ** 0.5
        got = run(mid)
        if len(got[2]) > target:
            lo_c = mid
        else:
            hi_c, best = mid, got
    first, inv, T2 = best if best is not None else run(hi_c)
    n = len(first)
    acc = np.zeros((n, 3))
    np.add.at(acc, inv, P)
    P2 = acc / np.maximum(np.bincount(inv, minlength=n), 1)[:, None]
    used = np.unique(T2)
    remap = np.full(n, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return P2[used], (UV[first][used] if UV is not None else None), remap[T2]


UV_SAMPLE_CAP = 20000


def uv_world_ratio(W, UV, T):
    """(world area, UV area) over the same sampled triangles, wherever it is asked for.

    Both the measurement that chooses a material's UV correction and the report that checks
    the result must sample identically: strided subsets of a mesh are not unbiased, because
    triangle size correlates with index (a LOD section is ordered by strip, and the strips
    are not the same size). Measuring with stride 2000 and checking with stride 4000 left
    CliffRock_02 reading 0.509 tiles/m where the exported geometry was 0.67.
    """
    if UV is None or not len(T):
        return 0.0, 0.0
    J = T if len(T) <= UV_SAMPLE_CAP else T[::max(1, len(T) // UV_SAMPLE_CAP)]
    q = len(T) / len(J)
    a, b, c = W[J[:, 0]], W[J[:, 1]], W[J[:, 2]]
    wa = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    d1, d2 = UV[J[:, 1]] - UV[J[:, 0]], UV[J[:, 2]] - UV[J[:, 0]]
    ua = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    ok = wa > 1e-9
    return float(wa[ok].sum()) * q, float(ua[ok].sum()) * q


def raw_density(placed):
    """(material, class) -> tiles/m of the game's own UV0, on the placed geometry.

    Authored UVs are mostly right - most materials come out at 0.49 against the 0.50 a
    2x2 m library material wants - but proxy and baked-atlas art carries UVs packed into
    a sub-rectangle, which tiles a single texture across tens of metres. Measuring is the
    only way to tell the two apart: the material name does not say.

    Keyed and filtered exactly as the correction is applied - per material AND class, with
    baked-atlas sections left out - because a measurement taken on a different basis than
    it is applied on lands the result somewhere else. Measuring across both left Paint at
    0.88 tiles/m after a correction that was supposed to put it on 0.50.
    """
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for _mesh, secs in placed:
        for cls, mat, W, UV, T, dk in secs:
            if UV is None or not len(T):
                continue
            if (KEY_TEX.get(str(dk) if dk else '') or MATERIAL_TEX.get(mat)) in ATLAS_BASES:
                continue
            wa, ua = uv_world_ratio(W, UV, T)
            r = acc[(mat, cls[0])]
            r[0] += wa
            r[1] += ua
    out = {}
    for key, (wa, ua) in acc.items():
        if wa >= UV_MEASURE_MIN_M2 and ua > 0:
            out[key] = (ua / wa) ** 0.5
    return out


def uv_correction(measured, mat, cname):
    """The factor that puts this material's texture back on a 2 m repeat."""
    d = measured.get((mat, cname))
    if d and d > 0:
        return float(np.clip(UV_TARGET / d, UV_FIX_LO, UV_FIX_HI))
    return UV_SCALE.get(cname)


UV_ACC = collections.defaultdict(lambda: [0.0, 0.0, 0])   # (mat, class, atlas, bound) -> [m2, uv, tris]


def uv_account(mat, cname, atlas, bound, baked, W, UV, T):
    """Record world area vs UV area exactly as written, so a stretched texture is a number.

    sqrt(uv_area / world_area) is tiles per metre; skate.'s materials are authored for 2x2 m,
    so the target is ~0.50. Measured here rather than on the GLB because this is the last point
    where the material name is still known.
    """
    if UV is None or not len(T):
        return
    wa, ua = uv_world_ratio(W, UV, T)
    r = UV_ACC[(mat, cname, bool(atlas), bool(bound), bool(baked))]
    r[0] += wa
    r[1] += ua
    r[2] += len(T)


UV_RAW_SEEN = {}
UV_FACTOR = {}
UV_TRIM = {}


def uv_report(path):
    """Write the density record beside the GLBs and into work/ (see hot_textures)."""
    rows = []
    for (mat, cname, atlas, bound, baked), (wa, ua, n) in UV_ACC.items():
        rows.append({'material': mat, 'class': cname, 'atlas': atlas, 'bound': bound,
                     'baked': baked, 'm2': wa, 'uv_area': ua, 'tris': n,
                     'measured': UV_RAW_SEEN.get((mat, cname)),
                     'factor': UV_FACTOR.get((mat, cname)),
                     'trim': UV_TRIM.get((mat, cname)),
                     'tiles_per_m': (ua / wa) ** 0.5 if wa > 0 else 0.0})
    rows.sort(key=lambda r: -r['m2'])
    for dst in (path, os.path.join(WORK, 'uv_density.json')):
        with open(dst, 'w') as fh:
            json.dump(rows, fh, indent=1)
    # a baked atlas is judged by its sheet, not by tiles/m
    bad = [r for r in rows if r['m2'] > 500 and r['tiles_per_m'] < 0.25 and not r['baked']]
    print(f"UV density: {len(rows)} material/class pairs; "
          f"{len(bad)} stretched (>500 m2 and <0.25 tiles/m) holding "
          f"{sum(r['m2'] for r in bad):,.0f} of {sum(r['m2'] for r in rows):,.0f} m2")
    off = [r for r in rows if r['m2'] > 500 and not r['baked']
           and not 0.40 <= r['tiles_per_m'] <= 0.62]
    off.sort(key=lambda r: -r['m2'])
    if off:
        print(f"  {len(off)} pair(s) over 500 m2 landed off the 0.40-0.62 band:")
        for r in off[:10]:
            print(f"   measured {r['measured'] if r['measured'] is None else round(r['measured'], 3)}"
                  f"  x{r['factor'] if r['factor'] is None else round(r['factor'], 2)}"
                  f"  -> {r['tiles_per_m']:.3f}  {r['m2']:>8,.0f} m2  {r['material'][:44]}")
    for r in bad[:12]:
        print(f"   {r['tiles_per_m']:6.3f} t/m {r['m2']:>10,.0f} m2 "
              f"{'atlas' if r['atlas'] else '     '} {r['class']:<9} {r['material'][:58]}")


# skate. flags the collision part a board grinds on: every authored localrail carries one part
# whose packed material has bit 19 set (0x2380E60 / 0x2380BA0) beside its body (0x2300...), and
# a few ledges and props carry it too. Measured 2026-09-22: most localrails shipped with NO
# spline (their render materials are not `_Grindable`, and the edge tracer skipped them).
GRIND_PART_FLAG = 0x80000
# ...but bit 19 alone is not "grind": grass shells and generated roads carry it in other families
# (0x2F8.., 0x1A8.., 0x388..) and yielded 6,000 bogus ridges. The grind family is 0x238xxxx:
# every localrail, the skatepark rails, three plot pieces with built-in rails, gondola cables.
GRIND_PART_MASK, GRIND_PART_FAMILY = 0xFF80000, 0x2380000


def is_grind_part(material):
    return (int(material) & GRIND_PART_MASK) == GRIND_PART_FAMILY
# Thin props whose top is a grind in skate. (railings, fences, barriers): narrow, long, knee- to
# chest-high. Measured: patio railings 2/32, jersey barriers 1/51 had a spline along their top.
THIN_MAX_WIDTH, THIN_MIN_LENGTH, THIN_HEIGHT = 0.7, 1.2, (0.2, 3.5)


def edge_points(W, T, step=0.1):
    """Vertices plus points every `step` m along every triangle edge (long rails have vertices only at their ends)."""
    if T is None or not len(T):
        return W
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    a, b = W[e[:, 0]], W[e[:, 1]]
    n = np.clip(np.ceil(np.linalg.norm(b - a, axis=1) / step).astype(int), 1, 2000)
    rep_ = np.repeat(np.arange(len(e)), n)
    f = (np.arange(len(rep_)) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    return np.concatenate([W, a[rep_] + (b[rep_] - a[rep_]) * f[:, None]])


def ridge_polyline(W, step=None):
    """Top ridge of a long object as a list of polylines, following curves and bends.

    The points are gridded at `step` in XZ keeping each cell's highest point, then walked from
    one end to its nearest unvisited neighbour; a gap wider than 3 cells starts a new polyline
    (two separate rails in one part). Binning along one axis zigzagged on L-shaped rails.
    """
    if len(W) < 2:
        return []
    if step is None:
        # cell = the object's own width (PCA minor axis), so a wide top does not zigzag
        xz = W[:, [0, 2]] - W[:, [0, 2]].mean(axis=0)
        ev = np.linalg.eigh(np.cov(xz.T))[1]
        width = float(np.ptp(xz @ ev[:, 0]))
        step = float(np.clip(width * 1.1, 0.3, 0.9))
    key = np.floor(W[:, [0, 2]] / step).astype(np.int64)
    best = {}
    for i, k in enumerate(map(tuple, key.tolist())):
        if k not in best or W[i, 1] > W[best[k], 1]:
            best[k] = i
    P = W[list(best.values())]
    if len(P) < 2:
        return []
    from scipy.spatial import cKDTree
    tree = cKDTree(P[:, [0, 2]])
    left = np.ones(len(P), bool)
    out = []
    c = P[:, [0, 2]].mean(axis=0)
    while left.any():
        idx = np.nonzero(left)[0]
        cur = idx[np.argmax(np.linalg.norm(P[idx][:, [0, 2]] - c, axis=1))]
        line = [cur]
        left[cur] = False
        while True:
            d, j = tree.query(P[cur, [0, 2]], k=min(12, len(P)))
            nxt = next((jj for dd, jj in zip(np.atleast_1d(d), np.atleast_1d(j))
                        if left[jj] and dd <= 3.5 * step), None)
            if nxt is None:
                break
            left[nxt] = False
            line.append(nxt)
            cur = nxt
        L = P[line]
        if len(L) >= 2 and length(L) >= THIN_MIN_LENGTH * 0.5:
            out.append(smooth_rail(L))
    return out


def round_corners(L, min_deg=5.0, iters=2):
    """Chaikin corner cutting where a grind line bends more than `min_deg` degrees; the end points
    and straight runs are kept, so a kinked join rides as a curve instead of a snag."""
    for _ in range(iters):
        if len(L) < 3:
            return L
        d0 = L[1:-1] - L[:-2]
        d1 = L[2:] - L[1:-1]
        c = (d0 * d1).sum(1) / np.maximum(np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1), 1e-12)
        bent = c < np.cos(np.radians(min_deg))
        if not bent.any():
            return L
        out = [L[0]]
        for i in range(1, len(L) - 1):
            if bent[i - 1]:
                out.append(0.75 * L[i] + 0.25 * L[i - 1])
                out.append(0.75 * L[i] + 0.25 * L[i + 1])
            else:
                out.append(L[i])
        out.append(L[-1])
        L = np.array(out)
    return L


def smooth_rail(L):
    """A grind line a board can ride: straight runs become one straight segment (top height
    fitted linearly, so a stair rail slopes evenly), bent ones are smoothed (3-point average)
    and simplified. Cell-by-cell ridges wobbled across posts and bars, and wobble kicks the
    skater off a grind."""
    xz = L[:, [0, 2]]
    c = xz.mean(axis=0)
    if len(L) >= 3:
        u, sv, vt = np.linalg.svd(xz - c, full_matrices=False)
        ax = vt[0]
        t = (xz - c) @ ax
        off = (xz - c) @ vt[1]
        if np.abs(off).max() < 0.12:
            # Robust height line: posts and finials stick up above a handrail at its ends and
            # landings, and a plain fit (or the max-height ridge) climbed over them - the kinks the
            # skater was thrown off. Fit to the rail itself: the line most points lie on (RANSAC).
            best = None
            n_ = len(t)
            for i in range(n_):
                for j in range(i + 1, n_):
                    if abs(t[j] - t[i]) < 0.3:
                        continue
                    k_ = (L[j, 1] - L[i, 1]) / (t[j] - t[i])
                    b_ = L[i, 1] - k_ * t[i]
                    inl = np.abs(L[:, 1] - (k_ * t + b_)) < 0.05
                    if best is None or inl.sum() > best[0]:
                        best = (int(inl.sum()), inl)
            if best is not None and best[0] >= max(2, 0.6 * n_):
                inl = best[1]
                A = np.vstack([t[inl], np.ones(inl.sum())]).T
                k, b = np.linalg.lstsq(A, L[inl, 1], rcond=None)[0]
                # the points off the line may only be posts/finials ABOVE the rail, and few: an
                # arched rail (Summer Camp's rainbow rails, straight in plan) has its off-line
                # points BELOW, and flattening it put the grind line up to a metre off the rail
                res = L[:, 1] - (k * t + b)
                off_ = np.abs(res) >= 0.05
                if (res[off_] > 0).all() and off_.mean() <= 0.25:
                    res = np.zeros(1)
            else:
                A = np.vstack([t, np.ones_like(t)]).T
                k, b = np.linalg.lstsq(A, L[:, 1], rcond=None)[0]
                res = L[:, 1] - (k * t + b)
            if np.abs(res).max() < 0.10:
                t0, t1 = float(t.min()), float(t.max())
                p0 = np.array([c[0] + ax[0] * t0, k * t0 + b, c[1] + ax[1] * t0])
                p1 = np.array([c[0] + ax[0] * t1, k * t1 + b, c[1] + ax[1] * t1])
                return np.array([p0, p1])
        S = L.copy()
        S[1:-1] = (L[:-2] + L[1:-1] + L[2:]) / 3.0
        return rdp(S, 0.05)
    return L


def join_splines(lines, tol=0.08, max_deg=35.0):
    """Join grind lines whose ends meet (within `tol` m) and continue in the same direction.

    Edges traced per piece stop at every seam - a ledge made of three slabs was three rails, and
    Skate 3 drops you off a grind at the end of each one. Greedy: repeatedly glue the pair of
    free ends that are closest and bend least, until nothing qualifies.
    """
    from scipy.spatial import cKDTree
    L = [np.asarray(l, np.float64) for l in lines if len(l) >= 2]
    cos_max = np.cos(np.radians(max_deg))
    alive = [True] * len(L)
    changed = True
    joins = 0
    while changed:
        changed = False
        ends, own = [], []
        for i, l in enumerate(L):
            if alive[i]:
                ends += [l[0], l[-1]]
                own += [(i, 0), (i, 1)]
        if len(ends) < 2:
            break
        tree = cKDTree(np.array(ends))
        used = set()
        for a, b in sorted(tree.query_pairs(tol), key=lambda ab: np.linalg.norm(ends[ab[0]] - ends[ab[1]])):
            (i, ei), (j, ej) = own[a], own[b]
            if i == j or i in used or j in used or not alive[i] or not alive[j]:
                continue
            li = L[i] if ei == 1 else L[i][::-1]          # li ends at the joint
            lj = L[j] if ej == 0 else L[j][::-1]          # lj starts at the joint
            di = li[-1] - li[-2]
            dj = lj[1] - lj[0]
            ni, nj = np.linalg.norm(di), np.linalg.norm(dj)
            if ni < 1e-9 or nj < 1e-9 or (di @ dj) / (ni * nj) < cos_max:
                continue
            L[i] = np.vstack([li[:-1], (li[-1] + lj[0]) / 2, lj[1:]])
            alive[j] = False
            used.update((i, j))
            joins += 1
            changed = True
    return [l for l, a in zip(L, alive) if a], joins


# Movable props (lamp posts, bike racks, trash cans) carry only a stub part in their .physics -
# a 1.3 m x 7 cm box at the origin for a 5 m lamp post - so they had no usable collision and the
# skater rode through them. Where skate.'s collision is this much smaller than the drawn mesh,
# the collision is the convex hull of the drawn mesh's lowest STUB_HULL_HEIGHT m instead (solid
# where a skater meets it, nothing under lamp arms or awnings).
STUB_RATIO, STUB_HULL_HEIGHT = 0.3, 2.2
# a hull of hanging wire is an invisible slab: cables, wires, ropes, string lights keep their stub
STUB_SKIP = re.compile(r'cable|wire|rope|stringlight|bunting|flag|crane|_arm_|sign_light',re.I)
# ...and only props that are solid obstacles to a skater. A row of ten tower-crane arm sign lights
# at ground level got hulls (2026-09-23) and became a wall across a fast line: skate.'s stub meant
# "no collision" for them.
STUB_ALLOW = re.compile(r'streetlight|lamppost|lamp_?post|floodlight|bikerack|bicycle|trashcan|trash_?bin|'
                        r'dumpster|bollard|hydrant|electricalmeter|meter_|trolley|mailbox|newspaper|'
                        r'barrierpedestrian|signpost|postgeneric|parkingmeter|phonebooth', re.I)


def component_hulls(P, T, max_parts=16):
    """One convex hull per connected piece of a drawn mesh (a picnic table's top, benches and
    legs), so the collision follows the shape instead of wrapping the gaps between the pieces.
    Returns (V, T) of all hulls together, or None."""
    if P is None or len(T) == 0:
        return None
    key = np.round(P / 0.01).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1)
    parent = np.arange(vid.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in vid[T]:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    root = np.array([find(v) for v in vid[T[:, 0]]])
    groups = collections.defaultdict(list)
    for ti, r in enumerate(root.tolist()):
        groups[r].append(ti)
    order = sorted(groups.values(), key=lambda g: -len(g))[:max_parts]
    Vs, Ts, off = [], [], 0
    for g in order:
        used = np.unique(T[g].reshape(-1))
        h = stub_hull(P[used])
        if h is None:
            continue
        Vs.append(h[0])
        Ts.append(h[1] + off)
        off += len(h[0])
    if not Vs:
        return None
    return np.concatenate(Vs), np.concatenate(Ts)


def stub_hull(P):
    """Convex hull (P, T) of the mesh's lowest STUB_HULL_HEIGHT m, or None."""
    from scipy.spatial import ConvexHull
    if len(P) < 4:
        return None
    Q = P[P[:, 1] <= P[:, 1].min() + STUB_HULL_HEIGHT]
    if len(Q) < 4:
        return None
    try:
        h = ConvexHull(Q)
    except Exception:  # noqa: BLE001 (degenerate, e.g. a flat sign)
        return None
    used, inv = np.unique(h.simplices.reshape(-1), return_inverse=True)
    T = inv.reshape(-1, 3)
    V = Q[used]
    # outward winding
    c = V.mean(axis=0)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    flip = ((V[T].mean(axis=1) - c) * n).sum(axis=1) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    return V, T


def tile_counts(W, T):
    c = W[T].mean(axis=1)
    return collections.Counter(zip((np.floor(c[:, 0] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int).tolist(),
                                   (np.floor(c[:, 2] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int).tolist()))


def tex_native(tk, normal=False):
    """Native edge of a blob key's full-resolution source (falls back to the preview blob)."""
    src = TEX_SRC.get(tk[4:])
    n = (src or {}).get('nnat' if normal else 'nat')
    if n:
        return int(n)
    blob = (TEX_NORMALS.get(tk[4:]) if normal else TEXTURE_BLOBS.get(tk))
    return max(Image.open(io.BytesIO(blob)).size) if blob else 64


def plan_global_textures(staged, args):
    """Which textures live ONCE in cPres_Global, and at what edge: {blob key: edge px}.

    cPres_Global is always resident and capped at 11 MB, and every image in it pays ~22.4 KB of
    fixed cost: the shipped Summer Camp pool was 262 images at a median 33 KB, 8 MB of mostly
    overhead, while the seven textures covering 40,000+ m2 each were held at 128 px. Here the
    pool is spent on purpose: textures on two or more tiles, largest covered area first, at
    `--hot-texture-px` if they cover HOT_TEXTURE_M2 or more, else `--global-edge-px`, until the
    estimate reaches `--global-pool-mb`. Everything else is copied per tile (see full_png) and sized
    by that tile's own budget.
    """
    area, tiles = collections.Counter(), collections.defaultdict(set)
    for e in staged:
        tk, rW, rT = e[2], e[6], e[8]
        if not tk or not tk.startswith(('tex:', 'dec:')) or not len(rT):
            continue
        a, b, c = rW[rT[:, 0]], rW[rT[:, 1]], rW[rT[:, 2]]
        area[tk] += float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
        cen = rW[rT].mean(axis=1)
        tiles[tk].update(zip(np.floor(cen[:, 0] / TILE_SIZE).astype(int).tolist(),
                             np.floor(cen[:, 2] / TILE_SIZE).astype(int).tolist()))
    pool = args.global_pool_mb * 1e6
    out, used = {}, 0.0
    for tk, a in area.most_common():
        if len(tiles[tk]) < 2:
            continue
        want = args.hot_texture_px if a >= HOT_TEXTURE_M2 else args.global_edge_px
        e = 64
        while e * 2 <= min(want, tex_native(tk)):
            e *= 2
        # normals only for the materials covering the most ground: the pool buys colour detail
        n = 1 if (tk.startswith('tex:') and tk[4:] in TEX_NORMALS and a >= HOT_TEXTURE_M2) else 0
        cost = (SHARED_IMAGE_FIXED + e * e * SHARED_BYTES_PER_PX) * (1 + n)
        if used + cost > pool:
            continue
        out[tk] = e
        used += cost
    hist = collections.Counter(out.values())
    shared = sum(1 for tk in tiles if len(tiles[tk]) >= 2)
    print(f"global textures: {len(out)} of {shared} shared texture(s) kept in cPres_Global "
          f"(~{used / 1e6:.2f} MB of {args.global_pool_mb:.1f}): "
          + ", ".join(f"{n} at {e} px" for e, n in sorted(hist.items(), reverse=True))
          + f"; covering {sum(area[t] for t in out):,.0f} of {sum(area.values()):,.0f} m2; the rest are per tile")
    return out


def fill_final_collision(blocks, args, rend_tris):
    """Draw the parts of the FINAL collision (after the 3x3 planner) that nothing drawn is near.

    The planner cuts rock and prop collision to fit Skate 3's 3x3 pool (Summer Camp: ten tiles,
    rock p99 up to 3.6 m), so a cut rock's collision bulges where its drawn mesh does not: 55,000
    m2 of Build A's collision had nothing drawn within 0.5 m, all of it in those ten tiles. Each
    such triangle is drawn with the material of the nearest drawn surface, triplanar at 2 m, so
    what you hit is what you see.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(11)
    lim = args.fill_hidden_collision
    labels, lab_of, pts, lab_pts = {}, [], [], []
    for bk, classes in blocks.items():
        for (cname, tk, is_proxy), parts in classes.items():
            if cname in (COLLISION_CLASS, DECAL_CLASS):
                continue
            lk = (cname, None if is_proxy else (tk.split('@')[0] if tk else tk))
            if lk not in labels:
                labels[lk] = len(lab_of)
                lab_of.append(lk)
            li = labels[lk]
            for W, _UV, T in parts:
                if not len(T):
                    continue
                a, b, c = W[T[:, 0]], W[T[:, 1]], W[T[:, 2]]
                q, _, _ = area_samples(a, b, c, 0.15, 10**7, rng)
                pts.append(q.astype(np.float32))
                lab_pts.append(np.full(len(q), li, np.int32))
    if not pts:
        return
    tree = cKDTree(np.concatenate(pts))
    lab_pts = np.concatenate(lab_pts)
    del pts
    added, m2 = 0, 0.0
    by_class = collections.Counter()
    for bk in list(blocks):
        for P, _UV, T in list(blocks[bk].get((COLLISION_CLASS, None, False), [])):
            if not len(T):
                continue
            a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
            q, w, rep_ = area_samples(a, b, c, 0.5, 10**6, rng)
            d, _ = tree.query(q, distance_upper_bound=lim, workers=-1)
            far = ~np.isfinite(d)
            if not far.any():
                continue
            farw = np.bincount(rep_, weights=w * far, minlength=len(T))
            tot = np.bincount(rep_, weights=w, minlength=len(T))
            hide = np.nonzero(farw > 0.5 * np.maximum(tot, 1e-12))[0]
            if not len(hide):
                continue
            cen = (a[hide] + b[hide] + c[hide]) / 3
            _, near = tree.query(cen, workers=-1)
            lab = lab_pts[near]
            for li in np.unique(lab):
                ids = hide[lab == li]
                Th = T[ids]
                used, inv = np.unique(Th.reshape(-1), return_inverse=True)
                TW, TUV, TT = triplanar(P[used], inv.reshape(-1, 3))
                cname, tk = lab_of[li]
                cc = TW.mean(axis=0)
                bk2 = (math.floor(cc[0] / args.block_size), math.floor(cc[2] / args.block_size))
                if args.tile_textures and tk and tk.startswith(('tex:', 'dec:')):
                    # per-tile copies must stay in their own tile (see full_png)
                    cen_ = TW[TT].mean(axis=1)
                    tu_ = np.floor(cen_[:, 0] / TILE_SIZE).astype(np.int64)
                    tv_ = np.floor(cen_[:, 2] / TILE_SIZE).astype(np.int64)
                    for u_, v_ in set(zip(tu_.tolist(), tv_.tolist())):
                        sel_ = (tu_ == u_) & (tv_ == v_)
                        T_ = TT[sel_]
                        us_, iv_ = np.unique(T_.reshape(-1), return_inverse=True)
                        k_ = tk if tk in GLOBAL_TEX_KEYS else f'{tk}@{u_}_{v_}'
                        blocks[bk2][(cname, k_, False)].append((TW[us_], TUV[us_], iv_.reshape(-1, 3)))
                else:
                    blocks[bk2][(cname, tk, False)].append((TW, TUV, TT))
                rend_tris.update(tile_counts(TW, TT))
                added += len(TT)
                am = float(tot[ids].sum())
                m2 += am
                by_class[cname] += am
    print(f"fill (final collision): drew {added:,} triangles ({m2:,.0f} m2) of planner-cut collision that had "
          f"nothing drawn within {lim} m; by class " + ", ".join(f"{k} {v:,.0f}" for k, v in by_class.most_common(6)))


FILLS = []           # [(polygon XZ (n,2) after shift, png, bbox)] in draw order
FILL_USED = collections.Counter()


def point_in_poly(P, poly):
    """Even-odd ray casting: which of the points P (n,2) lie inside polygon poly (m,2)."""
    x, z = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, zi = poly[i]
        xj, zj = poly[j]
        cross = ((zi > z) != (zj > z)) & (x < (xj - xi) * (z - zi) / ((zj - zi) + 1e-12) + xi)
        inside ^= cross
        j = i
    return inside


def load_fills(shift, mirror_z):
    p_ = os.path.join(WORK, 'fills.json')
    if not os.path.exists(p_):
        return
    rows = sorted((r for r in json.load(open(p_)) if r.get('png')), key=lambda r: r['order'])
    for r in rows:
        P = np.array(r['points'], float)
        if mirror_z:
            P[:, 2] = -P[:, 2]
        P = P + shift
        xz = P[:, [0, 2]]
        FILLS.append((xz, r['png'], (xz[:, 0].min(), xz[:, 1].min(), xz[:, 0].max(), xz[:, 1].max())))


ROCK_MAT = {}
ROCK_USED = collections.Counter()


SEG_MAT = {}


def load_rock_materials():
    p_ = os.path.join(WORK, 'rock_materials.json')
    if os.path.exists(p_):
        ROCK_MAT.update(json.load(open(p_)))
    p_ = os.path.join(WORK, 'segment_materials.json')
    if os.path.exists(p_):
        SEG_MAT.update(json.load(open(p_)))


def rock_key(png_):
    """Blob key for one baked rock-array slice, registered on first use."""
    base = png_.rsplit('/', 1)[-1][:-4]
    key = 'tex:' + base
    if key not in TEXTURE_BLOBS:
        fp = os.path.join(WORK, png_)
        TEXTURE_BLOBS[key] = _capped(fp, MAX_MATERIAL_PX)
        TEX_BLOBS[base] = TEXTURE_BLOBS[key]
        w_, h_ = Image.open(fp).size
        TEX_SRC[base] = {'fp': png_, 'tint': None, 'nfp': None, 'nat': max(w_, h_), 'nnat': None}
    return key


_DECAL_BY_BASE = {}


def decal_key_for_base(base):
    """'dec:<slug>' for a colour texture that tools/bake_decals.py baked with its mask, else None."""
    if not base.startswith('world/materials/decal/') and '/graffiti/' not in base:
        return None
    if not _DECAL_BY_BASE and os.path.exists(DECAL_TEXTURES):
        for e in json.load(open(DECAL_TEXTURES)).values():
            if e.get('base') and e.get('png'):
                _DECAL_BY_BASE.setdefault(e['base'], e)
        _DECAL_BY_BASE.setdefault('', None)
    e = _DECAL_BY_BASE.get(base)
    if not e:
        return None
    fp = os.path.join(WORK, e['png'])
    if not os.path.exists(fp):
        return None
    key = 'dec:' + e['png'].rsplit('/', 1)[-1][:-4]
    if key not in TEXTURE_BLOBS:
        TEXTURE_BLOBS[key] = open(fp, 'rb').read()
        DECAL_MODE[key] = e.get('mode')
        TEX_SRC[key[4:]] = {'fp': e['png'], 'tint': None, 'nfp': None, 'rgba': True,
                            'nat': max(e.get('w', 0), e.get('h', 0)) or None, 'nnat': None}
    return key


def load_decals(crop, mirror_z):
    """[(texkey, R rows 3x3, t)] for decal volumes inside the crop with a baked texture."""
    if not (os.path.exists(DECALS_JSON) and os.path.exists(DECAL_TEXTURES)):
        return []
    tex = json.load(open(DECAL_TEXTURES))
    x0, z0, x1, z1 = crop
    out, skipped = [], collections.Counter()
    for d in json.load(open(DECALS_JSON)):
        e = tex.get(str(d['key'])) or tex.get(str(d.get('tkey')))
        if e is None:
            skipped['no texture'] += 1
            continue
        R, t = np.array(d['T'][:3], float), np.array(d['T'][3], float)
        if not (x0 < t[0] < x1 and z0 < t[2] < z1):
            skipped['outside crop'] += 1
            continue
        if max(np.linalg.norm(R[0]), np.linalg.norm(R[2])) > 50.0:
            skipped['over 50 m'] += 1
            continue
        fp = os.path.join(WORK, e['png'])
        if not os.path.exists(fp):
            skipped['png missing'] += 1
            continue
        key = 'dec:' + e['png'].rsplit('/', 1)[-1][:-4]
        if key not in TEXTURE_BLOBS:
            TEXTURE_BLOBS[key] = open(fp, 'rb').read()
            DECAL_MODE[key] = e.get('mode')
            TEX_SRC[key[4:]] = {'fp': e['png'], 'tint': None, 'nfp': None, 'rgba': True,
                                'nat': max(e.get('w', 0), e.get('h', 0)) or None, 'nnat': None}
        if mirror_z:
            t = t * [1, 1, -1]
            R = R * [1, 1, -1]
        rows, cols = e.get('grid', (1, 1))
        cell = (int(d.get('row') or 0) % rows, int(d.get('col') or 0) % cols, rows, cols)
        out.append((key, R, t, cell))
    print(f"decals: {len(out):,} volume(s) with a baked texture; skipped {dict(skipped)}")
    return out


def clip_to_box(poly, lo=(-0.5, -0.5, -0.5), hi=(0.5, 0.5, 0.5)):
    """Sutherland-Hodgman: a polygon (n,3) in box-local coords clipped to lo <= p <= hi."""
    for ax in range(3):
        for sgn in (1.0, -1.0):
            if not len(poly):
                return poly
            d = (hi[ax] - poly[:, ax]) if sgn > 0 else (poly[:, ax] - lo[ax])
            out = []
            for i in range(len(poly)):
                a, b = poly[i], poly[(i + 1) % len(poly)]
                da, db = d[i], d[(i + 1) % len(poly)]
                if da >= 0:
                    out.append(a)
                if (da >= 0) != (db >= 0):
                    out.append(a + (b - a) * (da / (da - db)))
            poly = np.array(out) if out else np.zeros((0, 3))
    return poly


def mask_rects(alpha, cut=0.5):
    """Opaque cells of a small alpha grid as merged rectangles [(r0, r1, c0, c1)] (quadtree)."""
    op = alpha >= cut
    out = []

    def rec(r0, r1, c0, c1):
        blk = op[r0:r1, c0:c1]
        if not blk.any():
            return
        if blk.all() or (r1 - r0 == 1 and c1 - c0 == 1):
            out.append((r0, r1, c0, c1))
            return
        rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
        for a, b in ((r0, rm), (rm, r1)):
            for c_, d_ in ((c0, cm), (cm, c1)):
                if b > a and d_ > c_:
                    rec(a, b, c_, d_)
    rec(0, op.shape[0], 0, op.shape[1])
    return out


_DECAL_ALPHA = {}


def decal_mask(key, row, col, rows, cols, grid=DECAL_CUT_GRID):
    """Alpha of one atlas cell of a baked decal, box-resampled to grid x grid (row 0 = v 0)."""
    ck = (key, row, col, rows, cols)
    if ck not in _DECAL_ALPHA:
        src = TEX_SRC.get(key[4:])
        a = None
        if src:
            im = Image.open(os.path.join(WORK, src['fp'])).convert('RGBA')
            A = np.asarray(im, np.float32)[..., 3] / 255.0
            h, w = A.shape
            cell = A[row * h // rows:(row + 1) * h // rows, col * w // cols:(col + 1) * w // cols]
            gh, gw = min(grid, cell.shape[0]), min(grid, cell.shape[1])
            a = np.asarray(Image.fromarray((cell * 255).astype(np.uint8)).resize((gw, gh), Image.BOX),
                           np.float32) / 255.0
            # how well the cut cells stand for the art: IoU of the kept cells (upsampled) with
            # the art's own solid pixels. Thin line art (plot cracks) keeps a few junction cells
            # that draw as solid dark bars - the "black slits" on the chalet patio.
            solid = cell >= 0.5
            kept = np.asarray(Image.fromarray(((a >= 0.5) * 255).astype(np.uint8)).resize(
                (cell.shape[1], cell.shape[0]), Image.NEAREST)) > 0
            un = float((solid | kept).sum())
            _DECAL_IOU[ck] = float((solid & kept).sum()) / un if un else 0.0
        _DECAL_ALPHA[ck] = a
    return _DECAL_ALPHA[ck]


_DECAL_IOU = {}
DECAL_MIN_IOU = 0.4
# Ground grime (patches, stains, dirt, puddles) is a soft BLENDED overlay in skate.; cut out hard
# it drew solid pale/dark shapes that read as broken geometry (user, 2026-09-23: "geometry
# artifacts ... not solid"). Hairline art (cracks, tyre tracks) only ever drew as blocks.
LINE_DECAL = re.compile(r'crack|tiretrack|patch|stain|dirt|puddle|leak|grime|oil|challengespot', re.I)


# A mesh section lying on another section of the same mesh (skate. layers stickers, straps, trim
# and decal strips 1-12 mm over their base and offsets them in its shaders; Skate 3 draws them
# flat and they flash - 918k samples measured 2026-09-23) is lifted along its vertex normals.
OVERLAY_LIFT, OVERLAY_NEAR, OVERLAY_FRAC = 0.012, 0.012, 0.8
# A whole mesh that is a layer laid on another object (stickers on trash cans, billboard ad
# faces, posters): lifted OVERLAY_MESH_LIFT. 1.5 cm still flashed on stickers and billboards;
# graffiti stopped flashing at 5 cm (user, 2026-09-23).
OVERLAY_MESH = re.compile(r'decal|sticker|adspace|poster|wheatpaste|banner|graffiti', re.I)
OVERLAY_MESH_LIFT = 0.035
OVERLAYS_LIFTED = [0]


def _vertex_normals(P, T):
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    vn = np.zeros_like(P)
    for j in range(3):
        np.add.at(vn, T[:, j], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]


def lift_overlays(parts, whole=False):
    """Lift, in place, the sections of one mesh LOD that lie on a LARGER section of it (parallel,
    within OVERLAY_NEAR along the normal, over OVERLAY_FRAC of their area). A strip that only
    touches its neighbours along an edge is not an overlay: its triangle centres are not over
    them. `parts` = [(cls, material, P, UV, T, depot_key)]. Returns the number lifted."""
    if whole:
        for i, (cls, mat, P, UV, T, dk) in enumerate(parts):
            if len(T):
                parts[i] = (cls, mat, P + _vertex_normals(P, T) * OVERLAY_MESH_LIFT, UV, T, dk)
        return len(parts)
    live = [i for i, p in enumerate(parts) if len(p[4])]
    if len(live) < 2:
        return 0
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    area, S, SN, SL, cen, cn = {}, [], [], [], {}, {}
    tot = 0.0
    for i in live:
        P, T = parts[i][2], parts[i][4]
        n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        ln = np.linalg.norm(n, axis=1)
        area[i] = 0.5 * float(ln.sum())
        tot += area[i]
        cen[i], cn[i] = P[T].mean(axis=1), n / np.maximum(ln, 1e-12)[:, None]
    spacing = max(0.0004, tot / 200_000)
    for i in live:
        P, T = parts[i][2], parts[i][4]
        q, _, rep = area_samples(P[T[:, 0]], P[T[:, 1]], P[T[:, 2]], spacing, 10**6, rng)
        S.append(q), SN.append(cn[i][rep]), SL.append(np.full(len(q), i))
    S, SN, SL = np.concatenate(S), np.concatenate(SN), np.concatenate(SL)
    tree = cKDTree(S)
    reach = max(0.04, 1.5 * np.sqrt(spacing))
    n_lift = 0
    for i in live:
        c, nc = cen[i], cn[i]
        if len(c) > 20_000:
            sel = rng.choice(len(c), 20_000, replace=False)
            c, nc = c[sel], nc[sel]
        d, j = tree.query(c, k=16, distance_upper_bound=reach)
        ok = np.isfinite(d)
        j = np.where(ok, j, 0)
        other = ok & (SL[j] != i)
        par = np.abs((SN[j] * nc[:, None, :]).sum(axis=2)) > 0.9
        sep = np.abs(((c[:, None, :] - S[j]) * SN[j]).sum(axis=2))
        hit = other & par & (sep < OVERLAY_NEAR)
        on = hit.any(axis=1)
        if on.mean() < OVERLAY_FRAC:
            continue
        base = collections.Counter(SL[j][hit].tolist()).most_common(1)[0][0]
        if area[i] >= area[base]:
            continue
        # out along the base's normal: flip when the layer's winding faces into its base
        side = 1.0 if float(((SN[j] * nc[:, None, :]).sum(axis=2))[hit].mean()) >= 0 else -1.0
        cls, mat, P, UV, T, dk = parts[i]
        parts[i] = (cls, mat, P + _vertex_normals(P, T) * (side * OVERLAY_LIFT), UV, T, dk)
        n_lift += 1
    return n_lift


def project_decal(R, c, TR, cand, lift=None, rects=None, extra=0.0):
    """Clip the drawn triangles TR[cand] to the decal box (rows R, centre c): (W, UV, T) or None.

    This is what a decal volume IS in skate.: a box projected onto whatever surface lies inside
    it. Faces turned edge-on to the projection axis (|n.up| < 0.35) are skipped, as the engine's
    projection would stretch across them. UVs are the box coordinates, so the image lands where
    skate. puts it; each piece is lifted `lift` along its own face normal.
    """
    Rinv = np.linalg.inv(R)
    u = R[1] / max(np.linalg.norm(R[1]), 1e-9)
    Ws, UVs, Ts, off = [], [], [], 0
    tris = TR[cand].astype(np.float64)
    N = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(N, axis=1)
    Lall = (tris - c) @ Rinv
    keep = ((ln > 1e-9) & (np.abs(N @ u) >= 0.35 * np.maximum(ln, 1e-12))
            & ~(Lall > 0.5).all(axis=1).any(axis=1) & ~(Lall < -0.5).all(axis=1).any(axis=1))
    for i in np.nonzero(keep)[0]:
        n = N[i] / ln[i]
        L = Lall[i]
        polys = [clip_to_box(L)] if rects is None else [clip_to_box(L, lo, hi) for lo, hi in rects]
        for poly in polys:
            if len(poly) < 3:
                continue
            nn = n if float(n @ u) > 0 else -n
            lf = (lift if lift is not None
                  else DECAL_LIFT_FLOOR if abs(float(nn[1])) >= 0.7 else DECAL_LIFT_WALL) + extra
            W = poly @ R + c + nn * lf
            uv = np.stack([poly[:, 0] + 0.5, 0.5 - poly[:, 2]], axis=1)
            k = len(poly)
            Ws.append(W)
            UVs.append(uv)
            Ts.append(np.array([[off, off + i, off + i + 1] for i in range(1, k - 1)]))
            off += k
    if not Ws:
        return None
    return np.concatenate(Ws), np.concatenate(UVs).astype(np.float32), np.concatenate(Ts).astype(np.int64)


def emit_decals(decals, blocks, shift, tile_y, sliver, rend_tris, block_size, snap=True, tile_keys=False,
                cutout=True):
    """One double-sided quad per decal volume, snapped onto the surface it projects onto.

    The box's quad is right x forward and `up` is its projection depth (~1.2 m). A quad at the box
    centre can float 60 cm off the wall, so the centre is moved along `up` to the nearest drawn
    surface inside the box (ray against the render triangles near it), then lifted DECAL_LIFT.
    Goes straight into `blocks`: decals are never LOD-stepped, thinned or UV-corrected.
    """
    from scipy.spatial import cKDTree
    tris = []
    for classes in blocks.values():
        for (cname, _tk, _p), parts in classes.items():
            if cname in (COLLISION_CLASS, DECAL_CLASS):
                continue
            for W, _UV, T in parts:
                if len(T):
                    tris.append(W[T].astype(np.float32))
    TR = np.concatenate(tris) if tris else np.zeros((0, 3, 3), np.float32)
    del tris
    if len(TR) and decals:
        # only triangles that can reach a decal box: bounds within 40 m (XZ) of some decal centre
        dc = np.array([t_ + shift for _k, _R, t_, _c in decals])[:, [0, 2]]
        lo0, hi0 = TR.min(axis=1)[:, [0, 2]], TR.max(axis=1)[:, [0, 2]]
        near = np.zeros(len(TR), bool)
        for d_ in dc:
            near |= ((lo0 <= d_ + 40.0) & (hi0 >= d_ - 40.0)).all(axis=1)
        TR = TR[near]
    cen = TR.mean(axis=1)
    rad = np.linalg.norm(TR - cen[:, None, :], axis=2).max(axis=1) if len(TR) else np.zeros(0)
    tree = cKDTree(cen) if len(cen) else None
    rmax = float(rad.max()) if len(rad) else 0.0
    # triangles by the 4 m XZ cells their bounds cover: a decal only tests what its box touches
    # (a centroid search padded by the largest triangle - a 100 m lake sheet - took an hour)
    GC = 4.0
    lo_ = np.floor(TR.min(axis=1)[:, [0, 2]] / GC).astype(np.int64)
    hi_ = np.floor(TR.max(axis=1)[:, [0, 2]] / GC).astype(np.int64)
    grid = collections.defaultdict(list)
    for i in range(len(TR)):
        for gx in range(lo_[i, 0], hi_[i, 0] + 1):
            for gz in range(lo_[i, 1], hi_[i, 1] + 1):
                grid[(gx, gz)].append(i)
    corners = np.array([[-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5]])
    UV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]] * 2, np.float32)
    TT = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], np.int64)
    n_ok, n_snap, n_out, n_proj, n_empty = 0, 0, 0, 0, 0
    n_thin = 0
    # largest boxes first (lowest), each later one stacked over those whose box it overlaps
    decals = sorted(decals, key=lambda d: -float(np.linalg.norm(d[1][0]) * np.linalg.norm(d[1][2])))
    placed_boxes = []
    for key, R, t, (row, col, rows, cols) in decals:
        c = t + shift
        _half = 0.5 * np.abs(R).sum(axis=0)
        rank = sum(1 for c2, h2 in placed_boxes if (np.abs(c - c2) <= _half + h2).all())
        placed_boxes.append((c, _half))
        extra = DECAL_STACK * min(rank, 6)
        up = R[1]
        h = float(np.linalg.norm(up))
        u = up / max(h, 1e-9)
        best = None
        proj = None
        if snap and len(TR):
            half = 0.5 * np.abs(R).sum(axis=0)          # world AABB half-extent of the box
            g0 = np.floor((c[[0, 2]] - half[[0, 2]]) / GC).astype(int)
            g1 = np.floor((c[[0, 2]] + half[[0, 2]]) / GC).astype(int)
            cs = set()
            for gx in range(g0[0], g1[0] + 1):
                for gz in range(g0[1], g1[1] + 1):
                    cs.update(grid.get((gx, gz), ()))
            if cs:
                rects = None
                if cutout:
                    m_ = decal_mask(key, row, col, rows, cols)
                    if m_ is not None and (_DECAL_IOU.get((key, row, col, rows, cols), 1.0) < DECAL_MIN_IOU
                                           or LINE_DECAL.search(key)):
                        n_thin += 1
                        continue    # line art the cut cannot draw (cracks): leave it out
                    if m_ is not None:
                        gh, gw = m_.shape
                        # grid row r = image row = v; local x = u - 0.5, local z = 0.5 - v
                        rects = [((c0 / gw - 0.5, -0.5, 0.5 - r1 / gh), (c1 / gw - 0.5, 0.5, 0.5 - r0 / gh))
                                 for r0, r1, c0, c1 in mask_rects(m_)]
                if rects is None or rects:
                    proj = project_decal(R, c, TR, np.fromiter(cs, np.int64), rects=rects, extra=extra)
                if rects is not None and not rects:
                    n_empty += 1
                    continue        # nothing in the art is solid enough to draw
                if proj is None and rects:
                    # nothing drawn inside the box to project onto: cut the decal's own quad
                    # (the box's mid-plane) from the mask rather than draw it whole
                    q_ = np.array([[-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5]]) @ R + c
                    fake = np.array([[q_[0], q_[1], q_[2]], [q_[0], q_[2], q_[3]]], np.float32)
                    proj = project_decal(R, c, fake, np.arange(2), rects=rects, lift=DECAL_LIFT + extra)
        if proj is not None:
            W, uv, TTp = proj
            uv = uv.copy()
            uv[:, 0] = (col + uv[:, 0]) / cols
            uv[:, 1] = (row + uv[:, 1]) / rows
            cen_ = W[TTp].mean(axis=1)
            groups = collections.defaultdict(list)
            for ti, cc_ in enumerate(cen_):
                groups[(int(np.floor(cc_[0] / TILE_SIZE)), int(np.floor(cc_[2] / TILE_SIZE)))].append(ti)
            placed_any = False
            for (gu, gv), tis in groups.items():
                tk = (int(gu * TILE_SIZE + TILE_SIZE / 2), int(gv * TILE_SIZE + TILE_SIZE / 2))
                if tk not in tile_y or tk in sliver:
                    continue
                Tg = TTp[tis]
                used_, inv_ = np.unique(Tg.reshape(-1), return_inverse=True)
                Wg, UVg, Tg = W[used_], uv[used_], inv_.reshape(-1, 3)
                tile_y[tk][0] = min(tile_y[tk][0], float(Wg[:, 1].min()))
                tile_y[tk][1] = max(tile_y[tk][1], float(Wg[:, 1].max()))
                rend_tris[tk] += len(Tg)
                bk = (math.floor(Wg[:, 0].mean() / block_size), math.floor(Wg[:, 2].mean() / block_size))
                k_ = f'{key}@{gu}_{gv}' if tile_keys else key
                blocks[bk][(DECAL_CLASS, k_, False)].append((Wg.astype(np.float32), UVg, Tg))
                placed_any = True
            if placed_any:
                n_proj += 1
                n_ok += 1
                continue
        if snap and tree is not None:
            reach = 0.5 * h + 0.5 * max(np.linalg.norm(R[0]), np.linalg.norm(R[2]))
            ii = tree.query_ball_point(c, min(reach + rmax, reach + 50.0))
            if ii:
                ii = np.asarray(ii)
                A, B, C = TR[ii, 0].astype(float), TR[ii, 1].astype(float), TR[ii, 2].astype(float)
                e1, e2 = B - A, C - A
                pv = np.cross(u, e2)
                det = (e1 * pv).sum(1)
                ok = np.abs(det) > 1e-12
                inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
                tv = c - A
                bu = (tv * pv).sum(1) * inv
                qv = np.cross(tv, e1)
                bv = (u * qv).sum(1) * inv
                sd = (e2 * qv).sum(1) * inv
                hit = ok & (bu >= 0) & (bv >= 0) & (bu + bv <= 1) & (np.abs(sd) <= 0.5 * h)
                if hit.any():
                    j = np.argmin(np.where(hit, np.abs(sd), np.inf))
                    best = float(sd[j])
        if best is not None:
            c = c + u * best
            n_snap += 1
        W = (corners[:, [0]] * R[0] + corners[:, [2]] * R[2] + c
             + u * ((DECAL_LIFT_FLOOR if abs(float(u[1])) >= 0.7 else DECAL_LIFT_WALL) + extra))
        W = np.concatenate([W, W - u * 0.004])
        tk = (int(np.floor(c[0] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2),
              int(np.floor(c[2] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2))
        if tk not in tile_y or tk in sliver:
            n_out += 1
            continue
        tile_y[tk][0] = min(tile_y[tk][0], float(W[:, 1].min()))
        tile_y[tk][1] = max(tile_y[tk][1], float(W[:, 1].max()))
        rend_tris[tk] += len(TT)
        bk = (math.floor(c[0] / block_size), math.floor(c[2] / block_size))
        uv = UV.copy()
        uv[:, 0] = (col + uv[:, 0]) / cols
        uv[:, 1] = (row + uv[:, 1]) / rows
        if tile_keys:
            # key each triangle by the tile its centroid falls in (the arena builder's own rule), so a
            # quad across a tile border never shares one per-tile copy between two tiles
            groups = collections.defaultdict(list)
            for ti, tri in enumerate(TT):
                cc_ = W[tri].mean(axis=0)
                groups[(int(np.floor(cc_[0] / TILE_SIZE)), int(np.floor(cc_[2] / TILE_SIZE)))].append(ti)
            for (gu, gv), tis in groups.items():
                blocks[bk][(DECAL_CLASS, f'{key}@{gu}_{gv}', False)].append((W, uv, TT[tis].copy()))
        else:
            blocks[bk][(DECAL_CLASS, key, False)].append((W, uv, TT.copy()))
        n_ok += 1
    modes = collections.Counter(DECAL_MODE.get(k) for k, _R, _t, _c in decals)
    print(f"decals: drew {n_ok:,} decal(s): {n_proj:,} projected onto the surfaces inside their box, "
          f"{n_ok - n_proj:,} as snapped quads ({n_snap:,} snapped); {n_empty:,} with no solid art skipped; {n_thin:,} line-art (cracks) the cut cannot draw skipped; "
          f"{n_out:,} outside the drawn tiles; "
          f"textures by mode {dict(modes)}")


def fill_hidden_collision(args, staged, staged_owner, placed, placed_coll, shift, tile_y, sliver):
    """Draw every piece of skate.'s collision that has no drawn surface within `--fill-hidden-collision` m.

    Measured on the Summer Camp build of 2026-09-22 (tools/invisible_collision.py): 176,000 m2 of
    collision - 12.7% of the floor, 5.3% of the walls - had nothing drawn within 0.5 m. The
    render and the collision of one object are separate authored meshes, and wherever they part
    (a river sheet tessellated two ways, a far LOD, a prop thinned for the tile budget) the
    skater hits a surface nobody can see. The fix is to draw what is ridden: each such collision
    triangle is drawn with its own instance's material, triplanar-tiled at 2 m. It is compared
    against the FINAL render (after LOD stepping and thinning), so it also covers thinned props.
    Appends to `staged` in place.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(7)
    lim = args.fill_hidden_collision
    pts = []
    area = {}
    for j, (e, pi) in enumerate(zip(staged, staged_owner)):
        rW, rT = e[6], e[8]
        if not len(rT):
            continue
        a, b, c = rW[rT[:, 0]], rW[rT[:, 1]], rW[rT[:, 2]]
        q, w, _ = area_samples(a, b, c, 0.15, 10**7, rng)
        pts.append(q.astype(np.float32))
        pts.append(rW.astype(np.float32))
        tot = float(w.sum())
        if e[1] != COLLISION_CLASS and tot > area.get(pi, (0.0, -1))[0]:
            area[pi] = (tot, j)
    if not pts:
        return
    tree = cKDTree(np.concatenate(pts))
    del pts
    added_tris, added_m2, by_mesh = 0, 0.0, collections.Counter()
    for pi, coll in enumerate(placed_coll):
        if not coll or pi not in area:
            continue
        mesh = placed[pi][0]
        key, cname, tk, mat, atlas, baked, _rW, _rUV, _rT = staged[area[pi][1]]
        if baked or atlas:
            pm = PROXY_MAT.get(mesh)
            base = MATERIAL_TEX.get(pm) if pm else None
            if base and base in TEX_BLOBS:
                tk = 'tex:' + base
                TEXTURE_BLOBS.setdefault(tk, TEX_BLOBS[base])
            else:
                tk = None
        for Wc, Tc in coll:
            if not len(Tc):
                continue
            W = Wc + shift
            a, b, c = W[Tc[:, 0]], W[Tc[:, 1]], W[Tc[:, 2]]
            q, w, rep = area_samples(a, b, c, 0.5, 400, rng)
            d, _ = tree.query(q, distance_upper_bound=lim, workers=-1)
            far = ~np.isfinite(d)
            if not far.any():
                continue
            farw = np.bincount(rep, weights=w * far, minlength=len(Tc))
            tot = np.bincount(rep, weights=w, minlength=len(Tc))
            hide = farw > 0.5 * np.maximum(tot, 1e-12)
            if not hide.any():
                continue
            cen = W[Tc].mean(axis=1)
            tu = (np.floor(cen[:, 0] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int)
            tv = (np.floor(cen[:, 2] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int)
            ok = np.fromiter(((int(x), int(z)) in tile_y and (int(x), int(z)) not in sliver
                              for x, z in zip(tu.tolist(), tv.tolist())), bool, len(Tc))
            hide &= ok
            if not hide.any():
                continue
            Th = Tc[hide]
            used, inv = np.unique(Th.reshape(-1), return_inverse=True)
            TW, TUV, TT = triplanar(W[used], inv.reshape(-1, 3))
            staged.append((key, cname, tk, 'fill:' + mat, False, False, TW, TUV, TT))
            staged_owner.append(pi)
            added_tris += len(TT)
            m2 = float(tot[hide].sum())
            added_m2 += m2
            by_mesh[mesh.rsplit('/', 1)[-1]] += m2
    print(f"fill: drew {added_tris:,} collision triangles ({added_m2:,.0f} m2) that had nothing drawn within "
          f"{lim} m; top: " + ", ".join(f"{m} {v:,.0f}" for m, v in by_mesh.most_common(6)))


def main():
    global MAX_MATERIAL_PX, MAX_TEXTURE_PX, HOT_MATERIAL_PX, HOT_TEXTURE_M2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(ROOT, 'work', 'IsleOfGrom', 'glb'))
    ap.add_argument('--block-size', type=float, default=400.0)
    ap.add_argument('--shared-small-texture-m2', type=float, default=0.0,
                    help='textures shared by two or more tiles and covering less than this (m2) are served at 64 px, '
                         'to keep the always-resident cPres_Global under its limit')
    ap.add_argument('--normal-min-m2', type=float, default=0.0,
                    help='textures covering less surface than this (m2, island-wide) use their class normal map; '
                         'trims the always-resident cPres_Global when many textures are shared between tiles')
    ap.add_argument('--grind-source', choices=('collision', 'render'), default='collision',
                    help="where 90-degree grind edges are traced: skate.'s own collision (default) or the render meshes")
    ap.add_argument('--decals', action=argparse.BooleanOptionalAction, default=False,
                    help='draw decal volumes (graffiti) from decals.json + decal_textures.json')
    ap.add_argument('--decal-family', default='environmentsimple.alphatest',
                    help='shading family for decal materials (see tools/probe_alpha.py)')
    ap.add_argument('--decal-cutout', action=argparse.BooleanOptionalAction, default=True,
                    help='cut each decal to its art in GEOMETRY (blocky; for when the engine '
                         'ignores alpha). Off: full projected quads cut by the alpha of the art, which '
                         'needs the no_specular material flag and the patched arena builder')
    ap.add_argument('--decal-snap', action=argparse.BooleanOptionalAction, default=True,
                    help='move each decal onto the nearest drawn surface inside its box')
    ap.add_argument('--rock-lod', choices=('finest', 'physics'), default='physics',
                    help="rocks: the LOD matching skate.'s rock collision (default, what you hit is what you see) "
                         "or 'finest' resident LOD")
    ap.add_argument('--fill-hidden-collision', type=float, default=FILL_DIST,
                    help="draw collision triangles with no drawn surface within this many metres "
                         "(0 = off); see fill_hidden_collision")
    ap.add_argument('--visual-follows-collision', type=float, default=0.03,
                    help="metres: ride geometry whose drawn surface sits further than this off skate.'s "
                         "collision is drawn from the collision shape instead (0 disables)")
    ap.add_argument('--mirror-z', action='store_true',
                    help='negate Z when converting skate. space. Off by default: with it on, the island '
                         'played MIRRORED in Skate 3 (2026-09-14), so the two spaces share handedness')
    ap.add_argument('--grind-max', type=int, default=0,
                    help='keep at most this many detected grind edges, longest first (0 = every one)')
    ap.add_argument('--tile-textures', action='store_true',
                    help='give every streaming tile its own copy of each texture it uses, so nothing is '
                         'promoted to cPres_Global and textures stream at full resolution per tile')
    ap.add_argument('--tile-texture-budget-mb', type=float, default=4.0,
                    help='most texture bytes one tile stream may hold under --tile-textures')
    ap.add_argument("--render-window-mb", type=float, default=60.0,
                    help='most render-stream MB in any 3x3 tile window (a warp loads the 3x3 at once)')
    ap.add_argument('--tile-stream-mb', type=float, default=7.5,
                    help='target size of one render stream (mesh + textures) under --tile-textures')
    ap.add_argument('--tile-texture-px', type=int, default=1024,
                    help='largest edge of a per-tile texture copy (source textures reach 1024-2048)')
    ap.add_argument('--global-pool-mb', type=float, default=4.0,
                    help='estimated cPres_Global bytes spent on shared textures under --tile-textures')
    ap.add_argument('--global-edge-px', type=int, default=256,
                    help='edge of a shared texture in cPres_Global below HOT_TEXTURE_M2 of coverage')
    ap.add_argument('--tex-cache-mb', type=int, default=300,
                    help='decoded full-resolution textures kept in memory while writing')
    ap.add_argument('--shared-texture-budget-mb', type=float, default=0.0,
                    help='size textures shared by two or more tiles to fit this estimate of cPres_Global, '
                         'giving resolution where it covers the most area (0 = the old flat caps)')
    ap.add_argument('--texture-px', type=int, default=MAX_MATERIAL_PX,
                    help='largest side a game texture is shipped at (full-res mips are 512-2048 px since '
                         'the TOC chunk-table fix; the per-tile stream budget decides how high this can go)')
    ap.add_argument('--hot-texture-px', type=int, default=HOT_MATERIAL_PX,
                    help='cap for textures covering more than --hot-texture-m2 of the map')
    ap.add_argument('--hot-texture-m2', type=float, default=HOT_TEXTURE_M2)
    ap.add_argument('--grind-max-per-tile', type=int, default=1800,
                    help='most splines any 100 m tile may carry (the engine crashed at 2,338; 1,658 loads)')
    ap.add_argument('--grind-angle', default='40,140',
                    help='min,max dihedral degrees for a grindable edge; 40 catches bevelled and '
                         'rounded ledges, whose corner turns in two ~45 degree steps')
    ap.add_argument('--grind-min-len', type=float, default=0.5, help='shortest grind edge kept, metres')
    ap.add_argument('--smooth-ramps', type=int, default=2,
                    help='Loop subdivision levels for the collision of curved skate features (quarter pipes, '
                         'banks, kickers, bowls): skate. installs only their faceted LOD2 (0 disables)')
    ap.add_argument('--seat-lip', type=float, default=0.12,
                    help='metres: ramp/bank toes sitting up to this far above the surface under them are '
                         'lowered onto it, so the wheels do not catch on the step (0 disables)')
    ap.add_argument('--seam-ramps', type=float, default=0.08,
                    help='ramp every drop of 1.5 cm up to this many m between rideable collision surfaces '
                         '(0 = off)')
    ap.add_argument('--seam-lip', type=float, default=0.10,
                    help='metres: a flat piece edge within this of a neighbouring surface (above or below) is '
                         'moved onto it, so road/slab edges ride over (real curbs are taller)')
    ap.add_argument('--weld', type=float, default=0.04,
                    help='metres: collision vertices of different pieces this close are merged into one')
    ap.add_argument('--tile-size', type=float, default=100.0,
                    help='streaming tile edge in metres; must match the arena builder --tile-size and the packer')
    ap.add_argument('--crop', default=None,
                    help='Frostbite x0,z0,x1,z1 to keep (default: the whole island)')
    ap.add_argument('--min-tile-triangles', type=int, default=500,
                    help='drop tiles receiving fewer triangles than this, so no CSPA cell and no '
                         'cPres/cSim stream is declared for a sliver (0 disables)')
    ap.add_argument('--collision-tile-budget', type=int, default=140_000,
                    help='collision triangles per 100 m tile. Measured on the shipped cSim streams: '
                         '31 B per triangle (p10 28, p90 34), so the 5 MB stream ceiling is ~160k; '
                         '140k is ~4.4 MB. Only prop collision is thinned to meet it.')
    ap.add_argument('--bake-light', action=argparse.BooleanOptionalAction, default=False,
                    help='bake sun shadows, sky occlusion and slope shading into a per-tile lightmap '
                         '(tools/lightmap.py) instead of one flat grey swatch')
    ap.add_argument('--sun-az', type=float, default=135.0, help='baked sun azimuth, degrees from +z towards +x')
    ap.add_argument('--sun-el', type=float, default=50.0, help='baked sun elevation, degrees')
    ap.add_argument('--lightmap', type=float, default=LIGHTMAP_GREY,
                    help='grey level for the LIGHTMAP channel (glTF emissive). 0 supplies '
                         'none, which is NOT "unlit": the engine then binds its own '
                         'default_white and every surface renders fully lit.')
    ap.add_argument('--collision-source', choices=('skate', 'derived'), default='skate',
                    help="skate: skate.'s own .physics collision, placed per instance (default). "
                         "derived: the old copy built from the render meshes")
    ap.add_argument('--prop-collision-floor', type=float, default=0.05,
                    help='props keep at least this fraction of their collision before ride geometry is cut')
    ap.add_argument('--rock-collision-floor', type=float, default=0.12,
                    help='rocks keep at least this fraction of their collision before ride geometry is cut')
    ap.add_argument('--collision-window-mb', type=float, default=14.0,
                    help='collision allowed in any 3x3 block of tiles: Skate 3 streams the collision '
                         'around the player into one pool that stalls between 14.87 and 15.05 MB')
    ap.add_argument('--collision-stream-mb', type=float, default=2.9,
                    help='one collision stream: 3.045 MB loaded, ~3.5 MB stalled')
    ap.add_argument('--collision-bytes-per-tri', type=float, default=15.3,
                    help='cSim bytes per collision triangle (fitted with --collision-bytes-per-vert on '
                         '39 shipped streams, max error 0.19 MB)')
    ap.add_argument('--collision-bytes-per-vert', type=float, default=28.1,
                    help='cSim bytes per collision vertex')
    ap.add_argument('--prune-unreachable', action=argparse.BooleanOptionalAction, default=True,
                    help='drop downward collision faces with no floor within 3 m below them')
    ap.add_argument('--grid-search', action=argparse.BooleanOptionalAction, default=True,
                    help='slide the 100 m tile grid (0-95 m, 5 m steps) to minimise the worst tile')
    ap.add_argument('--tile-vertex-budget', type=int, default=105_000,
                    help='render vertices per tile the grid search aims under: ~40 B/vertex measured, '
                         'so 105k is ~4.2 MB plus ~0.5 MB of materials and local textures')
    ap.add_argument('--tile-budget', type=int, default=70_000,
                    help='render triangles per 100 m tile: over it, meshes drop to their coarsest LOD, '
                         'then non-rideable geometry is decimated (~55 B/tri, so 70k ~ 3.9 MB)')
    args = ap.parse_args()
    global TILE_SIZE, GRIND_MAX_TOTAL, GRIND_MIN_ANGLE, GRIND_MAX_ANGLE, GRIND_MIN_LEN
    MAX_MATERIAL_PX = MAX_TEXTURE_PX = args.texture_px
    TEX_CACHE_BYTES[0] = args.tex_cache_mb * 1024 * 1024
    HOT_MATERIAL_PX, HOT_TEXTURE_M2 = max(args.hot_texture_px, args.texture_px), args.hot_texture_m2
    TILE_SIZE = float(args.tile_size)
    GRIND_MAX_TOTAL = args.grind_max
    GRIND_MIN_ANGLE, GRIND_MAX_ANGLE = (float(v) for v in args.grind_angle.split(','))
    GRIND_MIN_LEN = args.grind_min_len
    crop = tuple(float(v) for v in args.crop.split(',')) if args.crop else CROP
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.out):
        if f.endswith('.glb'):
            os.remove(os.path.join(args.out, f))

    G = json.load(open(os.path.join(WORK, 'global_assets.json')))
    RES = G['res']
    CH = json.load(open(os.path.join(WORK, 'global_chunks.json')))
    cas = Cas()

    global TEXTURES, TEXTURE_BLOBS
    TEXTURES = load_textures()
    HOT_TEX.update(hot_textures())
    if HOT_TEX:
        print(f"high-coverage textures kept at {HOT_MATERIAL_PX} px "
              f"(over {HOT_TEXTURE_M2:,.0f} m2 each): "
              + ", ".join(sorted(b.rsplit('/', 1)[-1] for b in HOT_TEX)))
    MATERIAL_TEX.update(load_material_textures())
    KEY_TEX.update(load_key_textures())
    load_rock_materials()
    if ROCK_MAT:
        print(f"rock materials: {len(ROCK_MAT)} depot key(s) drawn from skate.'s rock/grass texture array; "
              f"{len(SEG_MAT)} slab/panel sheet material(s)")
    if os.path.exists(DEPOT_KEY_TINTS):
        KEY_TINT.update(json.load(open(DEPOT_KEY_TINTS)))
    _alb, _nrm = load_class_textures()
    CLASS_BLOBS.update(_alb)
    CLASS_NORMALS.update(_nrm)
    TEXTURE_BLOBS = {k: v for k, v in (TEXTURES.values() if TEXTURES else [])}
    if TEXTURES:
        print(f"textures: {len(TEX_BLOBS):,} distinct game textures "
              f"({len(TEX_NORMALS):,} with normal maps, {len(ATLAS_BASES):,} baked atlases) "
              f"bound by {len(KEY_TEX):,} shader-block keys and {len(MATERIAL_TEX):,} "
              f"material names, "
              f"{len(TEXTURE_BLOBS):,} per-mesh images on {len(TEXTURES):,} meshes, "
              f"{len(CLASS_BLOBS):,} class textures, {len(CLASS_NORMALS):,} class normal maps")
    else:
        # the per-mesh table is a retired path (nothing writes its inputs); binding is per section
        print(f"textures: {len(TEX_BLOBS):,} distinct game textures ({len(TEX_NORMALS):,} with normal "
              f"maps, {len(ATLAS_BASES):,} baked atlases) bound by {len(KEY_TEX):,} shader-block keys "
              f"and {len(MATERIAL_TEX):,} material names; {len(CLASS_BLOBS):,} class textures, "
              f"{len(CLASS_NORMALS):,} class normal maps")

    inst = load_instances(crop)
    dropped = collections.Counter(i[0] for i in inst if DROP.search(i[0]) or i[0] not in RES)
    inst = [i for i in inst if not DROP.search(i[0]) and i[0] in RES]
    print(f"instances {len(inst)} ({sum(dropped.values())} clutter/unresolved dropped), "
          f"sources {dict(collections.Counter(i[3] for i in inst))}")

    names = sorted({i[0] for i in inst})
    meshsets = {n: meshset.parse(b) for n, b in zip(names, decompress_many(
        [cas.read(*RES[n][1:5]) for n in names]))}

    PROXY_MAT.update(proxy_materials(meshsets, CH))
    PHYS = {}
    if args.collision_source == 'skate':
        pn = sorted({physics_name(n) for n in names} & set(RES))
        for n, b in zip(pn, decompress_many([cas.read(*RES[n][1:5]) for n in pn])):
            try:
                PHYS[n] = physics.parse(b)
            except ValueError as e:
                print(f"  physics {n}: {e}")
        _wp = sum(1 for i in inst if physics_name(i[0]) in PHYS)
        print(f"skate. collision: {len(PHYS):,} physics assets; {_wp:,} of {len(inst):,} instances "
              f"carry skate.'s own collision, {len(inst) - _wp:,} have none in skate. and get none here")
    _pm_bound = sum(1 for v in PROXY_MAT.values() if v in MATERIAL_TEX)
    print(f"proxy shells: {len(PROXY_MAT)} mesh(es) whose only resident LOD is a proxy atlas; "
          f"{_pm_bound} take the material their finest LOD names")

    def lods(n):
        return [k for k, l in enumerate(meshsets[n].lods) if l.chunk in CH]

    def tris(n, k):
        return sum(s.primitives for s in meshsets[n].lods[k].sections
                   if not s.is_shadow and classify(s.material)[0] not in DROP_CLASSES)

    # The LOD budget has to be measured on the tiles the arena builder actually cuts. It used to key
    # instances on Frostbite t/100, which the recentre shift (1224.166, 3149.832 last build) moves
    # off the shipped grid - so the "hot" tiles it stepped were never the tiles that shipped, and
    # cPres_50_250 went out at 6.49 MB untouched. X/Z of the shift now come from the instance
    # positions, known before any LOD is chosen, so both use one grid. Y still comes from geometry.
    _ix = np.array([i[2][0] for i in inst])
    _iz = (-1.0 if args.mirror_z else 1.0) * np.array([i[2][2] for i in inst])
    sx = -float(np.percentile(_ix, 1) + np.percentile(_ix, 99)) / 2
    sz = -float(np.percentile(_iz, 1) + np.percentile(_iz, 99)) / 2

    def tkey(t):
        return (math.floor((t[0] + sx) / TILE_SIZE),
                math.floor(((-t[2] if args.mirror_z else t[2]) + sz) / TILE_SIZE))

    choice = []
    rock_phys = 0
    for mesh, R, t, src in inst:
        av = lods(mesh)
        k = (av[-1] if COARSE.search(mesh) else av[0]) if av else None
        if k is not None and args.rock_lod == 'physics' and collision_tier(mesh) == 1:
            # skate.'s rock collision is one of the rock's own LODs (cliffrock: LOD4, 1,099 tris;
            # scree: LOD2, 1,100). Drawing that LOD makes the drawn rock the ridden rock exactly.
            pt = sum(len(p_.triangles) for p_ in PHYS.get(physics_name(mesh), ()))
            if pt:
                best = min(av, key=lambda j: abs(tris(mesh, j) - pt))
                if abs(tris(mesh, best) - pt) <= 0.05 * pt:
                    k = best
                    rock_phys += 1
        choice.append(k)
    if args.rock_lod == 'physics':
        print(f"rocks: {rock_phys:,} instance(s) drawn at the LOD their skate. collision was built from")
    tile = collections.Counter()
    for (mesh, R, t, _), k in zip(inst, choice):
        if k is not None:
            tile[tkey(t)] += tris(mesh, k)
    hot = {key for key, v in tile.items() if v > args.tile_budget}
    # Step the heaviest meshes down ONE authored LOD at a time until the tile fits. This used
    # to drop EVERY mesh in a hot tile to its coarsest LOD: whole blocks turned into boxes, 8
    # tiles were still over budget afterwards (worst 179k tris), and vertex clustering then
    # shredded what was left -- the "geometry is not perfect" a player sees. EA's own LODs
    # are clean, so spend the budget where it buys the least visible loss.
    # Ride geometry (plots and generated ground) is stepped only once the props in the tile have
    # nothing left to give, and then a whole plot at a time: stepping plot pieces one by one put
    # neighbouring pieces of one surface on different LODs, and two LODs of a surface do not
    # meet - that is a crack under the wheels that skate. itself never has.
    import heapq
    steps = unfloored_tiles = 0
    for key in hot:
        heap = []
        members = collections.defaultdict(list)

        floored = [2]     # 2: every prop floored, 1: only small props, 0: none

        def nxt(i):
            av = lods(inst[i][0])
            j = av.index(choice[i])
            nk = av[j + 1] if j + 1 < len(av) else None
            # First pass: a prop keeps PROP_LOD_FLOOR of its finest LOD's triangles. Summer Camp's
            # bikes went out at LOD3 (638 of 6,048 triangles: six-sided wheels, no spokes) while
            # the tile still had bigger savings to take. The floor is lifted only if the tile is
            # still over budget once everything else has stepped - big props first, and the small
            # ones (a bike is 6,048 triangles) only last, since they have the least to give.
            if (nk is not None and ride_group(inst[i][0]) is None
                    and (floored[0] == 2 or DETAIL_PROP.search(inst[i][0])
                         or (floored[0] == 1 and tris(inst[i][0], av[0]) <= PROP_SMALL_TRIS))
                    and tris(inst[i][0], nk) < PROP_LOD_FLOOR * tris(inst[i][0], av[0])):
                return None
            return nk

        def push(g):
            if g[0] == 'inst':
                i = g[1]
                nk = nxt(i)
                if nk is not None:
                    saving = tris(inst[i][0], choice[i]) - tris(inst[i][0], nk)
                    if saving > 0:
                        heapq.heappush(heap, (0, -saving, g))
                return
            saving = 0
            for i in members[g]:
                nk = nxt(i)
                if nk is not None:
                    saving += tris(inst[i][0], choice[i]) - tris(inst[i][0], nk)
            if saving > 0:
                heapq.heappush(heap, (1, -saving, g))

        for i, (mesh, R, t, _) in enumerate(inst):
            if choice[i] is not None and tkey(t) == key:
                g = ride_group(mesh)
                if g is None:
                    push(('inst', i))
                else:
                    members[g].append(i)
        for g in members:
            push(g)
        total = tile[key]
        while total > args.tile_budget:
            # the floor goes before any ride geometry is stepped (heap priority 1)
            if floored[0] and (not heap or heap[0][0] == 1):
                floored[0] -= 1
                unfloored_tiles += floored[0] == 0
                for i, (mesh, R, t, _) in enumerate(inst):
                    if choice[i] is not None and tkey(t) == key and ride_group(mesh) is None:
                        push(('inst', i))
                continue
            if not heap:
                break
            _, _, g = heapq.heappop(heap)
            for i in ([g[1]] if g[0] == 'inst' else members[g]):
                nk = nxt(i)
                if nk is None:
                    continue
                total += tris(inst[i][0], nk) - tris(inst[i][0], choice[i])
                choice[i] = nk
                steps += 1
            push(g)
        tile[key] = total
    print(f"graduated LOD: {steps:,} single-step reductions across {len(hot)} hot tile(s); small props "
          f"kept {PROP_LOD_FLOOR:.0%} of their finest LOD except in {unfloored_tiles} tile(s) still over")
    tile2 = collections.Counter()
    for (mesh, R, t, _), k in zip(inst, choice):
        if k is not None:
            tile2[tkey(t)] += tris(mesh, k)
    print(f"tiles over {args.tile_budget:,} triangles: {len(hot)} -> after coarsening "
          f"{sum(1 for v in tile2.values() if v > args.tile_budget)} (max {max(tile2.values()):,})")

    need = sorted({(inst[i][0], k) for i, k in enumerate(choice) if k is not None})
    guids = sorted({meshsets[m].lods[k].chunk for m, k in need})
    chunk = dict(zip(guids, decompress_many([cas.read(*CH[g][2:6]) for g in guids])))
    geom = {}
    for m, k in need:
        lod = meshsets[m].lods[k]
        parts = []
        for s in lod.sections:
            if s.is_shadow or s.primitives == 0 or classify(s.material)[0] in DROP_CLASSES:
                continue
            P, UV, T = meshset.decode(chunk[lod.chunk], lod, s)
            cls = classify(s.material)
            CLASS_SOURCES[cls[0]].add(s.material)
            parts.append((cls, s.material, P.astype(np.float64), UV, T, s.depot_key))
        OVERLAYS_LIFTED[0] += lift_overlays(parts, whole=bool(OVERLAY_MESH.search(m.rsplit('/', 1)[-1])))
        geom[(m, k)] = parts
    if OVERLAYS_LIFTED[0]:
        print(f"overlays: {OVERLAYS_LIFTED[0]:,} layered section(s) (decals, stickers, straps, trims lying "
              f"on their own mesh) lifted {OVERLAY_LIFT * 100:.1f} cm so they stop flashing")

    # World space, glTF handedness. Triangles are clipped to the crop: an instance can sit on
    # the island while its mesh does not -- iog_plt_11_base_a_decal_concrete_02 spans 3.2 km,
    # and one such mesh dragged the recentre 1.8 km out to sea.
    x0, z0, x1, z1 = crop
    placed = []     # (mesh, [(cls, material, P, UV, T)])
    SMOOTH_COLL = {}  # mesh -> (P, T) smoothed collision of a curved skate feature, mesh frame
    skate_coll = []  # (ride, W, T): skate.'s own collision, same frame and handedness as `placed`
    skate_coll_mesh = []
    placed_lod = {}  # mesh -> {LOD index used}
    placed_coll = []  # skate.'s collision parts of each `placed` instance, same order
    grind_ridge = []  # ('flag'|'thin', world points): grind tops traced from skate.'s collision
    clipped = 0
    # stacked sheets: same orientation (up axes within 8 deg), centres within SHEET_STACK_XZ m
    _sh = [i for i, ((m_, _R, _t, _), k_) in enumerate(zip(inst, choice))
           if k_ is not None and SHEET_PROP.search(m_.rsplit('/', 1)[-1])]
    sheet_under = set()
    for i in _sh:
        Ri, ti = inst[i][1], inst[i][2]
        ui = Ri[1] / max(np.linalg.norm(Ri[1]), 1e-9)
        for j in _sh:
            Rj, tj = inst[j][1], inst[j][2]
            uj = Rj[1] / max(np.linalg.norm(Rj[1]), 1e-9)
            if (j != i and np.hypot(*(ti - tj)[[0, 2]]) < SHEET_STACK_XZ and abs(ui @ uj) > 0.99
                    and (tj[1], j) > (ti[1], i)):
                sheet_under.add(i)
                break
    if sheet_under:
        print(f"stacked sheets: {len(sheet_under)} sheet(s) under another sheet drawn without collision")
    for _ii, ((mesh, R, t, _), k) in enumerate(zip(inst, choice)):
        if k is None:
            continue
        inst_coll = []
        smooth = SMOOTH_COLL.get(mesh) if args.smooth_ramps > 0 else None
        if smooth is None and args.smooth_ramps > 0 and CURVED_FEATURE.search(mesh.rsplit('/', 1)[-1])                 and PHYS.get(physics_name(mesh)):
            # The render LOD is ONE sheet with the same facets as skate.'s collision, which is
            # built from separate convex slices whose shared walls read as creases - so the
            # sheet is what gets smoothed (quarter pipe: 14 slope kinks over 3 deg -> 1, the
            # coping; shape moves at most 3.6 cm).
            secs_ = [(P_, T_) for _c, _m, P_, _uv, T_, _dk in geom[(mesh, k)] if len(T_)]
            if secs_:
                off, PP, TT = 0, [], []
                for P_, T_ in secs_:
                    PP.append(P_)
                    TT.append(T_ + off)
                    off += len(P_)
                smooth = SMOOTH_COLL[mesh] = loop_subdivide(np.concatenate(PP), np.concatenate(TT),
                                                           iters=args.smooth_ramps)
        parts_src = ([types.SimpleNamespace(positions=smooth[0], triangles=smooth[1])]
                     if smooth is not None else PHYS.get(physics_name(mesh), ()))
        # Movable props (a kind-6 rigid-body part) store their shape relative to a centre of mass
        # in the asset's body frame, which this parser does not apply: every picnic table, bench,
        # can and crate had its collision slab lying on the ground at its feet (the bumps
        # reported 2026-09-23). They get the hull of their drawn mesh instead.
        _dyn = any(getattr(p_, 'kind', 0) == 6 for p_ in (parts_src or ()))
        if (smooth is None and parts_src and collision_tier(mesh) == 0 and not STUB_SKIP.search(mesh)
                and (_dyn or STUB_ALLOW.search(mesh.rsplit('/', 1)[-1]))):
            _pp = np.concatenate([p_.positions for p_ in parts_src if len(p_.positions)] or [np.zeros((1, 3))])
            _rp = np.concatenate([edge_points(g_[2], g_[4], 0.25) for g_ in geom[(mesh, k)] if len(g_[4])]
                                 or [np.zeros((1, 3))])
            _secs = [(g_[2], g_[4]) for g_ in geom[(mesh, k)] if len(g_[4])]
            _pd = float(np.linalg.norm(_pp.max(0) - _pp.min(0)))
            _rd = float(np.linalg.norm(_rp.max(0) - _rp.min(0)))
            if _rd > 0.3 and (_dyn or _pd < STUB_RATIO * _rd):
                if mesh not in STUB_HULLS:
                    _o, _PP, _TT = 0, [], []
                    for _p, _t in _secs:
                        _PP.append(_p)
                        _TT.append(_t + _o)
                        _o += len(_p)
                    STUB_HULLS[mesh] = (component_hulls(np.concatenate(_PP), np.concatenate(_TT))
                                        if _PP else None) or stub_hull(_rp)
                if STUB_HULLS[mesh] is not None:
                    parts_src = [types.SimpleNamespace(positions=STUB_HULLS[mesh][0], triangles=STUB_HULLS[mesh][1],
                                                       material=0)]
        if _ii in sheet_under:
            parts_src = ()
        # Flat clutter lying on the ground (tarps, plywood sheets, trim, litter) under
        # FLAT_PROP_MAX m tall: its collision was a 1-8 cm lip on smooth ground that Skate 3's
        # wheels catch on (the worst ride-surface seams, 2026-09-23). Drawn, not collided.
        if smooth is None and collision_tier(mesh) == 0 and parts_src:
            _allp = np.concatenate([p_.positions for p_ in parts_src if len(p_.positions)] or [np.zeros((1, 3))])
            if float((_allp @ R)[:, 1].max() - (_allp @ R)[:, 1].min()) <= FLAT_PROP_MAX:
                FLAT_PENDING.append((mesh, parts_src, R, t, inst_coll))
                parts_src = ()
        for part in parts_src:
            if not len(part.triangles):
                continue
            Wc = part.positions @ R + t
            cc = Wc[part.triangles].mean(axis=1)
            kc = (cc[:, 0] > x0) & (cc[:, 0] < x1) & (cc[:, 2] > z0) & (cc[:, 2] < z1)
            if not kc.any():
                continue
            Tc = part.triangles[kc]
            if args.mirror_z:
                Wc[:, 2] = -Wc[:, 2]
            Tc = Tc if (np.linalg.det(R) < 0) == args.mirror_z else Tc[:, [0, 2, 1]]
            skate_coll.append((collision_tier(mesh), Wc, Tc))
            skate_coll_mesh.append(mesh)
            inst_coll.append((Wc, Tc))
            if is_grind_part(getattr(part, 'material', 0)):
                grind_ridge.append(('flag', edge_points(Wc, Tc)))
        if parts_src and smooth is None and collision_tier(mesh) == 0:
            allp = (np.concatenate([edge_points(p_.positions, p_.triangles) for p_ in parts_src if len(p_.positions)])
                    if parts_src else None)
            if allp is not None and len(allp):
                ext = allp.max(0) - allp.min(0)
                wide, long_ = sorted((ext[0], ext[2]))
                if (wide <= THIN_MAX_WIDTH and long_ >= THIN_MIN_LENGTH
                        and THIN_HEIGHT[0] <= ext[1] <= THIN_HEIGHT[1]
                        and not any(is_grind_part(getattr(p_, 'material', 0)) for p_ in parts_src)):
                    Wa = allp @ R + t
                    if args.mirror_z:
                        Wa[:, 2] = -Wa[:, 2]
                    grind_ridge.append(('thin', Wa))
        placed_lod.setdefault(mesh, set()).add(k)
        secs = []
        for cls, mat, P, UV, T, dk in geom[(mesh, k)]:
            W = P @ R + t
            c = W[T].mean(axis=1)
            keep = (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 2] > z0) & (c[:, 2] < z1)
            if not keep.all():
                clipped += int((~keep).sum())
                T = T[keep]
                if len(T) == 0:
                    continue
                used = np.unique(T)
                remap = np.full(len(W), -1, dtype=np.int64)
                remap[used] = np.arange(len(used))
                W, T = W[used], remap[T]
                UV = UV[used] if UV is not None else None
            if args.mirror_z:
                W[:, 2] = -W[:, 2]
            # A mirrored instance matrix (det < 0, 4 measured) flips winding once, and --mirror-z
            # flips it once more; reverse the triangles exactly when one of the two applies,
            # so faces stay outward (checked on a tetrahedron for all four combinations).
            secs.append((cls, mat, W, UV,
                         T if (np.linalg.det(R) < 0) == args.mirror_z else T[:, [0, 2, 1]], dk))
        if secs:
            placed.append((mesh, secs))
            placed_coll.append(inst_coll)

    if STUB_HULLS:
        print(f"stub collision: {sum(1 for v in STUB_HULLS.values() if v is not None):,} prop mesh(es) whose .physics "
              f"was a stub got a hull of their lowest {STUB_HULL_HEIGHT} m: "
              + ", ".join(m.rsplit('/', 1)[-1] for m, v in list(STUB_HULLS.items())[:8] if v is not None))
    _sup = support_index(skate_coll) if FLAT_PENDING else None
    for mesh, parts_src, R, t, inst_coll in FLAT_PENDING:
        pieces = []
        for part in parts_src:
            if not len(part.triangles):
                continue
            Wc = part.positions @ R + t
            if not (x0 < Wc[:, 0].mean() < x1 and z0 < Wc[:, 2].mean() < z1):
                continue
            if args.mirror_z:
                Wc[:, 2] = -Wc[:, 2]
            pieces.append((Wc, part.triangles if (np.linalg.det(R) < 0) == args.mirror_z
                           else part.triangles[:, [0, 2, 1]]))
        if (pieces and SHEET_PROP.search(mesh.rsplit('/', 1)[-1])
                and flat_unsupported(pieces, _sup) > FLAT_UNSUPPORTED):
            for Wc, Tc in pieces:
                skate_coll.append((0, Wc, Tc))
                skate_coll_mesh.append(mesh)
                inst_coll.append((Wc, Tc))
            FLAT_KEPT[mesh.rsplit('/', 1)[-1]] += 1
        else:
            FLAT_DROPPED[mesh.rsplit('/', 1)[-1]] += 1
    if FLAT_KEPT:
        print(f"flat clutter: {sum(FLAT_KEPT.values()):,} sheet(s) bridge a gap and keep their collision: "
              + ", ".join(f"{k} {v}" for k, v in FLAT_KEPT.most_common(6)))
    if FLAT_DROPPED:
        print(f"flat clutter: {sum(FLAT_DROPPED.values()):,} instance(s) under {FLAT_PROP_MAX * 100:.0f} cm drawn "
              f"without collision: " + ", ".join(f"{k} {v}" for k, v in FLAT_DROPPED.most_common(6)))
    # The drawn ride surface must be the surface you ride. skate. installs only far LODs for most
    # meshes (85% of the declared detail is not on disk) while the collision is its full-detail
    # shape, so the texture floated off the skating path: median 1 cm on asphalt and concrete,
    # 5-6 cm on grass and rock, p95 20-40 cm (tools/ride_audit.py). For ride geometry whose drawn
    # surface sits more than --visual-follows-collision metres off its own collision, the
    # collision shape is drawn instead, with the section's own material tiled at 2 m. A baked
    # far-LOD atlas cannot be tiled, so those take the material the mesh's finest LOD names.
    if args.visual_follows_collision > 0 and placed_coll:
        swapped, checked, tri_before, tri_after, devs = 0, 0, 0, 0, []

        def up_area(W, T):
            if not len(T):
                return 0.0
            n = np.cross(W[T[:, 1]] - W[T[:, 0]], W[T[:, 2]] - W[T[:, 0]])
            return float(np.clip(n[:, 1], 0, None).sum())

        forced = 0
        for i, ((mesh, secs), coll) in enumerate(zip(placed, placed_coll)):
            force = bool(coll) and bool(DRAW_FROM_COLLISION.search(mesh))
            if not force and (not coll or collision_tier(mesh) != 2
                              or CURVED_FEATURE.search(mesh.rsplit('/', 1)[-1])):
                continue
            samples = []
            for _cls, _mat, W, _UV, T, _dk in secs:
                if not len(T):
                    continue
                n = np.cross(W[T[:, 1]] - W[T[:, 0]], W[T[:, 2]] - W[T[:, 0]])
                ln = np.linalg.norm(n, axis=1)
                sel = (ln > 1e-9) & (n[:, 1] > 0.7 * ln)
                if sel.any():
                    samples.append(W[T[sel]].mean(axis=1))
            if not samples and not force:
                continue
            CW = np.concatenate([w for w, _ in coll])
            off, CTs = 0, []
            for w, t_ in coll:
                CTs.append(t_ + off)
                off += len(w)
            CT = np.concatenate(CTs)
            if force:
                d = np.zeros(1)
                forced += 1
            else:
                q = np.concatenate(samples)
                q = q[::max(1, len(q) // 200)]
                d = surface_distance(q, CW, CT)
                d = d[np.isfinite(d)]
                checked += 1
                if not len(d) or float(np.percentile(d, 90)) <= args.visual_follows_collision:
                    continue
            cls, mat, _W, _UV, _T, dk = max(secs, key=lambda sec: up_area(sec[2], sec[4]))
            kt = KEY_TEX.get(str(dk) if dk else '')
            if ATLAS_RX.search(mat.lower()) or (kt in ATLAS_BASES) or (MATERIAL_TEX.get(mat) in ATLAS_BASES):
                pm = PROXY_MAT.get(mesh)
                if pm:
                    mat, dk = pm, None
            used = np.unique(CT)
            remap = np.full(len(CW), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            TW, TUV, TT = triplanar(CW[used], remap[CT])
            tri_before += sum(len(sec[4]) for sec in secs)
            tri_after += len(TT)
            placed[i] = (mesh, [(cls, mat, TW, TUV.astype(np.float32), TT, dk)])
            if force:
                continue
            swapped += 1
            devs.append(float(np.percentile(d, 90)))
        print(f"visual follows collision: {swapped:,} of {checked:,} ride instances drawn from skate.'s collision "
              f"(their drawn surface sat over {args.visual_follows_collision * 100:.0f} cm off it, p90 median "
              f"{np.median(devs) * 100 if devs else 0:.1f} cm); render triangles {tri_before:,} -> {tri_after:,}"
              + (f"; {forced} water sheet(s) drawn from their collision" if forced else ''))

    # Where the 100 m grid falls is ours to choose - the recentre is arbitrary - and it decides
    # whether a dense cluster lands in one stream or straddles two. The first build with skate.'s
    # own collision shipped cPres_50_250 at 7.33 MB (a downtown cluster of props, each already on
    # its only resident LOD) and cSim_150_-50 at 5.05 MB (five 11k-triangle generated ledges),
    # while every other tile sat under 4.8 MB. Sliding the grid costs no geometry at all, so the
    # offset that minimises the worst tile is chosen before anything is thinned. Load is measured
    # the way the streams are: render by VERTICES (a render stream is ~40 B per vertex, fitted on
    # the shipped streams to 0.17 MB), collision by triangles (~31 B each).
    if args.grid_search:
        rc, rw = [], []
        for _m, _secs in placed:
            for _cls, _mat, W, _uv, T, _dk in _secs:
                if len(T):
                    rc.append(W[T].mean(axis=1)[:, [0, 2]])
                    rw.append(np.full(len(T), len(W) / len(T)))
        rc, rw = np.concatenate(rc), np.concatenate(rw)
        cc = (np.concatenate([W[T].mean(axis=1)[:, [0, 2]] for _r, W, T in skate_coll])
              if skate_coll else np.zeros((0, 2)))

        def loads(ox, oz):
            def mx(c, w=None):
                if not len(c):
                    return 0.0
                kx = np.floor((c[:, 0] + sx + ox) / TILE_SIZE).astype(np.int64)
                kz = np.floor((c[:, 1] + sz + oz) / TILE_SIZE).astype(np.int64)
                key = (kx - kx.min()) * 4096 + (kz - kz.min())
                return float(np.bincount(key, weights=w).max())
            return mx(rc, rw), mx(cc)

        def score(l):
            return max(l[0] / args.tile_vertex_budget, l[1] / args.collision_tile_budget)

        base = loads(0, 0)
        best = (score(base), 0, 0, base)
        for ox in range(0, int(TILE_SIZE), 5):
            for oz in range(0, int(TILE_SIZE), 5):
                l = loads(ox, oz)
                if score(l) < best[0] - 1e-9:
                    best = (score(l), ox, oz, l)
        _, ox, oz, l = best
        sx, sz = sx + ox, sz + oz
        print(f"grid offset ({ox}, {oz}) m: worst tile {base[0]:,.0f} -> {l[0]:,.0f} render vertices "
              f"(budget {args.tile_vertex_budget:,}), {base[1]:,.0f} -> {l[1]:,.0f} collision triangles "
              f"(budget {args.collision_tile_budget:,})")

    ys = np.concatenate([s[2][:, 1] for _, secs in placed for s in secs if s[0][2]])
    xs = np.concatenate([s[2][:, 0] for _, secs in placed for s in secs])
    zs = np.concatenate([s[2][:, 2] for _, secs in placed for s in secs])
    shift = np.array([sx, -float(np.percentile(ys, 0.5)) + 1.0, sz])
    print(f"clipped {clipped:,} triangles outside the crop; x {xs.min():.0f}..{xs.max():.0f} "
          f"z {zs.min():.0f}..{zs.max():.0f}; recentre shift {np.round(shift, 2)}")
    with open(os.path.join(args.out, 'recentre.json'), 'w') as fh:
        json.dump({'shift': [float(v) for v in shift], 'level': LEVEL}, fh)

    # Tile budget, measured on the FINAL 100 m tiles the arena builder will cut.
    for rnd in range(4):
        per_inst, totals, dec, col = [], collections.Counter(), collections.Counter(), collections.Counter()
        for mesh, secs in placed:
            tc = collections.Counter()
            for cls, mat, W, UV, T, dk in secs:
                cnt = tile_counts(W + shift, T)
                tc.update(cnt)
                if cls[0] not in PROTECTED:
                    dec.update(cnt)
                if cls[2]:
                    col.update(cnt)
            per_inst.append(tc)
            totals.update(tc)
        hot = {k: v for k, v in totals.items() if v > args.tile_budget}
        worst = totals.most_common(1)[0]
        print(f"  budget pass {rnd}: {len(totals)} tiles, {len(hot)} over {args.tile_budget:,}; worst "
              f"{worst[0]} {worst[1]:,} tris ~{worst[1] * PRES_BYTES_PER_TRI / 1e6:.1f} MB cPres, "
              f"max cSim ~{max(col.values()) * SIM_BYTES_PER_TRI / 1e6:.1f} MB")
        if not hot:
            break
        factor = {k: max(PROP_THIN_FLOOR, 1 - (totals[k] - args.tile_budget) / max(dec[k], 1)) for k in hot}
        nxt, nxt_c = [], []
        for (mesh, secs), tc, pc in zip(placed, per_inst, placed_coll):
            f = min((factor[k] for k in tc if k in factor), default=1.0)
            if f >= 0.999:
                nxt.append((mesh, secs))
                nxt_c.append(pc)
                continue
            ns = []
            for cls, mat, W, UV, T, dk in secs:
                if cls[0] not in PROTECTED:
                    W, UV, T = cluster_simplify(W, UV, T, max(1, int(len(T) * f)))
                if len(T):
                    ns.append((cls, mat, W, UV, T, dk))
            if ns:
                nxt.append((mesh, ns))
                nxt_c.append(pc)
        placed, placed_coll = nxt, nxt_c
    # A tile that receives only a handful of triangles still gets a declared CSPA cell and its
    # own cPres/cSim streams. On the Isle of Grom crop three such tiles shipped for 28 triangles
    # between them - one of them for THREE - as 2.5-3.3 KB collision streams, against a smallest
    # control stream of 11 KB. The engine then re-probed exactly those three on a ~5.2 s metronome
    # for the whole run, seven times each, while the four real tiles were read once and never
    # revisited. A duration-matched DM Jumpline boot (75 s past its world switch) re-probes
    # nothing, never wipes out and renders 12/12 frames, where ours renders 2/10. Dropping the
    # slivers costs 0.015% of the geometry, so the triangles go and the tile is never declared.
    pre_tris, pre_m2 = collections.Counter(), collections.Counter()
    for _mesh, _secs in placed:
        for _cls, _mat, W, _uv, T, _dk in _secs:
            Ws = W + shift
            cen = Ws[T].mean(axis=1)
            keys_ = list(zip((np.floor(cen[:, 0] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int).tolist(),
                             (np.floor(cen[:, 2] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int).tolist()))
            pre_tris.update(keys_)
            ar_ = 0.5 * np.linalg.norm(np.cross(Ws[T[:, 1]] - Ws[T[:, 0]], Ws[T[:, 2]] - Ws[T[:, 0]]), axis=1)
            for k_, a_ in zip(keys_, ar_.tolist()):
                pre_m2[k_] += a_
    # A sliver is few triangles AND little surface. Counting triangles alone dropped 35 edge tiles
    # of Summer Camp hillside once its rocks were drawn at their coarse physics LOD (500 triangles
    # can cover a whole cliff).
    sliver = ({tk for tk, n in pre_tris.items() if n < args.min_tile_triangles and pre_m2[tk] < SLIVER_M2}
              if args.min_tile_triangles > 0 else set())
    if sliver:
        print(f"dropping {len(sliver)} sliver tile(s) under {args.min_tile_triangles:,} tris: "
              f"{sorted(sliver)} -- {sum(pre_tris[t] for t in sliver)} triangles, "
              f"{100 * sum(pre_tris[t] for t in sliver) / max(1, sum(pre_tris.values())):.4f}% of the map")

    # Tiles the instance-level budget could not bring down are the ones dominated by
    # PROTECTED ground, and that geometry is exactly what must stay dense in the COLLISION
    # copy: a thinned floor is a hole the skater falls through. The 5 MB cPres ceiling is
    # measured on the VISUAL copy alone, though, so the visual copy is thinned on its own
    # here and collision keeps the full shell. Tile 50_250 sat at 92k triangles (~5.1 MB
    # estimated, 5.65 MB as built) through all four rounds for this reason.
    # Only non-ride classes are thinned: a thinned floor drawn over an unthinned collision
    # floor is a visible surface the board does not sit on.
    final_tiles, prot_tiles = collections.Counter(), collections.Counter()
    for _m, _secs in placed:
        for _c, _mat, W, _u, T, _dk in _secs:
            cnt = tile_counts(W + shift, T)
            final_tiles.update(cnt)
            if _c[0] in PROTECTED:
                prot_tiles.update(cnt)
    render_fix = {k: max(PROP_THIN_FLOOR, (args.tile_budget - prot_tiles[k]) / max(1, v - prot_tiles[k]))
                  for k, v in final_tiles.items() if v > args.tile_budget}
    if render_fix:
        still = [k for k in render_fix if prot_tiles[k] > args.tile_budget]
        print(f"render-only thinning of NON-ride classes on {len(render_fix)} tile(s) over "
              f"{args.tile_budget:,} tris (x{min(render_fix.values()):.2f}.."
              f"x{max(render_fix.values()):.2f}); ride surfaces untouched"
              + (f"; {len(still)} tile(s) over on ride geometry alone: {sorted(still)}" if still else ''))

    over_tiles = {k for k, v in final_tiles.items() if v > args.tile_budget}
    if over_tiles:
        contrib = collections.defaultdict(collections.Counter)
        for _m, _secs in placed:
            for _c, _mat, W, _u, T, _dk in _secs:
                for tk, n in tile_counts(W + shift, T).items():
                    if tk in over_tiles:
                        contrib[tk][(_m, _c[0])] += n
        for tk in sorted(over_tiles, key=lambda t: -final_tiles[t]):
            print(f"  over-budget tile {tk[0]}_{tk[1]}: {final_tiles[tk]:,} render tris; top contributors:")
            for (m, c), n in contrib[tk].most_common(10):
                av = lods(m)
                print(f"     {n:7,} {c:<9} LOD {sorted(placed_lod.get(m, []))} of resident {av}  {m.rsplit('/', 1)[-1]}")

    tile_tris = collections.Counter()
    rend_tris = collections.Counter()

    UV_RAW = raw_density(placed)
    _fix = {m: UV_TARGET / d for m, d in UV_RAW.items() if not 0.4 <= d <= 0.62}
    print(f"UV: measured {len(UV_RAW)} material/class pairs; {len(_fix)} off the "
          f"{UV_TARGET:.2f} tiles/m target get a per-material correction "
          f"(x{min(_fix.values(), default=1):.2f}..x{max(_fix.values(), default=1):.2f})")

    blocks = collections.defaultdict(lambda: collections.defaultdict(list))
    # The render copy is measured again after it is built, because decimation changes UV
    # density: welding vertices shrinks a patch's world area faster than its UV area, so a
    # thinned tile comes out finer than it went in. CliffRock_02 - 262,791 m2, the largest
    # single material on the island - was corrected onto 0.50 tiles/m and still exported at
    # 0.657. Density scales linearly with the UV factor, so one more multiply lands it exactly.
    load_fills(shift, args.mirror_z)
    if FILLS:
        print(f"terrain fills: {len(FILLS)} asphalt/concrete polygon(s) painted onto skate.'s terrain meshes")
    staged = []
    staged_owner = []   # index into `placed` of each staged entry (the fill pass needs it)
    post = collections.defaultdict(lambda: [0.0, 0.0])
    tile_y = {}
    splines = []
    collide_parts = []
    grind_acc, edge_splines = [], []
    for pi, (mesh, secs) in enumerate(placed):
        allP = np.concatenate([s[2] for s in secs]) + shift
        c = (allP.min(0) + allP.max(0)) / 2
        key = (math.floor(c[0] / args.block_size), math.floor(c[2] / args.block_size))
        for (cname, rgb, collide), mat, W, UV, T, dk in secs:
            W = W + shift
            cen = W[T].mean(axis=1)
            tu = (np.floor(cen[:, 0] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int)
            tv = (np.floor(cen[:, 2] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int)
            if sliver:
                keepm = np.fromiter(((int(a), int(b)) not in sliver
                                     for a, b in zip(tu.tolist(), tv.tolist())), bool, len(T))
                if not keepm.all():
                    T, cen, tu, tv = T[keepm], cen[keepm], tu[keepm], tv[keepm]
                    if len(T) == 0:
                        continue
            tile_tris.update(zip(tu.tolist(), tv.tolist()))
            # Tile bounds must cover exactly the tiles that RECEIVE triangles. Keying them off
            # instance bounding boxes declared 43 cells against 41 render / 40 collision streams;
            # a CSPA cell with no stream behind it is one the loader describes but never fills,
            # which the d2s3 README records as a load that never finishes. DM Jumpline, which
            # loads, matches 140/140/140 exactly.
            ylo, yhi = W[T][:, :, 1].min(axis=1), W[T][:, :, 1].max(axis=1)
            for tk in {(int(a), int(b)) for a, b in zip(tu, tv)}:
                sel = (tu == tk[0]) & (tv == tk[1])
                lo_, hi_ = float(ylo[sel].min()), float(yhi[sel].max())
                cur = tile_y.setdefault(tk, [lo_, hi_])
                cur[0] = min(cur[0], lo_)
                cur[1] = max(cur[1], hi_)
            if UV is None:
                UV = (W[:, [0, 2]] / 2.0).astype(np.float32)
            # The shader block is the identity; the material name is only a fallback for a
            # section whose block carries no colour slot.
            mkey, kstr = None, str(dk) if dk else ''
            atlas_mat = bool(ATLAS_RX.search(mat.lower()))
            tbase = None
            if cname != COLLISION_CLASS:
                tbase = KEY_TEX.get(kstr)
                if atlas_mat:
                    # `m_proxy_atlas` is one name over 237 meshes, so binding it BY NAME is
                    # meaningless - it resolves to a near-white aluminium (mean 242) and put
                    # that on 27,644 m2 of plate. The mesh's own finest LOD names the real
                    # material; where it does not, the class texture is the honest answer.
                    pm = PROXY_MAT.get(mesh)
                    if tbase is None and pm:
                        tbase = MATERIAL_TEX.get(pm)
                elif tbase is None:
                    tbase = MATERIAL_TEX.get(mat)
                # skate. shades water procedurally; its shader block binds only the effects
                # placeholder (`t_vfx_default_cu`), which drew the river as a grey smear.
                if cname == 'Water' and tbase and re.search(r'(^|/)effects/|t_vfx_default', tbase):
                    tbase = None
            decal_mesh = False
            if tbase:
                mkey = 'tex:' + tbase
                if mkey not in TEXTURE_BLOBS:
                    TEXTURE_BLOBS[mkey] = TEX_BLOBS[tbase]
                # A mesh section painted with decal art (concrete patches, cracks, stains) is a
                # cut-out in skate.: its colour map is opaque and the shape is in the _osk mask.
                # Drawn as a plain material it was a solid dark rectangle. Use the decal bake.
                dk_ = decal_key_for_base(tbase.split('|')[0]) if args.decals else None
                if dk_ and LINE_DECAL.search(dk_) and not collide:
                    continue        # ground grime / hairline art: see LINE_DECAL
                if dk_:
                    mkey, cname, decal_mesh = dk_, DECAL_CLASS, True
            # skate.'s slab/panel sheet materials (tools/bake_materials.py): one baked sheet
            # repeat per material on the section's own uv0 - never rescaled
            seg_mesh = False
            if kstr in SEG_MAT and cname != COLLISION_CLASS and not atlas_mat:
                mkey, seg_mesh = rock_key(SEG_MAT[kstr]['png']), True
            # A baked atlas's UVs index a packed sheet: rescaling them to a tiling target
            # would sample the wrong cell of the sheet. Only library materials get corrected.
            baked = tbase in ATLAS_BASES or decal_mesh or seg_mesh
            # A prop's own texture (cmn_prp_rug_01_c, stickers_01_c: no `_2x2m` in the name) is a
            # unique unwrap, not a tiling material: rescaling its UVs crops or repeats the picture.
            fixed_uv = baked or bool(tbase and not LIBRARY_TEX_RX.search(tbase.split('|')[0])
                                     and 'TerrainMesh_FetchVirtualTexture' not in mat)
            f = None if fixed_uv else uv_correction(UV_RAW, mat, cname)
            UV_RAW_SEEN[(mat, cname)] = UV_RAW.get((mat, cname))
            UV_FACTOR[(mat, cname)] = f
            if f is not None and UV is not None:
                UV = (UV * f).astype(np.float32)
            # The visual copy may be thinned past the collision copy (see render_fix).
            vW, vUV, vT = W, UV, T
            if render_fix and cname not in PROTECTED:
                rf = min((render_fix.get((int(a), int(b)), 1.0)
                          for a, b in zip(tu.tolist(), tv.tolist())), default=1.0)
                if rf < 0.999:
                    vW, vUV, vT = cluster_simplify(W, UV, T, max(1, int(len(T) * rf)))
                    if not len(vT):
                        vW, vUV, vT = W, UV, T
            atlas = atlas_mat and not baked
            if atlas:
                # render copy only: collision and grind detection keep the indexed shell
                rW, rUV, rT = triplanar(vW, vT)
                tex = None
            elif 'TerrainMesh_FetchVirtualTexture' in mat and not baked and cname != COLLISION_CLASS:
                # skate. paints terrain meshes from its virtual texture, which it projects in
                # WORLD space at each layer's tiling (0.5 tiles/m: one 2x2 m texture per 2 m,
                # tag 0xD591). The mesh's own uv0 is not a tiling layout - drawn with it, sand
                # and grass smeared one texture across a whole hillside (user, 2026-09-23).
                rW, rUV, rT = triplanar(vW, vT, scale=2.0)
                tex = TEXTURES.get(mesh)
            else:
                rW, rUV, rT = vW, vUV, vT
                tex = None if cname == COLLISION_CLASS else TEXTURES.get(mesh)
            tk = mkey or (tex[0] if tex else None)
            if tk is None and cname != COLLISION_CLASS:
                _t = KEY_TINT.get(kstr)
                if _t:
                    tk = 'paint:' + ','.join(f'{v:.3f}' for v in _t)
            rk = ROCK_MAT.get(kstr) if (kstr and cname != COLLISION_CLASS and not atlas and not baked) else None
            if FILLS and len(rT) and 'TerrainMesh_FetchVirtualTexture' in mat and not atlas and not baked:
                # skate.'s terrain virtual texture paints asphalt/concrete fills (roads, paths)
                # onto these meshes: triangles inside a fill take its texture (last draw wins)
                cen_ = rW[rT].mean(axis=1)[:, [0, 2]]
                which = np.full(len(rT), -1)
                for fi, (poly, _png, bb) in enumerate(FILLS):
                    m_ = ((cen_[:, 0] >= bb[0]) & (cen_[:, 0] <= bb[2]) & (cen_[:, 1] >= bb[1]) & (cen_[:, 1] <= bb[3]))
                    if m_.any():
                        idx_ = np.nonzero(m_)[0]
                        inside = point_in_poly(cen_[idx_], poly)
                        which[idx_[inside]] = fi
                for fi in np.unique(which):
                    sel_ = which == fi
                    tk_ = tk if fi < 0 else rock_key(FILLS[fi][1])
                    staged.append((key, cname, tk_, mat, atlas, baked, rW, rUV, rT[sel_]))
                    staged_owner.append(pi)
                    if fi >= 0:
                        FILL_USED[FILLS[fi][1]] += int(sel_.sum())
            elif rk and len(rT):
                # skate.'s rock material: rock slice on steep faces, the grass slice (tag 0xAAEF)
                # on faces flatter than its slope angle (tag 0xC89D) - tools/bake_materials.py
                fn_ = np.cross(rW[rT[:, 1]] - rW[rT[:, 0]], rW[rT[:, 2]] - rW[rT[:, 0]])
                ln_ = np.maximum(np.linalg.norm(fn_, axis=1), 1e-12)
                ang = float(rk['slope_deg'])
                flat = (fn_[:, 1] / ln_ >= np.cos(np.radians(ang))) if ang < 90 else np.zeros(len(rT), bool)
                for sel_, png_ in ((~flat, rk['base']), (flat, rk['top'])):
                    if sel_.any():
                        staged.append((key, cname, rock_key(png_), mat, atlas, baked, rW, rUV, rT[sel_]))
                        staged_owner.append(pi)
                        ROCK_USED[png_] += int(sel_.sum())
            else:
                staged.append((key, cname, tk, mat, atlas, fixed_uv, rW, rUV, rT))
                staged_owner.append(pi)
            if not fixed_uv:
                w_, u_ = uv_world_ratio(rW, rUV, rT)
                p_ = post[(mat, cname)]
                p_[0] += w_
                p_[1] += u_
            if physics_name(mesh) in PHYS or (PHYS and args.collision_source == 'skate'):
                pass    # skate.'s own collision, or skate. gives this mesh none
            elif collide:
                collide_parts.append((cname in PROTECTED, W, T))
            elif cname not in NEVER_COLLIDE and len(T):
                # Safety net for a misclassified floor. Next to the crop's spawn every one of
                # the 201 cells with visible ground but no collision belonged to IOG_Sign: 63
                # triangles carrying 2,025 m2 of up-facing surface, non-collidable only because
                # the class table says signage is. A class name is a guess about geometry, and
                # a wrong guess here is a hole the skater falls through - so anything large,
                # flat and facing up gets collision whatever its class claims.
                nrm = np.cross(W[T[:, 1]] - W[T[:, 0]], W[T[:, 2]] - W[T[:, 0]])
                ln = np.linalg.norm(nrm, axis=1)
                flat = (ln > 2.0 * FLOOR_TRI_AREA) & (nrm[:, 1] > 0.5 * ln)
                if flat.any():
                    collide_parts.append((False, W, T[flat]))
                    RESCUED[cname] += int(flat.sum())
            if cname == 'Rail' and 'localrail' in mesh:
                splines.extend(rail_polylines(W, T))
            elif (args.grind_source == 'render' and cname in GRIND_CLASSES and len(T)
                    and 'terrain' not in mesh and not atlas_mat):
                grind_acc.append((W, T))
        # One grind pass per instance over all its man-made sections together, so an edge
        # between two materials (a concrete top on a metal side) is still found.
        if grind_acc:
            off, gp, gt = 0, [], []
            for w_, t_ in grind_acc:
                gp.append(w_); gt.append(t_ + off); off += len(w_)
            try:
                got = d2s3_splines.grind_edges(
                    np.concatenate(gp).astype(np.float64), np.concatenate(gt).astype(np.int64),
                    min_angle=GRIND_MIN_ANGLE, max_angle=GRIND_MAX_ANGLE,
                    min_length=GRIND_MIN_LEN)
            except Exception:  # noqa: BLE001
                got = []
            edge_splines.extend(np.asarray(l, dtype=np.float64) for l in got if len(l) >= 2)
            grind_acc = []

    if args.fill_hidden_collision > 0 and placed_coll:
        fill_hidden_collision(args, staged, staged_owner, placed, placed_coll, shift, tile_y, sliver)

    trim = {}
    for k, (wa, ua) in post.items():
        if wa >= UV_MEASURE_MIN_M2 and ua > 0:
            d = (ua / wa) ** 0.5
            if d > 0 and not 0.49 <= d <= 0.51:
                trim[k] = float(np.clip(UV_TARGET / d, UV_FIX_LO, UV_FIX_HI))
    GLOBAL_TEX = {}
    if args.tile_textures:
        GLOBAL_TEX = plan_global_textures(staged, args)
    GLOBAL_TEX_KEYS.clear()
    GLOBAL_TEX_KEYS.update(GLOBAL_TEX)
    for key, cname, tk, mat, atlas, baked, rW, rUV, rT in staged:
        t = None if baked else trim.get((mat, cname))
        if t is not None and rUV is not None:
            rUV = (rUV * t).astype(np.float32)
            UV_TRIM[(mat, cname)] = t
        rend_tris.update(tile_counts(rW, rT))
        uv_account(mat, cname, atlas, tk is not None, baked, rW, rUV, rT)
        # A proxy shell with its own baked sheet (`baked`) is still far-LOD scenery, so it keeps
        # the `__proxylod` tag the spawn search uses to refuse it.
        if (args.tile_textures and tk and tk.startswith(('tex:', 'rip:', 'dec:'))
                and tk not in GLOBAL_TEX and len(rT)):
            # one texture copy per streaming tile: a texture no two tiles share is never promoted
            # into the always-resident cPres_Global, so it can stream at full resolution
            cen = rW[rT].mean(axis=1)
            tu_ = np.floor(cen[:, 0] / TILE_SIZE).astype(np.int64)
            tv_ = np.floor(cen[:, 2] / TILE_SIZE).astype(np.int64)
            for u_, v_ in set(zip(tu_.tolist(), tv_.tolist())):
                sel = (tu_ == u_) & (tv_ == v_)
                Ts_ = rT[sel]
                used_ = np.unique(Ts_)
                remap_ = np.full(len(rW), -1, dtype=np.int64)
                remap_[used_] = np.arange(len(used_))
                blocks[key][(cname, f'{tk}@{u_}_{v_}', atlas or (baked and not (tk or '').startswith('tex:SEG_')))].append(
                    (rW[used_], rUV[used_] if rUV is not None else None, remap_[Ts_]))
        else:
            blocks[key][(cname, tk, atlas or (baked and not (tk or '').startswith('tex:SEG_')))].append((rW, rUV, rT))
    staged.clear()
    if args.decals:
        _dec = load_decals(crop, args.mirror_z)
        if _dec:
            emit_decals(_dec, blocks, shift, tile_y, sliver, rend_tris, args.block_size, args.decal_snap,
                        tile_keys=args.tile_textures, cutout=args.decal_cutout)
    print(f"UV: second pass re-scaled {len(trim)} material/class pair(s) measured on the "
          f"FINAL geometry (x{min(trim.values(), default=1):.2f}..x{max(trim.values(), default=1):.2f})")

    over = sum(1 for v in tile_tris.values() if v > args.tile_budget)
    print(f"{TILE_SIZE:g} m tiles {len(tile_tris)}; over {args.tile_budget:,} tris: {over}; heaviest "
          f"{[(f'{k[0]}_{k[1]}', v) for k, v in tile_tris.most_common(6)]}")
    rworst = rend_tris.most_common(3)
    print(f"  render copy: heaviest {[(f'{k[0]}_{k[1]}', v) for k, v in rworst]}; "
          f"biggest cPres ~{(rworst[0][1] if rworst else 0) * PRES_BYTES_PER_TRI / 1e6:.2f} MB")
    if RESCUED:
        print("collision safety net rescued "
              + ", ".join(f"{n:,} {c}" for c, n in RESCUED.most_common()))
    with open(os.path.join(args.out, 'class_sources.json'), 'w') as fh:
        json.dump({k: sorted(v) for k, v in sorted(CLASS_SOURCES.items())}, fh, indent=1)

    # skate.'s collision joins the collision copy only in tiles that also draw something, so a
    # declared cell always has both streams behind it (see the sliver-tile note above).
    grind_coll = []
    if skate_coll:
        kept_c = dropped_c = 0
        coll_contrib = collections.defaultdict(collections.Counter)
        for (ride, W, T), cm in zip(skate_coll, skate_coll_mesh):
            W = W + shift
            cen = W[T].mean(axis=1)
            tu = (np.floor(cen[:, 0] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int)
            tv = (np.floor(cen[:, 2] / TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2).astype(int)
            ok = np.fromiter(((int(a), int(b)) in tile_y for a, b in zip(tu.tolist(), tv.tolist())),
                             bool, len(T))
            dropped_c += int((~ok).sum())
            if not ok.any():
                continue
            T, tu, tv = T[ok], tu[ok], tv[ok]
            kept_c += len(T)
            for tk2, n2 in collections.Counter(zip(tu.tolist(), tv.tolist())).items():
                coll_contrib[tk2][cm] += n2
            ylo, yhi = W[T][:, :, 1].min(axis=1), W[T][:, :, 1].max(axis=1)
            for tk in {(int(a), int(b)) for a, b in zip(tu, tv)}:
                sel = (tu == tk[0]) & (tv == tk[1])
                cur = tile_y[tk]
                cur[0] = min(cur[0], float(ylo[sel].min()))
                cur[1] = max(cur[1], float(yhi[sel].max()))
            collide_parts.append((ride, W, T))
            # Grind edges come from the surface you RIDE, not the one you see. Render meshes are
            # often a far-LOD proxy shell (a plaza's ledges exist only as a silhouette, and atlas
            # shells were excluded from edge detection outright), and even real render LODs sit
            # centimetres off skate.'s collision. Ride geometry and clutter (ledges, benches,
            # planters, barriers) qualify; rocks and terrain are creases, not rails, and the
            # authored localrail rails already have their own splines.
            if (args.grind_source == 'collision' and ride != 1 and len(T)
                    and 'terrain' not in cm and 'localrail' not in cm):
                grind_coll.append((W, T))
        print(f"skate. collision: {kept_c:,} triangles placed, {dropped_c:,} outside the drawn tiles dropped")
        if grind_coll:
            n_before = len(edge_splines)
            for W, T in grind_coll:
                try:
                    got = d2s3_splines.grind_edges(W.astype(np.float64), T.astype(np.int64),
                                                   min_angle=GRIND_MIN_ANGLE, max_angle=GRIND_MAX_ANGLE,
                                                   min_length=GRIND_MIN_LEN)
                except Exception:  # noqa: BLE001
                    got = []
                edge_splines.extend(np.asarray(l, dtype=np.float64) for l in got if len(l) >= 2)
            print(f"grind edges from skate.'s collision: {len(edge_splines) - n_before:,} convex 60-120 degree "
                  f"edges over {len(grind_coll):,} collision parts")
        for tk2, cnt in sorted(coll_contrib.items(), key=lambda kv: -sum(kv[1].values())):
            if sum(cnt.values()) <= args.collision_tile_budget:
                continue
            print(f"  over-budget collision tile {tk2[0]}_{tk2[1]}: {sum(cnt.values()):,} tris; top contributors:")
            for m, n in cnt.most_common(8):
                print(f"     {n:7,} {'ride' if RIDE_COLLISION.search(m) else 'prop'}  {m.rsplit('/', 1)[-1]}")

    if args.seat_lip > 0:
        seated, welded = seat_and_weld(collide_parts, lip=args.seat_lip, weld=args.weld, flat_lip=args.seam_lip)
        print(f"collision: seated {seated:,} ramp/bank toe vertices onto the surface under them "
              f"(lips up to {args.seat_lip * 100:.0f} cm), welded {welded:,} vertices across pieces")
    if os.environ.get('SKATE_DUMP_COLL'):
        # the collision before seam ramps, for offline seam work (work/scratch/ramp_test.py)
        _t = [int(p[0]) if not isinstance(p[0], bool) else -1 for p in collide_parts]
        np.savez_compressed(os.environ['SKATE_DUMP_COLL'], tier=np.array(_t),
                            nv=np.array([len(p[1]) for p in collide_parts]),
                            nt=np.array([len(p[2]) for p in collide_parts]),
                            W=np.concatenate([np.asarray(p[1], np.float32) for p in collide_parts]),
                            T=np.concatenate([np.asarray(p[2], np.int32) for p in collide_parts]))
    if args.seam_ramps > 0:
        ramps = ramp_seams(collide_parts, hi=args.seam_ramps)
        for W, T in ramps:
            collide_parts.append((True, W, T))
        print(f"collision: {sum(len(T) for _, T in ramps) // 2:,} seam ramp(s) ({RAMP_STATS}) of <=2.5 deg, >=1 m over 1.5-"
              f"{args.seam_ramps * 100:.0f} cm drops between rideable surfaces")

    # Collision copy: one mesh per 100 m tile, ride surfaces kept whole and everything else
    # decimated to fit the budget, routed collision-only.
    per_tile = collections.defaultdict(list)
    for protected, W, T in collide_parts:
        cen = W[T].mean(axis=1)
        u = np.floor(cen[:, 0] / TILE_SIZE).astype(int)
        v = np.floor(cen[:, 2] / TILE_SIZE).astype(int)
        for k in {(int(a), int(b)) for a, b in zip(u, v)}:
            per_tile[k].append((protected, W, T[(u == k[0]) & (v == k[1])]))
    if args.prune_unreachable:
        pruned = 0
        for k, parts in per_tile.items():
            if not parts:
                continue
            Pk = [W for _, W, _ in parts]
            off, Tk, owner = 0, [], []
            for j, (_, W, T) in enumerate(parts):
                Tk.append(T + off)
                owner.append(np.full(len(T), j))
                off += len(W)
            drop = unreachable_faces(np.concatenate(Pk), np.concatenate(Tk))
            if drop.any():
                own = np.concatenate(owner)
                parts[:] = [(p, W, T[~drop[own == j]]) for j, (p, W, T) in enumerate(parts)]
                pruned += int(drop.sum())
        print(f"collision: {pruned:,} unreachable faces pruned (downward with no floor within 10 m "
              f"below, or sideways, roofed over and floorless: nothing a skater can reach)")
    before = {k: sum(len(t) for _, _, t in parts) for k, parts in per_tile.items()}
    after = {}
    over_ride = []
    prop_cut = []
    qem_tiles = []
    # Skate 3 keeps the collision of the 3x3 tiles around the player resident in one pool, and sits
    # on the loading screen when it overflows: 14.87 MB over nine tiles loaded, 15.05 MB did not.
    # One stream alone also stalls above ~3.05 MB. Planning from triangles alone missed by 7-14%,
    # because the reduction changes vertex density - so the plan is CLOSED-LOOP: reduce, measure
    # the result (15.3 B/triangle + 28.1 B/vertex, fitted on 39 shipped streams), and tighten the
    # heaviest tile of every window still over, until all fit.
    reduced = {}
    if args.collision_source == 'skate':
        bpt, bpv = args.collision_bytes_per_tri, args.collision_bytes_per_vert

        def tv(prts):
            return (sum(len(T) for _, _, T in prts),
                    sum(len(np.unique(T)) for _, _, T in prts if len(T)))

        orig = {k: tv(prts) for k, prts in per_tile.items()}
        est = {k: bpt * t + bpv * v for k, (t, v) in orig.items()}
        alloc = {k: t for k, (t, v) in orig.items()}
        cap, limit = args.collision_stream_mb * 1e6, args.collision_window_mb * 1e6

        def window(c):
            return [(c[0] + dx, c[1] + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)
                    if (c[0] + dx, c[1] + dz) in alloc]

        def current_tris(k):
            return tv(reduced[k][1])[0] if k in reduced else orig[k][0]

        for k in alloc:
            if est[k] > cap:
                alloc[k] = int(orig[k][0] * cap / est[k] * 0.98)
        rounds = 0
        for rounds in range(1, 25):
            for k in alloc:
                if alloc[k] < orig[k][0] and (k not in reduced or reduced[k][0] != alloc[k]):
                    prts, errs, tier_counts = reduce_tile(per_tile[k], alloc[k], args.prop_collision_floor,
                                                     args.rock_collision_floor)
                    reduced[k] = (alloc[k], prts, errs, tier_counts)
                    t_, v_ = tv(prts)
                    est[k] = bpt * t_ + bpv * v_
            wins = sorted(((sum(est[t] for t in window(c)), c) for c in alloc), reverse=True)
            over_stream = [k for k in alloc if est[k] > cap]
            if wins[0][0] <= limit and not over_stream:
                break
            # Tighten from what the tile ACTUALLY holds or was allowed, whichever is smaller: a tile
            # whose clutter and rocks already sit at their floors keeps more than its allocation,
            # and basing the next step on that count stalled the loop over budget (15.18 MB).
            want = {}
            for w_, c in wins:
                if w_ <= limit:
                    break
                # A tile already pinned at its reduction floors keeps more than it is allotted;
                # cutting it again changes nothing, and always picking it stalled the loop at
                # 15.19 MB. Take the heaviest tile in the window that still honours its budget.
                can = [t for t in window(c) if alloc[t] > 1 and current_tris(t) <= alloc[t] * 1.05]
                heavy = max(can or window(c), key=lambda t: est[t])
                frac = min(0.6, (w_ - limit) / est[heavy])
                base = min(alloc[heavy], current_tris(heavy))
                want[heavy] = min(want.get(heavy, 1 << 60), int(base * (1 - frac) * 0.98))
            for k in over_stream:
                base = min(alloc[k], current_tris(k))
                want[k] = min(want.get(k, 1 << 60), int(base * cap / est[k] * 0.98))
            changed = False
            for k, nt in want.items():
                nt = max(1, min(nt, int(min(alloc[k], current_tris(k)) * 0.98)))
                if nt < alloc[k]:
                    alloc[k] = nt
                    changed = True
            if not changed:
                break
        worst = max((sum(est[t] for t in window(c)), c) for c in alloc)
        print(f"collision: {rounds} planning round(s); heaviest 3x3 window {worst[0] / 1e6:.2f} MB around "
              f"{int(worst[1][0] * TILE_SIZE + TILE_SIZE / 2)}_{int(worst[1][1] * TILE_SIZE + TILE_SIZE / 2)} (target {args.collision_window_mb:.1f}), "
              f"heaviest stream {max(est.values()) / 1e6:.2f} MB (cap {args.collision_stream_mb:.2f}); "
              f"{len(reduced)} tile(s) reduced")
    for k, parts in per_tile.items():
        if k in reduced:
            _, parts, errs, tier_before = reduced[k]
            qem_tiles.append((k, tier_before, sum(len(T) for _, _, T in parts), errs))
        keep = [(W, T) for p, W, T in parts if p and len(T)]
        rest = [(W, T) for p, W, T in parts if not p and len(T)]
        budget = max(args.collision_tile_budget - sum(len(T) for _, T in keep), 0)
        have = sum(len(T) for _, T in rest)
        f = 1.0 if have <= budget else max(budget / have, 0.02)
        merged = []
        for W, T in rest:
            if f < 0.999:
                if args.collision_source == 'skate':
                    W, T = qem_simplify(W, T, max(1, int(len(T) * f)))
                else:
                    W, _, T = cluster_simplify(W, None, T, max(1, int(len(T) * f)))
            merged.append((W, T))
        if f < 0.999:
            prop_cut.append((k, round(1 - f, 2)))
        # Ground is thinned only when clutter alone cannot bring the tile under budget, and
        # never past GROUND_FLOOR - a floor with holes is worse than a fat tile.
        kept_n = sum(len(T) for _, T in keep)
        gf = 1.0
        if args.collision_source == 'skate':
            if kept_n > args.collision_tile_budget:
                over_ride.append((k, kept_n))
        elif kept_n + sum(len(T) for _, T in merged) > args.collision_tile_budget and kept_n:
            gf = max(GROUND_FLOOR, (args.collision_tile_budget - sum(len(T) for _, T in merged)) / kept_n)
        for W, T in keep:
            if gf < 0.999:
                W, _, T = cluster_simplify(W, None, T, max(1, int(len(T) * gf)))
            merged.append((W, T))
        off, Ps, Ts = 0, [], []
        for W, T in merged:
            if len(T) == 0:
                continue
            Ps.append(W)
            Ts.append(T + off)
            off += len(W)
        if not Ps:
            continue
        P, T = np.concatenate(Ps), np.concatenate(Ts)
        after[k] = len(T)
        c = (P.min(0) + P.max(0)) / 2
        bk = (math.floor(c[0] / args.block_size), math.floor(c[2] / args.block_size))
        blocks[bk][(COLLISION_CLASS, None, False)].append((P, (P[:, [0, 2]] / 2.0).astype(np.float32), T))
    if args.fill_hidden_collision > 0:
        fill_final_collision(blocks, args, rend_tris)
    for k, tiers0, n1, errs in sorted(qem_tiles, key=lambda x: -sum(x[1].values())):
        dev = '; '.join(f"{g} p99 {e[0] * 100:.1f} cm max {e[1] * 100:.1f} cm" for g, e in errs.items())
        print(f"collision: tile {int(k[0] * TILE_SIZE + TILE_SIZE / 2)}_{int(k[1] * TILE_SIZE + TILE_SIZE / 2)} ride {tiers0['ride']:,} rock {tiers0['rock']:,} "
              f"prop {tiers0['prop']:,} -> {n1:,} tris ({'ride exact; ' if 'ride' not in errs else ''}{dev})")
    if qem_tiles:
        print(f"collision: {sum(1 for t in qem_tiles if 'ride' not in t[3])} of {len(qem_tiles)} reduced tile(s) "
              f"keep skate.'s ride collision exactly, {sum(1 for t in qem_tiles if 'rock' not in t[3])} keep rocks exactly")
    if prop_cut:
        print(f"collision: prop collision reduced by edge collapse in {len(prop_cut)} tile(s): "
              f"{sorted(prop_cut, key=lambda x: -x[1])[:8]}")
    if over_ride:
        print(f"!! {len(over_ride)} tile(s) carry more RIDE collision than --collision-tile-budget, "
              f"kept whole: {sorted(over_ride, key=lambda x: -x[1])[:6]}")
    print(f"collision: {sum(after.values()):,} tris over {len(after)} tiles "
          f"(was {sum(before.values()):,}); worst tile {max(after.values(), default=0):,} "
          f"~{max(after.values(), default=0) * SIM_BYTES_PER_TRI / 1e6:.2f} MB")

    # Island-wide surface each texture actually covers, on the final render geometry. Textures used
    # by two or more tiles are promoted into the always-resident cPres_Global (11 MB limit), and at
    # 50 m tiles far more of them are shared: 439 promoted images, 342 of them 128 px, normals ~3.8 MB.
    # A normal map on a material covering a few hundred square metres buys almost nothing visible,
    # so below --normal-min-m2 a texture falls back to its class normal (still shared, one per class).
    tex_area = collections.Counter()
    tex_tiles = collections.defaultdict(set)
    for _bk, _classes in blocks.items():
        for (_cn, _tk, _px), _parts in _classes.items():
            if _tk:
                for _W, _UV, _T in _parts:
                    if len(_T):
                        tex_area[_tk] += float(0.5 * np.linalg.norm(np.cross(
                            _W[_T[:, 1]] - _W[_T[:, 0]], _W[_T[:, 2]] - _W[_T[:, 0]]), axis=1).sum())
                        _c = _W[_T].mean(axis=1)
                        tex_tiles[_tk].update(zip(np.floor(_c[:, 0] / TILE_SIZE).astype(int).tolist(),
                                                  np.floor(_c[:, 2] / TILE_SIZE).astype(int).tolist()))
    # Measured on the 50 m build: cPres_Global held 241 promoted textures at 33 KB and 69 at 25 KB
    # (9.7 of 11.05 MB), so the pool is the long tail of small shared textures, not the big ones.
    # A texture used by two or more tiles and covering under --shared-small-texture-m2 is served at
    # 64 px; single-tile textures are left alone, since they never reach the global pool.
    small_shared = {tk for tk, a in tex_area.items()
                    if a < args.shared_small_texture_m2 and len(tex_tiles[tk]) >= 2
                    and not (tk or '').startswith('dec:')}

    # Budgeted resolution for the shared pool. With full-resolution mips reachable (the TOC
    # chunk-table fix) a flat cap put 39 MB into cPres_Global. Instead every shared texture starts
    # at 64 px and the one buying the most visible area per extra byte doubles, repeatedly, up to
    # its native size, until the pool estimate reaches --shared-texture-budget-mb.
    shared_edge = {}
    if args.shared_texture_budget_mb > 0:
        import heapq as _hq
        shared = [tk for tk in tex_area if len(tex_tiles[tk]) >= 2 and tk in TEXTURE_BLOBS]

        def _native(blob):
            return max(Image.open(io.BytesIO(blob)).size)

        def _cost(tk, e):
            n = 1 if (tk[4:] in TEX_NORMALS and tex_area.get(tk, 0.0) >= args.normal_min_m2) else 0
            return (SHARED_IMAGE_FIXED + e * e * SHARED_BYTES_PER_PX) * (1 + n)

        nat = {tk: _native(TEXTURE_BLOBS[tk]) for tk in shared}
        for tk in shared:
            shared_edge[tk] = min(64, nat[tk])
        left = args.shared_texture_budget_mb * 1e6 - sum(_cost(tk, shared_edge[tk]) for tk in shared)
        heap = [(-tex_area[tk] / max(1.0, _cost(tk, 2 * shared_edge[tk]) - _cost(tk, shared_edge[tk])), tk)
                for tk in shared if 2 * shared_edge[tk] <= nat[tk]]
        _hq.heapify(heap)
        while heap:
            _, tk = _hq.heappop(heap)
            e = shared_edge[tk]
            extra = _cost(tk, 2 * e) - _cost(tk, e)
            if extra > left:
                continue
            left -= extra
            shared_edge[tk] = 2 * e
            if 4 * e <= nat[tk]:
                _hq.heappush(heap, (-tex_area[tk] / max(1.0, _cost(tk, 4 * e) - _cost(tk, 2 * e)), tk))
        small_shared = set()
        hist = collections.Counter(shared_edge.values())
        print(f"shared texture budget {args.shared_texture_budget_mb:.1f} MB over {len(shared)} shared textures: "
              + ", ".join(f"{n} at {e} px" for e, n in sorted(hist.items(), reverse=True)))

    # Per-tile textures (--tile-textures). Inside each tile the texture covering the most area per
    # extra byte doubles, up to min(native, --tile-texture-px), until the tile's texture bytes reach
    # its budget: --tile-stream-mb minus the tile's own mesh bytes, clamped to
    # [0.5, --tile-texture-budget-mb] MB. Same image cost model as the global pool.
    tile_edge = {}
    if args.tile_textures:
        for tk, e in GLOBAL_TEX.items():
            shared_edge[tk] = e
        small_shared = set()
        import heapq as _hq2
        by_tile = collections.defaultdict(list)
        for tk in tex_area:
            if tk and '@' in tk and tk.split('@')[0] in TEXTURE_BLOBS:
                by_tile[tk.rsplit('@', 1)[1]].append(tk)

        def _nat(tk):
            return min(tex_native(tk.split('@')[0]), args.tile_texture_px)

        def _tcost(tk, e):
            base = tk.split('@')[0]
            c = SHARED_IMAGE_FIXED + e * e * SHARED_BYTES_PER_PX
            if (TEX_SRC.get(base[4:]) or {}).get('nfp'):
                ne = max(64, e // 2)              # its own normal map, at half the colour edge
                c += SHARED_IMAGE_FIXED + ne * ne * SHARED_BYTES_PER_PX
            return c

        budgets, est = [], {}
        # A warp loads the whole 3x3 of render tiles around the landing spot at once: Summer Camp
        # HD hung on the loading screen warping to a spot whose 3x3 held 77.5 MB. Cap every 3x3
        # window's estimate at --render-window-mb by shrinking the texture budgets inside it.
        tbud = {}
        mesh_of = {}
        for tile_ in by_tile:
            u_, v_ = (int(x) for x in tile_.split('_'))
            ck = (int(u_ * TILE_SIZE + TILE_SIZE / 2), int(v_ * TILE_SIZE + TILE_SIZE / 2))
            mesh_of[(u_, v_)] = MESH_BYTES_PER_TRI * rend_tris.get(ck, 0)
            tbud[(u_, v_)] = float(np.clip(args.tile_stream_mb * 1e6 - mesh_of[(u_, v_)], 0.5e6,
                                           args.tile_texture_budget_mb * 1e6))
        allmesh = {}
        for ck, n in rend_tris.items():
            allmesh[(int((ck[0] - TILE_SIZE / 2) // TILE_SIZE), int((ck[1] - TILE_SIZE / 2) // TILE_SIZE))] =                 MESH_BYTES_PER_TRI * n
        limit = args.render_window_mb * 1e6 / TEX_EST_FUDGE
        for _round in range(30):
            worst = 0.0
            for (u_, v_) in set(allmesh) | set(tbud):
                win = [(u_ + a, v_ + b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
                tot = sum(allmesh.get(w, 0.0) + tbud.get(w, 0.0) for w in win)
                worst = max(worst, tot)
                if tot > limit:
                    tex = sum(tbud.get(w, 0.0) for w in win)
                    if tex > 0:
                        f = max(0.0, 1 - (tot - limit) / tex)
                        for w in win:
                            if w in tbud:
                                tbud[w] = max(0.25e6, tbud[w] * f)
            if worst <= limit:
                break
        for tile_, tks in by_tile.items():
            u_, v_ = (int(x) for x in tile_.split('_'))
            ck = (int(u_ * TILE_SIZE + TILE_SIZE / 2), int(v_ * TILE_SIZE + TILE_SIZE / 2))
            mesh_b = mesh_of[(u_, v_)]
            budget = tbud[(u_, v_)]
            budgets.append(budget)
            for tk in tks:
                tile_edge[tk] = min(128 if tk.startswith('dec:') else 64, _nat(tk))
            left = budget - sum(_tcost(tk, tile_edge[tk]) for tk in tks)
            heap = [(-tex_area[tk] / max(1.0, _tcost(tk, 2 * tile_edge[tk]) - _tcost(tk, tile_edge[tk])), tk)
                    for tk in tks if 2 * tile_edge[tk] <= _nat(tk)]
            _hq2.heapify(heap)
            while heap:
                _, tk = _hq2.heappop(heap)
                e = tile_edge[tk]
                extra = _tcost(tk, 2 * e) - _tcost(tk, e)
                if extra > left:
                    continue
                left -= extra
                tile_edge[tk] = 2 * e
                if 4 * e <= _nat(tk):
                    _hq2.heappush(heap, (-tex_area[tk] / max(1.0, _tcost(tk, 4 * e) - _tcost(tk, 2 * e)), tk))
            est[ck] = mesh_b + sum(_tcost(tk, tile_edge[tk]) for tk in tks)
        area_by_edge = collections.Counter()
        for tk, e in tile_edge.items():
            area_by_edge[e] += tex_area[tk]
        tot = sum(area_by_edge.values()) or 1.0
        worst = max(est.items(), key=lambda kv: kv[1]) if est else ((0, 0), 0)
        print(f"per-tile textures: {len(tile_edge):,} tile copies over {len(by_tile)} tiles; texture budget "
              f"min {min(budgets, default=0) / 1e6:.2f} / median {float(np.median(budgets)) / 1e6 if budgets else 0:.2f} "
              f"/ max {max(budgets, default=0) / 1e6:.2f} MB; surface by resolution: "
              + ", ".join(f"{e} px {100 * a / tot:.0f}%" for e, a in sorted(area_by_edge.items(), reverse=True))
              + f"; heaviest estimated stream {worst[1] / 1e6:.2f} MB at {worst[0][0]}_{worst[0][1]}")

    def shrink_png(blob, edge=64):
        im = Image.open(io.BytesIO(blob))
        w, h = im.size
        while max(w, h) > edge and min(w, h) > 1:
            w, h = max(1, w // 2), max(1, h // 2)
        if (w, h) == im.size:
            return blob
        buf = io.BytesIO()
        im.resize((w, h), Image.LANCZOS).save(buf, format='PNG')
        return buf.getvalue()
    normals_dropped = set()
    rgb_of = {n: rgb for n, _, rgb, _ in CLASSES}
    rgb_of[MISC[0]] = MISC[1]
    rgb_of[DECAL_CLASS] = (128, 128, 128)
    rgb_of[COLLISION_CLASS] = COLLISION_RGB
    collide_of = {n: c for n, _, _, c in CLASSES}
    collide_of[MISC[0]] = MISC[2]
    collide_of[COLLISION_CLASS] = True
    spec_png, nrm_png = png((SPECULAR_GREY,) * 3), png((128, 128, 255))
    lm_png = lightmap_swatch(args.lightmap) if args.lightmap > 0 else None
    bake = None
    tile_frames = {}
    if args.bake_light and lm_png is not None:
        import lightmap as _lightmap
        bake = _lightmap.LightBake(blocks, skip_classes=(COLLISION_CLASS, DECAL_CLASS, 'Water'),
                                   sun_az=args.sun_az, sun_el=args.sun_el)
        bake.preview(os.path.join(WORK, 'scratch', 'lightmap_preview.png'))
        # one frame per streaming tile, the same whichever GLB block a triangle arrives in
        _box = {}
        for _classes in blocks.values():
            for (_cn, _tk, _px), _parts in _classes.items():
                if _cn == COLLISION_CLASS:
                    continue
                for _W, _UV, _T in _parts:
                    if not len(_T):
                        continue
                    _c = _W[_T].mean(axis=1)
                    _u = np.floor(_c[:, 0] / TILE_SIZE).astype(np.int64)
                    _v = np.floor(_c[:, 2] / TILE_SIZE).astype(np.int64)
                    for u_, v_ in set(zip(_u.tolist(), _v.tolist())):
                        q = _W[_T[(_u == u_) & (_v == v_)].ravel()][:, [0, 2]]
                        lo_, hi_ = _box.get((u_, v_), (np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])))
                        _box[(u_, v_)] = (np.minimum(lo_, q.min(axis=0)), np.maximum(hi_, q.max(axis=0)))
        for (u_, v_), (lo_, hi_) in _box.items():
            t0 = np.array([u_ * TILE_SIZE, v_ * TILE_SIZE])
            tile_frames[(u_, v_)] = bake.tile_frame(np.maximum(lo_, t0 - 30.0),
                                                    np.minimum(hi_, t0 + TILE_SIZE + 30.0))
    lm_stats = collections.Counter()
    stats = {'blocks': 0, 'triangles': 0, 'vertices': 0, 'files': [], 'materials': set(),
             'collision_triangles': 0, 'visual_triangles': 0, 'degenerate_dropped': 0}
    for (bu, bv), classes in sorted(blocks.items()):
        glb = GlbBuilder()
        spec, nrm = glb.add_image('specular', spec_png), glb.add_image('normal', nrm_png)
        lmap = glb.add_image('lightmap', lm_png) if lm_png is not None else None
        tile_lm = {}

        def lit_prim(P, T, UV0, mat, cname):
            """Primitives for one material: smooth normals and, under --bake-light, one lightmap
            per streaming tile with the UVs into it (the material is cloned per tile)."""
            N = vertex_normals(P.astype(np.float64), T)
            if bake is None or cname == COLLISION_CLASS:
                stats['materials'].add(mat.name)
                return [Primitive(positions=P, indices=T, normals=N, uv0=UV0, material=glb.add_material(mat))]
            cen = P[T].mean(axis=1)
            tu_ = np.floor(cen[:, 0] / TILE_SIZE).astype(np.int64)
            tv_ = np.floor(cen[:, 2] / TILE_SIZE).astype(np.int64)
            out = []
            for u_, v_ in sorted(set(zip(tu_.tolist(), tv_.tolist()))):
                frame = tile_frames.get((u_, v_))
                if frame is None:
                    frame = tile_frames[(u_, v_)] = bake.tile_frame(
                        (u_ * TILE_SIZE, v_ * TILE_SIZE), ((u_ + 1) * TILE_SIZE, (v_ + 1) * TILE_SIZE))
                if (u_, v_) not in tile_lm:
                    tile_lm[(u_, v_)] = glb.add_image(f'lightmap_{u_}_{v_}', bake.tile_png(frame))
                Ts = T[(tu_ == u_) & (tv_ == v_)]
                vidx, T2, uv1, npl = bake.map_prim(P, Ts, frame, N)
                lm_stats['tris'] += len(Ts); lm_stats['planar'] += npl
                lm_stats['v_in'] += len(np.unique(Ts)); lm_stats['v_out'] += len(vidx)
                name = f'{mat.name}__L{u_}_{v_}'
                stats['materials'].add(name)
                mi = glb.add_material(dataclasses.replace(
                    mat, name=name, emissive_texture=tile_lm[(u_, v_)], emissive_texcoord=1))
                out.append(Primitive(positions=P[vidx], indices=T2, normals=N[vidx], uv0=UV0[vidx], uv1=uv1,
                                     material=mi))
            return out
        img_cache = {}
        prims = []
        for (cname, texkey, is_proxy), parts in sorted(
                classes.items(), key=lambda kv: (kv[0][0], kv[0][1] or '', kv[0][2])):
            off, Ps, UVs, Ts = 0, [], [], []
            for W, UV, T in parts:
                Ps.append(W); UVs.append(UV); Ts.append(T + off)
                off += len(W)
            P = np.concatenate(Ps).astype(np.float32)
            T = np.concatenate(Ts).astype(np.int64)
            # `__proxylod` marks a far-LOD shell however it ended up textured, so the
            # spawn search still refuses to stand the skater on scenery.
            tag = '__proxylod' if is_proxy else ''
            _base, _, _tsfx = (texkey or '').partition('@')
            if texkey and _base in TEXTURE_BLOBS:
                _b, _, _paint = _base.replace('tex:', '').partition('|')
                mname = (f'IOG_{cname}{tag}__' + _b.replace('/', '.')[-52:]
                         + ('__p' + _paint.replace('.', '').replace(',', '')[:9]
                            if _paint else '') + (f'__t{_tsfx}' if _tsfx else ''))
                if texkey not in img_cache:
                    blob = TEXTURE_BLOBS[_base]
                    if args.tile_textures and _base[4:] in TEX_SRC and texkey in tile_edge:
                        blob = full_png(_base[4:], tile_edge[texkey])
                    elif args.tile_textures and _base[4:] in TEX_SRC and texkey in shared_edge:
                        blob = full_png(_base[4:], shared_edge[texkey])
                    elif texkey in tile_edge:
                        # baked sheets (SEG_, TERR_, FILL_) have no TEX_SRC entry
                        blob = shrink_png(blob, tile_edge[texkey])
                    elif texkey in small_shared:
                        blob = shrink_png(blob)
                    elif texkey in shared_edge:
                        blob = shrink_png(blob, shared_edge[texkey])
                    img_cache[texkey] = glb.add_image(texkey, blob)
                img = img_cache[texkey]
            else:
                # A block with no texture still states its colour; without it these land on
                # an untinted class swatch, which is more of the same grey.
                paint = texkey[6:] if (texkey or '').startswith('paint:') else None
                ck = cname if paint is None else f'{cname}|{paint}'
                mname = f'IOG_{cname}{tag}' + ('__p' + paint.replace('.', '').replace(',', '')[:9]
                                               if paint else '')
                if ck not in img_cache:
                    blob = CLASS_BLOBS.get(cname) or png(rgb_of[cname])
                    if args.shared_texture_budget_mb > 0 and cname in CLASS_BLOBS:
                        blob = shrink_png(blob, 256)   # class fallbacks are on every tile: global
                    if paint:
                        blob = _tinted(blob, [float(v) for v in paint.split(',')])
                    img_cache[ck] = glb.add_image(ck, blob)
                img = img_cache[ck]
            mnorm = TEX_NORMALS.get(_base[4:]) if texkey and texkey.startswith(('tex:', 'rip:')) \
                else None
            # Per-tile copies keep their OWN normal map (half the colour edge). When they collapsed
            # into cPres_Global every per-tile texture took its class normal instead: 1,426 of 1,710
            # materials drew one generic bump map per class and lost skate.'s per-material detail -
            # grain, pores, cracks, plank gaps (user, 2026-09-30).
            if (mnorm is not None and args.tile_textures and texkey in GLOBAL_TEX_KEYS
                    and tex_area.get(texkey, 0.0) < HOT_TEXTURE_M2):
                mnorm = None
            if mnorm is not None and tex_area.get(texkey, 0.0) < args.normal_min_m2:
                normals_dropped.add(texkey)
                mnorm = None
            if mnorm is not None:
                nk2 = '__n_' + texkey[4:]
                _src = TEX_SRC.get(_base[4:]) or {}
                if nk2 not in img_cache:
                    if args.tile_textures and _src.get('nfp') and texkey in tile_edge:
                        mnorm = full_png(_base[4:], max(64, tile_edge[texkey] // 2), normal=True)
                    elif args.tile_textures and _src.get('nfp') and texkey in shared_edge:
                        mnorm = full_png(_base[4:], shared_edge[texkey], normal=True)
                    elif texkey in tile_edge:
                        mnorm = shrink_png(mnorm, tile_edge[texkey])
                    elif texkey in shared_edge:
                        mnorm = shrink_png(mnorm, shared_edge[texkey])
                    img_cache[nk2] = glb.add_image(nk2, mnorm)
                nrm_img = img_cache[nk2]
                mat_ = (Material(
                    name=mname, base_color_texture=img,
                    metallic_roughness_texture=None if cname == DECAL_CLASS else spec,
                    normal_texture=nrm_img,
                    emissive_texture=None if cname == COLLISION_CLASS else lmap,
                    alpha_mode=DECAL_MODE.get(_base) if cname == DECAL_CLASS else None))
                prims += lit_prim(P, T, np.concatenate(UVs).astype(np.float32), mat_, cname)
                stats['triangles'] += len(T); stats['vertices'] += len(P)
                stats['visual_triangles'] += len(T)
                continue
            nkey = f'__n_{cname}'
            if cname in CLASS_NORMALS:
                if nkey not in img_cache:
                    # every tile uses the class normals, so they all sit in cPres_Global
                    img_cache[nkey] = glb.add_image(nkey, shrink_png(CLASS_NORMALS[cname], 128)
                                                    if args.shared_texture_budget_mb > 0 else CLASS_NORMALS[cname])
                nrm_img = img_cache[nkey]
            else:
                nrm_img = nrm
            mat_ = (Material(
                name=mname, base_color_texture=img,
                    metallic_roughness_texture=None if cname == DECAL_CLASS else spec,
                normal_texture=nrm_img,
                emissive_texture=None if cname == COLLISION_CLASS else lmap,
                alpha_mode=DECAL_MODE.get(_base) if cname == DECAL_CLASS else None))
            prims += lit_prim(P, T, np.concatenate(UVs).astype(np.float32), mat_, cname)
            stats['triangles'] += len(T)
            stats['vertices'] += len(P)
            if cname == COLLISION_CLASS:
                stats['collision_triangles'] += len(T)
            else:
                stats['visual_triangles'] += len(T)
        glb.add_mesh(f'block_{bu}_{bv}', prims)
        name = f'block_{bu:+03d}_{bv:+03d}.glb'
        size = glb.write(os.path.join(args.out, name))
        stats['degenerate_dropped'] += glb.degenerate_dropped
        del glb, img_cache
        import gc
        gc.collect()
        stats['files'].append(name)
        stats['blocks'] += 1
        print(f"  {name}: {len(prims)} materials, {sum(len(p.indices) for p in prims):,} tris, {size / 1e6:.1f} MB")

    if bake is not None and lm_stats['tris']:
        print(f"lightmap: {lm_stats['planar']:,} of {lm_stats['tris']:,} drawn triangles lit from their tile's "
              f"top-down map, the rest take one shade each; vertices {lm_stats['v_in']:,} -> {lm_stats['v_out']:,}")
    if small_shared:
        print(f"shared small textures: {len(small_shared)} texture(s) on two or more tiles and under "
              f"{args.shared_small_texture_m2:,.0f} m2 served at 64 px ({sum(tex_area[t] for t in small_shared):,.0f} m2)")
    if normals_dropped:
        print(f"normal maps: {len(normals_dropped)} texture(s) covering under {args.normal_min_m2:,.0f} m2 use their "
              f"class normal instead of their own ({sum(tex_area[t] for t in normals_dropped):,.0f} m2 of "
              f"{sum(tex_area.values()):,.0f} m2 textured surface)")
    doc = {}
    for m in sorted(stats['materials']):
        cname = m[4:].split('__', 1)[0]
        doc[m] = {
            'collision': {'physics_surface': '0', 'audio_surface': '0', 'surface_pattern': '0'},
            # skate. ships no separate render/collision split we have decoded, so solid
            # classes ride on their render triangles; decals, glass and foliage do not.
            # Every visual class is render-only; collision rides on the decimated copy.
            'exclude_collision': cname != COLLISION_CLASS,
            'exclude_pres': cname == COLLISION_CLASS,
            'roughness': 0.05,
            'material_name': ATTRIBUTOR,
        }
        if cname == DECAL_CLASS:
            # the arena builder ignores glTF alphaMode: the family and this flag are the whole alpha path
            doc[m]['material_name'] = args.decal_family
            doc[m]['diffuse_has_alpha'] = True
            # The recomp's world shader skips the alpha test whenever a spec mask is bound
            # (skate3_native_scene_gpu.cpp: env families bind spec in the decal slot, overlay.w
            # == 3; scene.hlsl clips only at overlay.w < 0.5). the old arena tool (branch
            # no-specular-for-alphatest) then emits no specular channel at all, so the art's
            # own alpha cuts at the game's ALPHAREF 30.
            doc[m]['no_specular'] = True
    json.dump(doc, open(os.path.join(args.out, 'blenrose_materials.json'), 'w'), indent=1)

    ridge_n = collections.Counter()
    for kind, Wr in grind_ridge:
        for pl in ridge_polyline(Wr + shift, None if kind == 'flag' else 0.9):
            splines.append(pl)
            ridge_n[kind] += 1
    print(f"grind ridges: {ridge_n['flag']:,} from skate.'s flagged grind parts, "
          f"{ridge_n['thin']:,} along the tops of thin props (railings, fences, barriers)")
    # rails bend at stair landings (flat -> down the stairs): join through bends up to 60 deg
    splines, _ja = join_splines(splines, tol=0.4, max_deg=60.0)
    # A traced collision edge lying along a clean rail/ridge line is the same grind, but wobbly
    # (it follows the rail's faceted collision): the game latched onto it. Keep the clean line.
    if splines and edge_splines:
        from scipy.spatial import cKDTree
        _cp = []
        for l in splines:
            for a_, b_ in zip(l[:-1], l[1:]):
                n_ = max(2, int(np.linalg.norm(b_ - a_) / 0.1) + 1)
                _cp.append(a_ + (b_ - a_) * np.linspace(0, 1, n_)[:, None])
        _ct = cKDTree(np.concatenate(_cp))
        _kept = []
        for l in edge_splines:
            d_, _ = _ct.query(l)
            if (d_ < 0.3).mean() < 0.5:
                _kept.append(l)
        print(f"grind: {len(edge_splines) - len(_kept):,} traced edge line(s) dropped where a clean rail/ridge "
              f"line already runs within 30 cm")
        edge_splines = _kept
    # A traced edge with floor at its own height on BOTH sides is a seam between boards, tiles or
    # slabs, not a ledge: the chalet patio's 4x4 m decking grinded along every tile joint and the
    # board snapped onto them rolling across (user, 2026-09-23). Ledges and curbs fall away on
    # one side and stay.
    if edge_splines and collide_parts:
        _fl = flush_splines(edge_splines, collide_parts)
        print(f"grind: {len(edge_splines) - len(_fl):,} traced edge line(s) dropped that have floor flush "
              f"on both sides or lie buried under a deck (seams between boards/tiles, not ledges)")
        edge_splines = _fl
    edge_splines = [smooth_rail(np.asarray(l, np.float64)) if len(l) >= 3 else l for l in edge_splines]
    # lines whose ends nearly meet are one grind (user, 2026-09-23: "make all the splines
    # connected if they are close together and smooth"); 8 cm left ledges in pieces
    edge_splines, _je = join_splines(edge_splines, tol=0.35, max_deg=40.0)
    print(f"grind joins: {_ja:,} authored/ridge and {_je:,} traced line ends glued where they meet in line")
    edge_splines = [l for l in edge_splines if length(l) >= GRIND_MIN_LEN]
    if GRIND_MAX_TOTAL and len(edge_splines) > GRIND_MAX_TOTAL:
        edge_splines.sort(key=length, reverse=True)
        print(f"grind edges: {len(edge_splines):,} found, keeping the longest {GRIND_MAX_TOTAL:,}")
        edge_splines = edge_splines[:GRIND_MAX_TOTAL]
    # Skate 3 crashes loading a tile that carries too many splines: Isle of Grom with 2,338 in
    # one 100 m tile died with an access violation as its streams loaded, while 1,658 per tile
    # (Summer Camp) loads - consistent with a 2,048-entry table. Detected edges are capped per
    # tile, longest first, counting every tile a spline touches; authored rails count first.
    if args.grind_max_per_tile:
        def tiles_of(l):
            return {(int(math.floor(p[0] / TILE_SIZE)), int(math.floor(p[2] / TILE_SIZE))) for p in l}
        per = collections.Counter()
        for l in splines:
            for t_ in tiles_of(l):
                per[t_] += 1
        kept, cut = [], 0
        for l in sorted(edge_splines, key=length, reverse=True):
            ts = tiles_of(l)
            if all(per[t_] < args.grind_max_per_tile for t_ in ts):
                for t_ in ts:
                    per[t_] += 1
                kept.append(l)
            else:
                cut += 1
        if cut:
            print(f"grind edges: {cut:,} short edges dropped to keep every tile under "
                  f"{args.grind_max_per_tile:,} splines (busiest now {max(per.values()):,})")
        edge_splines = kept
    print(f"grind splines: {len(splines):,} authored rails + {len(edge_splines):,} "
          f"90-degree edges ({round(sum(length(l) for l in edge_splines)):,} m)")
    splines = splines + edge_splines
    splines = [round_corners(np.asarray(l, np.float64)) for l in splines if length(l) >= 1.0]
    json.dump({'splines': [{'points': [[round(float(v), 4) for v in p] for p in l]} for l in splines]},
              open(os.path.join(args.out, 'splines.json'), 'w'))
    json.dump({f'{u}_{v}': [round(a, 3), round(b, 3)] for (u, v), (a, b) in sorted(tile_y.items())},
              open(os.path.join(args.out, 'd2s3_tiles.json'), 'w'), indent=0)


    metres = round(sum(length(l) for l in splines), 1)
    stats['materials'] = sorted(stats['materials'])
    stats['options'] = {'source': 'skate. dingolevel_isle_of_grom', 'crop': crop,
                        'block_size': args.block_size, 'tile_budget': args.tile_budget,
                        'shift': shift.round(3).tolist()}
    stats['counters'] = {'grind_splines': {'rails': len(splines), 'metres': metres}}
    stats['events'] = []
    json.dump(stats, open(os.path.join(args.out, 'd2s3_export.json'), 'w'), indent=1)
    print(f"\nblocks {stats['blocks']}  triangles {stats['triangles']:,}  materials {len(stats['materials'])}  "
          f"tiles {len(tile_y)}  rails {len(splines)} ({metres} m)  degenerate dropped {stats['degenerate_dropped']}")
    uv_report(os.path.join(args.out, 'uv_density.json'))
    # Spawns come last: chosen on the FINAL geometry, with clearance checks. A failure here
    # used to abort the whole export through SystemExit, taking the diagnostics with it and
    # leaving a staged GLB set with no spawn.txt and no explanation.
    try:
        find_spawns.START_NAME = LEVEL_TITLES.get(LEVEL, find_spawns.START_NAME)
        # Spawn at skate spots, not at the highest point or the median rail: skate.'s own
        # skatepark kit, DIY pieces, ramps and rails, in the exported frame.
        feats = [[float(t[0]) + shift[0], float(t[1]) + shift[1],
                  (-float(t[2]) if args.mirror_z else float(t[2])) + shift[2]]
                 for mesh, _R, t, _src in inst if SKATE_FEATURE.search(mesh.rsplit('/', 1)[-1])]
        json.dump(feats, open(os.path.join(args.out, 'skate_features.json'), 'w'))
        spawns = find_spawns.run(args.out, features=feats)
    except SystemExit as e:
        spawns = []
        print(f"!! NO SPAWN: {e}. The pack will build with the default spawn.")
    if spawns:
        s = spawns[0]
        print(f"startline --spawn={s['x']},{s['y']},{s['z']}  (also in spawn.txt)")
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
