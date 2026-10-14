"""In-game probe: which Skate 3 shading family draws a cut-out (graffiti, decals)?

Nothing in the arena builder or d2s3 records which attributor families alpha-test or blend, and
The arena builder ignores glTF alphaMode: transparency can only come from the family named in
`blenrose_materials.json` plus `diffuse_has_alpha` (which makes it encode the diffuse as BC3 with
alpha). So we ask the game. Six labelled boards stand in a ring 5 m around the spawn, each facing
it, each with a different family:

    1  environmentsimple.alphatest   diffuse_has_alpha
    2  environment.decal_tileable    diffuse_has_alpha
    3  environment.decal             diffuse_has_alpha
    4  environment.transparent       diffuse_has_alpha
    5  environmentsimple.default     diffuse_has_alpha   (control: the family every map uses)
    6  environmentsimple.alphatest   NO alpha flag       (control: proves the flag matters)

Each board is magenta with its number in white, a round HOLE (alpha 0, right) and a SOFT RING
(alpha ramp, bottom). Report per number: hole see-through? ring soft, hard-edged, or solid?
Board invisible or black? The answer sets `--decal-family` for the graffiti build.

    python tools/probe_alpha.py <glb_dir>        # after export_glb, before d2s3 stages 2-7
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_glb as eg  # noqa: E402
from d2s3.glb_writer import GlbBuilder, Material, Primitive  # noqa: E402

PROBES = [
    ('1', 'environmentsimple.alphatest', True),
    ('2', 'environment.decal_tileable', True),
    ('3', 'environment.decal', True),
    ('4', 'environment.transparent', True),
    ('5', 'environmentsimple.default', True),
    ('6', 'environmentsimple.alphatest', False),
]
RADIUS, WIDTH, HEIGHT = 5.0, 2.0, 2.0
PX = 256   # d2s3: images under 256 px arrive zero-padded and near-black


def board_png(label: str) -> bytes:
    rgb = np.zeros((PX, PX, 3), np.uint8)
    rgb[:] = (230, 40, 160)
    img = Image.fromarray(rgb, 'RGB').convert('RGBA')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=150)
    except TypeError:
        font = ImageFont.load_default()
    d.text((18, -10), label, fill=(255, 255, 255, 255), font=font)
    a = np.array(img)
    yy, xx = np.mgrid[0:PX, 0:PX]
    hole = (xx - 190) ** 2 + (yy - 70) ** 2 < 45 ** 2
    a[hole, 3] = 0
    r = np.sqrt((xx - 128) ** 2 + (yy - 200) ** 2)
    ring = (r > 20) & (r < 50)
    a[ring, 3] = np.clip((r[ring] - 20) / 30 * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a, 'RGBA').save(buf, format='PNG')
    return buf.getvalue()


def main():
    glb_dir = sys.argv[1]
    sx, sy, sz = (float(v) for v in open(os.path.join(glb_dir, 'spawn.txt')).read().split(','))
    glb = GlbBuilder()
    spec = glb.add_image('probe_spec', eg.png((eg.SPECULAR_GREY,) * 3))
    nrm = glb.add_image('probe_nrm', eg.png((128, 128, 255)))
    lmap = glb.add_image('probe_lightmap', eg.lightmap_swatch(0.6))
    prims, mats_doc = [], {}
    for j, (label, family, alpha) in enumerate(PROBES):
        ang = 2 * np.pi * j / len(PROBES)
        c = np.array([sx + RADIUS * np.sin(ang), sy, sz + RADIUS * np.cos(ang)])
        face = np.array([sx, sy, sz]) - c
        face[1] = 0
        face /= np.linalg.norm(face)
        right = np.array([face[2], 0.0, -face[0]])
        corners = [c - right * WIDTH / 2, c + right * WIDTH / 2,
                   c + right * WIDTH / 2 + [0, HEIGHT, 0], c - right * WIDTH / 2 + [0, HEIGHT, 0]]
        uv = [[0, 1], [1, 1], [1, 0], [0, 0]]
        # front and back sheets 1 cm apart, so the board reads from both sides whatever culling
        # the family does
        P = np.array(corners + [p - face * 0.01 for p in corners], np.float32)
        T = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], np.int64)
        N = np.array([face] * 4 + [-face] * 4, np.float32)
        name = f'PROBE_{label}_{family.replace(".", "_")}{"" if alpha else "_noflag"}'
        img = glb.add_image(name, board_png(label))
        mi = glb.add_material(Material(name=name, base_color_texture=img, metallic_roughness_texture=spec,
                                       normal_texture=nrm, emissive_texture=lmap,
                                       alpha_mode='MASK', double_sided=True))
        prims.append(Primitive(positions=P, indices=T, normals=N,
                               uv0=np.array(uv + uv, np.float32), material=mi))
        entry = {'collision': {'physics_surface': '0', 'audio_surface': '0', 'surface_pattern': '0'},
                 'exclude_collision': True, 'exclude_pres': False, 'roughness': 0.05,
                 'material_name': family}
        if alpha:
            entry['diffuse_has_alpha'] = True
        mats_doc[name] = entry
    glb.add_mesh('block_probe', prims)
    glb.write(os.path.join(glb_dir, 'block_probe.glb'))
    mp = os.path.join(glb_dir, 'blenrose_materials.json')
    doc = json.load(open(mp))
    doc.update(mats_doc)
    json.dump(doc, open(mp, 'w'), indent=1)
    # the boards stand up to HEIGHT above the spawn: keep them inside the tile's declared Y range
    tp = os.path.join(glb_dir, 'd2s3_tiles.json')
    tiles = json.load(open(tp))
    tk = f"{int(np.floor(sx / 100) * 100 + 50)}_{int(np.floor(sz / 100) * 100 + 50)}"
    if tk in tiles:
        tiles[tk][1] = max(tiles[tk][1], sy + HEIGHT + 1.0)
        json.dump(tiles, open(tp, 'w'), indent=0)
    print(f"alpha probe: {len(PROBES)} boards around the spawn ({sx}, {sy}, {sz}) in tile {tk}: "
          + ", ".join(f"{l}={f}{'' if a else ' (no flag)'}" for l, f, a in PROBES))


if __name__ == '__main__':
    main()
