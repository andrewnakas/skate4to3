# Skate 3 render arenas (cPres) - measured facts

Code: `sk3/render.py` (decode), `sk3/render_write.py` (encoders + builder),
`tests/test_render_roundtrip.py`. Container layout (header, dictionary, subrefs, graphics region):
`specs/SK3_FORMATS.md` "RW4 arena". Everything is big-endian (Xbox 360).

Evidence: retail `worldDIST_*.big` from the recomp's `game/data`, read with this repo's readers.
"Small worlds" = BlackBoxPark, MegaPark, StartPark, IndustrialSkatePark (280 cPres arenas) unless a
count says otherwise. Scratch scripts: `work/scratch/render/*.py` (not committed).

## Proof of the whole decode
`render_write.rebuild(x) == x` (decode every object to semantic fields - vertex attributes, index
lists, declarations, meshes, models, instances, materials, export tables, probes - re-encode, rebuild
the arena with `rw4_write.build`) holds for every cPres arena (`cPres_<x>_<z>_high.xsf` and
`cPres_Global.xsf`):

| worlds | arenas | pass |
|---|---|---|
| BlackBoxPark, MegaPark, StartPark, IndustrialSkatePark | 280 | 280 |
| MaloofMoneyCup, DownTownSkatePark, SkateSchool | 656 | 656 |
| DownTown, Industrial, University | 28642 | 28642 |

Only texture headers (0x200e8) and their texel base resources are carried raw (texture code owns
them). Visual check: top-down rasterisation of BlackBoxPark (roof cut at y = 9 m) shaded by the
decoded normals, a uv0 checker and lightmap uv as colour, and of four DownTown tiles: a skatepark
with quarterpipes/banks/bowl, continuous checker squares across ramps, smooth per-chart lightmap
gradients (`work/scratch/render/bbp_shade.png`, `bbp_uvlm.png`, `dt_shade.png`).

## Arena kinds
- `cPres_Global.xsf` and tile streams hold three kinds of arena, told apart by object types:
  texture arenas {0x10031, 0x200e8, 0xeb000b, 0xeb0008} (248 in the small worlds; 9-type registry),
  mesh arenas {0xeb0008, 0xeb0005, 0xeb000d, per mesh 7 objects, 0xeb0001, 0xeb000b} (20), and
  small material-only / probe arenas {0xeb0008, 0xeb0005 or 0xeb0024, 0xeb000b}.
- Big worlds add 0xeb0016, 0xeb0025, 0xeb0028 (DownTown: 114, 99, 98 objects) - see the end.
- Tile `cPres_<x>_<z>`: positions of four DownTown tiles named x in {50, 150}, z in {-50, 50} span
  x -0.3..200.6, z -102.4..100: tiles are 100 m squares centred on the name (small overhang).
  BlackBoxPark -50_-50 lies in x -12..1, z -36..0.

## Object order inside a mesh arena (all 20 small-world mesh arenas)
`eb0008, eb0005, eb000d`, then per mesh `10031 (vertices), 200ea, 10031 (indices), 200eb, eb0023,
200e9, 20081`, then `eb0001, eb000b`. Alignments: base resources 4 (texels 0x1000), 200e9/ea/eb/81
4, every eb00xx 0x10.

## 0x200e9 vertex declaration (425 small + 13734 DownTown)
`{u32 0, u32 mask, u16 count, u16 1, u32 0}` then count x 16 B elements
`{u16 stream 0, u16 offset, u32 D3DDECLTYPE, u8 method 0, u8 usage, u8 usage index, u8 id, u32 1}`,
then count bytes, each = the vertex stride. mask = OR of `1 << id` (all). Size = 16 + 17 n.
Element ids: 1 position, 6 TEX0, 7 TEX1, 9 TEX3, 0x15 tangent (usage 6). Seen formats:

| type | data | seen as |
|---|---|---|
| 0x002A23B9 | 3 x f32 | position |
| 0x002C235F | 2 x f16 | uv0 / uv1 (decal decl) / uv3 |
| 0x001A2360 | 4 x f16 | TEX0 = (uv0, uv2) |
| 0x001A215A | 4 x s16 / 32767 | TEX1 = lightmap uv + normal (below) |
| 0x002C2159 | 2 x s16 / 32767 | TEX1 lightmap uv (no normal); TEX0 in 4 DownTown decls |
| 0x002A2190 | u32 11:11:10 snorm, x in bits 0-10, y 11-21, z 22-31 | tangent |
| 0x001A2286, 0x002C2359 | UBYTE4, SHORT2 | one skinned mesh (University) |

Declarations by count (DownTown): pos+uv0+TEX1(s4)+tangent stride 28: 8437; pos+uv0+TEX1(s2) 20:
2891; pos+TEX0(h4)+TEX1(s4)+tangent 32: 1459; pos+uv0+TEX1(s4) 24: 907; pos+h4+TEX1 half2: 32;
5 rarer ones (<= 4 each), incl. TEX3 and pos+uv0 only.

### Vertex attribute meaning (measured on 12196 BlackBoxPark vertices, stride-28 decl)
- position: world space (bbox of the decoded vertices equals the mesh bbox; instance matrices are
  identity in all 20 small-world instances).
- TEX1 SHORT4N = `(lu * b, lv * s, nx, ny)`:
  - (s3, s4) correlate 0.99 with the area-weighted face normal x, y; nz = sqrt(1-nx^2-ny^2);
  - sign(TEX1.y) = sign(normal z) for 98.7 % of vertices with |nz| > 0.05;
  - sign(TEX1.x) = -sign(dot(cross(n, t), dP/dv)) for 99.6 % of 9266 triangles (b = binormal sign);
  - |TEX1.x|, |TEX1.y| = lightmap uv in [0, 1] (smooth per-chart gradients in the render).
- tangent: dot with normalised dP/du (of uv0) has p5 0.987, median 0.999; dot with the normal ~0.
- uv0: tiling uv (range -96..33), half floats.
- uv2 (second half of TEX0 h4): used exactly by the decal shaders (environment.decal*, 123 meshes).

## 0x200ea vertex buffer header (40 B; 425/425 small)
`{1, 1, 0, 0, 0, 0xFFFF0000, resIndex << 2 | 3, 0x10000000 | size | 2, size, 0}`: a D3D resource
header + Xbox GPU vertex fetch constant; resIndex = dictionary index of the 0x10031 holding the
vertices, size = its byte size = count x stride, 2 = endian swap 8-in-32.
## 0x200eb index buffer header (36 B)
`{0x20000002, 1, 0, 0, 0, 0xFFFF0000, resIndex, size, index count}`; 16-bit indices; base
resource size = align16(2 x count), zero padded (425/425).
## 0x20081 binding (16 B): `{1, 1, ib header index, vb header index}` (425/425).

## 0xeb0023 mesh (116 + 16 (n-1) B)
f32 bbox `(min x,y,z, 0, max x,y,z, 0)`, then u32: `0, material ref, decl, binding, ib, vb, n prims,
extra ref, bounds ref, 0x60, 1, 0x60 + 16 n, 0 x4`, n x `{4 = triangle list, 0 base vertex, start
index, index count}`, `0`. Prims are contiguous and cover the whole index buffer (425/425). Refs
with bit 0x00800000 are subref indices. 0x60 / 0x60+16n = start / end of the prim table.
- material ref -> (eb0005, table entry): ONE material per mesh; all prims use it (consecutive-entry
  reading overflows the table in 9 meshes; materials are shared by meshes, max 4 uses).
- multi-prim meshes (80 of 425): each prim has its own bounds/extra record - spatial sub-chunks.
- extra ref / bounds ref -> the model's per-prim arrays (prim k = record base + k).

## 0xeb0001 model
f32 bbox (8), then u32 `0x40 matrix, 0x80 mesh list, name-pointer offset (= 0x80 + 8 n), name
offset (= pointer + 4), n meshes << 16 | n meshes, 1 << 16 | n prims, bounds offset, extra offset`.
Matrix 4x4 identity written with signed zeros `1,-0,0,-0, -0,1,-0,0, 0,-0,1,-0, -0,0,-0,1` (20/20).
Mesh list n x `{mesh object index, 0}`; u32 at the pointer offset = name offset; name NUL-terminated;
bounds at align16(name end); both arrays padded to a multiple of 4 records with zeros.
- bounds record 32 B: `(min, r, max, r)` of the prim's vertices, r = half the bbox diagonal
  (660/660 prims).
- extra record 16 B `(ax, ay, az, 0)`: per-axis projected area of the prim. `sum |area vector|`
  matches exactly for 50 % of prims; the rest are smaller (occlusion-aware?) - rule open; the builder
  writes the sum (upper bound).

## 0xeb000d instance(s)
`{0xACB31C9A, count, count + 1, 0x20, string pool offset, 0, 0, 0}` + count x 0xA0 records
`{f32 matrix[16], f32 bbox[8] (= model bbox), u64 guid, 6 x 0xFFFFFFFF, model object index (0 = no
model), 0, 0, name off, 3 string offs, 0}` + string pool (NUL strings, deduplicated, first use first;
the 3 extra strings are "undefined" in all). Instance guids in one world are sequential
(f8ded31d3b0f6b00/02/04/06). The hash word is the same in all 708 DownTown instances.

## 0xeb0005 material table (23 small)
`{n materials, total params, 0x14, params offset = align16(0x14 + 12 n), strings offset = params +
32 x total}`, n x `{param count, 0, offset of first param}`, params 32 B each:
`{name off, kind, 0xDEADBEEF, 0xDEADC0DE, value 16 B}`. kind 0xFFFF: `{u64 guid, string off, 0}`;
kind 0x20000: `{f32, 0, 0, 0}`. String pool: deduplicated, first use first (name, then value).
After the pool, 16 zero bytes in most tables (16 of 23 small; rule unknown; 0 is also retail).
- "Name": guid = the material's export guid, value = long generated material name.
- "AttribulatorMaterialName": guid 0, value = shader (Attribulator material) name (434/434).
- Texture params: guid = texture guid, value = `<texture name>_0x<guid as 16 hex>`.
- Unused materials: a table may hold materials no mesh uses (26 entries / 25 meshes in BBP -50_-50);
  they are still exported.

Shader census (434 materials, small worlds) with parameter order and the declaration they pair with:

| shader | n | params after Name, AttribulatorMaterialName | decl |
|---|---|---|---|
| environmentsimple.default | 138 | lightmap specular diffuse normal | pos uv0 TEX1s4 tangent |
| environment.default | 116 | lightmap detailNormalUVScale specular diffuse macroOverlayOpacity detail macrooverlay normal macroOverlayUVScale | same |
| environment.decal_tileable | 78 | environment lightmap detailNormalUVScale decal specular diffuse macroOverlayOpacity detail embeddedDecal macrooverlay normal macroOverlayUVScale | pos h4(uv0,uv2) TEX1s4 tangent |
| environment.decal | 45 | (same as decal_tileable) | same |
| incandescent.default | 25 | exposure diffuse | pos uv0 [TEX1s2] |
| dynamicobject.default | 9 | detailNormalUVScale gamePlayDesignate specular diffuse detail normal | - |
| advertisement.default | 9 | lightmap diffuse | pos uv0 TEX1s4 |
| incandescent.backlituvscroll | 8 | uAnimationSpeed lightmap vAnimationSpeed diffuse | pos uv0 TEX1s2 |
| environment.transparent | 4 | lightmap transparent specular diffuse | pos uv0 TEX1s4 |
| environmentsimple.diffuse, incandescent.videoscreen | 1, 1 | lightmap diffuse / diffuse | pos uv0 TEX1s2 |

Float values (most common): detailNormalUVScale 8 (241/248), macroOverlayOpacity 1 (172/239),
macroOverlayUVScale 0.3 (223/239), embeddedDecal 1 (123/123), exposure 10 (13/25),
uAnimationSpeed -0.25, vAnimationSpeed 0, gamePlayDesignate 0.

## How materials reach textures
A texture param's guid is looked up in export tables (0xeb000b) of other arenas: texture arenas
export `{name "0x<guid 16 hex>.Texture", hash 0x9B0F1678, guid, type 0xAC462E4A, value 1 = the
0x200e8 object}`. Over the 4 small worlds, of the material texture refs: lightmap -> cPres_Global
377, tile cPres 22; diffuse/normal/specular/detail/macrooverlay/environment/decal/transparent ->
cPres_Global (most) or cTex (57); 24 refs resolve in no arena of the world (e.g.
obj_railsfences_gl_rails_01_*, default_normal_0x0000043d03e3870a) - presumably shared packs outside
the world BIG. 12 BlackBoxPark texture exports carry guid 0x6c70... while their name says 0x2c70...,
and 5 Industrial ones hash 0x9B0F2374 (texture code to explain).

## 0xeb000b export table
`{n, 0x14, table end = 0x14 + 24 n, n index pairs, index offset}` + n x `{name off or 0, hash or
0xFEFFFFFF, u64 guid, type, value}` + string pool + bytes up to the index (retail leaves garbage
there: in BlackBoxPark's texture tables the 1-byte gap is 0x00 in 20, 0xFF in 13, 0x3F in 10, other
values in the rest; carried as `string_pad`) + index pairs `{type, value}`.
- Mesh arenas export: every material (type 0xEB0066, value = the first subref that points at its
  table entry, materials in table order), the table (0xEB0005, value 1 = object index), the instance
  matrix (0xEB0069, guid = instance guid, value = subref (eb000d, 0x20)), the instance object
  (0xEB000D, value 2). Texture arenas: one entry per texture, no index.
- Index: always these 25 types in this order `eb0066 eb0005 eb0067 eb0006 eb0001 eb000a eb0065
  eb0007 eb0069 eb000d eb006b eb0019 eb0064 eb0004 eb0068 eb0009 eb0016 eb0013 eb0014 eb0018 eb0017
  eb0020 eb0024 eb0026 eb0027`; value = first entry index for eb0066 (0, or n if none), the eb0066
  count for eb0005 (n if no eb0005 entry), n for all others (rule holds in every arena of the
  round trip).
## Subref order in mesh arenas
Per mesh in order: (eb0005, material entry), (model, extra record), (model, bounds record); then the
unused materials' entries; then (eb000d, 0x20).

## 0xeb0008: `{0x19, 0x0D}` (8 B in texture arenas) or `{0x19, 0x0D, 0, 0}` (16 B) - every arena.
## 0xeb0024 probes: `{n, 0x10, 0xDEDEDEDE, 0xDEDEDEDE}` + n x 40 f32 (160 B), e.g. 102 records.
## Big-world extras (structure measured on DownTown samples, round trip exact)
- 0xeb0016: `{n, 0x10, 0, 0}` + n x 112 B `{5 x (x, y, z, 1) - 4 corners + centre, vec4 axis,
  u64 guid (0x2c70170a...), 0, 0xEFBEADDE}`; arenas with it have instance records with model 0
  (`MassiveInstance_prop_DT_adBoard...`): billboard/ad quads.
- 0xeb0025: f32 bbox (8) + `{n, 0x30, 0, 0}` + n x 4 points (x, y, z, 0): quads.
- 0xeb0028: `{n, 0x10, 0, 0}` + n x 4 u64 (guid-like) records.

## Builder choices (render_write.build_render_arena)
Layout, counts, refs, subref and export order, index, decl, padding follow the rules above; a tile
rebuilt from its own decoded parts matches retail in type order, every eb0023/200e9/200ea/200eb/
20081 byte, positions/normals exactly, tangent dot p5 0.998, binormal sign 99 %. Choices retail
does not decide for us: material/instance/table guids (FNV-1a of names), material "Name" string,
the eb0005 16-byte tail (0), the per-prim extra vec4 (sum |area vector|), one prim per mesh, unused
materials (none).
