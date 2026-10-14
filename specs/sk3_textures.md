# Skate 3 (Xbox 360) textures - measured facts

Code: `sk3/texture.py` (decode), `sk3/texture_write.py` (encode), `tests/test_texture_roundtrip.py`.
Retail data: `C:\dev\skate3recomp\out\build\windows-release\game\data`. Scratch measurement scripts:
`work/scratch/tex/` (census.py -> census.json, survey.py, ebb.py, refs.py, tailmap.py, mipsearch.py,
sizes.py, fcstats.py, unres.py). Only general public knowledge used besides retail bytes: the Xenos
texture fetch-constant bit layout, the Xenos 2D tiled address formula, BC1/BC2/BC3 block formats.

## Census
`census.py` walks every non-audio archive (big/*.big, content/*.big, fe/**/*.big), decompressing
every entry (see "Compressed BIG entries" below), and records every RW4 object of type 0x200e8:
63,280 texture objects. 51,316 of them sit in SFIL world streams (worldDIST_*, worlddmo, worldmisc,
proxy*_100_Proxy); 11,964 in loose `.rx2` arenas (fe, createacharacter, parkassets, marquee, ...).
Everything below is about the world-stream textures unless it says otherwise.

## Texture arena (world streams)
Measured on all 51,316: exactly four objects, dictionary order and alignment fixed:
`[0] 0x10031 align 0x1000` base resource (pixels), `[1] 0x200e8 align 4` (52 B),
`[2] 0xeb000b align 0x10` (72 B), `[3] 0xeb0008 align 0x10` (8 B, always `00000019 0000000d`).
One texture per arena, no subrefs. 0xeb0008 is not texture-specific: in the four smallest worlds
it is also in 57 cSim and 32 non-texture cPres arenas, there 16 B `00000019 0000000d` + 8 zero
bytes; meaning of 0x19/0x0D unknown (likely a version stamp). The RW4 layer is `sk3/rw4_write` unchanged.
(.rx2 arenas: up to 23 textures share one arena; some are untiled, and their base_page field is
non-zero because several textures share one base resource; not needed for world packs.)

## 0x200e8 texture object (52 bytes, big-endian u32s)
- words 0-6: `3, 1, 0, 0, 0, 0xFFFF0000, 0xFFFF0000` in all 63,280 (a D3D resource header:
  common/type, refcount, fences, identifier, base/mip flush).
- words 7-12: the 24-byte Xenos texture fetch constant (bit fields per the public layout).
  Over all 51,316 world textures:
  - dword0 = `0x80000002 | pitch << 22`: type 2, tiled bit 31 always set, all sign/clamp bits 0.
    pitch field (unit 32 texels) = align32(width in blocks) x block width / 32
    (DXT: 64-wide -> 128 texels; 2805 textures are DXT narrower than 128).
  - dword1 = format | endian << 6; base address 0 (patched at load).
    Formats: 18 DXT1 (27,844), 4 R5G6B5 (17,356), 20 DXT4_5 = DXT5 (3,200), 6 A8R8G8B8 (2,916).
    Endian: 1 (8-in-16) for 4/18/20, 2 (8-in-32) for 6.
  - dword2 = (width-1) | (height-1) << 13 | depth << 26 (depth 5 = 6 faces for the 301 cube maps).
  - dword3 = swizzle << 1 only: DXT (X,Y,Z,W), 565 (Z,Y,X,1), 8888 (Z,Y,X,W). num_format 0,
    exp_adjust 0, every filter field 0.
  - dword4 = (levels-1) << 6 only (mip_min 0).
  - dword5 = dimension << 9 | packed_mips << 11 | mip_page << 12. dimension 1 (2D) or 3 (cube,
    301). packed_mips = 1 exactly when levels > 1 (43,195; 8,121 single-level textures have 0).
  - Proven: `texture_write.fetch_words(format, width, height, levels, cube)` reproduces all six
    words of all 51,316 world textures (fcstats.py / extract never needs `fc_override` there).
  - Sizes: 51 distinct, 4x32 .. 1024x1024, all powers of two. Levels: full chain
    (log2(max side)+1) or 1.

## Pixel layout (base resource)
All levels tiled with the public Xenos 2D tiling formula (`texture.tiled_offsets`), on blocks
(4x4 for DXT, texels otherwise), bytes in the format's endian swap (undo with `endian_fix`).
- Level k > 0 dims: max(1, side >> k).
- Each surface is padded to 32x32 blocks and its byte size to 4 KB.
- Level 0 at offset 0; level 1 at `mip_page << 12` = 4 KB-aligned level-0 size x faces.
- The first level whose SHORT side is <= 16 texels ("packed start", possibly level 0) and every
  smaller level share one surface the size of the packed-start level's padded surface. Offsets
  (texels, t = level - packed start, w x h = packed-start level):
  wide (w > h): t < 3 -> (0, 16 >> t); t >= 3 -> (4 * (w >> t), 0).
  square/tall: t < 3 -> (16 >> t, 0); t >= 3 -> (0, 4 * (h >> t)).
  Measured with tailmap.py (non-zero block maps of the packed surface for 256x256, 512x128,
  128x256 DXT1 and a flat 16x16 565: every predicted rectangle is filled, all other blocks zero)
  and confirmed by mip-chain contact sheets (each level is the box-downsample of the previous).
- Cube maps: per level, the 6 faces back to back, each face 4 KB-aligned (level-major, face-minor);
  checked visually (each face column matches across levels 0-3) and by the round trip.
- Every byte not covered by a level is zero: `build(extract(x)) == x` (below) re-creates the
  base resource from the levels into a zero buffer.
- Size check (sizes.py): computed surface end == base resource size for all 260 small-world textures.

## 0xeb000b: the arena's export (name) table
Generic per-arena table (also in cPres/cSim arenas for eb0005/eb000d/eb0064/... objects, those
with empty names and word 0xFEFFFFFF). Layout: header `{u32 count, u32 entries at (0x14),
u32 strings at, u32 0, u32 total size}`, entries 24 B `{u32 name offset, u32 word, u64 id,
u32 type hash, u32 dictionary index}`, then NUL-ended names, padded to 4.
For textures (all 51,316): count 1, entries 0x14, strings 0x2C, size 0x48; name
`0x<16 hex>.Texture`; type hash 0xAC462E4A = FNV-1 32 of b"Texture" (measured: computed);
dictionary index 1 (the 0x200e8 object). The name's 16 hex digits are the texture's identity
("name id"). The word at +0x18 is 0x9B0F1678 in 50,710 and 0x9B0F??74 in the rest (not
derivable, carried). The last (pad) byte after the NUL is not zero in about half the arenas
(00 68, ff 63, 3f 20, aa 12, ... of 260): carried in `TexSpec.pad`.

### Low-res stand-ins (the 0x40 flag)
The entry id equals the name id, except that bit 62 (`0x4000000000000000`) is set in 18,516 world
textures. Every flagged texture (all in cPres streams; 565/8888, mostly 64x64/32x32) has an
unflagged twin with the same name in a cTex stream of the same world, always larger
(2,264/2,264 names checked by world). So: cTex holds the full texture, cPres a small stand-in
under the same name id; the flag tells them apart.

## How other arenas reference a texture
- By name id only. Material objects (type 0xeb0005, in cPres) carry records
  `{u32 ?, u32 kind, DEADBEEF DEADC0DE, u64 id}` (16-aligned); the DEADBEEF DEADC0DE pair is a
  pointer slot filled at load. BlackBoxPark: 1,479 records; kind 0xFFFF with a texture name id in
  815; kind 0xFFFF but not a texture of any archive in 306 (141 of them id 0, rest look like
  64-bit hashes - not textures; owned by the material side); kind 0x20000 in 358.
- Searching every BlackBoxPark object for every texture's name id, table id and SFIL asset id
  (refs.py): only eb0005 (872 hits on name ids, 815 on table ids = the unflagged ones) and the
  eb000b tables themselves. SFIL asset ids are not referenced from objects; they appear in the
  stream manifests (`DIST_<Map>_Pres.xst` holds asset id 0x70D72AE0FF9ED2BA, the cPres_Global
  texture arena of 0x2c70170a0007000c).
- Within each small world every texture-shaped referenced id resolves inside that world's own
  archive (unres.py: BlackBoxPark, MegaPark, StartPark, IndustrialSkatePark).
- SFIL asset id is not a function of the name: 2,044 of 8,486 names occur under more than one
  asset id; FNV-1/1a 64 and MD5 of the name strings do not give it.
- Id families: `0x0000xxxx03e3870a` (cPres_Global of every world: shared engine-default textures)
  and `0x2c70170a00xxxxxx` (world content). .rx2 textures use real paths as names
  (e.g. `skatereel_video_textures\1.Texture`) with hash ids.

## The four "default" textures
Located by name id in the census (all retail archives, after decompression):
| id | where | what |
|---|---|---|
| 0x0000043d03e3870a | cPres_Global.xsf of all 10 worlds (asset 0x75C38E99585BDB61) | 16x16 R5G6B5, 5 levels |
| 0x2c70170a000b0040 | cPres_Global.xsf of 7 worlds (not BlackBoxPark, MegaPark, StartPark) | 16x16 R5G6B5 |
| 0x2c70170a0007000c | asset 0x70D72AE0FF9ED2BA: BlackBoxPark cPres_Global + 29 DownTown/Industrial/University/IndustrialSkatePark/SkateSchool streams | 512x512 DXT1 |
| 0x054a480902fd9a8d | SkateSchool cPres_Global only (asset 0x55553D5FE6F42590) | 512x512 DXT1 |
None of them exists in any non-world archive (big/*.big, fe, parkassets, createacharacter,
livingworld, marquee, missions, worlddmo, worldmisc): they are not base-game resident data, each
world ships its own copy, and worlds that do not reference one do not carry it. A DLC map cannot
count on them being loaded; it must carry its own copies (same name id is fine - every world
repeats them).

## Compressed BIG entries (needed to read non-world archives)
`sk3/big.py` reads record word1 as "offset high". In the 0x00100600 archives it is the stored
(compressed) size: offset = word0 << ((flags >> 8) & 0xFF) (shift 6 in 0x00100600, 4 in
0x00100400 world archives, where word1 is always 0), size = word2. A stored entry is either
RefPack (`10 FB`) or a `chunkref` container: `"chunkref", u32 2, u32 total size, u32 chunk size
0x20000, u32 chunk count, u32 align 0x10, 3 x 0`, then per chunk a header `{u32 stored size,
u32 kind (2 = RefPack)}` placed so the chunk data starts 16-aligned (first data at 0x30).
Measured: every entry of every non-audio archive decompresses to its word2 size
(`work/scratch/tex/bigx.py`, used by census.py; 27,654/27,654 entries). Note worlddmo.big and
worldmisc.big are 0x00100600 (shift 6) with word1 = 0, so `big.py`'s offset*16 is wrong there too.

## Round trip and encoder
- `texture_write.build(extract(x)) == x` for every world texture arena: small worlds
  BlackBoxPark 90, MegaPark 48, StartPark 45, IndustrialSkatePark 77, MaloofMoneyCup 170,
  DownTownSkatePark 153, SkateSchool 188; large DownTown 23,639, Industrial 11,419,
  University 13,525; worlddmo 281, worldmisc 281, proxyDowntown 558, proxyIndustrial 384,
  proxyUniversity 458 = all 51,316 world-stream textures. The spec holds: arena id, name id, format, width, height, levels, cube,
  linear block data per (level, face), plus the carried name-table word, pad byte and (for stand-ins)
  the flagged id.
- `from_image`: power-of-two RGBA -> box-filtered full mip chain -> our numpy BC1 (principal axis +
  2 least-squares passes, 4-colour mode forced) / BC3 (8-level alpha) -> tiled arena. Decoding
  our own output: re-encoding decoded retail DXT1 images gives mean abs error 0.19 (max 0.63) per
  channel; smooth synthetic images < 5 on levels >= 32 px. `lowres_standin` makes the cPres copy.
