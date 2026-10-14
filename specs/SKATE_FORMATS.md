# skate. (Steam app 3354750) — container formats

Reverse-engineered 2026-09-11 from the Steam install. Engine is **Frostbite**
(not Unreal): `Data/Win32/` layout, CAS content store, Oodle (`oo2core_9_win64.dll`).

## Install layout
```
Skate/Skate.exe                                   137 MB
Skate/Data/layout.toc                             DbObject: superbundle + install manifest
Skate/Data/Win32/levels/game/<level>/<level>.toc  binary superbundle TOC (per level)
Skate/Data/Win32/configurations/layout/
    defaultinstallpackage/cas_NN.cas              74 files, 11.6 GB total, up to ~1 GB each
    initialinstallpackage/<lang>/cas_NN.cas
```
There are **no `.cat` catalog files** — CAS resolution is embedded in the TOCs.

Levels present: `bam_levelroot`, `dingolevel_ftue_island`,
**`dingolevel_isle_of_grom`**, `dingolevel_root`, `dingolevel_sdm_int_001/002`,
`dingolevel_splash`.

## Common TOC header
```
0x000  u32   magic 0x01CED100
0x004  u32   0
0x008  256B  signature block
0x108  'x' + 256 ASCII hex chars + 'x'    (128-byte signature)
       zero padding
0x22C  payload
```
The payload is **not** encrypted or obfuscated — earlier Frostbite XOR
deobfuscation does not apply here. Dispatch on the first payload byte:
`& 0x80` set => DbObject, otherwise => binary superbundle TOC.

## DbObject (layout.toc) — see `tools/dbo.py`
Entry = `u8 type` (bit `0x80` = unnamed, i.e. array element or root),
optional NUL-terminated name, then value. Containers are LEB128 size-prefixed.
Types confirmed: `01` array, `02` object, `07` string (LEB128 len incl. NUL).
Root size lands exactly on EOF — used to locate it without hard-coding 0x22C.

## Binary superbundle TOC (level TOCs) — big-endian
15 × u32 header (60 bytes), offsets relative to payload base 0x22C:
```
[ 0] bundleHashMapOffset   60        header is exactly 60 bytes
[ 1] bundleDataOffset      1064      = 60 + bundlesCount*4
[ 2] bundlesCount          251
[ 3] chunkHashMapOffset    5080      = 1064 + bundlesCount*16
[ 4] chunkGuidOffset       91444     = 5080 + chunksCount*4
[ 5] chunksCount           21591
[ 6] 523264   = 91444 + chunksCount*20 (chunk records)
[ 7] 523264
[ 8] 868720
[ 9] 523264
[10] 86364    = chunksCount*4
[11] 4
[12] 4585
[13] 76
[14] 887060
```
All arithmetic relations above were verified against `dingolevel_isle_of_grom.toc`
(1,998,268 bytes). Hash-map slots are int32; negative = empty.

Bundle record (16 B): `[u32 a, u32 b|0x40000000, u32 0, u32 c]`.
### TOC chunk table (measured 2026-10-14, `tools/measure/toc_chunk_table.py`)

`chunkGuidOffset` holds `chunksCount` x **20 B** entries (the section is exactly 20 B/chunk in
every TOC, e.g. items.toc 311,620 B / 15,581): 16 B guid then u32 BE `idx`.
- The guid is stored **byte-reversed** relative to the order DxTexture/MeshSet descriptors use.
  Measured: descriptor guid `2819ead3...f63f` (texture `metal_steel_02_2x2m_c`) is absent in
  descriptor order and present reversed at +172,220 = 8,611 x 20 (an entry boundary).
- `idx` always has the high bit set (0 exceptions over 24 TOCs); `idx & 0x7FFFFFFF` is a byte
  offset into 4-byte words (low 2 bits always 0), so the location record is
  `(idx & 0x7FFFFFFF) >> 2`. Oracle: that record's size = 388,756 = ReSkate Studio's `encoded`
  size for the same chunk; the other candidate mappings land on unrelated records.
- Over all 24 TOCs: no record index out of range, no record shared by two entries.
- Studio prints guids in Microsoft text form = `uuid.UUID(bytes_le=<descriptor-order bytes>)`.
- Localisation TOCs reference extra package ids (e.g. 0xEAF2/0x7217) with no package on disk.

### Chunk CAS records — CONFIRMED

`chunkRecordsOffset` (header field 6) holds `chunksCount` × 16 B, big-endian:

```
u32 packageId   u32 (packageHi<<16 | casIndex)   u32 offset   u32 size
```

`(field0, field1>>16)` takes exactly two values across all 21,591 chunks, and each
maps to one install package:

| field0 | hi16 | package | cas indices used |
|---|---|---|---|
| 0xBC80 | 0xE699 | `initialinstallpackage` | 28 distinct, all ≤ 45 |
| 0x7837 | 0x152C | `defaultinstallpackage` | 15 distinct, all ≤ 25 |

Every referenced index exists as a root-level `cas_NN.cas`, and every
`offset+size` lands inside its file. Language subdirectories only ever hold
`cas_01`/`cas_02`, which no level chunk references, so the root file is
unambiguous.

## Block compression — CONFIRMED

Each chunk is a sequence of blocks with an 8-byte big-endian header:

```
u32 decompressedSize   u8 compressionType   u8 flags(0x70)   u16 compressedSize
```

Walking `8 + compressedSize` lands exactly on the chunk end for 88.5% of chunks.
Only two type bytes occur:

- `0x00` — stored; `compressedSize == decompressedSize`
- `0x11` — **Oodle Kraken**; every one of 1,084 sampled payloads begins `0x8c`

The other 11.5% (354 of 3,085 sampled) are a different container, all sharing the
identical 16-byte prefix `00010000 0071 0000 4800 000e 342ee010`. Not block
headers — `00010000` recurs at +0 and +48 but not on a 64 KiB stride. Unidentified;
possibly textures or a resource table. Geometry has been found without them.

### Decompressing

Oodle is proprietary and the open-source reimplementations (`powzix/ooz` and its
forks) are all gone — every candidate 404s. The game ships `oo2core_9_win64.dll`,
which is the exact codec version the data was packed with, so we call that:
`tools/oodle_batch.c` → Win64 shim run under wine, batched (wine startup would
otherwise dominate at tens of thousands of blocks per level). See `tools/oodle.py`.

Verified: 41 sampled Kraken blocks decompress with byte-exact output sizes.

## Geometry

Decompressed blocks contain vertex buffers as **half-float4** `(x, y, z, 1.0)` —
8 B stride. A block holds many assets concatenated, so the half4 run ends and
other data follows; asset boundaries must come from the bundle index, not from
scanning.

No plaintext asset names anywhere in the level TOC — names are hashed; real
names live inside EBX assets in the bundles, after decompression.

## Bundle manifest (entry 0 of every bundle) — CONFIRMED on all 251

Stored raw. `u32 BE size`, `u32 BE magic 0xD68E799D`, then **little-endian**
`totalCount, ebxCount, resCount, chunkCount, stringsOffset, chunkMetaOffset,
chunkMetaSize` (offsets relative to +4), then sha1 x total, ebx {nameOff, size},
res {nameOff, size}, resType u32, resMeta 16 B, resRid u64, chunks {guid, logicalOffset,
logicalSize}, strings, chunkMeta DbObject. Entries 1.. are ebx, res, chunks in that order.
The flags byte low nibble in each block header is bits 16-19 of compressedSize.

Level superbundle TOCs size the bundle hash map at nb or nb+1 slots.

## RIFF EBX

`RIFF` form `EBX\0` or `EBXS`; chunks `EBXD`, `EFIX`, `EBXX`, `RFL2`. EBXD offsets are
relative to its payload rounded up to 16.

RFL2 is self-contained reflection — no SDK dump needed:
```
u32 n, n x {guid16, u32 sig}                                      type GUIDs
u32 n, n x {nameHash, fieldStart u32, fieldCount u16, flags u16, size u16, align u16}
u32 n, n x {nameHash, offset u32, flags u16, typeRef u16}          fields
u32 n, n x {nameHash, valueCount u32, valueStart u32}              enums
u32 n, n x {nameHash, value u32}                                   enum values
u32 n, n x {nameHash, stringOffset u32}                            names (sorted by hash)
strings
```
Field kind = `(flags >> 5) & 0x1F` (Frostbite EbxFieldType), `flags & 0x08` = array.

EFIX: partition guid; type guids; type sigs; exported count; data container offsets;
pointer offsets; resource ref offsets; imports {partition, instance}; import offsets;
typeinfo offsets; arrayOffset, boxedOffset, stringOffset (+1 trailing u32).

Data container = 24-byte header whose first u64 indexes EFIX type guids. Pointer,
CString and array fields hold a **signed int32** self-relative offset in the low half of
8 bytes (arrays routinely point backwards). Odd pointer = import, index `>> 1`. Arrays
carry a u32 count before the first element. `BoxedValueRef` = `{u64 type, i32 rel @+8}`.

## MeshSet (res 0x49B156D4)

Pointers relative to +0x10. LOD record 0xB0: sectionCount +0x08, sections +0x0C,
IB size +0x58, VB size +0x5C, chunk guid +0x74, name +0x8C. Section 0x180: material
+0x08, primitives +0x20, startIndex +0x24, vertexOffset (bytes) +0x28, vertices +0x2C,
16 x {usage, format, offset, stream} +0x70, strides +0xB0, counts +0xD0, AABB +0x150.
Multi-stream VB; IB at VB size; u16 indices. Position usage 1 format 8 (half4), UV0
usage 0x21 format 6 (half2). `_Shadow` sections are position-only proxies.
Validated on 40 random meshes: every index in range, positions inside section AABBs.

## World placement

- `StaticModelGroupEntityData.MemberDatas[]` → `MeshAsset` import, `InstanceTransforms`
  (LinearTransform rows right/up/forward/trans), `InstanceCount`. Transforms are world
  space and orthonormal to 1e-4.
- Root layer: `SpatialReferenceObjectData` (generated roads), `ObjectReferenceObjectData`
  (plots) → `ObjectBlueprint` → `StaticModelEntityData` → one mesh.
- ECS prefabs: `EcsRuntimePrefabAsset` entity table (ParentIndex), component segments by
  GUID with Static/DynamicInstances (1-based, 0 = default, 0xFFFFFFFF = absent) into
  edits of key/boxed-value pairs. `ca3770bf…` LocalTransform (Sqt: scale, quat xyzw,
  translation), `2fc5fe8d…` EcsMeshComponentAsset, `b6cd360d…` spline control point,
  `2e6a66cc…` spline (Tension/IsClosed).
- Grindability is in material names: `…_Grindable` / `…_NoneGrindable`.

### MeshSet section +0x138 = ShaderBlockDepot key (re-measured 2026-10-14)
`tests/test_oracle_meshes.py`: for 400 MeshSets dumped by `reskate_cli dump-res` (filter `_mesh`), the
u64 at section +0x138 equals the `key=` Studio's `mesh-sections` prints for every section of every LOD,
in order (with material names).

## ShaderBlockDepot (res 0x73312045) - measured 2026-10-14 (`sk4/depot.py`)
Relocatable blob. resMeta = u32 relocOffset, u32 relocSize, u16, u16, u32 keyCount.
- +0x00 u64 -> key table (0x10 in all 2,475 depots). Key table: keyCount x {u64 materialKey, u64 ->
  descriptor}. Several keys share a descriptor (e.g. a material and its `_Shadow` variant).
- descriptor: {u64 -> block, u64 hash, u64 size}; blocks are packed back to back.
- block: u32 recordCount, then records (NOT sorted by hash):
  `u64 paramHash, u32 type, u16 count, u16 tag`, then the payload:
  - type 0x14C5FC30 (texture): u8 flag (1), then {16 B instance guid, 16 B partition guid};
  - every other type: u32 count (equal to the header count in every record, including 256), then
    count x element. Element sizes: 0x566E2DD0 = 1;
    0x57F970CA, 0x2BB65C2D (float), 0x36ED3444, 0x8E4A817E, 0x7826E998 = 4; 0x2A36E9D5 = 8;
    0x885EFF59 (vec3) = 12; 0x7F999829 = 16 (vec4).
  With exactly these sizes all 64,330 blocks walk to their end (`work/scratch/learn_depot.py`).
- Oracle: record hashes, types and order == Studio `shader-params` for 157 sampled keys; vec3/float
  values equal (`tests/test_depot.py`).
- Texture binding: both guids of a record belong to one TextureAsset EBX - the partition guid is the
  EBX's EFIX partition guid and the instance guid occurs inside that EBX (400/400 sampled,
  `tests/test_depot_guids.py`). Studio's *printed texture names* are not a usable oracle: they equal
  ours for 45/168 keys, are shifted by one record for 43, and are blank/different for the rest
  (Studio leaves global textures such as panoramas and default masks unnamed).
- The u16 tag is constant per parameter across shaders, the u64 hash differs per shader (our own
  2026-09-23 notes, re-checked below). Roles are learned per tag from bound texture name suffixes.
