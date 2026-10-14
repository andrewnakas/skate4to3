# STATUS (loop state)

Plan: C:\Users\andre\.claude\plans\splendid-toasting-music.md (P0 setup -> P6 publish).
Old pipeline (reference for OUR code only): C:\Users\andre\Documents\skate4maps\skate4mapsconversionpipeline,
history bundle D:\n64work\archive\skate4mapsconversionpipeline.bundle (+ .uncommitted.diff).

## Done
- P0: repo, CLEANROOM.md, THIRD_PARTY.md, tools/taint_scan.py (0 hits). Front end carried over into sk4/
  without the third-party-derived files; stubs that raise until P1 re-measures them:
  `superbundle.Toc.chunk` (TOC chunk table), `collect.extract_decals` (decal volumes).
  Still missing (P1): depot texture/tint binder, terrain presets, decal bake.
  `meshset.py` section +0x138 depot key: carried, must be confirmed against Studio `mesh-sections` in P1.
  grind_edges.py + glb_writer.py vendored from our own d2s3.

- P1.1 TOC chunk table re-measured (tools/measure/toc_chunk_table.py, spec "TOC chunk table");
  tests/test_oracle_chunks.py: 36 chunks over 12 TOCs == Studio encoded+decoded sizes.
- P1.2 section +0x138 depot key: tests/test_oracle_meshes.py, 400 MeshSets == Studio mesh-sections.

## Next
P1.3 ShaderBlockDepot decode. Measured so far (bench depot, work/oracle/depot1):
- relocatable blob; header u64 ptr->key table at 0x10; key table = {u64 key, u64 ptr->descriptor};
  descriptor = {u64 ptr block, u64 hash, u64 size}; meta = reloc off, reloc size, u16 nDescriptors, u16 ?, u32 nKeys.
- block = u32 count, then records sorted by u64 hash: hash8 type4 u16 count u16 id u8 n, payload.
  tex 0x14C5FC30: 2x16 B guids (no pad). Value types: 3 zero bytes + count*elem (4 B scalars
  0x8E4A817E/0x2BB65C2D(float)/0x36ED3444/0x7826E998, 12 B vec3 0x885EFF59); other types unknown size.
- 2,475 Studio-dumped depots in work/oracle/depots (index.tsv) to learn sizes; oracle `shader-params <key>`.
Then: semantics of param hashes (colour/normal/mask/tint per layer) learned against Studio texture names.

## Owner decisions pending
- none

## Notes
- Studio CLI: "D:\ReSkate Studio\reskate_cli.exe"; game root "C:\Program Files (x86)\Steam\steamapps\common\Skate".
- One export at a time (RAM). Don't run while the owner plays skate. or Skate 3.
