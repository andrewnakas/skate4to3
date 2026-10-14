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

- P1.3a depot parser (sk4/depot.py, spec "ShaderBlockDepot"): all 64,330 blocks walk exactly; oracle
  tests vs Studio listing + guid membership. Studio's printed texture NAMES are off-by-one/blank
  for ~70% of keys - not an oracle; our partition-guid index resolves 100%.
- P1.3b binder v1 (sk4/bind_keys.py -> depot_textures.json): role per param hash learned from bound
  texture suffixes (169 hashes), primary layer = shader's base colour slot. 52,653/63,805 keys bound.
  Param names (Param_Base_C ...) exist in shader EBX but the 64-bit hash function is unknown
  (FNV/djb2/sdbm/crc/md5 all miss) - not needed.

## Next
P1.3c calibrate layer choice + tints against skate.'s baked far-LOD proxy atlases
  (`*_proxy_c` / M_Proxy_Atlas sections = the game's own baked look of the mesh): for meshes with a
  proxy, fit which layer/vec3 reproduces the proxy colour; learn per-hash tint roles from that.
  Known wrong today: bench QD_BenchPark_01_Wood_M picks paint over lumber; no tints at all.
P1.4 terrain presets + terrain quads; P1.5 decal volumes + bake. Then P1 gate (full Summer Camp
  front-end run vs old pipeline numbers), then P2.

## Owner decisions pending
- none

## Notes
- Studio CLI: "D:\ReSkate Studio\reskate_cli.exe"; game root "C:\Program Files (x86)\Steam\steamapps\common\Skate".
- One export at a time (RAM). Don't run while the owner plays skate. or Skate 3.
