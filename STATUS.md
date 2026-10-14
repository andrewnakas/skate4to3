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

- P1.3c layer/tint rules: tools/calibrate_layers.py samples skate.'s baked proxy atlases (4,000 meshes),
  tools/fit_layer_rules.py keeps per-shader rules only where 2-fold held-out error beats the default by
  >=10% (18 shaders, 4,222 sections: 0.925 -> 0.549). bind_keys applies them (30k tints, 5k layer picks).
  CAUTION: tint hash 0x67978f09c3b45f30 x2 is tiny in some shaders - check renders in P5.
- P1.4 terrain presets: surface name -> world/materials/terrain family by token overlap (9 surfaces);
  bake_materials (ours) re-bakes to the preset colour. Level filter now includes road-generator meshes.
- P1.5 decals: collect.extract_decals rewritten (658/658 volumes match the old extraction by position,
  row/col, template; keys changed with the 10-13 game update); sk4/decal_sheets.py bakes 61/64 templates
  (opacity = osk/osm RED, grid cols tag 0x406E / rows 0xF7A1, colour tag 0x8C54 x2).
- tools/front_end.sh <level> <dir>: index -> partitions -> guid index -> bind_keys -> collect -> level_keys ->
  extract -> inherit -> bind_by_name -> build_key_textures -> decal_sheets -> bake_materials.
- P2 started: sk3/big.py (all 28 named retail archives), sk3/sfil.py, sk3/refpack.py, sk3/rw4.py,
  tools/measure/sk3_census.py. A background agent is writing sk3/rw4_write.py (byte-exact round trip).

## Next
- finish the Summer Camp front-end run (extract with the wider level filter), then export GLBs
  (sk4/export_glb.py) and compare verify/audit numbers with the old v6b build logs.
- P2/P3: arena writer (agent), then mesh/material/texture/collision/spline object specs + builders.

## Owner decisions pending
- none

## Notes
- Studio CLI: "D:\ReSkate Studio\reskate_cli.exe"; game root "C:\Program Files (x86)\Steam\steamapps\common\Skate".
- One export at a time (RAM). Don't run while the owner plays skate. or Skate 3.
