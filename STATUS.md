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

- P3 done by agents, all byte-exact on retail: RW4 arena writer (56,215 arenas), render meshes/materials
  (29,578 cPres arenas) + builder, textures (51,316) + BC1/BC3 encoder, collision mesh/KD/splines (all
  10 worlds) + builder, packaging (SFIL 7,089 streams, manifests, BIG incl. 3.9 GB pack, DLC metadata).
- Vault/locator name hash found in the retail code (lookup8, a=b=0xABCDEF0011223344, c=golden ratio).
- Template-free DLC generator (sk3/vault.py, dlc.py): no community pack needed.
- P4: sk3/assemble.py (GLB blocks -> 100 m tile arenas), tools/build_pack.py, tools/pack.py.
- IN GAME (2026-10-14 ~06:30): Summer Camp built 100% by skate4to3 loads and plays on the owner's test
  rig (tools/boot_test.sh: rig at D:/n64work/sanvan/colldiag/run, travel via injected pause-menu inputs).
  Root cause of the first hang: one 7 MB render arena per tile; retail arenas are <= 921 KB / 60 meshes,
  so assemble now splits (<= 48 meshes, ~750 KB, <= 8,000-vertex parts).
  Bisection packs (work/hyb): C0 (old tiles + our packager) OK, A (our collision) OK, B2 (our textures) OK.

- Isle of Grom built end to end (40 tiles, 390 MB) but CRASHES in game ~60 s after travelling to it.
  The OLD pipeline's Isle of Grom pack crashes the same way, so it predates skate4to3.
  Bisection (work/var, tools/variant_boot.sh): collision-only packs crash too; any 39 of the 40 tiles
  load, all 40 crash (the full 2x2 block at the far corner 250..350 x 550..650 is needed) - preload
  cap, texture budget, boundary, instance-free defer and heap quarantine do not help.
  --skate3_heap_check=true reports a corrupt free-list link after "250 instance teardowns ran with
  job(s) executing" = the recomp's own known WorldPresentation teardown race; faulting function
  sub_82990768 (allocator). Needs engine-side debugging (owner's recomp) - see Owner decisions.
- Rig incident (2026-10-14 ~09:10, fixed): a failed variant build made smoke_boot.sh run with an empty
  pack path and wipe the rig's dlc/ folder; NEW_SAN_VAN (hard link to packs/fixed.big) and TESTPACK
  (copied back from %APPDATA%\skate3\dlc\SKATE_2_DLC) restored; script now refuses empty/non-pack paths.

## Next
- P5 quality on Summer Camp: colours/tints vs skate. (owner eye), lightmap brightness, decals.
- P6: push the private repo, final docs.

## Owner decisions pending
- Isle of Grom crash: engine-side heap race in the recomp (old pack crashes too) - debug in the recomp, or ship Summer Camp first?
- Summer Camp colours: whitish props / pale ramps vs the old pack's black/orange - which matches skate.?

## Notes
- Studio CLI: "D:\ReSkate Studio\reskate_cli.exe"; game root "C:\Program Files (x86)\Steam\steamapps\common\Skate".
- One export at a time (RAM). Don't run while the owner plays skate. or Skate 3.
