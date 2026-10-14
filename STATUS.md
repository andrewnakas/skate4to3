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

## Next
P1.1 measure the TOC chunk table from bytes; oracle = `reskate_cli dump-res`.

## Owner decisions pending
- none

## Notes
- Studio CLI: "D:\ReSkate Studio\reskate_cli.exe"; game root "C:\Program Files (x86)\Steam\steamapps\common\Skate".
- One export at a time (RAM). Don't run while the owner plays skate. or Skate 3.
