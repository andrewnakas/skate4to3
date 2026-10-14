# Clean-room rules

skate4to3 converts skate. (EA, 2025) levels into Skate 3 DLC packs for the Xbox 360 recomp. Every line
of code here is written by this project. Community tools that other people asked us not to republish,
or that are not ours to republish, are never read while writing code here.

## Never read while writing code
- the community skate. map extractor and anything derived from it (including our own old pipeline files
  that ported it: the old depot binder, terrain-preset binder, decal bake, TOC chunk reader, decal-volume
  reader, extractor-texture importer);
- the community Skate 3 arena / DLC builder tool chain and the scripts it was ported from;
- third-party Skate 3 reader code in other engines.
`tools/taint_scan.py` lists the exact names and must report 0 before every commit.

## Allowed evidence
- Retail game files (skate. and Skate 3), measured byte by byte with our own code.
- ReSkate Studio's command-line tool run as a black box on the owner's machine (its printed output
  only - never its binary internals): `mesh-sections`, `shader-params`, `dump-res`, `levels`, ...
- How the Skate 3 recomp behaves when it loads our files.
- Our own code from earlier projects (front-end readers, stream/manifest/BIG code, `.header` writer).
- Permissively licensed libraries listed in `THIRD_PARTY.md`.

## Specs first
Every format fact goes into `specs/` with how it was measured (file, offset, count, oracle) before
code relies on it. Facts that were known before this repo existed are re-measured, not copied.

## Outputs
Built packs contain the owner's own game data and are for personal use only: never commit or publish
them, `work/` or `out/`.
