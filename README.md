# skate4to3

Converts levels from **skate.** (EA, 2025, Frostbite) into DLC packs for the **Skate 3** Xbox 360 recomp:
skate. files -> per-tile scene -> Skate 3 arenas (render, textures, collision, grind splines) ->
streams, manifests, vaults -> `.big` + `.header`. Every stage is this repo's own code, written from
measurements of the games' own files (see `CLEANROOM.md`, `specs/`).

## Build a map
Requirements: Windows, Python 3.12 with numpy, Pillow, pyfqmr; skate. installed through Steam;
the Skate 3 recomp (for its `game/data`, used only to test).

```bash
bash tools/build.sh dingolevel_mpr mpr "Summer Camp"      # ~1.5 h; output work/mpr/pack/<ID>/
```
Copy `work/<dir>/pack/<ID>/` into the recomp's `dlc/` folder and pick the map in the startup picker,
then travel to it from the pause menu's Locations tab.

Built packs contain the owner's own game data: personal use only, never commit or publish them.

## Layout
`sk4/` reads skate., `sk3/` writes Skate 3, `tools/` builds and checks (`boot_test.sh` boots a pack
unattended on a test rig), `specs/` holds every measured format fact, `tests/` (pytest) holds the
round-trip and oracle tests (`python -m pytest -q tests`).
