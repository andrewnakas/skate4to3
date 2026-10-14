# skate4to3

Converts levels from **skate.** (EA, 2025, Frostbite) into DLC packs for the **Skate 3** Xbox 360 recomp,
in one self-contained Python pipeline: skate. files -> per-tile scene -> Skate 3 arenas (render,
collision, grind splines) -> streams/manifests -> `.big` + `.header` pack.

Work in progress; see `STATUS.md`. Rules for what code may be read while writing this: `CLEANROOM.md`.

Layout: `sk4/` reads skate., `sk3/` writes Skate 3, `tools/` builds and checks, `specs/` holds every
measured format fact, `tests/` holds round-trip and oracle tests.
