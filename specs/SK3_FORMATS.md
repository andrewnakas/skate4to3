# Skate 3 (Xbox 360) formats - measured facts

Retail data: the recomp's `game/data` (C:\dev\skate3recomp\out\build\windows-release\game\data).
World maps ship as `content/worldDIST_<Map>.big` with root `data/content/world/stream/DIST_<Map>`.

## BIG archive ("EB\0\3", big-endian) - `sk3/big.py`
Header: +0x04 entry count, +0x08 flags (0x00100400 world archives, 0x00100600 multi-root content,
0x00500400 DLC per our earlier work), +0x0C name table offset, +0x10 name table length,
+0x14 name layout = slotStride<<24 | rootStride<<16 | rootCount, +0x1C total size,
+0x20 path-directory length. Records at 0x30, 16 B: u32 offset/16, u32 offset-high, u32 size,
u32 id = djb2("<root>/<name>") (h = h*33 + byte from 5381), ascending id.
Name table: one zero byte, then `count` slots of `slotStride` bytes in record order:
[u8 rootIndex][name NUL-padded]; root region from the next 32-byte boundary, `rootCount` slots of
`rootStride` bytes.
- Measured over all 28 named retail archives (content/, big/): the header stride is exact in every
  one; every record's slot hashes to its id once the root index is read modulo 256 (missions.big has
  401 roots: slot byte 30 = root 286). Audio archives (data/audio) carry no names at all.
