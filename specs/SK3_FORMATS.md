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

## RW4 arena ('\x89RW4' 'xb2\0', big-endian) - `sk3/rw4.py`, `sk3/rw4_write.py`
Measured with our readers over every arena in the SFIL streams (cPres_*/cSim_*/cTex_*, cPres_Global,
cSim_Global) of the 7 small world archives (BlackBoxPark 110, MegaPark 77, StartPark 71,
IndustrialSkatePark 91, MaloofMoneyCup 214, DownTownSkatePark 205, SkateSchool 549 = 1317 arenas;
scratch scripts work/scratch/rw4_survey.py, rw4_types.py, rw4_h6c.py) and then proven by
`tests/test_rw4_roundtrip.py`: `rw4_write.build(spec_from_arena(parse(x))) == x` for every arena,
where the spec holds only the arena id, the objects in dictionary order as (type id, align, data)
and the subref records. Every rule below is therefore exact on all of them.
The same round trip then passed on the three large worlds too (DownTown 25617, Industrial 14105,
University 15176): 56215/56215 retail world arenas, so the two registries, the 0x6C pairing and every
layout rule below also hold there (the counts quoted below are from the 1317 small-world arenas).

File order: header 0xAC | section block | main objects | dictionary | subref records + runtime
table | graphics region (base resources). The main region ends at h44; file length = h44 + h54.

Header (u32 at offset; "always" = identical in all 1317):
- 0x00 '\x89RW4', 0x04 'xb2\0', 0x08-0x17 always `0d0a1a0a 01200400 34353400 30303000` ("454", "000"),
  0x18 always 0.
- 0x1C arena id (also twice in section 0x10006).
- 0x20 = 0x24 = object count (dictionary entries).
- 0x28 always 0x10; 0x2C, 0x38, 0x3C, 0x40 always 0.
- 0x30 dictionary offset; 0x34 section block offset, always 0xAC.
- 0x44..0x93: ten (size, alignment) pairs:
  (0x44 main region size = end of subref runtime table, 0x48 always 0x10 = also the largest main
  object alignment in every arena), (0, 1),
  (0x54 graphics region size = end of last base resource, no tail padding,
  0x58 largest base-resource alignment, 1 when there are none), (0, 1), (0, 1),
  (0x6C, 0x70 always 4), then (0, 1) x4. 0x94-0xAB always 0.
- 0x6C is 0x304 in every arena with the 9-entry type registry and 0xAC in every arena with the
  63-entry one (fit over all 1317: no dependence on object/type/subref counts or sizes).

Section block at 0xAC, packed, offsets in the directory relative to 0xAC:
- 0x10004 directory {id, 4, 0x0C, offsets of the next four sections} (0x1C bytes; it does not list itself).
- 0x10005 types {id, count, 0x0C, type ids}. Only two registries occur:
  9 types `0 10030 10031 10032 10033 10010 200e8 eb0008 eb000b` (771 arenas: cTex, most cPres_Global,
  some cPres) and 63 types starting `0 10030 10031 10032 10033 10010 eb0000 ...` (546: cSim,
  cSim_Global, the rest; full list in rw4_write.REGISTRY_FULL). Every arena uses the 9-type registry
  exactly when all its object types are in it (771/771 and 546/546). The 63 list is not sorted and
  does not contain the 9 list in the same order (200e8 comes after eb0008/eb000b).
- 0x10006 arena guid {id, 3, 0x18, arena id, 0xFFB00000, arena id, 0, 0, 0} (0x24 bytes).
- 0x10007 subrefs {id, n, 0, 0, records end, records start, n} (0x1C bytes); records start =
  dictionary end.
- 0x10008 atoms {id, 0, 0} (0x0C bytes), always empty.

Main objects: dictionary order = file order; the first starts at the end of the atoms section
aligned to its own alignment, each next one aligned to its alignment after the previous one's end;
padding is always zero bytes. Alignments seen: 4 and 0x10. Sizes need not be multiples of 4
(1184 odd-size objects), but the last main object always ends 4-aligned and the dictionary follows
it directly (so "dictionary aligned to 4" and "dictionary right after" are not distinguishable).

Dictionary: 24 bytes per object {ptr, 0, size, align, index into the type registry, type id};
+0x04 is 0 in every entry. Base resources (types 0x10030-0x1003F; only 0x10031 occurs) are
mixed into the dictionary order (in 140 arenas they are not first); their ptr is relative to h44.

Subref records: n x {u32 dictionary index, u32 byte offset inside that object} (offset < object size
in all 8110 records; targets are types eb0001/eb0005/eb0004/eb000d/eb0019/eb0009), not always
sorted (169 arenas), followed by n x 24 zero bytes (a runtime table); h44 = that table's end.

Graphics region: base resources in dictionary order, each aligned (relative to h44, not to the file)
to its own alignment, zero padding. Alignment 0x1000 (771 arenas, the 9-type registry) or 4
(140 arenas); 406 arenas have no base resource (h54 = 0, h58 = 1). h44 itself is not 16-aligned
(only 4-aligned), so on disk a 0x1000-aligned resource is not page-aligned; the Xbox two-section
SFIL form stores this region separately (BlackBoxPark: main raw size = h44 and graphics raw size =
h54 in all 110 payloads).
