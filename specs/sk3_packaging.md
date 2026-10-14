# Skate 3 packaging: streams, manifests, archives, DLC packs - measured facts

Scope: from "a set of arenas per tile" to "an installed DLC pack". Code: `sk3/sfil.py`, `sk3/refpack.py`,
`sk3/manifests.py`, `sk3/big.py`, `sk3/dlc.py`, `tools/pack.py`. Every fact below says how it was
measured; "all N" means every instance in the corpus, checked by a test or a scratch survey.

Corpus (bytes only, read with our own readers):
- Retail: the 10 `content/worldDIST_*.big` of the recomp's `game/data`, plus `big/db.big`,
  `big/miscboot.big` (vault and string-table oracles).
- DLC packs that load in the recomp (whoever built them): SUMMERCAMP (146 MB, 2 copies), SUMMERCAMPHD
  (2 builds, 332/338 MB), ISLEOFGROM, SKATE_2_DLC (5 worlds), SAN VANELONA (2.2 GB), NEW_SAN_VAN
  (3.9 GB), found under `%APPDATA%\skate3` and `D:\skate3-engine-0.1.5-windows-x86_64\dlc`.
- The owner's recomp picker (`C:\dev\skate3recomp\src\skate3_dlc_picker.cpp`) for loader rules.

## RefPack (`sk3/refpack.py`)
- Stream: u8 flags 0x10 (bit 0: 4-byte sizes; bit 7: compressed-size field first), 0xFB, 3-byte
  big-endian raw size; opcodes as in the module docstring. Every retail section starts `10 FB`.
- Our parse: greedy; at each position every earlier match of the next 3 bytes within 128 KiB is
  tried nearest first; best = most bytes saved (len - opcode bytes, 2/3/4 for the three copy
  forms), ties to the longer copy; nothing is copied into the last 4 bytes. Measured against EA's
  op streams (`work/pkg/rp_*.py`): the rule "longest match" diverged after 2356 ops in
  cSim_50_-50 (BlackBoxPark); "most saved" fixed that; the 4-byte tail rule fixed every stream end.
  Result: 26 of 68 StartPark RefPack sections (< 200 KB) are reproduced byte for byte. Every
  remaining divergence is a copy of > 16 KiB distance (4-byte opcode) that EA skips for a literal
  plus a nearer copy one byte later - EA's match finder does not see some far candidates (a
  chain or bucket limit we have not identified). Decoding is unaffected: `test_fresh_compression_decodes`,
  `test_refpack_roundtrip_synthetic`.
- EA's opcode-length census (StartPark): 4-byte copies of length 5 are the most common (29839),
  so EA has no minimum length for far copies.

## SFIL streams (`.xsf`, `sk3/sfil.py`, `tests/test_sfil_roundtrip.py`)
- Header 0x00 'SFIL', version 3, 0, group word, align, group count 1, platform, tail word; zero
  padding to `align`. Two forms in the corpus:
  - EA form (all retail streams, SKATE_2_DLC): align 0x80, platform 2, two-section payloads.
  - raw form (SUMMERCAMP, SUMMERCAMPHD, ISLEOFGROM, SAN VANELONA, NEW_SAN_VAN): align 0x100,
    platform 0, payload = the RW4 arena itself.
- Chunk: u64 asset id, u32 payload size, u32 payload offset (= align), u32 chunk size, zeros.
  Chunk size = align_up(offset + size, align) in EA streams; the raw-form packs add one more
  alignment block when the size is already a multiple of the alignment (SUMMERCAMP 15/15,
  ISLEOFGROM 11/11, SUMMERCAMPHD 7/7 such chunks; SKATE_2_DLC 50/50 use align_up).
- EA two-section payload: 0x50-byte descriptor, main {raw = arena h44, stored, RefPack flag} at
  0x00, graphics {raw = arena length - h44, stored, flag} at 0x20, everything else zero (807/807
  BlackBoxPark..SkateSchool chunks); stored main then stored graphics back to back.
- EA compresses a section iff its raw size >= 2048: over all 10 worlds the largest stored-raw
  section is 2044 bytes and the smallest compressed one 2056 (`work/pkg/sfil_thr.py`).
- Tail word: retail values look like stack garbage (0, 0x10100810, 0xC0920910, ...); raw-form packs
  use (tile count of the group) << 24, except SUMMERCAMPHD whose Sim streams carry 87 (the Pres
  count). The writer takes it as an input.
- The group word (+0x0C) equals the group's manifest group word (all streams checked by the
  manifest rebuild).
- Rewrite: `read` keeps each RefPack section's stored bytes; `write` reuses them while the arena is
  unchanged. Result: every stream of the 7 small worlds and of 4 DLC packs re-writes byte for byte
  (pytest), and DownTown 637/637, Industrial 1552/1552, University 635/635 (SK3_SFIL_ALL run).

## Manifests (`sk3/manifests.py`, `tests/test_manifests.py`)
Byte-exact read/write over every manifest in all 10 worlds and 6 DLC packs: 60 MMAP, 60 CMAP,
60 CSPA, 60 ATOC, 201 location tables (+2 loose retail `.loc`).
- All start {magic, u32 3, u32 0, u32 group word}. MMAP is 24 bytes, rest zero.
- CMAP: +0x10 entry count (= (len-0x20)/0x68 everywhere), +0x14 platform (2 EA form; 0 raw-form
  groups; 2 for an empty group even in raw-form packs; 3 in the empty mission manifests), +0x18
  0xDEADC0DE, +0x1C 0. Entry 0x68: u64 asset id, u64 group id, u32 alignment (0x80 / 0x100),
  u32 stream size, u32 chunk count, name[0x40] zero-padded, 0xFFFFFFFF, 0, 0.
  - asset id = FNV-1 64 (multiply then xor) of the tile name: 6778/6778 tiles.
  - *_Global id = FNV-1 64 of "<name>_<DIST>": 23/34 (the 11 others are packs whose DIST was
    renamed after hashing); Global group id = 0xE171E1ABFEDB69ED (all).
  - other tiles' group id = their kind's Global id (2326) or the root id (3319, retail Sim/Tex and
    one pack); 1133 use a Global id hashed under another DIST name. Both load.
  - stream size = file length - alignment (retail, SKATE_2_DLC) or the full length (raw packs).
  - order: *_Global first (all), tiles otherwise by name in packs; retail order not derived.
- CSPA: +0x10 cell count, +0x14 0; 80-byte cells {u64 asset id, 8 bytes, 4 float4}: one per CMAP
  entry except *_Global, same order (all 60). The 8 bytes are zero or stale ASCII in EA files
  ("\0k8jobs\"). Vectors: tile box min, tile box max (x/z = the grid cell, y = content), content
  box min, max, w = 1 (retail BlackBoxPark; the raw packs put the cell centre in vector 3).
  Owner's recomp traces (memory, NSV collision work): the first cell's box x-extent sets the
  group's cell size and the game waits only for cells whose box 1 overlaps the focus.
- ATOC: +0x10 record count, +0x14 0; 64-byte records {u64 asset id, 0xAB329A6A, u32 arena length,
  the arena header words +0x44..+0x68, u32 0, 4 bytes zero or stale (CDCDCDCD, "_Pro", ...)}.
  One record per distinct asset id in first-appearance order over the CMAP entries' chunks
  (University Pres 2665 records for 7736 chunks; all 60 match).
- Location table (`.loc`, text, CRLF, 4-space indent): `<LocationTable>`, per location Name,
  `<Category />` (packs) or `<Category></Category>` (EA), Transform = 16 floats "%.6f" joined by
  ", " (row-major, translation in the last row).
- `generate()` rebuilds MMAP/CMAP/ATOC byte for byte and the CSPA cell list for SummerCamp from its
  streams alone, and the retail SkateSchool ATOC/CMAP fields (test).

## BIG archives (`sk3/big.py`, `tests/test_big.py`)
Rebuilt byte for byte: all 10 retail world archives and all 6 DLC packs (incl. 3.9 GB NEW_SAN_VAN),
the DLC ones with data order and path directory GENERATED from the path list alone.
- Header: count, flags, name-table offset/length, layout word = nameSlot<<24 | rootSlot<<16 |
  rootCount, 0, total size (= file size), path-directory length, 12 zero bytes.
- Record: u32 offset >> shift, u32 stored size, u32 size, u32 id = djb2(root/name); shift =
  (flags >> 8) & 0xFF: 4 in world (0x00100400) and DLC (0x00500400) archives, 6 in the 0x00100600
  ones, whose word 1 is a compressed size (RefPack or 'chunkref' containers; the texture work's
  measurement, folded into the reader). Records ascend by id.
- After the records: align16(n) bytes, zero in world/DLC archives; then the name table.
- Name table: u8 0; n slots of nameSlot = longest name + 3 bytes in record order,
  [u8 root index][name, NUL-padded]; the roots at align16(n * nameSlot) of the table (this can
  overlap the last slot's final pad byte: Industrial), rootSlot = longest root + 1; table
  zero-padded to 16. Roots are listed in the order the data first uses them (all 16 archives).
- Data: from align16(name table end + directory), each entry 16-aligned, zero padding, file padded
  to 16 (zero-length entries share the next offset).
- Data order: DLC packs store entries in path-directory order (6/6); retail world archives use an
  order of EA's tool (Pres manifests xmm, xst, xsm, xss, then Sim ..., loc, Tex ..., then streams),
  which the rebuild takes from the archive.
- Path directory (DLC): see `build_directory`. Node list = pre-order walk, per directory its files
  then its subdirectories, each sorted by the UPPER-cased path (lower-case sort fails on
  'cPres_*' vs 'DIST_*'; upper-case reproduces all 6 packs, 350..2923 nodes). Header word 0 =
  36 + 20*N + 4, word 4 = flavour 1 (all 6). Node +0: first child -> parent's child count, else
  running count of non-first nodes; +4: first child -> index of the parent's run in the id list,
  else 0xFFFFFFFF; +8 pool offset; +C file flag; +10 parent id. Id list = 1, then per parent id
  ascending its children descending. Pool = top name then leaf names at first use.

## DLC pack contents (`sk3/dlc.py`, `tests/test_dlc.py`)
Inventory of a single-world pack (SUMMERCAMP, 415 files): 9 boundaries, 36 locator files
(18 .loc + 18 .rx2), 108 empty mission manifests, 173 streams + 12 DIST manifests + 1 Sim.loc,
61 vault files (20 x .vlt/.bin/.opt + vaultlist), 2 language files, 11 stream XML, 2 unlocks.
Byte-exact codecs over all 6 packs: boundary 95, neighbour tables 75, preload lists 95, language
strings 6, histograms 6, product unlocks 6, locator objects 190(+18), locator tables 190,
mission manifests 1140, vault/bin/opt chunk streams 161 each.

- Boundary XML (CRLF): one rectangle per location = tile-centre extents +- half a cell, points
  (minx,minz)(maxx,minz)(maxx,maxz)(minx,maxz) - every location of every pack.
- Neighbour table `stream/<DIST>_{Pres,Sim}.xml` (identical files): per tile (sorted by x, z) the
  existing tiles of its 3x3 ring sorted by (x, z) (86/86 SummerCamp rows, all packs regenerate).
  The owner's recomp traces (memory, NSV) show the game reads `stream/dist_<slug>_<key>dlc_<Type>.xml`
  per location; only SKATE_2_DLC ships those, so the generator writes them too.
- Preload list `stream/freeskate_dlc_<slug>_<key>.xml` (`Count="1"`): the 63 tiles nearest the
  centre (centre included) by distance, x, z. The packs' centre is round(p/cell)*cell - cell/2 of
  the location's position (all locations); the generator defaults to the tile containing it.
- Mission manifests: 12 per location, constant bytes (group word 0x5634DD36 Pres, 0x5634DD26
  Sim/Tex; CMAP platform 3; every pack).
- Locators: per location `freeskate_<slug>_<key>_locator` (6 spawns) and `Z_<slug>_<key>_Start`
  (none); file `<DIST[:12]>_[0x<attrib_key(name)>].{loc,rx2}`. The .rx2 is a 3-object RW4 arena,
  one subref (object 1, +0x20): eb0008 = 00000019 0000000D + 8 zero bytes (190/190); eb0009 locator
  (layout in `Locator`; +0x90 = attrib_key(name)); eb000b = import/export table {2, 0x14, 0x44, 25,
  0x44}, import {0, 0xFEFFFFFF, attrib_key(name), 0xEB0068, 0x00800000}, export {0, 0xFEFFFFFF,
  export id, 0xEB0009, 1}, then the 25 cSim table types with count 2 (190/190 equal but for the two
  ids). Spawn i of 6 = position + 2 m along the locator's x, -x, z, -z, x+z, -x-z columns (104/104).
  The export id and the arena id (header 0x1C) are not hashes of any name or path tried (lookup8,
  lookup2, FNV-1/1a 32/64, CRC32 over names, file stems, paths) and occur nowhere outside their own
  arena (all pack bytes searched); retail global locators (content/global_locators, 23 arenas) use
  unrelated ids too (file id != attrib_key(name) there). Generated arenas therefore take their own:
  arena id = attrib_key(name + '.arena') & 0xFFFFFFFF, export id = attrib_key(name + '.export')
  (`dlc.locator_own_ids`). Built from first principles (`dlc.build_locator_arena`), every pack
  locator equals the old template-based build byte for byte given its two ids (190/190). The packs'
  arenas use a non-EA arena layout (dictionary +4 = ptr, 16-aligned dictionary, longer subref
  tail); we write EA's.
- Language table (little-endian): u32 0x00039000, u32 size-8, u32 count, u32 0x1C, u32 pool offset
  (from +8), name[16], count x {u32 hash, u32 pool offset} ascending by hash, pool of NUL-terminated
  texts (deduplicated, first-use order), padded to 4. Key = h=0xFFFFFFFF; h=h*33+byte over the
  label: 3141/3343 retail ID_ labels found in the retail English table (the rest absent).
  Per location 5 labels (ID_LOCATION_, ID_WORLD_, dist_<s>_<k>dlc, ID_DLC_..._SUMMARY_TITLE ->
  "<Name> - <Title>"; ID_MAP_LOCATION_DESC_ -> "... freeskate location."), per pack 3
  (ID_MAP_SECTION_<S>_MAPS "Maps", ID_MAP_HELPER_<S>, ID_MAP_CATEGORY_<S> = name): 48/48 resolved.
- Histogram: u32 0x00039001, 6148, count, 256 x 24-byte char table; packs: count 256, all zero
  (raw ASCII texts); retail English: 271 entries.
- Unlocks: `<slug>.unlock` empty (or a WORLDS list in SAN VANELONA);
  `<slug>0000_product00000000.unlock` = "DLC_PRODUCT," + upper(slug) zero-padded to 16 (+ LF).
- Vaults: see "Vaults (AttribSys)" below. SUMMERCAMPHD's two builds have identical metadata, so
  the keys are deterministic.


## Vaults (AttribSys) (`sk3/vault.py`, `tests/test_vault.py`)
Corpus: retail db.big (509 .vlt/.bin, 3 .opt, 509 summary reports naming the collections) and
the 161 vault triples of the 6 packs. Measured with `work/vault/*.py` (scratch) and the tests.

File codec (byte-exact: 670/670 .vlt, 670/670 .bin, 164/164 .opt):
- .vlt = chunks Vers, StrN, DepN, DatN, ExpN, PtrN, EndC; u32 tag, u32 size incl. the head; every
  chunk padded with zeros to a multiple of 16. Vers = u64: 0x693CDCC3F57ADB28 in all 161 pack
  vaults, 0xB2157AD88EF77187 in 505/509 retail (3 other values in 4). StrN = 8 zero bytes (670/670).
  EndC = 8 zero bytes. DepN = u64 n, u64 hashes[n] = attrib_key(file name), u32 name offsets[n]
  padded to 8, NUL-terminated names (the vault's own .vlt and .bin names). DatN = the export records
  back to back, in ExpN order. ExpN = u64 n, n x {u64 id, u64 type, u32 size, u32 file offset};
  types: 0xAD303B8F42B3307E collection (14319 retail, 1248 pack), 0x2A7895AC4A876152 class (329)
  and 0x0B38846845E9C175 database (1), both only in skaterschema.vlt. PtrN = 16-byte entries
  {u32 offset, u16 kind, u16 arg, u64 target}: kind 2 = section (arg 1: the slots that follow are
  in the .bin, arg 0: in the .vlt); kind 3 arg 1 = slot at offset -> .bin offset target; kind 0 ends.
- .bin = StrE chunk, EndC chunk, then (EA) the data area. Community builders put everything into
  StrE and end at EndC. .opt = EndC only (16 bytes).
- Export id of a collection = attrib_key('<class name>/<collection name>'): 14319/14319 retail
  exports (names from the summary reports and .bin strings) and 1103/1103 named pack exports.

Schema (retail skaterschema.vlt/.bin, `read_schema`):
- Database record: u32 0x202, u32 0xC04, u32 type count (711), u32 0, u32 type sizes[711]; its
  pointer at +0x0C leads to the 711 type names (NUL-separated, same order); type key =
  attrib_key(type name) (all collection type keys of all vaults resolve).
- Class record (0x28): u64 key, u32 (unidentified), u32 field count, u32 -> field table, 0, 0,
  u32 layout size, u32 static field count, 0. Field = 24 bytes {u64 key, u64 type, u16 layout
  offset, u16 size, u16 max count (1, n, 0xFFFF), u8 flags, u8 log2 alignment}, ascending by key;
  flags 0x01 array, 0x02 static (in the layout), 0x20 complex type. Class names: every class
  has a type `Attrib::Gen::ClassRefSpec_<name>` (329/329; plus challenge_local_data).
  challenge_local_data: layout 4 bytes (one static eSk8Characters), 314 fields.

Collection record (all 14319 + 1248):
- +0 u64 key = attrib_key(name), +8 class key, +0x10 parent key (0 = none), +0x18 u32 capacity
  (= entry count, 14319/14319, 1248/1248), +0x1C u32 w1c (0..4; meaning unidentified - not a
  modulo/probe statistic of the keys we tried; identical content carries 0 in retail and 1..2 in the
  packs, so the generator writes the packs' value per collection), +0x20 u32 entry count, +0x24 u16
  used type count, +0x26 u16 stored type count, +0x28 u32 layout pointer (PtrN slot; EA stores 0,
  packs the .bin offset), +0x2C 0, +0x30 u64 type keys, then entries 16 B {u64 field key, u32
  value, u16 type index, u16 flags}.
- Type list = the class's static-field types in layout-offset order, then the entry types in
  first-use order; EA pads it with a zero key to an even count (14224/14319 retail; the packs do not
  pad). Entry order is the source order (no hash-table rule found: < 18% of keys at key % n).
- Entry flags: 0x4000 inline (value word; a Text / tLocationID value is a PtrN slot to a string),
  0x0800 complex (schema flag 0x20), 0x0200 array (schema 0x01). Without 0x4000 the value is a PtrN
  slot to a data block of the field's size, or for arrays an 8-byte header {u16 count, u16 capacity,
  u16 element size, 0} followed by the elements.
- EA's layout (`encode`, 474/508 retail vaults re-encode byte for byte from the decoded content):
  StrE = every string once in first-use order (collections in record order: layout slots, then entry
  by entry), chunk padded to 16; data area after EndC: per collection its layout (aligned to its
  largest static field alignment) then its out-of-line values in entry order, each aligned to the
  field's alignment; pointer slots store 0. PtrN: section 1 (only when there are .bin slots or a
  complex entry) sorted by (slot offset inside its array element / block, address), then section 0
  sorted by (slot offset inside its record, address). The other 34 retail vaults (stategraphs,
  skatercollections, challengebanks/main, ...) have .bin slots pointing at data blocks (nested
  structures); `decode` refuses them rather than mis-read them.
- Community builders' layout (all 6 packs): data inside StrE, unaligned, identical blocks shared,
  slots store their target offset, types unpadded, and three struct sizes too small
  (tHighlightDefinition 0x70 of 0x90, tObjectiveDefinition 0x10 of 0x14, tVaultedRefSpec 0x18 of
  0x20: the game reads the next bytes) and 4 values per pack pointing at .bin offset 0 (the StrE
  head; `BIN0`). The decoder resolves what the game reads, so content compares across layouts.

DLC content (single-world packs; SUMMERCAMP, ISLEOFGROM, SUMMERCAMPHD, every name resolved by hash
except one): names derive from slug S, s4 = S[:4], keys K (K[0] main), DIST D.
- 61 files = 20 vault triples + `dlc_<S>_minimal.vaultlist.xml` (CRLF; lists dlc_<S>_minimal,
  dlc_<s4>_local_data_framework, challengebanks\dlc_<s4>, progressionbanks\dlc_<s4>; the
  challenge_local_data vaults are reached through tVaultedRefSpecs). 3 x (11 + len(K)) + 1 in general.
- challenge_local_data/dlc_<s4>_freeskate_{activities, activities_<gap_tag|accumulation|simultrick|
  tricklist|survival>, locations}.vlt: each holds only a copy of retail challenge_local_data
  'default' (45 entries, equal to retail's).
- challenge_local_data/freeskate_dlc_<S>_<K>.vlt: one empty collection per location, parent
  dlc_<s4>_freeskate_locations.
- dlc_<s4>_local_data_framework.vlt: challenge_local_data dlc_<s4>, dlc_<s4>_freeskate_locations,
  dlc_<s4>_freeskate_activities(+5) = retail's freeskate_* collections renamed.
- challengebanks/dlc_<s4>.vlt: challenges + challenge_objective 'default' copies; challenges /
  global data / failure objective / objective / objectives group dlc_<s4>; challenges + global data
  for dlc_<s4>_freeskate_locations, _activities (+5 activities with failure objectives); per
  location challenges + global data freeskate_dlc_<S>_<K> (locator freeskate_<S>_<K>_locator, texts
  ID_MAP_LOCATION_DESC_/ID_LOCATION_<S>_<K>, world ref dist_<S>_<K>dlc).
- progressionbanks/dlc_<s4>.vlt: 6 'default' copies, dlc_<s4> state/handler/rewards/stategraph/
  action/group, rewards dlc_<s4>_challenges, dlc_<s4>_ots_unlock, dlc_<s4>_milestones and one
  milestone reward whose name is NOT known (keys 0xAA86FA1E66596509 summ, 0xC84467CE580C5F5D isle;
  ~50 M candidate names tried; referenced only as a parent link inside its own vault). The generator
  names it `dlc_<s4>_milestone_01`.
- dlc_<S>_minimal.vlt: per location world / worldpainter_fe_location_triggers / aud_worlddata /
  aud_bigfiles 'dist_<S>_<K>dlc' (texts D, s4, world<D>.big, ID_WORLD_<S>_<K>; constant tMapInfo
  'map1024' {0x802, -850, 1050, -1250, 512, 512}; 4 constant world RefSpecs), map_category <S>dlc,
  <S>dlc_maps, fe_locations per location (layout = dist_<S>_<K>dlc, Z_<S>_<K>_Start,
  ID_LOCATION_<S>_<K>), map_filter <S>dlc and freeskate_locations_<S>dlc (Lua filter source listing
  every dist_<S>_<K>dlc; tCompiledLua = {ptr source, ptr '', u32 source length}), map_listing
  progression_locations_dlc_<S> and online_freeskate_<S>dlc, dlc_mapping pack_<S> (product text =
  upper alphanumerics of S, first 7, padded with '0' to 16: 6/6 packs), progression_summary_group
  dlc_<S> and <S>_<K>_dlc per location (ID_DLC_<S>_<K>_SUMMARY_TITLE).
- Nothing depends on spawn positions; everything else is constant across the three packs.
- Generator (`vault.dlc_vaults(slug, keys, dist)`): emits all 61 files in EA's layout with the packs'
  content (their 3 short structs' trailing bytes and their 4 BIN0 values included, so the game reads
  the same values). Result (`test_generated_vaults_match_pack`): .opt and vault list byte-equal,
  19/20 vaults with identical decoded content, the 20th (progressionbanks) identical except the
  milestone reward's key and export id.


## Name hash (`sk3/attrib_hash.py`)
- Function: Bob Jenkins' lookup8 `hash()` over the name's bytes (no case folding, no prefix, no
  NUL): 24-byte blocks read as 3 little-endian u64, mix shifts 43/9/8/38/23/5/35/49/11/12/18/22,
  c += length before the tail (tail bytes 16..22 go to c from bit 8), result c. Start state
  a = b = 0xABCDEF0011223344, c = 0x9E3779B97F4A7C13.
- Found in the retail .text (recomp's lifted C++, generated/skate3_recomp.70.cpp, block loop at
  0x82B73B0C): lis/ori/rldimi build 0xABCDEF00'11223344 into a/b and 0x9E3779B9'7F4A7C13 into c;
  r4 = length, divwu by 24, byte loads assembled little-endian. The earlier search missed it by
  trying 0x11223344ABCDEF00 and golden in a/b. A sibling (skate3_recomp.71.cpp) is 32-bit lookup2
  with initval 0xABCDEF00.
- Verified (`tests/test_attrib_hash.py`): 1012/1012 DepN name/hash pairs from every retail vault in
  db.big; challenge_local_data = 0x46A56ADDF284C381; every locator file id in the DLC packs
  (36+36+20+36...) equals the hash of a locator name in its .loc/.rx2 (`Z_summercamp_spot8_Start`
  -> 0xED172AC1DC8BC492, capitals kept). Over all retail vault strings, 6404 of 50008 hash to a
  64-bit word present in the vault files (record/collection/field keys).
- Still unknown: the names of the ExpN type words, the locator arenas' arena / export ids (own ids
  generated, see Locators), one milestone reward collection name (see Vaults).
## Generator (`dlc.build_files`, `tools/pack.py`)
Template-free (default): everything is generated from the inputs (tiles, spots, display name,
optional CSPA content boxes) and the internal names (`PackNames`: slug, DIST name, location keys;
defaults slug = display letters/digits lower-cased, DIST_<CamelCaseName>, keys = slug, spot1, ...;
`--slug`, `--dist`, map.json "slug"/"dist"/per-spot "key"): streams (raw form by default, EA form on
request), all DIST manifests, the Sim location table, locator tables and arenas (ids from names,
first-principles objects, EA layout via rw4_write), boundaries, neighbour tables (+ the dist_*dlc
tables), preload lists, mission manifests, language strings, histogram, unlocks, the 3 x (11 + N)
+ 1 vault files (`vault.dlc_vaults`), archive (data order + path directory), header. One location
per spot, any count.
Optional `--template [PACK.big]` (off by default): names, the 61 vault files and the locator arenas'
objects 0/2 and ids taken from a working pack (the pre-hash behaviour, kept for comparison).
Proof (`tests/test_dlc.py`), SUMMERCAMP, ISLEOFGROM and SUMMERCAMPHD rebuilt from their own exported
inputs (tiles, spot matrices, display name, CSPA content boxes, group/tail words, internal names)
with no template: 415/415, 323/323, 415/415 files present; streams, DIST manifests, location table,
locator tables, boundaries, preload lists, neighbour tables, missions, language, histogram, unlocks,
.opt and vault list byte-identical; 19/20 vaults with identical decoded content and the 20th equal
but for the milestone collection's key/export id; the 18 locator arenas with identical objects but
for our export id (and our arena id). With the packs' own bytes swapped in for those vault and
locator files the whole archives are identical to the shipped ones (146, 158, 332 MB;
`test_template_free_rebuild`). The template path still rebuilds SUMMERCAMP byte for byte
(`test_whole_pack_regenerates_byte_exact_with_template`).

## The recomp's loader (owner's `skate3_dlc_picker.cpp`)
- Search: `dlc/` beside the exe, under the game data root, the user data root, or
  `skate3_dlc_root`; recursive. Ignores .header/.txt/.rtf/.log/.png/.jpg/.md/.xcp/.iso/.zip/.rar/
  .7z and files < 1 MiB.
- A candidate is a signed STFS container (content type must be marketplace, title 0 or 454108E6)
  or a raw `.big` with a 328-byte header: `<stem>.header` beside it, else any `.header` in the
  folder, else `Headers/**` (the canonical tree). The header's content id (file name) must match
  ToUpperAlnum(parent folder name) to be chosen in a folder with several; without a header the
  package name is the upper-cased alphanumerics of the folder (or the stem when shorter than 3).
- One pack is staged per launch: others are moved to `addons/parked`; the chosen raw .big is
  linked into `<user>/0000000000000000/454108E6/00000002/<PKG>/` and the picker writes its own
  content header (marketplace, display name, file name = package). A flattened per-addon content
  root made the title exit after ~8 s (owner's measurement), hence the ordinary layout.
- Header bytes (owner's makeheader.py and the picker): u32 1, u32 2, UTF-16BE display name at 8,
  content id ASCII at 0x108 (42 bytes), title id 454108E6 12 bytes from the end. Our writer equals
  the 5 headers shipped beside packs byte for byte.
- `skate3_dlc_trace` hooks name the guest functions for content enumeration, archive mount
  (0x82A83080 AddArchiveFromFile) and async file open; a pack that never mounts parks the guest in
  a file wait.
