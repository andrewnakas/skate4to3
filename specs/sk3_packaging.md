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
  (pytest); `python tests/test_sfil_roundtrip.py` covers the 3 large worlds too.

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
  (none); file `<DIST[:12]>_[0x<64-bit id>].{loc,rx2}`. The .rx2 is a 3-object RW4 arena
  (eb0008 constant 16 bytes; eb0009 locator, layout in `Locator`; eb000b = constant table with the
  locator id at +0x1C and a second unidentified 64-bit word). Spawn i of 6 = position + 2 m along the
  locator's x, -x, z, -z, x+z, -x-z columns (104/104 locators). Arena id (header 0x1C) and both
  64-bit ids are not a hash we identified. The packs' arenas use a non-EA arena layout (dictionary
  +4 = ptr, 16-aligned dictionary, longer subref tail); we write EA's.
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
- Vaults (.vlt): chunks Vers, StrN, DepN, DatN, ExpN, PtrN, EndC (u32 tag, u32 size incl. head).
  DepN = u64 count, u64 hashes, u32 name offsets, names (the .vlt/.bin file names). ExpN = u64
  count, {u64 key, u64 type 0xAD303B8F42B3307E, u32 size, u32 file offset of the DatN record}.
  PtrN = 16-byte entries {u32 offset, u16 kind, u16 arg, u64 target}: kind 2 selects the section
  (arg 0 = the .vlt, 1 = the .bin), kind 3 = pointer slot at `offset` -> .bin offset `target`.
  .bin = StrE string chunk + EndC + data. Record = u64 key, u64 class key (challenge_local_data =
  0x46A56ADDF284C381 in 424/463 retail files), u64 parent, counts, layout pointer, layout keys,
  16-byte field entries {u64 key, u32 value, u16 type index, u16 flags (0x4000 inline)}.
  Keys are 64-bit hashes of names whose function we did NOT identify: FNV-1/1a 64, lookup8 (levels
  0, 0xABCDEF00, 0x11223344ABCDEF00, golden, both init orders, LE/BE words), lookup3 pairs,
  lookup2 pairs, xxh64, murmur64A, CRC-64 variants and h*33 (64-bit) all fail on oracle pairs
  (DepN file names, retail class/collection names from the *_summaryreport.txt files, the
  schema's type names). Per-location vault content is a fixed pattern per location (4 records of
  0x140/0x40/0x60/0xC0 bytes in `dlc_<slug>_minimal.vlt`, one `challenge_local_data` file).
- SUMMERCAMPHD's two builds have identical metadata, so the keys are deterministic.

## Generator (`dlc.build_files`, `tools/pack.py`)
Generated from inputs: streams (raw form by default, EA form on request), all DIST manifests,
the Sim location table, locator tables and arenas (positions, spawn rings, EA layout via
rw4_write), boundaries, neighbour tables (+ the dist_*dlc tables), preload lists, mission
manifests, language strings, histogram, unlocks, archive (data order + path directory), header.
Taken from a template pack (default SUMMERCAMP, a single-world pack with 9 locations): the
internal names (slug, DIST name, location keys), the 61 vault files verbatim, the locator ids and
locator arenas' objects 0 and 2. Reason: vault keys (and the locator/arena ids) are hashes of those
names whose function is unknown, so new names could not be keyed. Up to the template's location
count of spots; left-over locations reuse spot 0.
Proof: SUMMERCAMP, ISLEOFGROM and SUMMERCAMPHD rebuilt from their own inputs (tiles, spot
matrices, display name, CSPA content boxes, group/tail words) give byte-identical archives (146,
158, 332 MB), with the packs' own locator .rx2 bytes swapped in where our EA-layout arena decodes
to the same objects (`test_whole_pack_regenerates_byte_exact`).

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
