# Skate 3 collision + grind splines (cSim_* streams) - measured facts

Code: `sk3/collision.py` (decoders, query harness), `sk3/splines.py` (eb0004), `sk3/collision_write.py`
(encoders, mesh builder, tile-arena builder), `tools/collision_view.py` (top-down picture).
Tests: `tests/test_collision_roundtrip.py`, `tests/test_collision_builder.py`.
Evidence: retail `worldDIST_*.big` cSim streams read with our own BIG/SFIL/RW4 readers. Scratch
measurement scripts: `work/scratch/coll/*.py`. "Small worlds" = BlackBoxPark, MegaPark, StartPark,
IndustrialSkatePark, MaloofMoneyCup, DownTownSkatePark, SkateSchool; "big" = DownTown, Industrial,
University. All values big-endian.

## Arena contents
- Census (small worlds, 369 cSim arenas): types 80001 x218, 80004 x62, 80006 x146, eb0004 x48,
  eb0008 x369, eb0009 x3, eb000a x146, eb000b x369, eb000d x198, eb000f/eb0010/eb0011 x425,
  eb0019 x7, eb001d x8, eb0020 x62, eb0027 x136, eb0029 x1. Big worlds add eb0013, eb0014, eb001a
  (census_big.py: DownTown 1116 arenas, Industrial 2102, University 1013).
- Tile stream `cSim_<x>_<z>_high.xsf` (BlackBoxPark, all 4 tiles): arena 0 = collision
  `eb0008 eb000d 80001 80006 eb000a [eb001d] [eb0004] eb000b`; arena 1 = `eb0008 (eb0010 eb0011 eb000f) x3 eb000b`;
  arena 2 = `eb0008 eb0027 eb000b`. Subrefs of arena 0: (1, 0x20) = the instance matrix, then
  (splines, 0x10 + 32k) per spline.

## 0x80006 clustered triangle mesh (the collision)
Header (all 146 small + 2090 big meshes re-encode byte-exact from the decoded fields):
- 0x00 bbox min xyz + 0 pad, 0x10 bbox max xyz + 0 pad. Equal to the tree bbox (146/146) and tight to
  the dequantised vertices within 1.01 mm (vertices can lie up to one grid step outside: bounds are
  computed before quantisation).
- 0x20 0; 0x24 = bit_length(unitCount + 1) (all meshes; 255 units -> 9 decides "+1"); 0x28 unit count;
  0x2C 0; 0x30 tree offset 0x60; 0x34 cluster table offset = 0x90 + 32 * branchCount; 0x38 f32
  granularity 0.001 (all); 0x3C 0x00100202 (all); 0x40 = 0x44 cluster count; 0x48 = 0x4C unit count;
  0x50 object size; 0x54 0; 0x58 0x80000000; 0x5C 0.
- 0x60 KD tree: u32 node offset (0x90; when branchCount = 0 a stale non-offset value, 4/146 small,
  kept as a field), u32 branchCount, u32 entry count (= units), 0, bbox min/max (0x70, 0x80).
- 0x90 branch nodes, 32 B: u32 parent (root: 0), u32 axis (0 x, 1 y, 2 z; all three used),
  2 x (u32 content, u32 index), f32 left-child max, f32 right-child min on the axis. content
  0xFFFFFFFF = branch node `index`, else a leaf of `content` units starting at unit id `index`.
  Nodes are in preorder (child index > parent, 146/146); parent fields consistent (146/146).
  Child cell = parent cell with max[axis] = left max (left) / min[axis] = right min (right).
- Unit id = cluster << 16 | byte offset into that cluster's unit data (e.g. 0x20ba6). Leaves walked
  depth-first left-first visit unit ids in increasing order (21/21 checked) and never cross a cluster
  (42961 leaves); every unit is in exactly one leaf (146/146). Empty leaves (content 0) are 26% of leaves.
  Leaf sizes: mean 5.3, max 43 (small worlds).
- Leaf containment: every triangle lies inside its leaf cell within 1.01 mm (27798 leaves; 1255 exceed
  by up to one grid step, same cause as the bbox).
- Cluster table: u32 offset per cluster, zero padded to 16; clusters 16-aligned, zero padding; object
  size 16-aligned.
- Cluster: u16 unitCount, u16 unitBytes, u16 unitStart, u16 normalStart (= unitStart, 1274/1274),
  u16 clusterSize (= 16 + 16 unitStart + unitBytes), u8 vertexCount (max 255 seen), u8 normalCount
  (0 in all 1274), u8 vertex mode, 3 zero bytes. Vertices at +0x10:
  - mode 1 (1121 clusters): i32 x3 offset, then u16 x3 per vertex; offset = per-axis min - 1, so the
    smallest stored value is 1 on every axis (51/51 BlackBoxPark clusters);
  - mode 2 (153 clusters): i32 x3 per vertex (used when the span exceeds 16 bits);
  - world = (offset + v) x f32(0.001); unit data at +0x10 + 16 * unitStart.
- Units: 0xA1 = triangle + edge bytes + u16 surface (222078 small-world units), 0x21 = triangle + edge
  bytes, no surface (4023). No quads (0x02), no group ids (0x40) anywhere. Unit = flags, 3 vertex
  indices (cluster-local u8), 3 edge bytes, [u16 surface]: 9 or 7 bytes. Every cluster vertex is used.
- Edge byte k belongs to edge (v_k, v_k+1). Measured on BlackBoxPark + StartPark by welding vertices
  and comparing with the neighbour triangle (edge1/2/3.py):
  - low 5 bits = angle code; floor(-2 log2(theta / 90 deg)) + 2 clamped to 0..26 (theta = angle between
    face normals) matches 62664 of 69076 matched edges exactly, 3884 more within +-1 (quantised normals); 2240 are 26 where the formula gives a small code (open);
    26 = flat; edges with no neighbour are always 0x5A (6275/6275).
  - 0x20 = convex: set only when the neighbour's far vertex lies below this face (0 concave among
    set bytes; 0 convex among clear non-flat bytes).
  - 0x40 = per-vertex flag of v_k: all corners of a welded vertex carry the same value (count of
    clear corners per vertex is 0 or the full valence); set on 96% of corners; never clear on an open
    border vertex. Read as "vertex disabled". The rule choosing the ~4% enabled vertices (BlackBoxPark;
    ~20% in some DownTown tiles) is not pinned down (convex-corner rules tested: no clean match).
- Surface ids (u16): 55 distinct values over small worlds; most common 0xB700, 0x0201, 0x2A71, 0x8B00,
  0x0A03, 0x0401. Meaning not measured (no oracle yet).
- Size: 22.8 bytes per triangle on average (226101 small-world triangles), 0.26 branch nodes per
  triangle, up to 56 clusters per mesh, up to 501 units / 4509 unit bytes per cluster.

## 0x80001 volume (96 B; 3216 re-encode)
16 f32 matrix rows (x, y, z, translation; 4th column 0) | u32 type | payload | words.
- type 6 = aggregate: +0x44 dictionary index of the geometry (0x80006 or 0x80004), +0x48..0x57 0,
  +0x58 0xFFFFFFFF, +0x5C 1.
- type 4 = box: +0x44..0x4F f32 half extents, +0x58 0, +0x5C 1.

## 0x80004 box aggregate (906 big + 62 small re-encode)
bbox; {0, 2, n, 0}; +0x30 volume array offset 0x50; +0x40 tree offset; n inline 0x60-byte volume
bodies (type 4 boxes, rotated); tree header like 0x80006's (stale node pointer when no branches,
entry count, bbox) whose last pad word is NOT cleared (e.g. 0xbcd9ab01): kept raw.

## Link objects
- eb0008 (16 B): {0x19, 0x0D, 0, 0} in all 4231 cSim arenas.
- eb000a (20 B): {bounding volume dict index (0 = none, else a type-4 box volume), count 1, 0x0C,
  aggregate volume dict index, 0}.
- eb0020 (16 B): {volume dict index, eb000d dict index, 0, 0} (pairs a 0x80004 volume with its instance).

## eb000d instance list
{kind (0xACB31C9A or 0x7369D9B5), count, unique string count, 0x20, string pool offset, 0, 0, 0},
0xA0 B per instance: matrix 4x4 (subref target), bbox min/max, 4 x u64 (id; then 0xFF.. or two ids),
{0, flags (4 or 0), 0, name ptr}, {str1, str2, str3 ptrs, 0}; pool = NUL-terminated unique strings
in first-reference order (e.g. name, "", "undefined"). 3249 re-encode.

## eb000b import / export table (all cSim arenas)
{n, 0x14, t2, 25, t2}; n x {0, 0xFEFFFFFF, u64 id, u32 type, u32 value}; value = dict index (export)
or 0x00800000 | k (import k); then the same 25 type ids in every arena, each followed by u32 n
(4600 re-encode). Tile arena 0: one eb0069 import, one eb000d export, one eb0064 import per spline
(BlackBoxPark 50_-50: 50 splines, 50 imports), eb0004 / eb001d exports. Import targets: open.

## eb0004 grind splines (`sk3/splines.py`; 494 re-encode)
- 0x00 u16 kind (1 only in one SkateSchool cSim_Global object, whose splines are flat curves at y = 0),
  u16 spline count; 0x04 segment count; 0x08 table offset 0x10; 0x0C data offset = 0x10 + 32 n.
- Table 32 B: u64 id, u64 template (high word 0x2C701707 in all), 0, u32 first segment offset,
  u32 last segment offset (segments = (last - first) / 0x90 + 1), 0. Segments of consecutive splines
  are contiguous (700/700).
- Segment 0x90 B: rows A, B, C (w 0), D (start, w 1.0), 0, bbox min, bbox max, {0, 0, flags, 0}, 0
  (3454/3454). P(t) = A t^3 + B t^2 + C t + D; P(1) equals the next segment's D (2754/2754).
- Straight segments (all but 6): C = 0, A = f32(-2 dP), B = f32(3 dP) exactly, dP = next D - D
  (2750/2750), bbox = min/max(D, next D) exactly.
- bbox = min/max of the Bezier control points D, D + C/3, D + 2C/3 + B/3, D + A + B + C (6/6 curved,
  exact in f32).
- flags: bit 0x10 always; 0x20 and 0x40 vary per segment; bits 0x80..0x800 constant along a spline.
  Meanings open.
- 99% of spline knots lie within 1 cm of a collision vertex (568 knots, 3 small worlds): splines run on
  collision edges. Pictures: work/coll/bbp_50_50.png (retail), ours_bbp_50_50.png (builder output).

## Not decoded (carried as bytes; listed for the record)
- eb0010 (in tile arena 1, 3 per tile): header {64.0, 64.0, u64, 64.0, 64.0, u64, u32 node count,
  0x30, 0, 0} then node count x 5 u16 (4 child indices, 0xFFFF none; 5th word 0xFFFF on branches) -
  a quadtree; each comes with an eb0011 (id list) and an eb000f {u32, u32, u64}.
- eb0027 (tile arena 2), eb001d / eb0009 (named locators: matrix, bbox, ids, strings such as
  "ws_sksc_coachfrank_locator_01", "world"), eb0019, eb0029, eb0013, eb0014, eb001a.

## Builder (sk3/collision_write.build_mesh) - our own choices, retail layout rules
Quantise to the 1 mm grid (round), drop degenerate triangles, median split on triangle centroids along
the widest centroid axis until <= 8 per leaf, extents = exact child min/max, preorder nodes, leaves
packed into clusters in DFS order (<= 255 vertices, < 64 KiB units, a new cluster before a 16-bit span
overflow), unit ids as retail, edge bytes by the measured angle/convex rules with every vertex disabled
by default. Validation (tests/test_collision_builder.py): own output decodes to the same triangle set,
re-encodes exactly, every unit reached once, 0 triangles outside their cell, vertical rays through
our tree agree with brute force on every ray (BlackBoxPark tiles, 6 DownTown tiles, 200-400 rays each;
synthetic unquantised terrain: max height error 0.48 mm). Output size 0.87-1.05x retail.
