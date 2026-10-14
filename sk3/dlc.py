"""DLC pack metadata: every non-stream file of a Skate 3 map pack, decoded from bytes, written back
byte-exact (tests/test_dlc.py), and a generator for a new map (`build_pack`, tools/pack.py).
Facts and how they were measured: specs/sk3_packaging.md.

A pack is `<PKG>/<slug>_00000000.big` (EB v3, flags 0x00500400, with a path directory) plus a 328-byte
`.header` whose content id equals the package directory name. Its data tree (S = slug, e.g.
`summercamp`; L = location key, one per freeskate location, the first is the slug itself):

    boundary/freeskate_dlc_S_L.xml                  Boundary polygon (tile extents + half a cell)
    content/global_locators/DLC_BAM/<DIST>/<DIST[:12]>_[0x<id>].{loc,rx2}
                                                    one locator table + RW4 locator arena per
                                                    locator: freeskate_S_L_locator and Z_S_L_Start
    content/missions/freeskate_S_L/freeskate_S_L_{Pres,Sim,Tex}.{pmm,psm,pss,pst}
                                                    empty stream manifests (constant bytes)
    content/world/stream/<DIST>/                    streams + DIST manifests + <DIST>_Sim.loc
    db/...                                          AttribSys vaults (.vlt/.bin/.opt) + vaultlist
    fe/languages/english/LANGUAGE_English_S_skate3ng-dlc_S_minimal.BIN (+ _HISTOGRAM_ variant)
    stream/<DIST>_{Pres,Sim}.xml                    tile neighbour tables
    stream/dist_S_Ldlc_{Pres,Sim}.xml               the per-location tables the game actually
                                                    reads (owner's recomp traces; see spec)
    stream/freeskate_dlc_S_L.xml                    per-location preload list (63 nearest tiles)
    unlocks/S.unlock, unlocks/S0000_product00000000.unlock

Vault keys are 64-bit hashes of names whose function we have not identified (spec "Vaults"), so the
generator keeps a template pack's internal names (slug, location keys, DIST name) and its vault keys,
and generates everything that is ours: streams, manifests, tiles, boundaries, neighbour tables,
locators (positions), location table, language strings (display texts), unlocks, header, archive.
"""
from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass, field

from . import big, manifests as mf, rw4, rw4_write, sfil

XML_HEAD = '<?xml version="1.0" encoding="us-ascii"?>'


# ---- boundary (data/boundary/*.xml) -----------------------------------------------------------
def read_boundary(d: bytes) -> list:
    t = d.decode('ascii')
    if t != write_boundary([tuple(map(int, p.split(', '))) for p in
                            re.findall(r'<Point>(.*?)</Point>', t)]).decode('ascii'):
        raise ValueError('boundary: unexpected layout')
    return [tuple(map(int, p.split(', '))) for p in re.findall(r'<Point>(.*?)</Point>', t)]


def write_boundary(points) -> bytes:
    s = [XML_HEAD, '<Boundary>'] + [f'  <Point>{x}, {z}</Point>' for x, z in points] + ['</Boundary>', '']
    return '\r\n'.join(s).encode('ascii')


def boundary_rect(centers, cell: int) -> list:
    """The rectangle every pack measured uses: tile-centre extents widened by half a cell, listed
    (minx,minz) (maxx,minz) (maxx,maxz) (minx,maxz)."""
    h = cell // 2
    x0, x1 = min(c[0] for c in centers) - h, max(c[0] for c in centers) + h
    z0, z1 = min(c[1] for c in centers) - h, max(c[1] for c in centers) + h
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


# ---- stream tile tables (data/stream/*.xml) ---------------------------------------------------
@dataclass
class StreamTiles:
    tiles: list                   # [(center (x, z), [tile (x, z), ...])]
    count_attr: bool = False      # '<StreamTiles Count="n">' (per-location files)
    newline: str = '\r\n'


def write_stream_tiles(st: StreamTiles) -> bytes:
    nl = st.newline
    head = f'<StreamTiles Count="{len(st.tiles)}">' if st.count_attr else '<StreamTiles>'
    s = [XML_HEAD, head]
    for c, ts in st.tiles:
        s += ['  <StreamTile>', f'    <Center>{c[0]}, {c[1]}</Center>']
        s += [f'    <Tile>{x}, {z}</Tile>' for x, z in ts]
        s.append('  </StreamTile>')
    s += ['</StreamTiles>', '']
    return nl.join(s).encode('ascii')


def read_stream_tiles(d: bytes) -> StreamTiles:
    t = d.decode('ascii')
    nl = '\r\n' if '\r\n' in t else '\n'
    tiles = []
    for c, body in re.findall(r'<Center>(.*?)</Center>(.*?)</StreamTile>', t, re.S):
        tiles.append((tuple(map(int, c.split(', '))),
                      [tuple(map(int, x.split(', '))) for x in re.findall(r'<Tile>(.*?)</Tile>', body)]))
    st = StreamTiles(tiles, 'Count=' in t, nl)
    if write_stream_tiles(st) != d:
        raise ValueError('stream tiles: unexpected layout')
    return st


def neighbour_table(centers, cell: int) -> StreamTiles:
    """Per tile, the existing tiles of its 3x3 ring, both sorted by (x, z) (all 86 SummerCamp rows)."""
    S = sorted(set(centers))
    return StreamTiles([(c, [p for p in S if p != c and abs(p[0] - c[0]) <= cell and
                             abs(p[1] - c[1]) <= cell]) for c in S])


def preload_list(center, centers, limit: int = 63) -> StreamTiles:
    """A location's preload list: the `limit` nearest tiles (the centre included), by distance,
    then x, then z (every location file in every pack measured)."""
    order = sorted(set(centers), key=lambda p: (math.hypot(p[0] - center[0], p[1] - center[1]),
                                                p[0], p[1]))
    return StreamTiles([(center, order[:limit])], count_attr=True)


# ---- language strings (fe/languages/<lang>/LANGUAGE_*.BIN, little-endian) ---------------------
LANG_ID = 0x00039000
HIST_ID = 0x00039001


def string_id(text: str) -> int:
    """String-table key: h = 0xFFFFFFFF; h = h*33 + byte (3141 of 3343 retail ID_ labels found in the
    English table this way; the rest are not in it)."""
    h = 0xFFFFFFFF
    for b in text.encode('ascii'):
        h = (h * 33 + b) & 0xFFFFFFFF
    return h


@dataclass
class Language:
    name: str                     # the 16-byte table name ('Global', the pack slug, ...)
    entries: list                 # [(hash, text)] in pool order of first use
    w0c: int = 0x1C

    def by_hash(self) -> dict:
        return dict(self.entries)


def read_language(d: bytes) -> Language:
    lid, size, count, w0c, pool = struct.unpack_from('<5I', d, 0)
    if lid != LANG_ID or size != len(d) - 8:
        raise ValueError('not a language table')
    name = d[0x14:0x24].split(b'\0', 1)[0].decode('ascii')
    ents = [struct.unpack_from('<II', d, 0x24 + 8 * i) for i in range(count)]
    base = pool + 8
    texts = {}
    for h, o in ents:
        texts[h] = (o, d[base + o:d.index(b'\0', base + o)].decode('latin1'))
    order = sorted(ents, key=lambda e: (e[1], ents.index(e)))
    return Language(name, [(h, texts[h][1]) for h, _o in order], w0c)


def write_language(L: Language) -> bytes:
    pool, at = bytearray(), {}
    for _h, t in L.entries:
        if t not in at:
            at[t] = len(pool)
            pool += t.encode('latin1') + b'\0'
    ents = sorted((h, at[t]) for h, t in L.entries)
    head = 0x24 + 8 * len(ents)
    body = bytearray(struct.pack('<5I', LANG_ID, 0, len(ents), L.w0c, head - 8))
    body += L.name.encode('ascii').ljust(16, b'\0')
    for h, o in ents:
        body += struct.pack('<II', h, o)
    body += pool
    body += bytes(-len(body) % 4)
    struct.pack_into('<I', body, 4, len(body) - 8)
    return bytes(body)


def write_histogram(table=None) -> bytes:
    """Character table of a language: count then 256 x 24 bytes. Community packs store raw ASCII
    with count 256 and an all-zero table; retail English carries 271 entries."""
    body = struct.pack('<3I', HIST_ID, 6148, 256 if table is None else table[0])
    return body + (bytes(6144) if table is None else table[1])


def location_strings(slug: str, display: str, locations) -> Language:
    """The 5 strings per location + 3 per pack every community pack carries (decoded above):
    title under ID_LOCATION_/ID_WORLD_/dist_<s>_<l>dlc/ID_DLC_..._SUMMARY_TITLE, description under
    ID_MAP_LOCATION_DESC_; then ID_MAP_SECTION_<S>_MAPS, ID_MAP_HELPER_<S>, ID_MAP_CATEGORY_<S>."""
    S = slug.upper()
    ents = []
    for key, title in locations:
        K = key.upper()
        full = f'{display} - {title}'
        for lab in (f'ID_LOCATION_{S}_{K}', f'ID_WORLD_{S}_{K}', f'dist_{slug}_{key}dlc',
                    f'ID_DLC_{S}_{K}_SUMMARY_TITLE'):
            ents.append((string_id(lab), full))
        ents.append((string_id(f'ID_MAP_LOCATION_DESC_{S}_{K}'), f'{full} freeskate location.'))
    ents.append((string_id(f'ID_MAP_SECTION_{S}_MAPS'), 'Maps'))
    ents.append((string_id(f'ID_MAP_HELPER_{S}'), f'{display} freeskate maps.'))
    ents.append((string_id(f'ID_MAP_CATEGORY_{S}'), display))
    # pool order = first use: per location title then description, then the three pack strings
    first = {}
    for h, t in ents:
        first.setdefault(t, len(first))
    ents.sort(key=lambda e: first[e[1]])
    return Language(slug[:15], ents)


# ---- unlocks -----------------------------------------------------------------------------------
def product_unlock(slug: str) -> bytes:
    return f'DLC_PRODUCT,{slug.upper().ljust(16, "0")}\n'.encode('ascii')


# ---- vaults (.vlt / .bin / .opt): chunk codec ---------------------------------------------------
def chunks(d: bytes) -> list:
    """[(tag, payload)] of a vault, bin or opt file: u32 tag, u32 size (incl. the 8-byte head)."""
    out, off = [], 0
    while off < len(d):
        tag, size = d[off:off + 4].decode('latin1'), struct.unpack_from('>I', d, off + 4)[0]
        if size < 8 or off + size > len(d):
            raise ValueError(f'bad chunk at {off:#x}')
        out.append((tag, d[off + 8:off + size]))
        off += size
    return out


def join_chunks(cs) -> bytes:
    return b''.join(t.encode('latin1') + struct.pack('>I', len(p) + 8) + p for t, p in cs)


@dataclass
class Vault:
    """A .vlt as chunks, with its dependency names decoded (DepN: u64 count, u64 hashes[count],
    u32 name offsets[count] (padded to 8), the names NUL-terminated, zero padding to 16)."""
    chunks: list
    dep_hashes: list = field(default_factory=list)
    dep_names: list = field(default_factory=list)


def read_vault(d: bytes) -> Vault:
    cs = chunks(d)
    v = Vault(cs)
    for tag, p in cs:
        if tag == 'DepN':
            n = struct.unpack_from('>Q', p, 0)[0]
            v.dep_hashes = list(struct.unpack_from(f'>{n}Q', p, 8))
            offs = struct.unpack_from(f'>{n}I', p, 8 + 8 * n)
            base = 8 + 8 * n + (4 * n + 7) // 8 * 8
            v.dep_names = [p[base + o:p.index(b'\0', base + o)].decode('ascii') for o in offs]
    return v


def strings_of_bin(d: bytes) -> list:
    """(offset, text) of every NUL-terminated string in a .bin's StrE chunk (file offsets)."""
    tag, size = d[:4], struct.unpack_from('>I', d, 4)[0]
    if tag != b'StrE':
        return []
    out, i = [], 8
    while i < size:
        j = d.index(b'\0', i)
        if j > i:
            out.append((i, d[i:j].decode('latin1')))
        i = j + 1
    return out


# ---- locator arenas (global_locators/*.rx2) ----------------------------------------------------
@dataclass
class Locator:
    """The eb0009 object of a locator arena (all 208 in the packs measured):

        0x00 u32 1, u32 spawn count n, u32 0, u32 0x20 (root), u32 spawn records, u32 pool, 0, 0
        0x20 root: 4x4 matrix (row-major, translation in row 4) + AABB min/max rows (w 0, +-1 m)
        0x80 u32 n, u32 spawn records, u32 pool (= name), u32 world-name offset
        0x90 u64 locator id (also in the file name), 8 zero bytes
        0xA0 n x 0x70: matrix, AABB, u32 name offset, 12 zero bytes
        then the pool: name, 'world', each spawn name '<name>::<base>_spawn_<i>', NUL-terminated."""
    name: str
    matrix: tuple
    locator_id: int
    spawns: list = field(default_factory=list)    # [(name, matrix)]
    world: str = 'world'


def _box(m):
    x, y, z = m[12], m[13], m[14]
    return (x - 1, y - 1, z - 1, 0.0, x + 1, y + 1, z + 1, 0.0)


def write_locator_object(L: Locator) -> bytes:
    n = len(L.spawns)
    pool_at = 0xA0 + 0x70 * n
    pool = bytearray(L.name.encode('ascii') + b'\0')
    world_at = pool_at + len(pool)
    pool += L.world.encode('ascii') + b'\0'
    names = []
    for nm, _m in L.spawns:
        names.append(pool_at + len(pool))
        pool += nm.encode('ascii') + b'\0'
    out = bytearray(struct.pack('>8I', 1, n, 0, 0x20, 0xA0, pool_at, 0, 0))
    out += struct.pack('>16f', *L.matrix) + struct.pack('>8f', *_box(L.matrix))
    out += struct.pack('>4IQ8x', n, 0xA0, pool_at, world_at, L.locator_id)
    for (nm, m), no in zip(L.spawns, names):
        out += struct.pack('>16f', *m) + struct.pack('>8f', *_box(m)) + struct.pack('>I12x', no)
    return bytes(out + pool)


def parse_locator_object(d: bytes) -> Locator:
    one, n, _z, root, recs, pool_at = struct.unpack_from('>6I', d, 0)

    def s(o):
        return d[o:d.index(b'\0', o)].decode('ascii')
    m = struct.unpack_from('>16f', d, root)
    _n, _r, _p, world_at, lid = struct.unpack_from('>4IQ', d, 0x80)
    spawns = []
    for i in range(n):
        b = recs + 0x70 * i
        spawns.append((s(struct.unpack_from('>I', d, b + 0x60)[0]), struct.unpack_from('>16f', d, b)))
    return Locator(s(pool_at), m, lid, spawns, s(world_at))


def spawn_ring(name: str, base: str, m: tuple, step: float = 2.0) -> list:
    """The six spawn points community locators carry around a locator: +-x, +-z, then the two
    diagonals (+x,+z) and (-x,-z) at `step` metres (measured on every 'freeskate_*_locator')."""
    offs = [(step, 0), (-step, 0), (0, step), (0, -step), (step, step), (-step, -step)]
    out = []
    for i, (dx, dz) in enumerate(offs, 1):
        q = list(m)
        # the offset is taken along the matrix columns 0 and 2 (the locator's own x and z)
        q[12] = _f32(m[12] + dx * m[0] + dz * m[2])
        q[13] = _f32(m[13] + dx * m[4] + dz * m[6])
        q[14] = _f32(m[14] + dx * m[8] + dz * m[10])
        out.append((f'{name}::{base}_spawn_{i}', tuple(q)))
    return out


def _f32(v: float) -> float:
    return struct.unpack('>f', struct.pack('>f', v))[0]


def locator_arena(template: bytes, L: Locator) -> bytes:
    """A locator .rx2: the template arena's three objects with object 1 rebuilt from `L` and the
    locator id in object 2 (+0x1C) set; written by rw4_write."""
    a = rw4.parse(template)
    spec = rw4_write.spec_from_arena(a)
    objs = []
    for o, so in zip(a.objects, spec.objects):
        data = so.data
        if o.type_id == 0xEB0009:
            data = write_locator_object(L)
        elif o.type_id == 0xEB000B:
            data = data[:0x1C] + struct.pack('>Q', L.locator_id) + data[0x24:]
        objs.append(rw4_write.SpecObject(so.type_id, so.align, data))
    spec.objects = objs
    return rw4_write.build(spec)


# ---- the content header ------------------------------------------------------------------------
HEADER_SIZE = 328
TITLE_ID = 0x454108E6


def write_header(display: str, content_id: str) -> bytes:
    """The 328-byte XCONTENT header beside a raw .big (layout from the owner's makeheader.py and
    the recomp's picker: version 1, content type 2, UTF-16BE name at 8, content id at 0x108, title
    id 12 bytes from the end). The content id must equal the package directory name."""
    if len(content_id) > 42:
        raise ValueError('content id longer than 42')
    h = bytearray(HEADER_SIZE)
    struct.pack_into('>II', h, 0, 1, 2)
    nm = display.encode('utf-16-be')[:0x108 - 8 - 2]
    h[8:8 + len(nm)] = nm
    h[0x108:0x108 + len(content_id)] = content_id.encode('ascii')
    struct.pack_into('>I', h, HEADER_SIZE - 12, TITLE_ID)
    return bytes(h)


def content_id_for(name: str) -> str:
    return (re.sub(r'[^A-Za-z0-9]', '', name).upper() or 'SKATE3PACK')[:42]


# ---- inventory ---------------------------------------------------------------------------------
CATEGORIES = (
    (r'^data/boundary/freeskate_dlc_.*\.xml$', 'boundary'),
    (r'^data/content/global_locators/.*\.loc$', 'locator table'),
    (r'^data/content/global_locators/.*\.rx2$', 'locator arena'),
    (r'^data/content/missions/.*\.p(mm|sm|ss|st)$', 'mission manifest (empty)'),
    (r'^data/content/world/stream/.*\.xsf$', 'stream'),
    (r'^data/content/world/stream/.*\.x(mm|sm|ss|st)$', 'DIST manifest'),
    (r'^data/content/world/stream/.*\.loc$', 'location table'),
    (r'^data/db/.*\.vaultlist\.xml$', 'vault list'),
    (r'^data/db/.*\.vlt$', 'vault'),
    (r'^data/db/.*\.bin$', 'vault data'),
    (r'^data/db/.*\.opt$', 'vault options'),
    (r'^data/fe/languages/.*HISTOGRAM.*\.BIN$', 'language histogram'),
    (r'^data/fe/languages/.*\.BIN$', 'language strings'),
    (r'^data/stream/freeskate_dlc_.*\.xml$', 'location preload list'),
    (r'^data/stream/.*\.xml$', 'neighbour table'),
    (r'^data/unlocks/.*_product\d+\.unlock$', 'product unlock'),
    (r'^data/unlocks/.*\.unlock$', 'unlock list'),
)


def category(path: str) -> str:
    for rx, name in CATEGORIES:
        if re.match(rx, path):
            return name
    return 'other'


def inventory(pack_big: str) -> dict:
    """{category: [paths]} for every file of a pack archive."""
    a = big.read(pack_big)
    out: dict = {}
    for e in a.entries:
        out.setdefault(category(e.path), []).append(e.path)
    return out


# ---- template + generator ----------------------------------------------------------------------
@dataclass
class Spot:
    title: str
    pos: tuple                    # (x, y, z) metres, game space
    yaw: float = 0.0              # degrees about +Y
    matrix: tuple | None = None   # explicit 4x4 (row-major) instead of pos + yaw


@dataclass
class Template:
    """What a generated pack takes from a working pack: its internal names (slug, DIST name,
    location keys) and the files keyed by them that we cannot re-key (vaults, locator ids)."""
    path: str
    slug: str
    dist: str
    keys: list                    # location keys, main first
    vaults: dict                  # data/db/** path -> bytes
    locators: dict                # locator name -> (file stem, arena bytes)


def read_template(pack_big: str) -> Template:
    a = big.read(pack_big)
    files = {e.path: e for e in a.entries}
    dists = sorted({e.root.rsplit('/', 1)[1] for e in a.entries
                    if e.root.startswith('data/content/world/stream/')})
    if len(dists) != 1:
        raise ValueError(f'{pack_big}: {len(dists)} worlds; a template must hold exactly one')
    dist = dists[0]
    loc = next(p for p in files if p.endswith('_Sim.loc') and '/world/stream/' in p)
    unl = [p for p in files if p.startswith('data/unlocks/') and not re.search(r'_product\d+', p)]
    slug = unl[0].rsplit('/', 1)[1][:-len('.unlock')]
    keys = [m.group(1) for m in (re.match(rf'^freeskate_{re.escape(slug)}_(.+)_locator$', L.name)
                                 for L in mf.read_loc(a.read(files[loc]))) if m]
    vaults = {p: a.read(e) for p, e in files.items() if p.startswith('data/db/')}
    locs = {}
    for p, e in files.items():
        if p.startswith('data/content/global_locators/') and p.endswith('.loc'):
            nm = mf.read_loc(a.read(e))[0].name
            locs[nm] = (p.rsplit('/', 1)[1][:-4], a.read(files[p[:-4] + '.rx2']))
    return Template(pack_big, slug, dist, keys, vaults, locs)


def tile_bounds_default(name: str, cell: int, y=(-100.0, 300.0)):
    c = mf.tile_center(name)
    h = cell / 2
    return (c[0] - h, y[0], c[1] - h, c[0] + h, y[1], c[1] + h)


def spot_center(pos, cell: int, rule: str = 'containing'):
    if rule == 'containing':
        return (math.floor(pos[0] / cell) * cell + cell // 2, math.floor(pos[2] / cell) * cell + cell // 2)
    # what the community packs do: the tile whose far corner is nearest
    return (round(pos[0] / cell) * cell - cell // 2, round(pos[2] / cell) * cell - cell // 2)


def build_files(display: str, tiles: dict, spots: list, tpl: Template, cell: int = 100,
                content: dict | None = None, center_rule: str = 'containing',
                stream_form: str = 'raw', group: dict | None = None, tail: dict | None = None,
                dlc_tables: bool = True) -> dict:
    """Every file of a pack for a new map: {path: bytes}.

    tiles:   {'Pres'|'Sim'|'Tex': [(tile name, [(asset id, arena bytes)])]}; tile names
             c<Kind>_<x>_<z>_high (centres in metres) or c<Kind>_Global.
    spots:   [Spot]; spots[0] is the main location, the rest fill the template's other location
             keys in order; template locations left over reuse spots[0].
    content: optional {tile name: (minx, miny, minz, maxx, maxy, maxz)} content boxes for CSPA
             (default: the cell, y -100..300).
    group:   optional {kind: group hash} (default: derived from the DIST name).
    tail:    optional {kind: SFIL tail word} (default: tile count << 24, as most packs; the word
             is unused stack garbage in EA's own streams).
    dlc_tables: also write stream/dist_<slug>_<key>dlc_{Pres,Sim}.xml, the tables the game reads
             for a DLC location (community packs other than Skate 2 lack them)."""
    S, D, keys = tpl.slug, tpl.dist, tpl.keys
    root = f'data/content/world/stream/{D}'
    out = {}
    content = content or {}
    for kind, _prefix in mf.KINDS:
        tl = tiles.get(kind, [])
        g = (group or {}).get(kind, mf.group_hash(D, kind))
        mt = []
        for name, arenas in tl:
            assets = [sfil.Asset(aid, ar) for aid, ar in arenas]
            if stream_form == 'raw':
                x = sfil.raw_stream(g, assets, tail=(tail or {}).get(kind, (len(tl) & 0xFF) << 24))
            else:
                x = sfil.ea_stream(g, assets)
            out[f'{root}/{name}.xsf'] = x
            k = content.get(name) or (None if name.endswith('_Global') else tile_bounds_default(name, cell))
            mt.append(mf.Tile(name, arenas, len(x), k))
        # an empty group's CMAP says platform 2 even in raw-form packs (all community packs)
        gen = mf.generate(D, kind, mt, cell=cell, platform=0 if stream_form == 'raw' and mt else 2,
                          alignment=0x100 if stream_form == 'raw' else 0x80,
                          size_includes_header=stream_form == 'raw', group=g)
        for ext, b in gen.items():
            out[f'{root}/{D}_{kind}{ext}'] = b
    centers = sorted({mf.tile_center(n) for k in ('Pres', 'Sim') for n, _ in tiles.get(k, [])
                      if mf.tile_center(n)})
    locs = [(keys[0], spots[0])] + [(k, spots[i + 1] if i + 1 < len(spots) else spots[0])
                                    for i, k in enumerate(keys[1:])]
    titles = [(k, locs[i][1].title if i < len(spots) else f'{spots[0].title} {i + 1}')
              for i, (k, _sp) in enumerate(locs)]
    table = []
    nb = write_stream_tiles(neighbour_table(centers, cell))
    for key, sp in locs:
        m = sp.matrix or mf.spawn_transform(*sp.pos, yaw_deg=sp.yaw)
        base = f'freeskate_{S}_{key}'
        for nm, ring in ((f'{base}_locator', True), (f'Z_{S}_{key}_Start', False)):
            stem, arena = tpl.locators[nm]
            lid = int(re.search(r'\[0x([0-9a-f]+)\]', stem).group(1), 16)
            L = Locator(nm, m, lid, spawn_ring(nm, base, m) if ring else [])
            gl = f'data/content/global_locators/DLC_BAM/{D}/{stem}'
            out[gl + '.loc'] = mf.write_loc([mf.Location(nm, m)])
            out[gl + '.rx2'] = locator_arena(arena, L)
            table.append(mf.Location(nm, m))
        loc_name = f'freeskate_dlc_{S}_{key}'
        out[f'data/boundary/{loc_name}.xml'] = write_boundary(boundary_rect(centers, cell))
        out[f'data/stream/{loc_name}.xml'] = write_stream_tiles(
            preload_list(spot_center(sp.pos, cell, center_rule), centers))
        if dlc_tables:
            out[f'data/stream/dist_{S}_{key}dlc_Pres.xml'] = nb
            out[f'data/stream/dist_{S}_{key}dlc_Sim.xml'] = nb
        mis = f'data/content/missions/freeskate_{S}_{key}/freeskate_{S}_{key}'
        for kind in ('Pres', 'Sim', 'Tex'):
            g = 0x5634DD36 if kind == 'Pres' else 0x5634DD26
            out[f'{mis}_{kind}.pmm'] = mf.write_mmap(g)
            out[f'{mis}_{kind}.psm'] = mf.write_cmap(mf.Cmap(g, 3))
            out[f'{mis}_{kind}.pss'] = mf.write_cspa(mf.Cspa(g))
            out[f'{mis}_{kind}.pst'] = mf.write_atoc(mf.Atoc(g))
    out[f'{root}/{D}_Sim.loc'] = mf.write_loc(table)
    out[f'data/stream/{D}_Pres.xml'] = nb
    out[f'data/stream/{D}_Sim.xml'] = nb
    lang = location_strings(S, display, titles)
    out[f'data/fe/languages/english/LANGUAGE_English_{S}_skate3ng-dlc_{S}_minimal.BIN'] = write_language(lang)
    out[f'data/fe/languages/english/LANGUAGE_English_HISTOGRAM_skate3ng-dlc_{S}_minimal.BIN'] = write_histogram()
    out[f'data/unlocks/{S}.unlock'] = b''
    out[f'data/unlocks/{S}0000_product00000000.unlock'] = product_unlock(S)
    out.update(tpl.vaults)
    return out


def inputs_of_pack(pack_big: str, tpl: Template | None = None):
    """The generator inputs a working pack implies: (display name, tiles, spots, cell, group
    hashes). Used to rebuild a pack's metadata and compare it with the original."""
    tpl = tpl or read_template(pack_big)
    a = big.read(pack_big)
    files = {e.path: e for e in a.entries}
    S, D = tpl.slug, tpl.dist
    root = f'data/content/world/stream/{D}'
    lang = read_language(a.read(files[f'data/fe/languages/english/LANGUAGE_English_{S}_skate3ng-dlc_{S}_minimal.BIN']))
    hs = lang.by_hash()
    display = hs[string_id(f'ID_MAP_CATEGORY_{S.upper()}')]
    tiles, group, tail, content = {}, {}, {}, {}
    for kind, _p in mf.KINDS:
        cm = mf.read_cmap(a.read(files[f'{root}/{D}_{kind}.xsm']))
        cs = mf.read_cspa(a.read(files[f'{root}/{D}_{kind}.xss']))
        names = [e.name for e in cm.entries if not e.name.endswith('_Global')]
        for n, c in zip(names, cs.cells):
            content[n] = c.content
        group[kind] = cm.group
        if cm.entries:
            tail[kind] = sfil.read(a.read(files[f'{root}/{cm.entries[0].name}.xsf']))[0]['tail']
        tiles[kind] = [(e.name, [(s.id, s.arena) for s in sfil.read(a.read(files[f'{root}/{e.name}.xsf']))[1]])
                       for e in cm.entries]
    spots = []
    for k in tpl.keys:
        nm = f'freeskate_{S}_{k}_locator'
        m = parse_locator_object(rw4.parse(tpl.locators[nm][1]).objects[1].data).matrix
        t = hs[string_id(f'ID_LOCATION_{S.upper()}_{k.upper()}')]
        yaw = math.degrees(math.atan2(-m[2], m[0]))
        spots.append(Spot(t[len(display) + 3:] if t.startswith(display + ' - ') else t,
                          tuple(m[12:15]), yaw, tuple(m)))
    xs = sorted({mf.tile_center(n)[0] for n, _ in tiles['Sim'] if mf.tile_center(n)})
    cell = min((b - a2 for a2, b in zip(xs, xs[1:])), default=100)
    return display, tiles, spots, cell, group, tail, content


def write_pack(files: dict, out_dir: str, slug: str, display: str, content_id: str | None = None):
    """<out_dir>/<CONTENTID>/<slug>_00000000.big + .header beside it; returns the .big path."""
    cid = content_id or content_id_for(display)
    pkg = os.path.join(out_dir, cid)
    os.makedirs(pkg, exist_ok=True)
    stem = os.path.join(pkg, f'{slug}_00000000')
    srcs = [big.Source(*p.rsplit('/', 1), data=b) for p, b in files.items()]
    with open(stem + '.big', 'wb') as f:
        big.write(f, srcs, big.FLAGS_DLC)
    with open(stem + '.header', 'wb') as f:
        f.write(write_header(display, cid))
    return stem + '.big'
