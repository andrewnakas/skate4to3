"""Skate 3 AttribSys vaults: `.vlt` (records) + `.bin` (data) + `.opt` (empty), the db/ files of
retail db.big and of every DLC pack. Facts and how they were measured: specs/sk3_packaging.md
"Vaults (AttribSys)".

Layers:
  1. file codec  - `read_vlt`/`write_vlt`, `read_bin`/`write_bin`: chunks, DepN, ExpN, PtrN kept as
                   numbers; byte-exact over every vault in retail db.big and in the packs.
  2. schema      - `Schema` from retail skaterschema.vlt/.bin: type names + sizes, class layouts and
                   per-field descriptors {key, type, offset, size, max count, flags, align}.
  3. database    - `decode` turns a vault into collections whose fields hold numbers, strings or
                   data blocks (pointers resolved); `encode` lays them out with EA's rules (474/508
                   retail vaults re-encode byte for byte; the other 34 hold nested data pointers,
                   which `decode` refuses).
  4. generator   - `dlc_vaults(slug, keys, dist)`: every db/ file of a single-world DLC pack, from
                   its names alone (the content measured on three working packs).
  5. comparison  - `normalized` / `semantic_diff`: what the game reads, independent of layout.

All words are big-endian. Keys are `attrib_hash.attrib_key` of names.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .attrib_hash import attrib_key

# export record types (ExpN word 2); names unknown, values measured
EXP_COLLECTION = 0xAD303B8F42B3307E      # 1248/1248 pack exports, 14319 retail
EXP_CLASS = 0x2A7895AC4A876152           # 329 in skaterschema.vlt
EXP_DATABASE = 0x0B38846845E9C175        # 1 in skaterschema.vlt

# entry flag bits (u16 at entry +0xE)
F_INLINE = 0x4000      # value word holds the value (types of <= 4 bytes; Text = string pointer)
F_COMPLEX = 0x0800     # schema field flag 0x20
F_ARRAY = 0x0200       # schema field flag 0x01

VERSION_PACK = 0x693CDCC3F57ADB28        # Vers of all 161 pack vaults
VERSION_RETAIL = 0xB2157AD88EF77187      # Vers of 505/509 retail vaults


def _a(n, a):
    return (n + a - 1) // a * a


# ================================================================================ 1. file codec
@dataclass
class Vlt:
    version: int
    strn: bytes                 # StrN payload (8 zero bytes everywhere measured)
    deps: list                  # [(hash, name)]
    datn: bytes                 # DatN payload
    datn_at: int                # file offset of the DatN payload (export offsets are file offsets)
    exports: list               # [(id, type, size, offset)]
    ptrs: list                  # [(offset, kind, arg, target)] incl. the kind-0 terminator
    endc: bytes = bytes(8)


def chunk_list(d: bytes) -> list:
    out, off = [], 0
    while off < len(d):
        tag, size = struct.unpack_from('>4sI', d, off)
        if size < 8 or off + size > len(d):
            break
        out.append((tag.decode('latin1'), off, d[off + 8:off + size]))
        off += size
    return out, off


def _pad16(b: bytes) -> bytes:
    return b + bytes(-(len(b) + 8) % 16)


def _chunk(tag: str, payload: bytes) -> bytes:
    return tag.encode('latin1') + struct.pack('>I', len(payload) + 8) + payload


def depn_payload(deps) -> bytes:
    """u64 count, u64 hashes, u32 name offsets (padded to 8), names NUL-terminated."""
    n = len(deps)
    p = bytearray(struct.pack('>Q', n))
    p += b''.join(struct.pack('>Q', h) for h, _ in deps)
    offs, pool = [], bytearray()
    for _h, nm in deps:
        offs.append(len(pool))
        pool += nm.encode('latin1') + b'\0'
    p += struct.pack(f'>{n}I', *offs) + bytes(-4 * n % 8)
    return bytes(p + pool)


def read_vlt(d: bytes) -> Vlt:
    cs, end = chunk_list(d)
    if end != len(d):
        raise ValueError('vlt: trailing bytes')
    tags = [t for t, _o, _p in cs]
    if tags != ['Vers', 'StrN', 'DepN', 'DatN', 'ExpN', 'PtrN', 'EndC']:
        raise ValueError(f'vlt: chunks {tags}')
    by = {t: (o, p) for t, o, p in cs}
    version = struct.unpack_from('>Q', by['Vers'][1])[0]
    p = by['DepN'][1]
    n = struct.unpack_from('>Q', p)[0]
    hs = struct.unpack_from(f'>{n}Q', p, 8)
    offs = struct.unpack_from(f'>{n}I', p, 8 + 8 * n)
    base = 8 + 8 * n + _a(4 * n, 8)
    names = [p[base + o:p.index(b'\0', base + o)].decode('latin1') for o in offs]
    p = by['ExpN'][1]
    n = struct.unpack_from('>Q', p)[0]
    exports = [struct.unpack_from('>QQII', p, 8 + 24 * i) for i in range(n)]
    p = by['PtrN'][1]
    ptrs = []
    for i in range(len(p) // 16):
        e = struct.unpack_from('>IHHQ', p, 16 * i)
        ptrs.append(e)
        if e[1] == 0:
            break
    v = Vlt(version, by['StrN'][1], list(zip(hs, names)), by['DatN'][1], by['DatN'][0] + 8, exports,
            ptrs, by['EndC'][1])
    if write_vlt(v) != d:
        raise ValueError('vlt: layout not reproduced')
    return v


def write_vlt(v: Vlt) -> bytes:
    out = bytearray(_chunk('Vers', struct.pack('>Q', v.version)))
    out += _chunk('StrN', v.strn)
    out += _chunk('DepN', _pad16(depn_payload(v.deps)))
    if len(out) + 8 != v.datn_at:
        raise ValueError(f'vlt: DatN would start at {len(out) + 8:#x}, not {v.datn_at:#x}')
    out += _chunk('DatN', _pad16(v.datn))
    ex = struct.pack('>Q', len(v.exports)) + b''.join(struct.pack('>QQII', *e) for e in v.exports)
    out += _chunk('ExpN', _pad16(ex))
    out += _chunk('PtrN', _pad16(b''.join(struct.pack('>IHHQ', *e) for e in v.ptrs)))
    out += _chunk('EndC', v.endc)
    return bytes(out)


@dataclass
class Bin:
    """`.bin`: a StrE chunk, EndC, then (retail) the data area. Pack builders put everything inside
    StrE and end the file at EndC. Offsets in the vault's pointers are file offsets."""
    stre: bytes                 # StrE payload
    tail: bytes = b''           # bytes after EndC
    endc: bytes = bytes(8)


def read_bin(d: bytes) -> Bin:
    tag, size = struct.unpack_from('>4sI', d, 0)
    if tag != b'StrE' or d[size:size + 4] != b'EndC':
        raise ValueError('bin: not StrE + EndC')
    esz = struct.unpack_from('>I', d, size + 4)[0]
    return Bin(d[8:size], d[size + esz:], d[size + 8:size + esz])


def write_bin(b: Bin) -> bytes:
    return _chunk('StrE', b.stre) + _chunk('EndC', b.endc) + b.tail


OPT = _chunk('EndC', bytes(8))          # every .opt measured (161 pack + 3 retail)


# ================================================================================ 2. schema
@dataclass
class FieldDef:
    key: int
    type: int
    offset: int                 # in the class layout (static fields)
    size: int                   # element size
    max_count: int              # 1, n, or 0xFFFF (unbounded array)
    flags: int                  # 0x01 array, 0x02 static (in layout), 0x04 ?, 0x20 complex type
    align: int                  # log2

    @property
    def static(self):
        return bool(self.flags & 2)


@dataclass
class ClassDef:
    key: int
    w08: int                    # unidentified (not the collection count)
    layout_size: int
    nstatic: int
    fields: dict                # key -> FieldDef, in schema order (ascending key)


@dataclass
class Schema:
    types: dict                 # type key -> (name, size)
    classes: dict               # class key -> ClassDef

    def type_name(self, k):
        return self.types.get(k, (f'{k:016x}', 0))[0]


def _ptr_maps(ptrs):
    """{slot: target} for slots in the vlt (section 0) and in the bin (section 1)."""
    vs, bs, sect = {}, {}, None
    for o, k, a, t in ptrs:
        if k == 2:
            sect = a
        elif k == 3:
            if a != 1:
                raise ValueError(f'pointer kind 3 arg {a}')
            (vs if sect == 0 else bs)[o] = t
    return vs, bs


def read_schema(vlt: bytes, binf: bytes) -> Schema:
    """skaterschema.vlt/.bin: the database export holds {u32 0x202, u32, u32 type count, u32 0,
    u32 sizes[count]} and points at the type names (NUL-separated, same order); each class export
    is 0x28 bytes {u64 key, u32 w08, u32 field count, u32 -> field table, 0, 0, u32 layout size,
    u32 static field count, 0}; the field table is 24 bytes per field {u64 key, u64 type, u16
    offset, u16 size, u16 max count, u8 flags, u8 log2 align}, ascending by key."""
    v = read_vlt(vlt)
    vs, _bs = _ptr_maps(v.ptrs)
    types, classes = {}, {}
    D = vlt
    for k, ty, size, off in v.exports:
        if ty == EXP_DATABASE:
            _w0, _w1, nt, _w3 = struct.unpack_from('>4I', D, off)
            sizes = struct.unpack_from(f'>{nt}I', D, off + 16)
            at = vs[off + 0xC]
            names = binf[at:].split(b'\0')[:nt]
            for nm, sz in zip(names, sizes):
                n = nm.decode('latin1')
                types[attrib_key(n)] = (n, sz)
    for k, ty, size, off in v.exports:
        if ty == EXP_CLASS:
            ck, w08, nf, _p, _z1, _z2, lay, nst, _z3 = struct.unpack_from('>QIIIIIIII', D, off)
            fo = vs.get(off + 0x10)
            fields = {}
            for i in range(nf):
                f = FieldDef(*struct.unpack_from('>QQHHHBB', binf, fo + 24 * i))
                fields[f.key] = f
            classes[ck] = ClassDef(ck, w08, lay, nst, fields)
    return Schema(types, classes)


def load_schema(db_big: str) -> Schema:
    from . import big
    a = big.read(db_big)
    f = {e.path: e for e in a.entries}
    return read_schema(a.read(f['data/db/skaterschema.vlt']), a.read(f['data/db/skaterschema.bin']))


# ================================================================================ 3. database
@dataclass
class Text:
    """A pointer to a NUL-terminated string in the .bin."""
    s: str

    def __hash__(self):
        return hash(('T', self.s))


@dataclass
class Blob:
    """A data block in the .bin (a layout, a value, or an array with its 8-byte header
    {u16 count, u16 capacity, u16 element size, u16 0}); `strings` = pointer slots in it
    {offset: text}."""
    data: bytes
    align: int = 4
    strings: dict = field(default_factory=dict)
    es: int = 0                 # element size of an array (0: not an array)

    def member(self, o):
        """Offset of a slot inside its element (EA sorts the .bin pointer table by it)."""
        return (o - 8) % self.es if self.es else o

    def key(self):
        return (self.data, tuple(sorted(self.strings.items())))


class AtBinStart:
    """An out-of-line value whose pointer is .bin offset 0 (the StrE chunk head). Community DLC
    builders write this for struct values they cannot serialize (4 per pack, spec "Vaults")."""

    def __eq__(self, o):
        return isinstance(o, AtBinStart)

    def __repr__(self):
        return 'BIN0'


BIN0 = AtBinStart()


@dataclass
class Entry:
    key: int
    type: int
    flags: int
    value: object               # int (inline word), Text (inline string pointer), Blob


@dataclass
class Collection:
    key: int
    cls: int
    parent: int = 0
    entries: list = field(default_factory=list)
    types: list = field(default_factory=list)   # stored type list (static-field types first)
    ntypes: int = 0             # u16 at +0x24 (types used by entries)
    capacity: int = 0           # u32 at +0x18
    w1c: int = 0                # u32 at +0x1C
    layout: Blob | None = None
    w2c: int = 0
    eid: int = 0                # ExpN id = attrib_key('<class name>/<collection name>')


@dataclass
class Database:
    """One vault: its file names, collections in DatN order, other exports kept raw."""
    vlt_name: str
    bin_name: str
    items: list                 # [Collection | (id, type, bytes)]
    version: int = VERSION_RETAIL
    form: str = 'ea'            # 'ea' (data after EndC) | 'pack'


def _cstr(b: bytes, o: int) -> str:
    return b[o:b.index(b'\0', o)].decode('latin1')


def blob_size(sch: Schema, cls: ClassDef | None, fk: int, ty: int, flags: int, binf: bytes, at: int):
    if flags & F_ARRAY:
        cnt, cap, es, _z = struct.unpack_from('>HHHH', binf, at)
        return 8 + max(cnt, cap) * es
    f = cls.fields.get(fk) if cls else None
    if f is not None:
        return f.size
    return sch.types[ty][1]


def blob_align(sch: Schema, cls: ClassDef | None, fk: int, ty: int, flags: int):
    f = cls.fields.get(fk) if cls else None
    a = 1 << f.align if f is not None else 4
    if flags & F_ARRAY:
        a = max(a, 4)
    return a


def layout_align(cls: ClassDef):
    return max([1 << f.align for f in cls.fields.values() if f.static] or [4])


def decode(vlt: bytes, binf: bytes, sch: Schema) -> Database:
    v = read_vlt(vlt)
    vs, bs = _ptr_maps(v.ptrs)
    items = []

    pool_end = 8 + len(read_bin(binf).stre) if read_bin(binf).tail else len(binf)
    nested = [t for t in bs.values() if t >= pool_end]
    if nested:      # a .bin slot pointing at data, not at a string: not modelled (34 retail vaults)
        raise ValueError(f'{len(nested)} nested data pointers')

    def strings_in(at, n):
        return {s - at: Text(_cstr(binf, t)) for s, t in bs.items() if at <= s < at + n}

    for eid, ty, size, off in v.exports:
        rec = vlt[off:off + size]
        if ty != EXP_COLLECTION:
            items.append((eid, ty, rec))
            continue
        k, c, par, cap, w1c, ne, nt, tl, lay, w2c = struct.unpack_from('>QQQIIIHHII', vlt, off)
        cls = sch.classes.get(c)
        col = Collection(k, c, par, capacity=cap, w1c=w1c, ntypes=nt, w2c=w2c, eid=eid)
        col.types = list(struct.unpack_from(f'>{tl}Q', vlt, off + 0x30))
        if off + 0x28 in vs:
            at = vs[off + 0x28]
            n = cls.layout_size
            col.layout = Blob(binf[at:at + n], layout_align(cls), strings_in(at, n))
        elif lay:
            raise ValueError('layout word without pointer')
        eb = off + 0x30 + 8 * tl
        for i in range(ne):
            fk, val, ti, fl = struct.unpack_from('>QIHH', vlt, eb + 16 * i)
            slot = eb + 16 * i + 8
            tk = col.types[ti]
            if slot in vs:
                at = vs[slot]
                if fl & F_INLINE:
                    value = Text(_cstr(binf, at))
                elif at == 0:
                    value = BIN0
                else:
                    n = blob_size(sch, cls, fk, tk, fl, binf, at)
                    es = struct.unpack_from('>H', binf, at + 4)[0] if fl & F_ARRAY else 0
                    value = Blob(binf[at:at + n], blob_align(sch, cls, fk, tk, fl), strings_in(at, n), es)
            else:
                value = val
            col.entries.append(Entry(fk, tk, fl, value))
        if 0x30 + 8 * tl + 16 * ne != size:
            raise ValueError('collection size')
        items.append(col)
    names = [n for _h, n in v.deps]
    # EA puts data blocks after EndC; community builders put them inside StrE (no tail)
    has_data = any(c.layout is not None or any(isinstance(e.value, Blob) for e in c.entries)
                   for c in items if isinstance(c, Collection))
    form = 'pack' if has_data and not read_bin(binf).tail else 'ea'
    return Database(names[0], names[1] if len(names) > 1 else '', items, v.version, form)


# ---- encoder ---------------------------------------------------------------------------------
def collection_record(col: Collection, layout_word: int = 0) -> bytes:
    tidx = {t: i for i, t in enumerate(col.types)}
    out = bytearray(struct.pack('>QQQIIIHHII', col.key, col.cls, col.parent, col.capacity, col.w1c,
                                len(col.entries), col.ntypes, len(col.types), layout_word, col.w2c))
    out += b''.join(struct.pack('>Q', t) for t in col.types)
    for e in col.entries:
        val = e.value if isinstance(e.value, int) else 0
        out += struct.pack('>QIHH', e.key, val, tidx[e.type], e.flags)
    return bytes(out)


def encode(db: Database) -> tuple:
    """(vlt bytes, bin bytes) in EA's layout:
    - DatN = the records back to back in item order; ExpN lists them in the same order.
    - .bin = StrE (every string once, first-use order, NUL-terminated, chunk padded to 16), EndC,
      then per collection its layout and its out-of-line values in entry order, each aligned.
    - PtrN = section 1 (.bin slots) then section 0 (.vlt slots), each ascending, zero terminator."""
    cols = [c for c in db.items if isinstance(c, Collection)]
    # string pool, first use
    pool, at = bytearray(), {}

    def use(t):
        if t.s not in at:
            at[t.s] = 8 + len(pool)
            pool.extend(t.s.encode('latin1') + b'\0')
    for c in cols:
        if c.layout:
            for _o, t in sorted(c.layout.strings.items()):
                use(t)
        for e in c.entries:
            if isinstance(e.value, Text):
                use(e.value)
            elif isinstance(e.value, Blob):
                for _o, t in sorted(e.value.strings.items()):
                    use(t)
    stre = bytes(pool) + bytes(-(len(pool) + 8) % 16)
    data = bytearray()
    base = 8 + len(stre) + 16
    bslots, place = {}, {}

    def put(blob):
        o = _a(base + len(data), blob.align) - base
        data.extend(bytes(o - len(data)))
        data.extend(blob.data)
        for so, t in blob.strings.items():
            bslots[base + o + so] = (blob.member(so), at[t.s])
        return base + o
    for c in cols:
        if c.layout is not None:
            place[(id(c), None)] = put(c.layout)
        for i, e in enumerate(c.entries):
            if isinstance(e.value, Blob):
                place[(id(c), i)] = put(e.value)
    binf = write_bin(Bin(stre, bytes(data)))
    # records

    deps = [(attrib_key(n), n) for n in (db.vlt_name, db.bin_name) if n]
    head = 16 + 16 + 8 + len(_pad16(depn_payload(deps)))
    datn_at = head + 8
    datn, exports, vslots, lslots = bytearray(), [], {}, {}
    for it in db.items:
        off = datn_at + len(datn)
        if isinstance(it, Collection):
            lw = 0
            rec = collection_record(it, lw)
            if it.layout is not None:
                vslots[off + 0x28] = place[(id(it), None)]
            eb = off + 0x30 + 8 * len(it.types)
            for i, e in enumerate(it.entries):
                if isinstance(e.value, Text):
                    vslots[eb + 16 * i + 8] = at[e.value.s]
                elif isinstance(e.value, Blob):
                    vslots[eb + 16 * i + 8] = place[(id(it), i)]
                elif e.value is BIN0:
                    vslots[eb + 16 * i + 8] = 0
            for sl in vslots:
                lslots.setdefault(sl, off)
            exports.append((it.eid, EXP_COLLECTION, len(rec), off))
        else:
            eid, ty, rec = it
            exports.append((eid, ty, len(rec), off))
        datn += rec
    ptrs = []
    # EA writes the .bin section header (even empty) when any entry has a complex type (0x0800)
    if bslots or (db.form == 'ea' and any(e.flags & F_COMPLEX for c in cols for e in c.entries)):
        ptrs.append((0, 2, 1, 0))
        ptrs += [(s, 3, 1, t) for s, (_m, t) in sorted(bslots.items(), key=lambda x: (x[1][0], x[0]))]
    ptrs.append((0, 2, 0, 0))
    # .vlt slots: by offset inside their record, then by address (as the .bin slots by member)
    ptrs += [(s, 3, 1, t) for s, t in sorted(vslots.items(), key=lambda x: (x[0] - lslots[x[0]], x[0]))]
    ptrs.append((0, 0, 0, 0))
    v = Vlt(db.version, bytes(8), deps, bytes(datn), datn_at, exports, ptrs)
    return write_vlt(v), binf


# ================================================================================ 4. DLC generator
# What a single-world DLC pack's db/ holds, as measured on SUMMERCAMP, ISLEOFGROM and SUMMERCAMPHD
# (spec "Vaults: DLC content"). Every key is attrib_key of a name built from:
#   slug  pack slug (summercamp)            s4    slug[:4] (summ) - the 'dlc_<s4>' prefix
#   keys  location keys, keys[0] = main     dist  DIST name (DIST_SummerCamp)
# Values are constants measured in all three packs (field keys are hashes whose names are not
# known; types are named by the retail schema). Most constant collections are copies of retail
# collections (challenge_local_data 'default', the freeskate activity collections, ...).

class _Spec:
    """Formatting context for the recipe (names of one pack)."""

    def __init__(self, slug, keys, dist):
        self.slug, self.keys, self.dist = slug, list(keys), dist
        self.s4 = slug[:4]
        alnum = ''.join(ch for ch in slug.upper() if ch.isalnum())
        self.product = alnum[:7].ljust(16, '0')         # dlc_mapping product id (6/6 packs)

    def fmt(self, s, key=None):
        k = key if key is not None else self.keys[0]
        return s.format(slug=self.slug, SLUG=self.slug.upper(), s4=self.s4, S4=self.s4.upper(),
                        dist=self.dist, key=k, KEY=k.upper(), key0=self.keys[0],
                        KEY0=self.keys[0].upper(), product=self.product)


def _t(s):
    return ('T', s)


def _r(cls, name):          # Attrib::RefSpec {u64 class, u64 collection, u64 0}
    return ('R', cls, name)


def _vr(cls, name):         # AttribSysUtils::tVaultedRefSpec {class, collection, 0, ptr -> ''}
    return ('VR', cls, name)


def _cr(name):              # Attrib::Gen::ClassRefSpec_* {u64 collection, u64 0}
    return ('CR', name)


def _arr(es, align):        # empty array: header {count 0, capacity 0, element size, 0}
    return ('B', '0000 0000 %04x 0000' % es, align, es, {})


def _b(hexs, align, es=0, s=None):
    return ('B', hexs, align, es, s or {})


R0 = ('B', '00' * 24, 8, 0, {})
CR0 = ('B', '00' * 16, 8, 0, {})
TRUE = 0x01000000           # Bool is the first byte of the inline word

# static (layout) field types per class, in layout offset order, and layout sizes - from the
# retail schema (skaterschema.vlt class records), so the generator needs no schema at run time
_STATIC = {
    'challenge_local_data': (['Sk8::Audio::eSk8Characters'], 4),
    'fe_locations': (['EA::Reflection::Text', 'Sk8::Challenge::tLocationID'], 12),
    'worldpainter_fe_location_triggers': (['EA::Reflection::Text'], 4),
}

# challenge_local_data 'default' (identical to retail challenge_local_data_framework.vlt 'default';
# every pack writes it into each of its 7 dlc_<s4>_freeskate_* vaults)
_CLD_DEFAULT = [
    (0x3b2db72e60aaa38e, 'EA::Reflection::Bool', 0x4000, 0),
    (0xdd99f44939e70af9, 'EA::Reflection::Bool', 0x4000, TRUE),
    (0xfc19949149740049, 'EA::Reflection::Int32', 0x0200, _arr(4, 4)),
    (0x117191727d1f3c41, 'EA::Reflection::Bool', 0x4000, TRUE),
    (0x4c89d137b3d2944e, 'EA::Reflection::Bool', 0x4000, 0),
    (0xf80366582ab8737f, 'EA::Reflection::Bool', 0x4000, 0),
    (0xbeafe2539cc67a68, 'EA::Reflection::Bool', 0x4000, 0),
    (0xd35023f4c43a6421, 'EA::Reflection::Bool', 0x4000, 0),
    (0xb01eaf6b0ea7373c, 'EA::Reflection::Bool', 0x4000, 0),
    (0x832e0fef78ebb33d, 'EA::Reflection::Text', 0x4000, _t('ID_CHALLENGE_CENTRAL_MESSAGE_OBJECTIVE_RETURNTOCHAL')),
    (0xecec13cea2899d81, 'Attrib::RefSpec', 0x0a00, _arr(24, 8)),
    (0xe14decee593e36d4, 'Attrib::RefSpec', 0x0a00, _arr(24, 8)),
    (0x0abdc707fc5ddf63, 'EA::Reflection::Bool', 0x4000, 0),
    (0x137f56d7c41456f1, 'EA::Reflection::Bool', 0x4000, TRUE),
    (0xbb120dfb026bb636, 'EA::Reflection::Bool', 0x4000, 0),
    (0xd46d537ac3775c5d, 'EA::Reflection::Bool', 0x4000, 0),
    (0x44754347a934fdcd, 'Sk8::Audio::eChallengeMusicType', 0x4000, 0),
    (0x7c99551d913fcb8f, 'EA::Reflection::Text', 0x4000, _t('')),
    (0x23f394473183c613, 'EA::Reflection::Bool', 0x4000, TRUE),
    (0xb017c5144a4be44f, 'EA::Reflection::UInt8', 0x4000, 0),
    (0x1a7274067ce13f22, 'EA::Reflection::Text', 0x4000, _t('')),
    (0x533baabdfe3b8a70, 'EA::Reflection::Bool', 0x4000, TRUE),
    (0x87db489867267a56, 'EA::Reflection::Float', 0x4000, 0x3fc00000),
    (0x16e7e79b28332502, 'EA::Reflection::Bool', 0x4000, TRUE),
    (0x53688086474e81eb, 'EA::Reflection::Bool', 0x4000, 0),
    (0x133aad27f05343f8, 'EA::Reflection::Bool', 0x4000, 0),
    (0xf3a508058ae5148d, 'EA::Reflection::Bool', 0x4000, 0),
    (0xa4a69fc7384cbf93, 'EA::Reflection::Float', 0x4000, 0x40533333),
    (0xdcc1cc2cefbd33be, 'Attrib::Gen::ClassRefSpec_livingworld_categorygroups', 0x0800,
     _b('f91e58de56973cb80000000000000000', 8)),
    (0x291d6d1b65aeffd2, 'Sk8::Challenge::tObjectiveStatusField', 0x4000, 0),
    (0xe82a28be11734cfa, 'EA::Reflection::Bool', 0x4000, TRUE),
    (0x8dd0bda08407e230, 'EA::Reflection::Int32', 0x4000, 5),
    (0x7b834a07530629fa, 'EA::Reflection::Float', 0x4000, 0x40466666),
    (0xbe79e831558f2113, 'Attrib::RefSpec', 0x0800, R0),
    (0x97cd1fc9b27a1e3c, 'EA::Reflection::Bool', 0x4000, 0),
    (0xe57648e80501973e, 'Attrib::RefSpec', 0x0a00, _arr(24, 8)),
    (0x1070acf7360e007e, 'EA::Reflection::Bool', 0x4000, 0),
    (0xb0d9b6b916d223e6, 'EA::Reflection::Bool', 0x4000, TRUE),
    (0x42cf56ddea069244, 'EA::Reflection::Bool', 0x4000, 0),
    (0xcaf082b3a11f4afb, 'EA::Reflection::Float', 0x4000, 0x40400000),
    (0x59cd0b9449d19c3c, 'EA::Reflection::Text', 0x4000, _t('ID_COMMON_NONE')),
    (0x48427451aef34906, 'Attrib::RefSpec', 0x0800, R0),
    (0xf59b3a4d0f2009e6, 'EA::Reflection::Bool', 0x4000, 0),
    (0x5f3d2f4ca4031e5b, 'EA::Reflection::Bool', 0x4000, 0),
    (0x59f6a63f5900a8c9, 'EA::Reflection::Bool', 0x4000, 0),
]

ACTIVITIES = ('gap_tag', 'accumulation', 'simultrick', 'tricklist', 'survival')
# eFreeskateType per activity in the challenge_global_data collections
_ACT_TYPE = {'gap_tag': 0, 'accumulation': 0, 'simultrick': 3, 'tricklist': 0, 'survival': 0}

_E = 'EA::Reflection::'
_C = 'Sk8::Challenge::'


def _challenge_entries(name):
    """challenges collection: its name, the global-data RefSpec and the local-data vault ref."""
    return [(0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t(name)),
            (0x24491bf00316aa45, 'Attrib::RefSpec', 0x0800, _r('challenge_global_data', name)),
            (0xd7bb2b160caf8ed6, 'AttribSysUtils::tVaultedRefSpec', 0x0800, _vr('challenge_local_data', name))]


def _challengebank(sp):
    """challengebanks/dlc_<s4>.vlt: 'default' copies, the dlc_<s4> tree, then 2 per location."""
    s4 = 'dlc_{s4}'
    hl_tail = sp.fmt('dlc_{s4}\0dlc_{s4}_freeskate_acti').encode('latin1')
    out = [
        ('challenges', 'default', None, 0, [
            (0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t('default')),
            (0x8c5669e5a6864831, _E + 'UInt64', 0x0000, _b('00' * 8, 8)),
            (0x24491bf00316aa45, 'Attrib::RefSpec', 0x0800, _r('challenge_global_data', 'default')),
            (0xd7bb2b160caf8ed6, 'AttribSysUtils::tVaultedRefSpec', 0x0800, _vr('challenge_local_data', 'default'))]),
        ('challenge_objective', 'default', None, 1, [
            (0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t('default')),
            (0xe73c9cf0f75a6fcd, _C + 'tChallengePresentationEvent', 0x0200, _arr(16, 4)),
            (0xd7332063be5cc0eb, _C + 'eTriggerActivationMode', 0x4000, 0),
            (0xd404ba3319e2c389, _C + 'tChallengePresentationEvent', 0x0200, _arr(16, 4)),
            (0xb053106604767956, _E + 'Bool', 0x4000, 0),
            (0x607ec72f932997c1, _C + 'tChallengePresentationEvent', 0x0200, _arr(16, 4)),
            (0x40d40861dd9a6cc6, _C + 'tObjectiveDefinition', 0x0200, _arr(20, 4)),
            (0xb69af085f7a6d275, 'Attrib::RefSpec', 0x0a00, _arr(24, 8)),
            # bytes 0x70.. of this 0x90-byte struct are the strings that follow it in the community
            # builders' .bin (they write 0x70 bytes); kept so the game reads the same value
            (0x928f9d693922d585, _C + 'tHighlightDefinition', 0x0000,
             _b('00' * 69 + '10000000000000001000000000000000100000000000000014000000000000001800000000000000500000'
                + hl_tail.hex(), 16, s={0: ''})),
            (0xbc7d59ecbb20868d, _E + 'Bool', 0x4000, 0),
            (0xe13d9c326e4d98d5, _C + 'tObjectiveDefinition', 0x0000, _b('00' * 20, 4)),
            (0xd4965ed3bd43ec44, _C + 'tObjectiveTriggers', 0x0000, _b('00' * 20, 4)),
            (0x41cbf3f0889de6b3, 'LuaState::tCompiledLua', 0x0000, _b('00' * 12, 4)),
            (0xe0b77cde569c8c0b, _E + 'Bool', 0x4000, 0),
            (0x16f350c53880eaa9, _C + 'eVisualIndicatorMode', 0x4000, 0),
            (0x7ae8fef77a54bb91, _C + 'tVisualEditorData', 0x0a00, _arr(80, 8))]),
        ('challenges', s4, 'default', 0, [
            (0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t(s4)),
            (0x8c5669e5a6864831, _E + 'UInt64', 0x0000, _b('0000000000000800', 8)),
            (0x24491bf00316aa45, 'Attrib::RefSpec', 0x0800, _r('challenge_global_data', s4)),
            (0xd7bb2b160caf8ed6, 'AttribSysUtils::tVaultedRefSpec', 0x0800, _vr('challenge_local_data', s4))]),
        ('challenge_global_data', s4, 'default', 0, []),
        ('challenge_failure_objective', s4, 'default', 0, [(0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t(s4))]),
        ('challenge_objective', s4, 'default', 0, [(0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t(s4))]),
        ('challenge_objectives_group', s4, 'default', 0, []),
        ('challenges', s4 + '_freeskate_locations', s4, 0, [
            (0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t(s4 + '_freeskate_locations')),
            (0x24491bf00316aa45, 'Attrib::RefSpec', 0x0800, _r('challenge_global_data', 'freeskate_locations')),
            (0xd7bb2b160caf8ed6, 'AttribSysUtils::tVaultedRefSpec', 0x0800, _vr('challenge_local_data', 'freeskate_locations')),
            (0x7867a00c3a6f893c, 'AttribSysUtils::tVaultedRefSpec', 0x0800, _vr('challenge_stategraph', 'freeskate'))]),
        ('challenge_global_data', s4 + '_freeskate_locations', s4, 1, [
            (0xf9a9a1dded84a643, _E + 'Bool', 0x4000, TRUE),
            (0x02b828ccdb4785e4, _C + 'eChallengeTypes', 0x4000, 0x16),
            (0xfd40e4a31027c984, _E + 'Bool', 0x4000, 0),
            (0x6b0bd2bae508b742, _E + 'Text', 0x4000, _t('')),
            (0x797427ed230c9980, 'Sk8::FE::tFEScreenShot', 0x4000, _t('default')),
            (0x122db7b0f026dbec, _C + 'eGlobalType', 0x4000, 0),
            (0x8022ba64cb8e431d, _E + 'Float', 0x4000, 0x40400000),
            (0xbca0ade94b04b8a7, _E + 'Text', 0x4000, _t('ID_MISSION_TEMPLATE_ONLINE_FREESKATE_TITLE')),
            (0xf1c6d5fa4a23951e, _C + 'tLocationID', 0x4000, _t('')),
            (0xa6e8924715a15cec, _C + 'tLocationID', 0x4000, _t('')),
            (0xa996afe8845cacab, 'Sk8::HUD::eHUDComponent', 0x0200, _b('000100010004000000000009', 4, es=4)),
            (0x876bf0819aa58858, _C + 'tRequiredChallengeHull', 0x0200, _arr(4, 4)),
            (0xe2a50788b6c20c92, _E + 'Text', 0x4000, _t('ID_LOCATION_{SLUG}_{KEY0}'))]),
        ('challenges', s4 + '_freeskate_activities', s4, 0, _challenge_entries(s4 + '_freeskate_activities')),
        ('challenge_global_data', s4 + '_freeskate_activities', s4, 2, [
            (0xf9a9a1dded84a643, _E + 'Bool', 0x4000, TRUE),
            (0x858d73cd6085c8ec, _E + 'Bool', 0x4000, TRUE),
            (0x02b828ccdb4785e4, _C + 'eChallengeTypes', 0x4000, 0x2a),
            (0xfd40e4a31027c984, _E + 'Bool', 0x4000, 0),
            (0x6b0bd2bae508b742, _E + 'Text', 0x4000, _t('')),
            (0x198a0e276e56a9d6, _E + 'Bool', 0x4000, TRUE),
            (0x89add2c42d6d58ae, _C + 'tWorldLocationID', 0x0800, _b('00' * 32, 8)),
            (0x89b3580572f41da3, _E + 'Bool', 0x4000, 0),
            (0x85d6b2c6ae61dd8b, _C + 'eFreeskateType', 0x4000, 1),
            (0x122db7b0f026dbec, _C + 'eGlobalType', 0x4000, 0),
            (0x8022ba64cb8e431d, _E + 'Float', 0x4000, 0x40400000),
            (0xf1c6d5fa4a23951e, _C + 'tLocationID', 0x4000, _t('default')),
            (0xc95dcfb60acf8393, _E + 'Text', 0x4000, _t('')),
            (0xa6e8924715a15cec, _C + 'tLocationID', 0x4000, _t('default')),
            (0xa996afe8845cacab, 'Sk8::HUD::eHUDComponent', 0x0200, _b('000100010004000000000009', 4, es=4)),
            (0x9233a3e52805d3fc, 'Attrib::Gen::ClassRefSpec_challenges', 0x0800, CR0),
            (0x876bf0819aa58858, _C + 'tRequiredChallengeHull', 0x0200, _arr(4, 4)),
            (0x4c807b9c7f6c6f47, _E + 'Bool', 0x4000, TRUE),
            (0xe2a50788b6c20c92, _E + 'Text', 0x4000, _t(''))]),
        ('challenge_failure_objective', s4 + '_freeskate_activities', s4, 0,
         [(0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t(s4 + '_freeskate_activities'))]),
    ]
    for act in ACTIVITIES:
        nm = f'{s4}_freeskate_activities_{act}'
        par = s4 + '_freeskate_activities'
        out += [
            ('challenges', nm, par, 0, _challenge_entries(nm)),
            ('challenge_global_data', nm, par, 0, [
                # {class f788504a8d922ced, collection 'challenge_spotfound', 0}
                (0xe616acdf5a36bfac, _C + 'tChallengeIconDefinition', 0x0800,
                 _b('f788504a8d922ced%016x0000000000000000' % attrib_key('challenge_spotfound'), 8)),
                (0x85d6b2c6ae61dd8b, _C + 'eFreeskateType', 0x4000, _ACT_TYPE[act]),
                (0x50d9293b818507ec, 'Attrib::Gen::ClassRefSpec_map_category', 0x0800, _cr('{slug}dlc_maps')),
                (0xc4b345415e8537be, 'Attrib::RefSpec', 0x0a00, _arr(24, 8))]),
            ('challenge_failure_objective', nm, par, 0, [(0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t(nm))]),
        ]
    for k in sp.keys:
        nm = 'freeskate_dlc_{slug}_' + k
        out += [
            ('challenges', nm, s4 + '_freeskate_locations', 0, _challenge_entries(nm)),
            ('challenge_global_data', nm, s4 + '_freeskate_locations', 0, [
                (0x6b0bd2bae508b742, _E + 'Text', 0x4000, _t('ID_MAP_LOCATION_DESC_{SLUG}_' + k.upper())),
                (0xf1c6d5fa4a23951e, _C + 'tLocationID', 0x4000, _t('freeskate_{slug}_%s_locator' % k)),
                (0x50d9293b818507ec, 'Attrib::Gen::ClassRefSpec_map_category', 0x0800, _cr('{slug}dlc_maps')),
                (0xa6e8924715a15cec, _C + 'tLocationID', 0x4000, _t('freeskate_{slug}_%s_locator' % k)),
                (0xe2a50788b6c20c92, _E + 'Text', 0x4000, _t('ID_LOCATION_{SLUG}_' + k.upper())),
                (0x7565cd2a80e93a5e, 'Attrib::Gen::ClassRefSpec_world', 0x0800, _cr('dist_{slug}_%sdlc' % k))]),
        ]
    return out


def _framework(sp):
    """dlc_<s4>_local_data_framework.vlt: the local data of the dlc_<s4> challenge tree (copies of
    retail challenge_local_data_framework.vlt's freeskate collections; the 4 struct values the
    community builders could not write point at .bin offset 0, kept as BIN0)."""
    s4 = 'dlc_{s4}'
    act = s4 + '_freeskate_activities'
    i16 = _E + 'Int16'
    return [
        ('challenge_local_data', s4, 'default', 0, [(0x9ddc567540219acc, _E + 'Bool', 0x4000, 0)]),
        ('challenge_local_data', s4 + '_freeskate_locations', s4, 0, [(0x9ddc567540219acc, _E + 'Bool', 0x4000, 0)]),
        ('challenge_local_data', act, s4, 1, [
            (0xa11c61a038b61108, _E + 'Bool', 0x4000, TRUE),
            (0x91506d129d7148b7, _E + 'Bool', 0x4000, TRUE),
            (0x7df9623ea16b2932, _C + 'tSpeechInfo', 0x0000, BIN0),
            (0x48843e3b3e4f7cbf, _E + 'Bool', 0x4000, TRUE),
            (0x242e4a29261117c8, i16, 0x4000, 0x50000),
            (0x928f9d693922d585, _C + 'tHighlightDefinition', 0x0000, BIN0),
            (0xe9d6e1f231d1c3a8, _E + 'Text', 0x4000, _t('ID_CHALLENGE_CENTRAL_MESSAGE_OBJECTIVE_OBJCOMPLETE')),
            (0x742d630146648434, i16, 0x4000, 0),
            (0xcb837eef50958fbb, _E + 'Bool', 0x4000, TRUE),
            (0xf89bfdfc88990f17, _E + 'Float', 0x4000, 0x43340000),
            (0xc468ed9ce2ff4984, _E + 'Float', 0x4000, 0x3f800000)]),
        ('challenge_local_data', act + '_gap_tag', act, 0, [(0x242e4a29261117c8, i16, 0x4000, 0xf0000)]),
        ('challenge_local_data', act + '_accumulation', act, 0, [
            (0xa74fec89c254dc43, _C + 'eFreeskateAccumulationType', 0x4000, 0),
            (0x242e4a29261117c8, i16, 0x4000, 0x320000),
            (0xe2210cb6d497fefb, _E + 'Int32', 0x4000, 0)]),
        ('challenge_local_data', act + '_simultrick', act, 0, [
            (0x242e4a29261117c8, i16, 0x4000, 0xf0000),
            (0x476fcc173ec88c7d, _E + 'Float', 0x4000, 0x40000000),
            (0x928f9d693922d585, _C + 'tHighlightDefinition', 0x0000, BIN0)]),
        ('challenge_local_data', act + '_tricklist', act, 1, [
            (0xa11c61a038b61108, _E + 'Bool', 0x4000, TRUE),
            (0x91506d129d7148b7, _E + 'Bool', 0x4000, TRUE),
            (0x48843e3b3e4f7cbf, _E + 'Bool', 0x4000, TRUE),
            (0x242e4a29261117c8, i16, 0x4000, 0xa0000),
            (0x928f9d693922d585, _C + 'tHighlightDefinition', 0x0000, BIN0),
            (0xe9d6e1f231d1c3a8, _E + 'Text', 0x4000, _t('ID_CHALLENGE_CENTRAL_MESSAGE_OBJECTIVE_OBJCOMPLETE')),
            (0x291d6d1b65aeffd2, _C + 'tObjectiveStatusField', 0x4000, 8),
            (0xe82a28be11734cfa, _E + 'Bool', 0x4000, TRUE),
            (0x2837e2dd59106c10, _C + 'tObjectiveStatusField', 0x4000, 6),
            (0x742d630146648434, i16, 0x4000, 0),
            (0xf89bfdfc88990f17, _E + 'Float', 0x4000, 0x43960000)]),
        ('challenge_local_data', act + '_survival', act, 0, [
            (0x242e4a29261117c8, i16, 0x4000, 0xa0000),
            (0xe2210cb6d497fefb, _E + 'Int32', 0x4000, 0),
            (0x047210029098973c, _E + 'Int32', 0x4000, 0),
            (0x476fcc173ec88c7d, _E + 'Float', 0x4000, 0x40000000)]),
    ]


# the 4 RefSpecs of every 'world' collection (class 9f0d024cd713a9dc, collections main, downtown,
# 4630e94e4119a5b6, 520c455a5ccd6776), measured constant in all packs
_WORLD_REFS = ('00040004001800009f0d024cd713a9dc316a03931530b5e600000000000000009f0d024cd713a9dc'
               '647c0aa898f5bfdf00000000000000009f0d024cd713a9dc4630e94e4119a5b6000000000000000'
               '09f0d024cd713a9dc520c455a5ccd67760000000000000000')

_LUA_FE = ('do\r\n    if( item.ClassKey == Attrib.ClassName.fe_locations ) then\r\n{body}    end\r\n'
           '    return false;\r\nend;')
_LUA_FE_ITEM = ('        if( LocationFilterHelpers.IsLocationInWorld( item, "{world}" ) ) then\r\n'
                '            return true;\r\n        end\r\n')
_LUA_ONLINE = ('do\r\n\tif( item.ClassKey == Attrib.ClassName.challenges )then\r\n'
               '\t\tif( item.GlobalData.ChallengeType == Challenge.eChallengeTypes.OnlineFreeSkate )then\r\n'
               '\t\t\tif( item.GlobalData.World.IsSkatePark == false and (\r\n{body}\r\n'
               '\t\t\t) )then\r\n\t\t\t\treturn true;\r\n\t\t\tend\r\n\t\tend\r\n\tend\r\n\t\r\n'
               '\treturn false;\r\nend;')
_LUA_ONLINE_ITEM = '\t\t\t\tLocationFilterHelpers.IsLocationInWorld( item, "{world}" )'


def _lua(code):
    """LuaState::tCompiledLua {ptr -> source, ptr -> '', u32 source length} (0xf9a = 3994 bytes in
    SUMMERCAMP: the length without the terminator)."""
    return _b('00' * 8 + '%08x' % len(code), 4, s={0: code, 8: ''})


def _minimal(sp):
    """dlc_<slug>_minimal.vlt: per location its world, triggers, audio and fe location records,
    the map category / filter / listing entries, the product mapping and the summary groups."""
    worlds = ['dist_{slug}_%sdlc' % k for k in sp.keys]
    out = []
    for k, w in zip(sp.keys, worlds):
        out += [
            ('world', w, 'default_dlc', 1, [
                (0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t(w)),
                (0x99d6e51c9e20a663, 'Attrib::RefSpec', 0x0800, R0),
                (0xd076621a896e5bfa, 'Attrib::RefSpec', 0x0a00, _b(_WORLD_REFS, 8, es=24)),
                (0xb88a4034d63e905e, 'Attrib::Types::Vector2', 0x0000, _b('00' * 8, 4)),
                # {ptr -> 'map1024', 0, 0x802, -850, 1050, -1250, 512, 512}: constant in all packs
                (0x20bbc594cd2faef8, 'Sk8::FE::tMapInfo', 0x0000,
                 _b('0000000000000802fffffcae0000041afffffb1e0000020000000200', 4, s={0: 'map1024'})),
                (0x095e1d9778f0c6f4, _E + 'Text', 0x4000, _t('ID_WORLD_{SLUG}_' + k.upper())),
                (0x50d9293b818507ec, 'Attrib::Gen::ClassRefSpec_map_category', 0x0800, _cr('{slug}dlc_maps')),
                (0x457c324498c79533, _E + 'Bool', 0x4000, 0),
                (0x095d7c0ca40a34ea, _E + 'Text', 0x4000, _t('{dist}')),
                (0xc978fe3dbaccbb4a, _E + 'Text', 0x4000, _t('{s4}')),
                (0xc7a0a84f018e87ba, _E + 'Text', 0x4000, _t('world/models/DIST_skybox_downtown')),
                (0x2a0bf629355afb15, _E + 'Text', 0x4000, _t('world/models/DIST_skybox_downtown_Textures')),
                (0x295c85d2b2997ed3, _E + 'Text', 0x4000, _t('world{dist}.big')),
                (0x2a655e8a29c1f642, _E + 'Text', 0x4000, _t('{dist}'))]),
            ('worldpainter_fe_location_triggers', w, 'default_dlc', 0, [], _b('00' * 4, 4)),
            ('aud_worlddata', w, 'default', 0, [
                (0x65fa976ef23a314e, _E + 'Text', 0x0200, _arr(4, 4)),
                (0x6e7e3bd211d61f25, 'Attrib::RefSpec', 0x0800, R0)]),
            ('aud_bigfiles', w, 'default', 0, [
                (0x16cec9e052a8b9ba, _E + 'Text', 0x4000, _t('data/audio/english/')),
                (0x12dfb2bdd9356640, _E + 'Text', 0x4000, _t('')),
                (0x1179ae4eb7022f99, _E + 'Text', 0x4000, _t('')),
                (0xcbdf09a8387553e2, _E + 'Text', 0x4000, _t('')),
                (0xdab4f65f7d523075, _E + 'Bool', 0x4000, 0),
                (0x83a22dd9978c45dc, _E + 'Bool', 0x4000, 0),
                (0x3c336b8972f26c0f, _E + 'Int32', 0x4000, 0)]),
        ]
    out += [
        ('map_category', '{slug}dlc', 'world', 0, [
            (0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t('{slug}dlc')),
            (0x465698dd3cc73ca2, _E + 'Text', 0x4000, _t('ID_MAP_CATEGORY_{SLUG}')),
            (0x1349bb6ec1d1e751, _E + 'Int32', 0x4000, 0x4e20)]),
        ('map_category', '{slug}dlc_maps', '{slug}dlc', 0, [
            (0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t('{slug}dlc_maps')),
            (0x465698dd3cc73ca2, _E + 'Text', 0x4000, _t('ID_MAP_SECTION_{SLUG}_MAPS')),
            (0x1349bb6ec1d1e751, _E + 'Int32', 0x4000, 0x4e84)]),
    ]
    for k, w in zip(sp.keys, worlds):
        lay = _b('00' * 12, 4, s={0: w, 4: 'Z_{slug}_%s_Start' % k, 8: 'ID_LOCATION_{SLUG}_' + k.upper()})
        out.append(('fe_locations', w, 'default_dlc', 0, [
            (0x6b0bd2bae508b742, _E + 'Text', 0x4000, _t('ID_MAP_LOCATION_DESC_{SLUG}_' + k.upper())),
            (0x786a2d24f475342b, _E + 'Text', 0x4000, _t('')),
            (0x50d9293b818507ec, 'Attrib::Gen::ClassRefSpec_map_category', 0x0800, _cr('{slug}dlc_maps')),
            (0x90e678b6961730e1, _E + 'Bool', 0x4000, TRUE),
            (0x7565cd2a80e93a5e, 'Attrib::Gen::ClassRefSpec_world', 0x0800, _cr(w))], lay))
    wn = [sp.fmt(w) for w in worlds]
    fe = _LUA_FE.replace('{body}', ''.join(_LUA_FE_ITEM.replace('{world}', w) for w in wn))
    onl = _LUA_ONLINE.replace('{body}', ' or\r\n'.join(_LUA_ONLINE_ITEM.replace('{world}', w) for w in wn))
    out += [
        ('map_filter', '{slug}dlc', 'default_dlc', 0, [
            (0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t('{slug}dlc')),
            (0x5fbb2c3f59fc6f1a, _E + 'Text', 0x4000, _t('ID_MAP_HELPER_{SLUG}')),
            (0x5604268225257009, 'LuaState::tCompiledLua', 0x0000, ('LUA', fe)),
            (0x6766efd84d74f4f0, _E + 'Text', 0x4000, _t('ID_MAP_CATEGORY_{SLUG}'))]),
        ('map_filter', 'freeskate_locations_{slug}dlc', 'freeskate_locations', 0, [
            (0x942ab8aee4b414ed, _E + 'Text', 0x4000, _t('freeskate_locations_{slug}dlc')),
            (0x5fbb2c3f59fc6f1a, _E + 'Text', 0x4000, _t('ID_MAP_HELPER_{SLUG}')),
            (0x5604268225257009, 'LuaState::tCompiledLua', 0x0000, ('LUA', onl)),
            (0x6766efd84d74f4f0, _E + 'Text', 0x4000, _t('ID_MAP_CATEGORY_{SLUG}'))]),
        ('map_listing', 'progression_locations_dlc_{slug}', 'progression_locations', 0, [
            (0xe0762ea3f90c9eeb, 'Attrib::RefSpec', 0x0800, _r('map_filter', '{slug}dlc')),
            (0x982bd1eae2334504, _E + 'Int16', 0x4000, 0x70000)]),
        ('dlc_mapping', 'pack_{slug}', 'default', 0, [
            (0x481d3fe849c4d988, _E + 'UInt16', 0x0200, _b('0001000100020000000b', 4, es=2)),
            (0xb080be5f9f596243, 'Sk8::BE::tUnlockableItemInfo', 0x0200, _arr(8, 4)),
            (0xad4e064704b8f1a9, _E + 'Text', 0x4000, _t('{product}')),
            (0xc90d657030918fdb, 'Sk8::BE::tUnlockableItemInfo', 0x0200, _arr(8, 4)),
            (0x2bdfe21a12094b1d, 'Sk8::BE::tUnlockableItemInfo', 0x0200, _arr(8, 4))]),
        ('progression_summary_group', 'dlc_{slug}', 'default', 0, []),
    ]
    for k in sp.keys:
        out.append(('progression_summary_group', '{slug}_%s_dlc' % k, 'dlc_{slug}', 0, [
            (0x4acbee23f4d5a7b5, 'Attrib::Gen::ClassRefSpec_progression_group', 0x0a00, _arr(16, 8)),
            (0x692f97ff95563bb4, _E + 'Int16', 0x4000, 0xe0000),
            (0x786255b8ff426a3d, 'Attrib::Gen::ClassRefSpec_fe_map_icons', 0x0800, _cr('teleport')),
            (0xe2a50788b6c20c92, _E + 'Text', 0x4000, _t('ID_DLC_{SLUG}_%s_SUMMARY_TITLE' % k.upper()))]))
    out.append(('map_listing', 'online_freeskate_{slug}dlc', 'online_freeskate', 0, [
        (0xe0762ea3f90c9eeb, 'Attrib::RefSpec', 0x0800, _r('map_filter', 'freeskate_locations_{slug}dlc')),
        (0x982bd1eae2334504, _E + 'Int16', 0x4000, 0xb0000)]))
    return out


# the milestone reward's collection name is not known (it is the only name of the 14 pack vaults
# not recovered; 'dlc_<s4>_milestone_01' style guesses did not hash to it), so new packs use ours
MILESTONE_NAME = 'dlc_{s4}_milestone_01'


def _progressionbank(sp, milestone=None):
    s4 = 'dlc_{s4}'
    T_ = _E + 'Text'
    return [
        ('progression_state', 'default', None, 0, [
            (0x9c374581090b6c85, _C + 'tChallengePresentationEvent', 0x0200, _arr(16, 4))]),
        ('progression_handler', 'default', None, 0, [(0xc3cd4b4183274819, 'Attrib::Blob', 0x0000, _b('000000fa00000000', 4))]),
        ('progression_rewards', 'default', None, 1, [
            (0x3dbc2f9c0083df86, 'Sk8::BE::eUNLOCK_CATEGORY', 0x4000, 0xc),
            (0x8915e3fcb76529ba, T_, 0x4000, _t('none')),
            (0xec2ed691df1c9f4c, T_, 0x4000, _t('#Description HAL ID')),
            (0xaa753a536859233d, T_, 0x4000, _t('#MISSING HALID')),
            (0x786255b8ff426a3d, T_, 0x4000, _t('missingtexture')),
            (0xf950c13d02f83230, 'Sk8::Audio::eSk8Characters', 0x4000, 0x68),
            (0xcbd4a095bed89d19, _E + 'Int32', 0x4000, 0)]),
        ('progression_stategraph', 'default', None, 0, [(0x942ab8aee4b414ed, T_, 0x4000, _t('default'))]),
        ('progression_action', 'default', None, 0, [
            (0x942ab8aee4b414ed, T_, 0x4000, _t('default')),
            (0xec7a860a5187b15d, _E + 'Bool', 0x4000, 0)]),
        ('progression_group', 'default', None, 0, [(0x942ab8aee4b414ed, T_, 0x4000, _t('default'))]),
        ('progression_state', s4, 'default', 0, []),
        ('progression_handler', s4, 'default', 0, []),
        ('progression_rewards', s4, 'default', 0, []),
        ('progression_stategraph', s4, 'default', 0, [(0x942ab8aee4b414ed, T_, 0x4000, _t(s4))]),
        ('progression_action', s4, 'default', 0, [(0x942ab8aee4b414ed, T_, 0x4000, _t(s4))]),
        ('progression_group', s4, 'default', 0, [(0x942ab8aee4b414ed, T_, 0x4000, _t(s4))]),
        ('progression_rewards', s4 + '_challenges', s4, 0, []),
        ('progression_rewards', s4 + '_ots_unlock', s4 + '_challenges', 0, [
            (0xaa753a536859233d, T_, 0x4000, _t('ID_DLC_{S4}_OTS_UNLOCK')),
            (0x786255b8ff426a3d, T_, 0x4000, _t('fe\\source\\images\\unlocks\\unlock_challenge_spot')),
            (0xe373a52fb7d81920, _E + 'Bool', 0x4000, TRUE)]),
        ('progression_rewards', s4 + '_milestones', s4, 0, []),
        ('progression_rewards', milestone or MILESTONE_NAME, s4 + '_milestones', 0, [
            (0x3dbc2f9c0083df86, 'Sk8::BE::eUNLOCK_CATEGORY', 0x4000, 0xc),
            (0x8915e3fcb76529ba, T_, 0x4000, _t('milestone_placeholder')),
            (0xec2ed691df1c9f4c, T_, 0x4000, _t('ID_DLC_{S4}_MILESTONE_1_DESC')),
            (0xaa753a536859233d, T_, 0x4000, _t('ID_DLC_{S4}_MILESTONE_1_TITLE')),
            (0x786255b8ff426a3d, T_, 0x4000, _t('fe\\source\\images\\unlocks\\unlock_milestone1')),
            (0xe373a52fb7d81920, _E + 'Bool', 0x4000, 0)]),
    ]


def _value(sp, v):
    if isinstance(v, int) or v is BIN0:
        return v
    kind = v[0]
    if kind == 'T':
        return Text(sp.fmt(v[1]))
    if kind == 'R':
        return Blob(struct.pack('>QQ8x', attrib_key(v[1]), attrib_key(sp.fmt(v[2]))), 8)
    if kind == 'VR':
        return Blob(struct.pack('>QQ16x', attrib_key(v[1]), attrib_key(sp.fmt(v[2]))), 8, {24: Text('')})
    if kind == 'CR':
        return Blob(struct.pack('>Q8x', attrib_key(sp.fmt(v[1]))), 8)
    if kind == 'LUA':
        return _value(sp, _lua(v[1]))
    _k, hexs, align, es, strs = v
    return Blob(bytes.fromhex(hexs.replace(' ', '')), align, {o: Text(sp.fmt(t) if '{' in t else t)
                                                           for o, t in strs.items()}, es)


def _types_of(cls, entries):
    """EA's type list: the class's static-field types (layout order), then the entry types in
    first-use order, padded with a zero to an even count (14224/14319 retail collections; the rest
    are retail collections the DLC recipe does not use)."""
    t = []
    for name in _STATIC.get(cls, ([], 0))[0] + [e[1] for e in entries]:
        k = attrib_key(name)
        if k not in t:
            t.append(k)
    return t, t + [0] * (len(t) % 2)


def _collection(sp, spec):
    cls, name, parent, w1c, entries = spec[:5]
    layout = spec[5] if len(spec) > 5 else None
    if layout is None and cls in _STATIC:
        layout = _b('00' * _STATIC[cls][1], 4)
    nm = sp.fmt(name)
    used, types = _types_of(cls, entries)
    c = Collection(attrib_key(nm), attrib_key(cls), attrib_key(sp.fmt(parent)) if parent else 0,
                   [Entry(k, attrib_key(t), fl, _value(sp, v)) for k, t, fl, v in entries],
                   types, len(used), len(entries), w1c, _value(sp, layout) if layout else None,
                   eid=attrib_key(f'{cls}/{nm}'))
    return c


def dlc_vaults(slug: str, keys, dist: str, version: int = VERSION_PACK, milestone: str | None = None) -> dict:
    """Every data/db/ file of a single-world DLC pack: {path under data/db/: bytes}.
    slug: pack slug; keys: location keys (keys[0] = main location); dist: DIST name."""
    sp = _Spec(slug, keys, dist)
    s4 = sp.s4
    out = {}

    def vault(path, specs):
        stem = path.rsplit('/', 1)[-1]
        db = Database(stem + '.vlt', stem + '.bin', [_collection(sp, x) for x in specs], version)
        v, b = encode(db)
        out[path + '.vlt'], out[path + '.bin'], out[path + '.opt'] = v, b, OPT

    for nm in [f'dlc_{s4}_freeskate_activities'] + [f'dlc_{s4}_freeskate_activities_{a}' for a in ACTIVITIES] \
            + [f'dlc_{s4}_freeskate_locations']:
        vault('challenge_local_data/' + nm, [('challenge_local_data', 'default', None, 2, _CLD_DEFAULT,
                                               _b('00000028', 4))])
    for k in keys:
        nm = f'freeskate_dlc_{slug}_{k}'
        vault('challenge_local_data/' + nm, [('challenge_local_data', nm, 'dlc_{s4}_freeskate_locations', 0, [])])
    vault(f'challengebanks/dlc_{s4}', _challengebank(sp))
    vault(f'progressionbanks/dlc_{s4}', _progressionbank(sp, milestone))
    vault(f'dlc_{s4}_local_data_framework', _framework(sp))
    vault(f'dlc_{slug}_minimal', _minimal(sp))
    out[f'dlc_{slug}_minimal.vaultlist.xml'] = vault_list(
        [f'dlc_{slug}_minimal.vlt', f'dlc_{s4}_local_data_framework.vlt',
         f'challengebanks\\dlc_{s4}.vlt', f'progressionbanks\\dlc_{s4}.vlt'])
    return out


def vault_list(files) -> bytes:
    """`<slug>.vaultlist.xml`: the vaults the game loads directly (CRLF, 2-space indent; 6/6
    packs). The challenge_local_data vaults are found through tVaultedRefSpecs instead."""
    s = ['<?xml version="1.0" encoding="utf-8"?>', '<VaultList>']
    s += [f'  <VaultFile>{f}</VaultFile>' for f in files]
    s += ['</VaultList>', '']
    return '\r\n'.join(s).encode('ascii')


# ================================================================================ 5. comparison
def _norm_value(v):
    if isinstance(v, Blob):
        d = bytearray(v.data)
        for o in v.strings:         # stored slot words: 0 (EA) or the target offset (packs)
            d[o:o + 4] = bytes(4)
        return ('B', bytes(d), tuple(sorted((o, t.s) for o, t in v.strings.items())), v.es)
    if isinstance(v, Text):
        return ('T', v.s)
    return v


def normalized(db: Database) -> list:
    """What the game reads from a vault, independent of the writer's layout: per collection its
    keys, counts, used types (EA pads the stored list to an even count, pack builders do not) and
    every value with pointers resolved."""
    out = []
    for c in db.items:
        if not isinstance(c, Collection):
            out.append(c)
            continue
        out.append(dict(key=c.key, cls=c.cls, parent=c.parent, eid=c.eid, w1c=c.w1c, ntypes=c.ntypes,
                        capacity=c.capacity, types=[t for t in c.types][:c.ntypes],
                        layout=_norm_value(c.layout) if c.layout else None,
                        entries=[(e.key, e.type, e.flags, _norm_value(e.value)) for e in c.entries]))
    return out


def semantic_diff(a: tuple, b: tuple, sch: Schema) -> list:
    """[(collection key, field, a, b)] between two (vlt, bin) pairs; [] = same content."""
    na, nb = normalized(decode(*a, sch)), normalized(decode(*b, sch))
    if len(na) != len(nb):
        return [(None, 'collections', len(na), len(nb))]
    out = []
    for x, y in zip(na, nb):
        for k in x:
            if x[k] != y[k]:
                out.append((x['key'], k, x[k], y[k]))
    return out
