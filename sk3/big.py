"""EA BIG archives ("EB\\0\\3"), the container Skate 3 ships worlds and DLC packs in.
Reader and byte-exact writer; facts and how they were measured: specs/sk3_packaging.md "BIG".

Layout (big-endian):

    0x00 'EB' 00 03        0x04 u32 entry count n    0x08 u32 flags (0x00100400 world, 0x00500400 DLC)
    0x0C u32 name table offset                       0x10 u32 name table length
    0x14 u32 name layout = nameSlot << 24 | rootSlot << 16 | rootCount
    0x18 0   0x1C u32 total size   0x20 u32 path-directory length (0 in world archives)   0x24-0x2F 0
    0x30 records, 16 B each, ascending by id: u32 offset >> shift, u32 stored size (0 = stored
         as-is), u32 size, u32 id; shift = (flags >> 8) & 0xFF (4 in world/DLC archives, 6 in the
         0x00100600 archives, whose stored entries are RefPack or 'chunkref' containers of
         0x20000-byte RefPack chunks); id = djb2("<root>/<name>")
    then align16(n) bytes (zero in world and DLC archives); the name table starts right after.
    name table: u8 0, then n slots of nameSlot bytes IN RECORD ORDER, each [u8 root index][name,
         NUL-padded], nameSlot = longest name + 3; at align16(n * nameSlot) (which can overlap the
         last slot's final pad byte) the roots, rootSlot = longest root + 1 bytes each, in the
         order they are first used by the data; the table is zero-padded to 16.
    path directory (DLC only) directly after the name table, already a multiple of 16.
    data from there on, each entry 16-aligned, zero padding, the file padded to 16.
         DLC packs store the data in path-directory order (a pre-order walk: per directory its
         files, then its subdirectories, each sorted by upper-cased name); EA's world archives
         use an order of their own, which the writer takes as an input.

Path directory: see build_directory().
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

MAGIC = b'EB\x00\x03'
RECORDS = 0x30
NUL = b'\x00'
FLAGS_WORLD = 0x00100400
FLAGS_DLC = 0x00500400


def djb2(text: str) -> int:
    h = 5381
    for b in text.encode('utf-8'):
        h = (h * 33 + b) & 0xFFFFFFFF
    return h


def _a16(x: int) -> int:
    return (x + 15) & ~15


@dataclass
class Entry:
    root: str
    name: str
    offset: int
    size: int
    id: int
    stored: int = 0       # record word 1: compressed size on disk, 0 when stored as-is

    @property
    def path(self) -> str:
        return f'{self.root}/{self.name}'


@dataclass
class Archive:
    path: str
    flags: int
    name_layout: int
    roots: list
    entries: list = field(default_factory=list)
    directory: bytes = b''
    name_table_offset: int = 0
    stride: int = 0
    entry_table: bytes = b''

    def read(self, e: Entry) -> bytes:
        """The entry's bytes, decoded when the archive stores it compressed."""
        with open(self.path, 'rb') as f:
            f.seek(e.offset)
            b = f.read(e.stored or e.size)
        return unpack_stored(b) if e.stored else b

    def read_raw(self, e: Entry) -> bytes:
        with open(self.path, 'rb') as f:
            f.seek(e.offset)
            return f.read(e.stored or e.size)

    def by_name(self) -> dict:
        return {e.name: e for e in self.entries}

    def data_order(self) -> list:
        return [e.path for e in sorted(self.entries, key=lambda e: (e.offset, e.size))]


def read(path: str) -> Archive:
    with open(path, 'rb') as f:
        h = f.read(RECORDS)
        if h[:4] != MAGIC:
            raise ValueError(f'{path}: not an EB v3 archive')
        count, flags, nt, nl, layout = struct.unpack_from('>IIIII', h, 4)
        dlen = struct.unpack_from('>I', h, 0x20)[0]
        f.seek(0)
        head = f.read(max(RECORDS + 16 * count, nt + nl + dlen))
    recs = [struct.unpack_from('>IIII', head, RECORDS + 16 * i) for i in range(count)]
    table = head[nt:nt + nl]
    nroots, rslot = layout & 0xFFFF, (layout >> 16) & 0xFF

    def roots_at(rstart):
        return [table[rstart + i * rslot:rstart + (i + 1) * rslot].split(NUL, 1)[0]
                .decode('ascii', 'replace') for i in range(nroots)]

    roots = roots_at(_a16((layout >> 24) * count))
    if not roots or not all(r.startswith('data/') for r in roots):
        roots = roots_at((1 + (layout >> 24) * count + 31) & ~31)
    if not roots or not all(r.startswith('data/') for r in roots):    # fall back to scanning
        roots = []
        for raw in table.split(NUL):
            if raw.startswith(b'data/'):
                r = raw.decode('ascii')
                if r not in roots:
                    roots.append(r)

    def slot_at(k, stride):
        """(root, name) for slot k. The root index is one byte: with more than 256 roots it wraps,
        so the root is the one among index, index+256, ... whose path hashes to record k's id."""
        s = table[1 + k * stride:1 + (k + 1) * stride]
        if len(s) < 2 or s[0] >= len(roots):
            return None
        name = s[1:].split(NUL, 1)[0].decode('ascii', 'replace')
        if not name or name.startswith('data/'):
            return None
        cands = roots[s[0]::256]
        if len(cands) > 1 and k < len(recs):
            for r in cands:
                if djb2(f'{r}/{name}') == recs[k][3]:
                    return r, name
        return cands[0], name

    def hits(st):
        n = 0
        for k in range(len(recs)):
            r = slot_at(k, st)
            if r and djb2(f'{r[0]}/{r[1]}') == recs[k][3]:
                n += 1
        return n

    stride = layout >> 24
    if hits(stride) != len(recs):
        best = -1
        for st in range(4, 321):
            n = hits(st)
            if n > best:
                best, stride = n, st
            if n == len(recs):
                break
    a = Archive(path, flags, layout, roots, directory=head[nt + nl:nt + nl + dlen],
                name_table_offset=nt, stride=stride,
                entry_table=head[RECORDS + 16 * count:nt])
    shift = (flags >> 8) & 0xFF
    for k, (w0, stored, size, rid) in enumerate(recs):
        r = slot_at(k, stride)
        if not r or djb2(f'{r[0]}/{r[1]}') != rid:
            r = ('', f'~unnamed_{rid:08X}')          # audio archives carry no names at all
        a.entries.append(Entry(r[0], r[1], w0 << shift, size, rid, stored))
    return a


def unpack_stored(b: bytes) -> bytes:
    """Decode a compressed entry: one RefPack stream, or a 'chunkref' container
    (+0x0C total, +0x10 chunk size, +0x14 chunk count, +0x18 alignment; chunks from 0x30, each
    preceded by u32 stored size, u32 type (2 = RefPack), aligned)."""
    from . import refpack
    if b[:8] != b'chunkref':
        return refpack.decompress(b) if refpack.looks_like(b) else b
    total, _csz, n, al = struct.unpack_from('>IIII', b, 0xC)
    pos, out = 0x30, bytearray()
    for _ in range(n):
        c, t = struct.unpack_from('>II', b, pos - 8)
        d = b[pos:pos + c]
        pos = (pos + c + 8 + al - 1) // al * al
        out += refpack.decompress(d) if t == 2 else d
    if len(out) != total:
        raise ValueError(f'chunkref produced {len(out)} bytes, header said {total}')
    return bytes(out)


# ---- path directory --------------------------------------------------------------------------
def plan_directory(file_paths) -> list:
    """[(path, is_file)] in the order DLC packs list their tree (and store their data): a pre-order
    walk from the top directory, per directory its files then its subdirectories, each group
    sorted by the upper-cased path. The top directory ('data') gets no node of its own."""
    files = set(file_paths)
    dirs = set()
    for f in files:
        parts = f.split('/')
        for k in range(1, len(parts)):
            dirs.add('/'.join(parts[:k]))
    kids: dict = {}
    for x in list(dirs) + list(files):
        if '/' in x:
            kids.setdefault(x.rsplit('/', 1)[0], []).append(x)
    top = min(dirs, key=len)
    out = []

    def walk(d):
        ks = kids.get(d, [])
        for k in sorted((x for x in ks if x in files), key=str.upper):
            out.append((k, True))
        for k in sorted((x for x in ks if x not in files), key=str.upper):
            out.append((k, False))
            walk(k)
    walk(top)
    return out


def build_directory(nodes, flavour: int = 1) -> bytes:
    """The DLC path directory for nodes [(path, is_file)] (plan_directory order). u32 big-endian:

        0x00 36 + 20*N + 4 (offset of the 0xFFFFFFFF word)   0x10 flavour (1 in every pack measured)
        0x24 N nodes of 20 bytes, node id = index + 2 (id 1 = the top directory):
             +0 first child of its parent: the parent's child count; else the count of non-first
                nodes before it          +4 first child: index of the parent's run in the id list;
                else 0xFFFFFFFF          +8 name offset in the pool   +C 1 file / 0 directory
             +10 parent id
        then u32 count of non-first nodes, 0xFFFFFFFF, u32 id-list bytes, the id list (1, then per
        parent id ascending its children's ids descending), u32 pool bytes, the pool (top name,
        then each leaf name at first use, NUL-terminated), zero-padded to 16."""
    idof = {p: i + 2 for i, (p, _f) in enumerate(nodes)}
    parent, children = [], {}
    for i, (p, _f) in enumerate(nodes):
        pid = idof.get(p.rsplit('/', 1)[0], 1)
        parent.append(pid)
        children.setdefault(pid, []).append(i + 2)
    ids, run = [1], {}
    for d in sorted(children):
        run[d] = len(ids)
        ids += sorted(children[d], reverse=True)
    pool, at = bytearray(), {}

    def name_off(s):
        if s not in at:
            at[s] = len(pool)
            pool.extend(s.encode('ascii') + NUL)
        return at[s]
    name_off(nodes[0][0].split('/')[0] if nodes else 'data')
    body = bytearray(struct.pack('>9I', 36 + 20 * len(nodes) + 4, 0, 0, 0, flavour, 0, 0, 0, 0))
    nonfirst = 0
    for i, (p, is_file) in enumerate(nodes):
        sib = children[parent[i]]
        first = sib[0] == i + 2
        body += struct.pack('>5I', len(sib) if first else nonfirst,
                            run[parent[i]] if first else 0xFFFFFFFF,
                            name_off(p.rsplit('/', 1)[1]), 1 if is_file else 0, parent[i])
        nonfirst += not first
    body += struct.pack('>III', nonfirst, 0xFFFFFFFF, 4 * len(ids)) + struct.pack(f'>{len(ids)}I', *ids)
    body += struct.pack('>I', len(pool)) + pool
    body += bytes(-len(body) % 16)
    return bytes(body)


def read_directory(d: bytes):
    """(flavour, [(path, is_file)]) from a path directory."""
    end = struct.unpack_from('>I', d, 0)[0]
    flavour = struct.unpack_from('>I', d, 16)[0]
    n = (end - 40) // 20
    nodes = [struct.unpack_from('>5I', d, 36 + 20 * i) for i in range(n)]
    p = 36 + 20 * n
    idlen = struct.unpack_from('>I', d, p + 8)[0]
    q = p + 12 + idlen
    plen = struct.unpack_from('>I', d, q)[0]
    pool = d[q + 4:q + 4 + plen]

    def nm(o):
        return pool[o:pool.index(NUL, o)].decode('ascii')
    out = []
    for c0, c1, no, isf, par in nodes:
        parts, q2 = [nm(no)], par
        while q2 >= 2:
            parts.append(nm(nodes[q2 - 2][2]))
            q2 = nodes[q2 - 2][4]
        parts.append(nm(0))
        out.append(('/'.join(reversed(parts)), bool(isf)))
    return flavour, out


# ---- writer ----------------------------------------------------------------------------------
@dataclass
class Source:
    """One file to store: bytes, or a (path, offset, size) slice of a file on disk."""
    root: str
    name: str
    data: bytes | None = None
    file: str | None = None
    offset: int = 0
    size: int | None = None

    @property
    def path(self) -> str:
        return f'{self.root}/{self.name}'

    @property
    def length(self) -> int:
        if self.data is not None:
            return len(self.data)
        return self.size if self.size is not None else os.path.getsize(self.file) - self.offset

    def chunks(self, block: int = 1 << 24):
        if self.data is not None:
            yield self.data
            return
        left = self.length
        with open(self.file, 'rb') as f:
            f.seek(self.offset)
            while left:
                b = f.read(min(block, left))
                if not b:
                    raise IOError(f'{self.file}: short read')
                left -= len(b)
                yield b


def sources_of(a: Archive) -> list:
    """Every entry of an archive as a Source reading from the archive file."""
    if any(e.stored for e in a.entries):
        raise ValueError('compressed entries: the writer stores entries as-is only')
    return [Source(e.root, e.name, file=a.path, offset=e.offset, size=e.size) for e in a.entries]


def write(out, sources, flags: int = FLAGS_DLC, data_order=None, directory=None,
          flavour: int = 1, entry_table: bytes | None = None) -> int:
    """Write an archive to the binary file object `out`; returns its size.

    data_order: list of paths giving the data order (default: path-directory order).
    directory:  None = generate the path directory (DLC), b'' = none (world archives),
                bytes = use as given."""
    srcs = {s.path: s for s in sources}
    if len(srcs) != len(sources):
        raise ValueError('duplicate paths')
    if directory is None:
        nodes = plan_directory(list(srcs))
        directory = build_directory(nodes, flavour)
        order = data_order or [p for p, f in nodes if f]
    else:
        order = data_order or [p for p, f in plan_directory(list(srcs)) if f]
    if sorted(order) != sorted(srcs):
        raise ValueError('data order does not list every entry exactly once')
    ids = {p: djb2(p) for p in srcs}
    if len(set(ids.values())) != len(ids):
        raise ValueError('two paths share a djb2 id')
    recs = sorted(srcs, key=ids.get)
    roots = list(dict.fromkeys(srcs[p].root for p in order))
    rindex = {r: i for i, r in enumerate(roots)}
    n = len(recs)
    slot = max(len(srcs[p].name) for p in recs) + 3 if n else 0
    rslot = max(len(r) for r in roots) + 1 if roots else 0
    if slot > 255 or rslot > 255 or len(roots) > 0xFFFF:
        raise ValueError('name or root too long for the header layout word')
    rstart = _a16(n * slot)
    table = bytearray(max(1 + n * slot, rstart + rslot * len(roots)))
    for k, p in enumerate(recs):
        s = srcs[p]
        b = 1 + k * slot
        table[b] = rindex[s.root] & 0xFF
        nm = s.name.encode('ascii')
        table[b + 1:b + 1 + len(nm)] = nm
    for i, r in enumerate(roots):
        rb = r.encode('ascii')
        table[rstart + i * rslot:rstart + (i + 1) * rslot] = rb + bytes(rslot - len(rb))
    table += bytes(-len(table) % 16)
    et = entry_table if entry_table is not None else bytes(_a16(n))
    nt = RECORDS + 16 * n + len(et)
    shift = (flags >> 8) & 0xFF
    gran = max(16, 1 << shift)
    up = lambda x: (x + gran - 1) // gran * gran
    pos = up(nt + len(table) + len(directory))
    offs = {}
    for p in order:
        offs[p] = pos
        pos = up(pos + srcs[p].length)
    pos = _a16(pos)
    total = pos
    if total >= 1 << 32:
        raise ValueError('archive past 4 GiB')
    head = bytearray(MAGIC + struct.pack('>IIIIIIII', n, flags, nt, len(table),
                                         slot << 24 | rslot << 16 | len(roots), 0, total,
                                         len(directory)) + bytes(12))
    for p in recs:
        head += struct.pack('>IIII', offs[p] >> shift, 0, srcs[p].length, ids[p])
    head += et + table + directory
    head += bytes(_a16(len(head)) - len(head))
    out.write(head)
    at = len(head)
    for p in order:
        if offs[p] != at:
            out.write(bytes(offs[p] - at))
            at = offs[p]
        for c in srcs[p].chunks():
            out.write(c)
            at += len(c)
    out.write(bytes(total - at))
    return total


def rebuild(a: Archive, out) -> int:
    """Re-write an archive from its own entries, layout and directory."""
    return write(out, sources_of(a), a.flags, a.data_order(), a.directory,
                 entry_table=a.entry_table)
