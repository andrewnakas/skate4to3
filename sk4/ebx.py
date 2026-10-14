"""Generic reader for skate.'s RIFF EBX partitions, driven by their own RFL2 reflection.

Container: RIFF / 'EBX\\0' or 'EBXS' with chunks EBXD (instance data), EFIX (fixups),
EBXX (unused here) and RFL2 (reflection). EBXD offsets are relative to the chunk
payload rounded up to 16.

RFL2:  u32 n, n x {guid16, u32 sig}       type GUIDs (same order as types)
       u32 n, n x {u32 nameHash, u32 fieldStart, u16 fieldCount, u16 flags, u16 size, u16 align}
       u32 n, n x {u32 nameHash, u32 offset, u16 flags, u16 typeRef}   fields
       u32 n, n x {u32 nameHash, u32 valueCount, u32 valueStart}        enums
       u32 n, n x {u32 nameHash, u32 value}                             enum values
       u32 n, n x {u32 nameHash, u32 stringOffset}                      names (sorted)
       strings
Field kind is (flags >> 5) & 0x1F (Frostbite EbxFieldType); flag 0x08 marks an array.

EFIX:  guid partition; typeGuids; typeSigs; u32 exported; dataContainer offsets;
       pointer offsets; resourceRef offsets; imports {guid partition, guid instance};
       import offsets; typeInfo offsets; u32 arrayOffset, boxedOffset, stringOffset.

Each data container starts with a 24-byte header whose first u64 indexes EFIX's
typeGuids. Pointer, CString and array fields are 8 bytes holding a SIGNED int32
self-relative offset in the low half: arrays inside the arrays region routinely
point backwards (0x00000000FFFFF840 is -1984, not 4 GiB). An odd value is an
import, (v >> 1) = index. Arrays point at their first element; u32 count before.
"""
from __future__ import annotations

import struct

K_INHERITED, K_STRUCT, K_POINTER, K_CSTRING, K_ENUM = 0, 2, 3, 7, 8
K_GUID, K_SHA1, K_RESREF = 21, 22, 23
ARRAY = 0x08
PRIM = {10: ('<B', 1), 11: ('<b', 1), 12: ('<B', 1), 13: ('<h', 2), 14: ('<H', 2),
        15: ('<i', 4), 16: ('<I', 4), 17: ('<q', 8), 18: ('<Q', 8), 19: ('<f', 4), 20: ('<d', 8)}
SCALAR_SIZE = {K_POINTER: 8, K_CSTRING: 8, K_ENUM: 4, K_GUID: 16, K_SHA1: 20, K_RESREF: 8}


class Import(tuple):
    __slots__ = ()

    @property
    def partition(self):
        return self[0]

    @property
    def instance(self):
        return self[1]


class Ref(int):
    """Pointer to an instance inside this partition."""


class Partition:
    def __init__(self, data: bytes):
        self.d = data
        # 'EBX\0' and 'EBXS' forms share the chunk layout; the root sublevel, root
        # layer and ECS prefabs are EBXS.
        if data[:4] != b'RIFF' or data[8:12] not in (b'EBX\0', b'EBXS'):
            raise ValueError('not a RIFF EBX partition')
        self.form = data[8:12]
        self.chunks, p = {}, 12
        while p + 8 <= len(data):
            tag, size = data[p:p + 4].decode('latin1'), struct.unpack_from('<I', data, p + 4)[0]
            self.chunks[tag] = (p + 8, size)
            p += 8 + size
            p += p & 1
        self.base = (self.chunks['EBXD'][0] + 15) & ~15
        self._rfl2(*self.chunks['RFL2'])
        self._efix(*self.chunks['EFIX'])
        by_guid = {g: i for i, g in enumerate(self.type_guids)}
        self.instance_type_map = [by_guid.get(g) for g in self.efix_type_guids]
        self.instances = []
        for off in self.data_containers:
            ti = struct.unpack_from('<Q', data, self.base + off)[0]
            self.instances.append((off, self.instance_type_map[ti]))
        self.addr = {self.base + off: i for i, (off, _) in enumerate(self.instances)}

    # -- tables ---------------------------------------------------------------
    def _rfl2(self, start, size):
        r, q = self.d, start
        u = lambda: struct.unpack_from('<I', r, q)[0]
        n = u(); q += 4
        self.type_guids = [r[q + 20 * i:q + 20 * i + 16].hex() for i in range(n)]; q += 20 * n
        n = u(); q += 4
        self.types = [struct.unpack_from('<IIHHHH', r, q + 16 * i) for i in range(n)]; q += 16 * n
        n = u(); q += 4
        self.fields = [struct.unpack_from('<IIHH', r, q + 12 * i) for i in range(n)]; q += 12 * n
        n = u(); q += 4
        enums = [struct.unpack_from('<III', r, q + 12 * i) for i in range(n)]; q += 12 * n
        n = u(); q += 4
        values = [struct.unpack_from('<II', r, q + 8 * i) for i in range(n)]; q += 8 * n
        n = u(); q += 4
        sbase = q + 8 * n
        self.names = {}
        for i in range(n):
            h, o = struct.unpack_from('<II', r, q + 8 * i)
            self.names[h] = r[sbase + o:r.index(b'\0', sbase + o)].decode('utf-8', 'replace')
        self.enums = {h: {v: self.names.get(vh, str(v)) for vh, v in values[s:s + c]} for h, c, s in enums}
        self.type_index = {self.names.get(t[0]): i for i, t in enumerate(self.types)}

    def _efix(self, start, size):
        e, q = self.d, start
        u = lambda o: struct.unpack_from('<I', e, o)[0]
        self.partition_guid = e[q:q + 16].hex(); q += 16
        n = u(q); q += 4
        self.efix_type_guids = [e[q + 16 * i:q + 16 * i + 16].hex() for i in range(n)]; q += 16 * n
        n = u(q); q += 4 + 4 * n
        self.exported = u(q); q += 4
        tabs = []
        for _ in range(3):
            n = u(q); q += 4
            tabs.append(struct.unpack_from(f'<{n}I', e, q)); q += 4 * n
        self.data_containers, self.pointer_offsets, self.resource_ref_offsets = tabs
        n = u(q); q += 4
        self.imports = [Import((e[q + 32 * i:q + 32 * i + 16].hex(), e[q + 32 * i + 16:q + 32 * i + 32].hex()))
                        for i in range(n)]; q += 32 * n
        n = u(q); q += 4
        self.import_offsets = struct.unpack_from(f'<{n}I', e, q); q += 4 * n
        n = u(q); q += 4 + 4 * n
        self.array_offset, self.boxed_offset, self.string_offset = struct.unpack_from('<3I', e, q)

    # -- reading --------------------------------------------------------------
    def type_name(self, t: int) -> str:
        return self.names.get(self.types[t][0], f'type{t}')

    def instances_of(self, name: str):
        t = self.type_index.get(name)
        return [i for i, (_, ti) in enumerate(self.instances) if ti == t]

    def read(self, i: int) -> dict:
        off, t = self.instances[i]
        out = {'$type': self.type_name(t)}
        self._read_type(t, self.base + off, out)
        return out

    def _read_type(self, t, addr, out):
        nh, fstart, fcount, _, _, _ = self.types[t]
        for fh, foff, flags, ref in self.fields[fstart:fstart + fcount]:
            kind = (flags >> 5) & 0x1F
            if kind == K_INHERITED:
                if ref != 0xFFFF:
                    self._read_type(ref, addr, out)
                continue
            out[self.names.get(fh, f'field_{fh:08x}')] = self._field(kind, flags & ARRAY, ref, addr + foff)

    def _field(self, kind, is_array, ref, a):
        if not is_array:
            return self._scalar(kind, ref, a)
        v = struct.unpack_from('<i', self.d, a)[0]
        if v == 0:
            return []
        data = a + v
        count = struct.unpack_from('<I', self.d, data - 4)[0]
        size = self.types[ref][4] if kind == K_STRUCT else PRIM.get(kind, (None, SCALAR_SIZE.get(kind)))[1]
        if not size:
            return f'<array of kind {kind}>'
        return [self._scalar(kind, ref, data + k * size) for k in range(count)]

    def _scalar(self, kind, ref, a):
        d = self.d
        if kind in PRIM:
            fmt = PRIM[kind][0]
            v = struct.unpack_from(fmt, d, a)[0]
            return bool(v) if kind == 10 else v
        if kind == K_STRUCT:
            o = {}
            self._read_type(ref, a, o)
            return o
        if kind == K_POINTER:
            v = struct.unpack_from('<i', d, a)[0]
            if v == 0:
                return None
            if v & 1:
                return self.imports[struct.unpack_from('<I', d, a)[0] >> 1]
            return Ref(self.addr.get(a + v, -1))
        if kind == K_CSTRING:
            v = struct.unpack_from('<i', d, a)[0]
            if v == 0:
                return ''
            s = a + v
            return d[s:d.index(b'\0', s)].decode('utf-8', 'replace')
        if kind == K_ENUM:
            v = struct.unpack_from('<I', d, a)[0]
            return self.enums.get(self.types[ref][0], {}).get(v, v) if ref != 0xFFFF else v
        if kind == K_GUID:
            return d[a:a + 16].hex()
        if kind == K_SHA1:
            return d[a:a + 20].hex()
        if kind == K_RESREF:
            return struct.unpack_from('<Q', d, a)[0]
        return f'<kind {kind}>'
