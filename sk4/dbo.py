"""Frostbite DbObject parser (skate. / Frostbite 2025 TOC format).

File layout observed in skate. (Steam app 3354750):
    0x000  u32  magic 0x01CED100
    0x004  u32  0
    0x008  256  signature block
    0x108  LEB128-prefixed ASCII hex signature string
    ...    zero padding
    0x22C  root DbObject

Entry encoding:
    u8 type        -- bit 0x80 set => entry is unnamed (array element / root)
    cstr name      -- only when 0x80 is clear
    <value>        -- per type
"""
import struct

# Types confirmed empirically against layout.toc / level TOCs.
T_EOF, T_ARRAY, T_OBJECT = 0x00, 0x01, 0x02
T_BOOL, T_STRING, T_INT, T_LONG = 0x06, 0x07, 0x08, 0x09
T_FLOAT, T_DOUBLE = 0x0B, 0x0C
T_GUID, T_SHA1, T_BLOB = 0x0F, 0x10, 0x13


class Reader:
    def __init__(self, data, pos=0):
        self.d = data
        self.p = pos

    def u8(self):
        v = self.d[self.p]
        self.p += 1
        return v

    def raw(self, n):
        v = self.d[self.p:self.p + n]
        self.p += n
        return v

    def leb128(self):
        result = shift = 0
        while True:
            b = self.u8()
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7

    def cstr(self):
        end = self.d.index(b'\x00', self.p)
        v = self.d[self.p:end].decode('utf-8', 'replace')
        self.p = end + 1
        return v

    def sized_str(self):
        n = self.leb128()
        s = self.d[self.p:self.p + n]
        self.p += n
        return s.rstrip(b'\x00').decode('utf-8', 'replace')


def read_entry(r):
    """Returns (name, value) or (None, None) at an object/array terminator."""
    t = r.u8()
    if t == T_EOF:
        return None, None
    name = None if (t & 0x80) else r.cstr()
    return name, read_value(r, t & 0x7F)


def read_value(r, t):
    if t == T_OBJECT:
        size = r.leb128()
        end = r.p + size
        obj = {}
        while r.p < end:
            n, v = read_entry(r)
            if n is None and v is None:
                break
            obj[n] = v
        r.p = end
        return obj
    if t == T_ARRAY:
        size = r.leb128()
        end = r.p + size
        arr = []
        while r.p < end:
            n, v = read_entry(r)
            if n is None and v is None:
                break
            arr.append(v)
        r.p = end
        return arr
    if t == T_STRING:
        return r.sized_str()
    if t == T_BOOL:
        return bool(r.u8())
    if t == T_INT:
        return struct.unpack('<i', r.raw(4))[0]
    if t == T_LONG:
        return struct.unpack('<q', r.raw(8))[0]
    if t == T_FLOAT:
        return struct.unpack('<f', r.raw(4))[0]
    if t == T_DOUBLE:
        return struct.unpack('<d', r.raw(8))[0]
    if t == T_GUID:
        return r.raw(16).hex()
    if t == T_SHA1:
        return r.raw(20).hex()
    if t == T_BLOB:
        return r.raw(r.leb128())
    raise ValueError(f'unknown DbObject type 0x{t:02X} at offset 0x{r.p - 1:X}')


def find_root(data):
    """Locate the root DbObject.

    The header (magic + signature block + 'x'-delimited ASCII hex signature +
    zero padding) varies in length between TOCs, so rather than hard-coding an
    offset we scan for a container whose LEB128 size lands exactly on EOF.
    """
    if data[:4] != b'\x00\xd1\xce\x01':
        return 0
    for off in range(0x100, min(len(data), 0x1000)):
        if data[off] not in (0x81, 0x82):
            continue
        try:
            r = Reader(data, off + 1)
            size = r.leb128()
        except (IndexError, ValueError):
            continue
        if r.p + size == len(data):
            return off
    raise ValueError('could not locate root DbObject')


def load(path):
    data = open(path, 'rb').read()
    r = Reader(data, find_root(data))
    t = r.u8()
    return read_value(r, t & 0x7F)
