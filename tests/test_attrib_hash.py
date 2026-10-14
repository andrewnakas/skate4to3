"""sk3.attrib_hash against retail keys: DepN name/hash pairs of every vault in db.big,
the challenge_local_data class key, and DLC locator file ids (= hash of the locator name)."""
import os, re, struct, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sk3 import big
from sk3.attrib_hash import attrib_key, hash_bytes

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')
DB = os.path.join(DATA, 'big', 'db.big')


def _entries(a):
    ents = a.entries if hasattr(a, 'entries') else list(a.by_name().values())
    return [(e.path() if callable(getattr(e, 'path', None)) else e.path, e) for e in ents]


def depn_pairs(vlt: bytes):
    o = 0
    while o + 8 <= len(vlt):
        tag, size = struct.unpack_from('>4sI', vlt, o)
        if size < 8:
            return
        if tag == b'DepN':
            c = vlt[o + 8:o + size]
            n = struct.unpack_from('>Q', c, 0)[0]
            hs = struct.unpack_from('>%dQ' % n, c, 8)
            offs = struct.unpack_from('>%dI' % n, c, 8 + 8 * n)
            base = 8 + 12 * n
            for h, so in zip(hs, offs):
                yield c[base + so:c.index(b'\0', base + so)].decode('latin1'), h
        o += size


def test_known_keys():
    assert attrib_key('challenge_local_data') == 0x46A56ADDF284C381
    assert attrib_key('main.vlt') == 0x5B0069F35EEA45C9
    assert attrib_key('Z_summercamp_spot8_Start') == 0xED172AC1DC8BC492
    # all tail lengths and block boundaries are exercised by the retail pairs below
    assert hash_bytes(b'') == hash_bytes(b'')


@pytest.mark.skipif(not os.path.exists(DB), reason='no retail Skate 3 data')
def test_retail_depn_pairs():
    a = big.read(DB)
    pairs = {}
    for p, e in _entries(a):
        if p.endswith('.vlt'):
            pairs.update(depn_pairs(a.read(e)))
    bad = [n for n, h in pairs.items() if attrib_key(n) != h]
    assert len(pairs) >= 1000 and not bad, (len(pairs), bad[:5])
    lens = {len(n) for n in pairs}
    assert min(lens) < 8 and max(lens) > 24  # short tails and multi-block names


def _dlc_packs():
    try:
        from test_sfil_roundtrip import dlc_packs
        return [p for p in dlc_packs(1 << 40) if os.path.getsize(p) < 600 << 20]
    except Exception:
        return []


@pytest.mark.skipif(not _dlc_packs(), reason='no DLC packs')
def test_locator_ids_are_name_hashes():
    total = 0
    for pack in _dlc_packs():
        a = big.read(pack)
        ents = _entries(a)
        names = {}
        for p, e in ents:
            if p.endswith(('.rx2', '.loc')):
                for m in re.finditer(rb'[\x20-\x7e]{4,}', a.read(e)):
                    s = m.group().decode()
                    names[attrib_key(s)] = s
        ids = [int(m.group(1), 16) for p, _ in ents
               for m in [re.search(r'\[0x([0-9a-f]{16})\]', p)] if m]
        missing = [hex(i) for i in ids if i not in names]
        assert not missing, (os.path.basename(pack), missing[:3])
        total += len(ids)
    assert total > 0
