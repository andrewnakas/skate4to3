"""sk3/vault.py: AttribSys vaults (.vlt/.bin/.opt).

- file codec: every vault of retail db.big and of every DLC pack re-writes byte for byte;
- schema: retail skaterschema types and class layouts;
- database: decode -> encode reproduces every retail vault without nested data pointers byte for
  byte (EA's layout rules); the rest are refused, never silently mis-read;
- generator: the vaults of SUMMERCAMP, ISLEOFGROM and SUMMERCAMPHD regenerate from their names alone
  with the same content (the one collection whose name is unknown aside), for any location count.

`python tests/test_vault.py` prints the counts.
"""
import collections, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sk3 import big, vault as V
from sk3.attrib_hash import attrib_key
from test_sfil_roundtrip import dlc_packs

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')
DB = os.path.join(DATA, 'big', 'db.big')
PACKS = dlc_packs(1 << 40)
need_db = pytest.mark.skipif(not os.path.exists(DB), reason='no retail db.big')

# the three single-world packs whose vaults the generator reproduces: (slug, keys, DIST)
GENERATED = {
    'summercamp': (['summercamp'] + [f'spot{i}' for i in range(1, 9)], 'DIST_SummerCamp'),
    'isleofgrom': (['isleofgrom'] + [f'spot{i}' for i in range(1, 8)] + ['rail8'], 'DIST_IsleOfGrom'),
    'summercamphd': (['summercamp'] + [f'spot{i}' for i in range(1, 9)], 'DIST_SummerCampHD'),
}


def db_files(path):
    """{path under data/db/: bytes} of an archive."""
    a = big.read(path)
    return {e.path[len('data/db/'):]: a.read(e) for e in a.entries if e.path.startswith('data/db/')}


_SCHEMA = []


def schema():
    if not _SCHEMA:
        _SCHEMA.append(V.load_schema(DB))
    return _SCHEMA[0]


def sources():
    out = [DB] if os.path.exists(DB) else []
    return out + PACKS


def file_codec(files):
    res = collections.Counter()
    for p, d in files.items():
        if p.endswith('.vlt'):
            ok = V.write_vlt(V.read_vlt(d)) == d
        elif p.endswith('.bin'):
            ok = V.write_bin(V.read_bin(d)) == d
        elif p.endswith('.opt'):
            ok = d == V.OPT
        else:
            continue
        res[(p.rsplit('.', 1)[1], ok)] += 1
    return res


@pytest.mark.parametrize('path', sources(), ids=os.path.basename)
def test_file_codec_exact(path):
    res = file_codec(db_files(path))
    assert sum(res.values()) and not [k for k in res if not k[1]], res


@need_db
def test_schema():
    s = schema()
    assert len(s.types) == 711 and len(s.classes) == 329
    c = s.classes[attrib_key('challenge_local_data')]
    assert (c.layout_size, c.nstatic, len(c.fields)) == (4, 1, 314)
    assert s.types[attrib_key('Attrib::RefSpec')] == ('Attrib::RefSpec', 24)
    # every class is named by a ClassRefSpec_<name> type (329/329)
    named = {attrib_key(n[len('Attrib::Gen::ClassRefSpec_'):]) for n, _ in s.types.values()
             if n.startswith('Attrib::Gen::ClassRefSpec_')} | {attrib_key('challenge_local_data')}
    assert set(s.classes) <= named


def reencode(files, sch):
    res = collections.Counter()
    bad = []
    for p, d in files.items():
        if not p.endswith('.vlt') or p.endswith('skaterschema.vlt'):
            continue
        b = files[p[:-4] + '.bin']
        try:
            db = V.decode(d, b, sch)
        except ValueError as e:
            res['nested data pointers (not modelled)' if 'nested' in str(e) else 'decode error'] += 1
            continue
        if db.form == 'ea':
            v2, b2 = V.encode(db)
            ok = (v2, b2) == (d, b)
            res['re-encodes byte for byte' if ok else 'differs'] += 1
            if not ok:
                bad.append(p)
        else:
            res['decoded (pack layout)'] += 1
    return res, bad


@need_db
def test_retail_reencode_exact():
    res, bad = reencode(db_files(DB), schema())
    assert not bad and res['re-encodes byte for byte'] == 474 and not res['decode error'], (res, bad[:4])
    assert res['nested data pointers (not modelled)'] == 34


@need_db
@pytest.mark.parametrize('path', PACKS, ids=os.path.basename)
def test_pack_vaults_decode(path):
    res, _bad = reencode(db_files(path), schema())
    assert res['decoded (pack layout)'] and not res['decode error'] and not res['differs'], res


def pack_named(slug):
    return next((p for p in PACKS if os.path.basename(p).lower() == f'{slug}_00000000.big'), None)


def generated_vs_pack(slug):
    keys, dist = GENERATED[slug]
    gen = V.dlc_vaults(slug, keys, dist)
    orig = db_files(pack_named(slug))
    res = collections.Counter()
    diffs = {}
    for p in sorted(set(gen) | set(orig)):
        if p not in gen or p not in orig:
            res['missing' if p not in gen else 'extra'] += 1
            continue
        if gen[p] == orig[p]:
            res['bytes equal'] += 1
        elif p.endswith('.vlt'):
            d = V.semantic_diff((orig[p], orig[p[:-4] + '.bin']), (gen[p], gen[p[:-4] + '.bin']), schema())
            if d:
                diffs[p] = d
            res['same content' if not d else 'differs'] += 1
    return res, diffs


@need_db
@pytest.mark.parametrize('slug', sorted(GENERATED))
def test_generated_vaults_match_pack(slug):
    """Every one of the 61 db/ files of the pack, generated from (slug, keys, DIST) alone: .opt and
    vault list byte-equal, the 20 vaults with the same decoded content. Only difference: the
    milestone reward collection under dlc_<s4>_milestones, whose name is not known (its key and
    export id differ; the content is equal)."""
    if not pack_named(slug):
        pytest.skip('pack not on this machine')
    res, diffs = generated_vs_pack(slug)
    assert res['bytes equal'] == 21 and res['same content'] + res['differs'] == 20, res
    pb = f'progressionbanks/dlc_{slug[:4]}.vlt'
    assert list(diffs) == [pb], list(diffs)
    assert len({x[0] for x in diffs[pb]}) == 1 and sorted(x[1] for x in diffs[pb]) == ['eid', 'key']


@pytest.mark.parametrize('n', [1, 3, 12])
def test_generator_any_location_count(n):
    keys = ['mymap'] + [f'spot{i}' for i in range(1, n)]
    files = V.dlc_vaults('mymap', keys, 'DIST_MyMap')
    assert len(files) == 3 * (11 + n) + 1
    for k in keys:
        assert f'challenge_local_data/freeskate_dlc_mymap_{k}.vlt' in files
    # our own output is EA layout: it decodes and re-encodes to the same bytes
    if os.path.exists(DB):
        for p, d in files.items():
            if p.endswith('.vlt'):
                db = V.decode(d, files[p[:-4] + '.bin'], schema())
                assert V.encode(db) == (d, files[p[:-4] + '.bin']), p
                for c in db.items:
                    assert c.capacity == len(c.entries)
    v = V.read_vlt(files['dlc_mymap_minimal.vlt'])
    assert v.deps == [(attrib_key('dlc_mymap_minimal.vlt'), 'dlc_mymap_minimal.vlt'),
                      (attrib_key('dlc_mymap_minimal.bin'), 'dlc_mymap_minimal.bin')]
    assert (attrib_key('world/dist_mymap_mymapdlc'), V.EXP_COLLECTION) == v.exports[0][:2]


def test_vault_list_format():
    assert V.vault_list(['a.vlt', 'b\\c.vlt']) == (
        b'<?xml version="1.0" encoding="utf-8"?>\r\n<VaultList>\r\n  <VaultFile>a.vlt</VaultFile>\r\n'
        b'  <VaultFile>b\\c.vlt</VaultFile>\r\n</VaultList>\r\n')


if __name__ == '__main__':
    tot = collections.Counter()
    for p in sources():
        tot.update(file_codec(db_files(p)))
    print('FILE CODEC', sorted(tot.items()))
    if os.path.exists(DB):
        res, bad = reencode(db_files(DB), schema())
        print('RETAIL', dict(res), bad[:3])
        tot = collections.Counter()
        for p in PACKS:
            tot.update(reencode(db_files(p), schema())[0])
        print('PACKS', dict(tot))
        for slug in sorted(GENERATED):
            if pack_named(slug):
                res, diffs = generated_vs_pack(slug)
                print('GENERATED', slug, dict(res), {k: [(hex(x[0]), x[1]) for x in v] for k, v in diffs.items()})
