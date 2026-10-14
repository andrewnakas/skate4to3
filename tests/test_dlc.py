"""sk3/dlc.py: every DLC metadata format re-writes byte for byte over every pack on this machine,
the layout rules regenerate the packs' files, and a pack rebuilt from its own inputs matches the
original file by file (except where the spec documents a deliberate difference).

`python tests/test_dlc.py` prints per-format and per-category counts.
"""
import collections, os, re, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sk3 import big, dlc, manifests as mf, rw4
from test_sfil_roundtrip import dlc_packs

PACKS = dlc_packs(1 << 40)
SMALL = [p for p in PACKS if os.path.getsize(p) < 600 << 20]


def pack_files(path, cats):
    a = big.read(path)
    with open(path, 'rb') as f:
        for e in a.entries:
            c = dlc.category(e.path)
            if c in cats:
                f.seek(e.offset)
                yield e.path, c, f.read(e.size)


def codec_check(path):
    res = collections.Counter()
    bad = []
    cats = {'boundary', 'neighbour table', 'location preload list', 'language strings',
            'language histogram', 'product unlock', 'locator arena', 'vault', 'vault data',
            'vault options', 'locator table', 'mission manifest (empty)'}
    for p, c, d in pack_files(path, cats):
        if c == 'boundary':
            ok = dlc.write_boundary(dlc.read_boundary(d)) == d
        elif c in ('neighbour table', 'location preload list'):
            ok = dlc.write_stream_tiles(dlc.read_stream_tiles(d)) == d
        elif c == 'language strings':
            ok = dlc.write_language(dlc.read_language(d)) == d
        elif c == 'language histogram':
            ok = dlc.write_histogram() == d
        elif c == 'product unlock':
            m = re.search(r'unlocks/(.+)0000_product', p)
            ok = d.rstrip(b'\n') == dlc.product_unlock(m.group(1)).rstrip(b'\n')
        elif c == 'locator arena':
            o = rw4.parse(d).objects[1].data
            ok = dlc.write_locator_object(dlc.parse_locator_object(o)) == o
        elif c in ('vault', 'vault data', 'vault options'):
            ok = dlc.join_chunks(dlc.chunks(d)) == d
        elif c == 'locator table':
            ok = mf.write_loc(mf.read_loc(d)) == d
        else:
            ext = p[-4:]
            rd = {'.pmm': mf.read_mmap, '.psm': mf.read_cmap, '.pss': mf.read_cspa, '.pst': mf.read_atoc}[ext]
            wr = {'.pmm': mf.write_mmap, '.psm': mf.write_cmap, '.pss': mf.write_cspa, '.pst': mf.write_atoc}[ext]
            ok = wr(rd(d)) == d
        res[(c, ok)] += 1
        if not ok:
            bad.append(p)
    return res, bad


@pytest.mark.parametrize('path', PACKS, ids=os.path.basename)
def test_codecs_exact(path):
    res, bad = codec_check(path)
    assert sum(res.values()) and not bad, f'{len(bad)} differ, e.g. {bad[:4]}'


def test_string_id_hash():
    assert dlc.string_id('ID_MAP_CATEGORY_SUMMERCAMP') == 0xDA60AB7F
    assert dlc.string_id('dist_summercamp_spot1dlc') == 0xEDD88055


DB = os.path.join(os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data'),
                  'big', 'db.big')
GENERATED = ('summercamp', 'isleofgrom', 'summercamphd')    # the single-world packs on this machine


def pack_named(slug):
    return next((p for p in SMALL if os.path.basename(p).lower() == f'{slug}_00000000.big'), None)


def regenerate(path, template=False):
    """A pack rebuilt from its own exported inputs (tiles, spots, display name, CSPA content boxes,
    group / tail words, internal names): template-free, or with the pack itself as the template."""
    pn = dlc.names_of_pack(path)
    names = dlc.read_template(path) if template else pn
    display, tiles, spots, cell, group, tail, content = dlc.inputs_of_pack(path, pn)
    return dlc.build_files(display, tiles, spots, names, cell=cell, center_rule='community', group=group,
                           tail=tail, content=content, dlc_tables=False)


def compare(path, files, sch=None):
    """Per file: 'bytes' (identical) or what differs and why, by category. Vault files compare by
    decoded content (our writer uses EA's layout, the packs a community builder's), locator arenas
    by object (EA arena layout vs the packs' builder's; template-free arenas carry their own arena
    and export ids). Returns (counts, unexplained differences, {path: pack bytes} for the files
    whose content is the same, to compare the whole archive)."""
    from sk3 import vault as V
    a = big.read(path)
    orig = {e.path: a.read(e) for e in a.entries}
    res, bad, swap = collections.Counter(), collections.defaultdict(list), {}
    for p in sorted(set(orig) | set(files)):
        c = dlc.category(p)
        if p not in files or p not in orig:
            res[(c, 'missing' if p not in files else 'extra')] += 1
            bad[c].append(p)
            continue
        o, g = orig[p], files[p]
        if o == g:
            res[(c, 'bytes')] += 1
        elif p.startswith('data/db/') and p.endswith('.bin'):
            continue                                        # judged with its .vlt
        elif p.startswith('data/db/') and p.endswith('.vlt') and sch is not None:
            d = V.semantic_diff((o, orig[p[:-4] + '.bin']), (g, files[p[:-4] + '.bin']), sch)
            if not d:
                v = 'same content'
            elif len({x[0] for x in d}) == 1 and sorted(x[1] for x in d) == ['eid', 'key']:
                v = 'same content, milestone collection name ours'
            else:
                v = None
            if v:
                res[(c, v)] += 1
                swap[p], swap[p[:-4] + '.bin'] = o, orig[p[:-4] + '.bin']
            else:
                res[(c, 'differs')] += 1
                bad[c].append((p, d[:2]))
        elif p.endswith('.rx2'):
            ao, ag = rw4.parse(o).objects, rw4.parse(g).objects
            if [x.data for x in ao] == [x.data for x in ag]:
                v = 'same objects'
            elif (len(ao) == len(ag) == 3 and ao[0].data == ag[0].data and ao[1].data == ag[1].data and
                  ao[2].data[:0x34] + ao[2].data[0x3C:] == ag[2].data[:0x34] + ag[2].data[0x3C:]):
                v = 'same objects, export id ours'
            else:
                v = None
            res[(c, v or 'differs')] += 1
            if v:
                swap[p] = o
            else:
                bad[c].append(p)
        else:
            res[(c, 'differs')] += 1
            bad[c].append(p)
    return res, dict(bad), swap


def whole_pack(path, files, swap):
    """The archive written from `files` with the pack's own bytes swapped in where the content is
    the same: equal to the shipped .big byte for byte?"""
    from test_big import Compare
    files = dict(files, **swap)
    sink = Compare(path)
    big.write(sink, [big.Source(*k.rsplit('/', 1), data=v) for k, v in files.items()], big.FLAGS_DLC)
    return sink.done(), sink.first_diff


def schema():
    from sk3 import vault as V
    return V.load_schema(DB) if os.path.exists(DB) else None


@pytest.mark.parametrize('slug', GENERATED)
def test_template_free_rebuild(slug):
    """The pack from its own inputs with no template: every file present; vaults with the same
    decoded content (one collection name of ours), locator arenas with the same objects (our export
    id), everything else byte for byte; with the pack's bytes swapped in for those, the whole .big
    is the shipped one byte for byte."""
    p = pack_named(slug)
    if not p or not os.path.exists(DB):
        pytest.skip('pack or retail db.big not on this machine')
    files = regenerate(p)
    res, bad, swap = compare(p, files, schema())
    assert not bad, {c: v[:3] for c, v in bad.items()}
    assert res[('vault', 'same content, milestone collection name ours')] == 1
    assert res[('locator arena', 'same objects, export id ours')] == 18
    ok, at = whole_pack(p, files, swap)
    assert ok, f'first difference at {at}'


def test_whole_pack_regenerates_byte_exact_with_template():
    """SummerCamp from its inputs with itself as the template (its vault files and locator
    arenas): the whole .big comes out identical to the shipped one (EA-layout locator arenas
    swapped for the pack's own where the objects are equal)."""
    p = pack_named('summercamp')
    if not p:
        pytest.skip('no SummerCamp pack')
    files = regenerate(p, template=True)
    res, bad, swap = compare(p, files)
    assert not bad, {c: v[:3] for c, v in bad.items()}
    ok, at = whole_pack(p, files, swap)
    assert ok, f'first difference at {at}'


def test_locator_arenas_from_first_principles():
    """Every pack locator arena (190): built from its decoded locator and its two own ids alone,
    it equals the template-based build byte for byte, and its file id is attrib_key(name)."""
    import struct
    from sk3 import rw4_write
    from sk3.attrib_hash import attrib_key
    if not PACKS:
        pytest.skip('no packs')
    n = 0
    for path in PACKS:
        a = big.read(path)
        for e in a.entries:
            if e.path.startswith('data/content/global_locators/') and e.path.endswith('.rx2'):
                d = a.read(e)
                ar = rw4.parse(d)
                L = dlc.parse_locator_object(ar.objects[1].data)
                aid = rw4_write.spec_from_arena(ar).arena_id
                exp = struct.unpack_from('>Q', ar.objects[2].data, 0x34)[0]
                assert dlc.build_locator_arena(L, aid, exp) == dlc.locator_arena(d, L), e.path
                assert ar.objects[0].data == dlc.LOCATOR_HEAD
                assert ar.objects[2].data == dlc.locator_table(L.locator_id, exp)
                assert e.name == dlc.locator_file_stem(e.root.rsplit('/', 1)[1], L.name) + '.rx2'
                assert L.locator_id == attrib_key(L.name)
                n += 1
    assert n >= 190


def test_default_names():
    spots = [dlc.Spot('Main', (0, 0, 0))] + [dlc.Spot(f'S{i}', (0, 0, 0)) for i in range(1, 4)]
    n = dlc.names_for('Summer Camp', spots)
    assert (n.slug, n.dist, n.keys) == ('summercamp', 'DIST_SummerCamp', ['summercamp', 'spot1', 'spot2', 'spot3'])
    spots[2].key = 'rail8'
    assert dlc.names_for('Isle of Grom', spots, slug='isleofgrom').keys[2] == 'rail8'


def test_template_free_build_any_location_count():
    """A map with 12 locations and no tiles: per location a locator pair, boundary, preload list,
    12 mission manifests and a challenge_local_data vault; nothing taken from a pack."""
    arena = dlc.build_locator_arena(dlc.Locator('x', tuple(float(i % 5 == 0) for i in range(16)), 1))
    tiles = {'Pres': [], 'Tex': [], 'Sim': [(f'cSim_{x}_{z}_high', [(0x1000 + x + z, arena)])
                                          for x in (-50, 50) for z in (-50, 50)]}
    spots = [dlc.Spot(f'Spot {i}', (10.0 * i, 0.0, 5.0)) for i in range(12)]
    files = dlc.build_files('My Map', tiles, spots, dlc.names_for('My Map', spots))
    cats = collections.Counter(dlc.category(p) for p in files)
    assert cats['locator arena'] == cats['locator table'] == 24
    assert cats['boundary'] == cats['location preload list'] == 12
    assert cats['vault'] == cats['vault data'] == cats['vault options'] == 11 + 12
    assert cats['mission manifest (empty)'] == 12 * 12


if __name__ == '__main__':
    tot = collections.Counter()
    for p in PACKS:
        r, bad = codec_check(p)
        tot.update(r)
        print(os.path.basename(p), sum(v for (c, ok), v in r.items() if ok), '/', sum(r.values()), bad[:2])
    print('CODECS', sorted(tot.items()))
    sch = schema()
    for slug in GENERATED:
        p = pack_named(slug)
        if not p:
            continue
        files = regenerate(p)
        res, bad, swap = compare(p, files, sch)
        print('TEMPLATE-FREE', slug, sorted(res.items()))
        print('   whole .big (same-content files swapped):', whole_pack(p, files, swap), 'unexplained:', bad)
