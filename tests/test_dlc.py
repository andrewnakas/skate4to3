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


def rebuild(path):
    tpl = dlc.read_template(path)
    display, tiles, spots, cell, group, tail, _content = dlc.inputs_of_pack(path, tpl)
    files = dlc.build_files(display, tiles, spots, tpl, cell=cell, center_rule='community',
                            group=group, tail=tail)
    a = big.read(path)
    orig = {e.path: e for e in a.entries}
    res = collections.Counter()
    diff = collections.defaultdict(list)
    for p, e in orig.items():
        c = dlc.category(p)
        if p not in files:
            res[(c, 'missing')] += 1
            diff[c].append(p)
            continue
        d = a.read(e)
        if c == 'locator arena':
            same = [o.data for o in rw4.parse(files[p]).objects] == [o.data for o in rw4.parse(d).objects]
        elif c == 'DIST manifest' and p.endswith('.xss'):
            same = [x.asset_id for x in mf.read_cspa(files[p]).cells] == [x.asset_id for x in mf.read_cspa(d).cells]
        else:
            same = files[p] == d
        res[(c, 'same' if same else 'differs')] += 1
        if not same:
            diff[c].append(p)
    for p in files:
        if p not in orig:
            res[(dlc.category(p), 'extra')] += 1
    return res, diff


@pytest.mark.skipif(not SMALL, reason='no DLC packs')
def test_rebuild_matches_pack():
    """SummerCamp rebuilt from its own inputs: every file present and equal, except the
    documented differences (EA-layout locator arenas compare by object; CSPA cells by id; the
    added dist_*dlc tables)."""
    p = next(p for p in SMALL if os.path.basename(p).startswith('summercamp_'))
    res, diff = rebuild(p)
    differs = {c: v for c, v in diff.items()}
    assert not differs, {c: v[:3] for c, v in differs.items()}


def test_whole_pack_regenerates_byte_exact():
    """SummerCamp from its inputs (tiles, spots, display name, CSPA content boxes, group words)
    plus its vaults and locator ids: the whole .big comes out identical to the shipped one."""
    from test_big import Compare
    p = next((p for p in SMALL if os.path.basename(p).startswith('summercamp_')), None)
    if not p:
        pytest.skip('no SummerCamp pack')
    ok, at = whole_pack(p)
    assert ok, f'first difference at {at}'


def whole_pack(p):
    from test_big import Compare
    tpl = dlc.read_template(p)
    display, tiles, spots, cell, group, tail, content = dlc.inputs_of_pack(p, tpl)
    files = dlc.build_files(display, tiles, spots, tpl, cell=cell, center_rule='community',
                            group=group, tail=tail, content=content, dlc_tables=False)
    # our locator arenas use EA's arena layout; the packs use another builder's (dictionary +4 =
    # ptr, dictionary 16-aligned, longer subref tail). Where the decoded objects are equal, put
    # the pack's own bytes back so the rest of the archive is compared byte for byte.
    a = big.read(p)
    orig = {e.path: e for e in a.entries}
    swapped = 0
    for k in [k for k in files if k.endswith('.rx2')]:
        o = a.read(orig[k])
        if [x.data for x in rw4.parse(o).objects] == [x.data for x in rw4.parse(files[k]).objects]:
            files[k] = o
            swapped += 1
    sink = Compare(p)
    big.write(sink, [big.Source(*k.rsplit('/', 1), data=v) for k, v in files.items()], big.FLAGS_DLC)
    return sink.done(), sink.first_diff


if __name__ == '__main__':
    tot = collections.Counter()
    for p in PACKS:
        r, bad = codec_check(p)
        tot.update(r)
        print(os.path.basename(p), sum(v for (c, ok), v in r.items() if ok), '/', sum(r.values()), bad[:2])
    print('CODECS', sorted(tot.items()))
    for p in SMALL:
        try:
            r, diff = rebuild(p)
        except ValueError as ex:
            print('REBUILD', os.path.basename(p), 'skipped:', ex)
            continue
        print('REBUILD', os.path.basename(p), sorted(r.items()))
        print('WHOLE PACK byte-exact', os.path.basename(p), whole_pack(p))
        for c, v in diff.items():
            print('   ', c, len(v), v[:2])
