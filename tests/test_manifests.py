"""sk3/manifests.py: every DIST manifest (.xmm .xsm .xss .xst) and location table (.loc) in the retail
worlds and the DLC packs on this machine re-writes byte for byte; `generate` rebuilds a group's
MMAP / CMAP / ATOC byte for byte from its tiles' streams and the CSPA cell list in order.

`python tests/test_manifests.py` prints pass counts per format.
"""
import collections, glob, os, struct, sys
import pytest


@pytest.fixture(autouse=True)
def _legacy_global_group(monkeypatch):
    """These tests regenerate community packs byte for byte, bug included."""
    import sk3.manifests as _mf
    monkeypatch.setattr(_mf, 'LEGACY_MISSING_GLOBAL_GROUP', True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sk3 import big, manifests as mf
from test_sfil_roundtrip import DATA, dlc_packs

WORLDS = sorted(glob.glob(os.path.join(DATA, 'content', 'worldDIST_*.big')))
LOOSE_LOC = sorted(glob.glob(os.path.join(DATA, 'content', 'global_locators', '**', '*.loc'),
                             recursive=True))
ARCHIVES = WORLDS + dlc_packs(1 << 40)

READ = {'.xmm': (mf.read_mmap, mf.write_mmap), '.xsm': (mf.read_cmap, mf.write_cmap),
        '.xss': (mf.read_cspa, mf.write_cspa), '.xst': (mf.read_atoc, mf.write_atoc),
        '.loc': (mf.read_loc, mf.write_loc)}


def manifest_files(path):
    a = big.read(path)
    with open(path, 'rb') as f:
        for e in a.entries:
            ext = os.path.splitext(e.name)[1]
            if ext in READ:
                f.seek(e.offset)
                yield e.path, ext, f.read(e.size)


def check(path):
    c = collections.Counter()
    bad = []
    for name, ext, d in manifest_files(path):
        rd, wr = READ[ext]
        ok = wr(rd(d)) == d
        c[(ext, ok)] += 1
        if not ok:
            bad.append(name)
    return c, bad


@pytest.mark.parametrize('path', ARCHIVES, ids=os.path.basename)
def test_manifests_rewrite_exact(path):
    c, bad = check(path)
    assert sum(c.values()) and not bad, f'{len(bad)} differ, e.g. {bad[:4]}'


def test_loose_locators_exact():
    if not LOOSE_LOC:
        pytest.skip('no retail data')
    for p in LOOSE_LOC:
        d = open(p, 'rb').read()
        assert mf.write_loc(mf.read_loc(d)) == d, p


def regenerate(path, dist_root):
    """Rebuild one archive's manifests from its streams; returns {ext+kind: (ours, theirs)}."""
    from sk3 import sfil
    a = big.read(path)
    files = {e.path: e for e in a.entries}
    dist = dist_root.rsplit('/', 1)[1]
    out = {}
    for kind, prefix in mf.KINDS:
        base = f'{dist_root}/{dist}_{kind}'
        cm = mf.read_cmap(a.read(files[base + '.xsm']))
        tiles = []
        for e in cm.entries:
            x = a.read(files[f'{dist_root}/{e.name}.xsf'])
            h, assets = sfil.read(x)
            tiles.append(mf.Tile(e.name, [(s.id, s.arena) for s in assets], len(x)))
        community = cm.platform == 0
        al = cm.entries[0].alignment if cm.entries else (0x100 if community else 0x80)
        globals_ = [e for e in cm.entries if e.name.endswith('_Global')]
        gen = mf.generate(dist, kind, tiles, platform=cm.platform, alignment=al,
                          size_includes_header=community, group=cm.group)
        for ext in ('.xmm', '.xsm', '.xss', '.xst'):
            out[kind + ext] = (gen[ext], a.read(files[base + ext]), bool(globals_))
    return out


def test_generate_matches_community_pack():
    packs = [p for p in dlc_packs() if 'summercamp_' in os.path.basename(p).lower()]
    if not packs:
        pytest.skip('no SummerCamp pack')
    res = regenerate(packs[0], 'data/content/world/stream/DIST_SummerCamp')
    for key, (ours, theirs, _g) in res.items():
        if key.endswith(('.xmm', '.xst', '.xsm')):
            assert ours == theirs, key
        else:
            assert [c.asset_id for c in mf.read_cspa(ours).cells] == \
                   [c.asset_id for c in mf.read_cspa(theirs).cells], key


def test_generate_matches_retail_atoc_and_ids():
    p = os.path.join(DATA, 'content', 'worldDIST_SkateSchool.big')
    if not os.path.exists(p):
        pytest.skip('no retail data')
    res = regenerate(p, 'data/content/world/stream/DIST_SkateSchool')
    for key, (ours, theirs, _g) in res.items():
        if key.endswith('.xmm'):
            assert ours == theirs, key
        if key.endswith('.xst'):     # EA's record tails hold stale bytes; everything else equal
            strip = lambda m: [(r.asset_id, r.tag, r.size, r.words) for r in m.records]
            assert strip(mf.read_atoc(ours)) == strip(mf.read_atoc(theirs)), key
        if key.endswith('.xsm'):
            o, t = mf.read_cmap(ours), mf.read_cmap(theirs)
            assert sorted((e.name, e.asset_id, e.group_id, e.size, e.chunks) for e in o.entries) == \
                   sorted((e.name, e.asset_id, e.group_id, e.size, e.chunks) for e in t.entries), key


if __name__ == '__main__':
    tot = collections.Counter()
    for p in ARCHIVES:
        c, bad = check(p)
        tot.update(c)
        print(os.path.basename(p), dict(c), bad[:3])
    for p in LOOSE_LOC:
        d = open(p, 'rb').read()
        tot[('.loc loose', mf.write_loc(mf.read_loc(d)) == d)] += 1
    print('TOTAL', sorted(tot.items()))
