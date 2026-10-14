"""Every cSim object with a codec in sk3/collision_write.py re-encodes byte for byte from its decoded
fields. Small worlds by default; SK3_COLL_ALL=1 adds DownTown / Industrial / University.
`python tests/test_collision_roundtrip.py [worlds]` prints pass counts per type.
"""
import collections, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import big, sfil, rw4, collision_write as cw

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')
SMALL = ['BlackBoxPark', 'MegaPark', 'StartPark', 'IndustrialSkatePark', 'MaloofMoneyCup',
         'DownTownSkatePark', 'SkateSchool']
LARGE = ['DownTown', 'Industrial', 'University']
WORLDS = SMALL + (LARGE if os.environ.get('SK3_COLL_ALL') else [])


def world_path(m):
    return os.path.join(DATA, 'content', f'worldDIST_{m}.big')


def sim_objects(m):
    a = big.read(world_path(m))
    for e in a.entries:
        b = os.path.basename(e.name)
        if b.endswith('.xsf') and b.startswith('cSim'):
            for i, x in enumerate(sfil.read(a.read(e))[1]):
                for o in rw4.parse(x.arena).objects:
                    yield f'{b}#{i}', o


def tally(m):
    c = collections.Counter()
    bad = []
    for name, o in sim_objects(m):
        try:
            r = cw.reencode(o.type_id, o.data)
        except Exception as ex:          # a decode failure counts as a failure
            r = ex
        key = 'opaque' if r is None else (r == o.data)
        c[(o.type_id, key)] += 1
        if key is False:
            bad.append((name, hex(o.type_id)))
    return c, bad


@pytest.mark.parametrize('m', WORLDS)
def test_csim_objects_roundtrip(m):
    if not os.path.exists(world_path(m)):
        pytest.skip('no retail Skate 3 data')
    c, bad = tally(m)
    assert sum(v for (t, k), v in c.items() if k is True) and not bad, bad[:5]


if __name__ == '__main__':
    total = collections.Counter()
    for m in sys.argv[1:] or SMALL + LARGE:
        c, bad = tally(m)
        total.update(c)
        p = sum(v for (t, k), v in c.items() if k is True)
        f = sum(v for (t, k), v in c.items() if k is False)
        o = sum(v for (t, k), v in c.items() if k == 'opaque')
        print(f'{m:20s} pass {p:6d} fail {f:4d} opaque {o:5d}  {bad[:2]}', flush=True)
    types = sorted({t for t, _k in total})
    for t in types:
        print(f'  {t:#08x} ' + ' '.join(f'{k}={total[(t, k)]}' for k in (True, False, 'opaque') if total[(t, k)]))
