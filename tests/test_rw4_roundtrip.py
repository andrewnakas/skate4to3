"""sk3/rw4_write.py rebuilds every retail world arena byte for byte from the generator-level spec
(arena id, objects in dictionary order as (type, align, data), subref records) alone.

Small worlds run by default; set SK3_RW4_ALL=1 to include DownTown / Industrial / University
(slow: hundreds of MB of RefPack). Main entry `python tests/test_rw4_roundtrip.py [maps]` prints
pass counts per archive.
"""
import collections, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import big, sfil, rw4, rw4_write

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')
SMALL = ['BlackBoxPark', 'MegaPark', 'StartPark', 'IndustrialSkatePark', 'MaloofMoneyCup',
         'DownTownSkatePark', 'SkateSchool']
LARGE = ['DownTown', 'Industrial', 'University']
WORLDS = SMALL + (LARGE if os.environ.get('SK3_RW4_ALL') else [])


def world_path(m):
    return os.path.join(DATA, 'content', f'worldDIST_{m}.big')


def arenas(m):
    a = big.read(world_path(m))
    for e in a.entries:
        if e.name.endswith('.xsf'):
            _h, assets = sfil.read(a.read(e))
            for i, x in enumerate(assets):
                yield f'{os.path.basename(e.name)}#{i}', x.arena


def roundtrip(d):
    return rw4_write.build(rw4_write.spec_from_arena(rw4.parse(d))) == d


@pytest.mark.parametrize('m', WORLDS)
def test_world_arenas_roundtrip(m):
    if not os.path.exists(world_path(m)):
        pytest.skip('no retail Skate 3 data')
    n, bad = 0, []
    for name, d in arenas(m):
        n += 1
        if not roundtrip(d):
            bad.append(name)
    assert n and not bad, f'{len(bad)}/{n} differ, e.g. {bad[:5]}'


def test_registry_is_derived():
    """The spec never carries the type registry: the shortest known one covering the objects."""
    o = rw4_write.SpecObject
    small = rw4_write.build(rw4_write.Spec(1, [o(0x10031, 0x1000, b'\0' * 16), o(0x200e8, 4, b'\0' * 4)]))
    full = rw4_write.build(rw4_write.Spec(1, [o(0x80006, 0x10, b'\0' * 16)]))
    assert rw4.parse(small).types == list(rw4_write.REGISTRY_SMALL)
    assert rw4.parse(full).types == list(rw4_write.REGISTRY_FULL)


if __name__ == '__main__':
    total = collections.Counter()
    for m in sys.argv[1:] or SMALL + LARGE:
        c = collections.Counter(roundtrip(d) for _n, d in arenas(m))
        total.update(c)
        print(f'{m:20s} pass {c[True]:5d}  fail {c[False]}', flush=True)
    print(f'{"total":20s} pass {total[True]:5d}  fail {total[False]}')
