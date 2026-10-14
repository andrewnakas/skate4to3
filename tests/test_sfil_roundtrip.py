"""sk3/sfil.py rewrites every retail .xsf byte for byte from its parsed header + assets, and the
DLC packs' streams (raw form and EA form) too. Our RefPack output always decodes back exactly.

Small worlds run by default; SK3_SFIL_ALL=1 adds DownTown / Industrial / University (slow).
`python tests/test_sfil_roundtrip.py [maps]` prints per-archive counts, including how many
RefPack sections our compressor reproduces byte for byte (an EA-likeness measure, not a test).
"""
import collections, glob, os, struct, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import big, sfil, refpack

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')
SMALL = ['BlackBoxPark', 'MegaPark', 'StartPark', 'IndustrialSkatePark', 'MaloofMoneyCup',
         'DownTownSkatePark', 'SkateSchool']
LARGE = ['DownTown', 'Industrial', 'University']
WORLDS = SMALL + (LARGE if os.environ.get('SK3_SFIL_ALL') else [])
DLC_ROOTS = [os.path.join(os.environ.get('APPDATA', ''), 'skate3'), r'D:\skate3-engine-0.1.5-windows-x86_64\dlc']


def dlc_packs(max_bytes=600 << 20):
    """Every *_00000000.big DLC pack on this machine (one per name), smallest first."""
    seen = {}
    for root in DLC_ROOTS:
        for p in glob.glob(os.path.join(root, '**', '*_00000000.big'), recursive=True):
            sz = os.path.getsize(p)
            if sz <= max_bytes:
                seen.setdefault(os.path.basename(p).lower(), p)
    return sorted(seen.values(), key=os.path.getsize)


DLC_PACKS = dlc_packs()


def world_path(m):
    return os.path.join(DATA, 'content', f'worldDIST_{m}.big')


def streams(path):
    a = big.read(path)
    for e in a.entries:
        if e.name.endswith('.xsf'):
            yield e.path, a.read(e)


def rewrite(d):
    h, assets = sfil.read(d)
    return sfil.write(h, assets)


def check(path):
    n, bad = 0, []
    for name, d in streams(path):
        n += 1
        if rewrite(d) != d:
            bad.append(name)
    return n, bad


@pytest.mark.parametrize('m', WORLDS)
def test_world_streams_rewrite_exact(m):
    if not os.path.exists(world_path(m)):
        pytest.skip('no retail Skate 3 data')
    n, bad = check(world_path(m))
    assert n and not bad, f'{len(bad)}/{n} differ, e.g. {bad[:5]}'


@pytest.mark.parametrize('path', DLC_PACKS, ids=os.path.basename)
def test_dlc_streams_rewrite_exact(path):
    n, bad = check(path)
    assert n and not bad, f'{len(bad)}/{n} differ, e.g. {bad[:5]}'


def test_fresh_compression_decodes():
    """Without the stored bytes the writer compresses itself; the stream reads back to the same
    arenas, with EA's framing (descriptor raw sizes, >= 2048 rule, alignment) unchanged."""
    if not os.path.exists(world_path('BlackBoxPark')):
        pytest.skip('no retail Skate 3 data')
    for name, d in list(streams(world_path('BlackBoxPark')))[:6]:
        h, assets = sfil.read(d)
        fresh = [sfil.Asset(a.id, a.arena, 'xbox2') for a in assets]
        out = sfil.write(h, fresh, depth=64)
        h2, back = sfil.read(out)
        assert h2 == h and [(a.id, a.arena) for a in back] == [(a.id, a.arena) for a in assets]
        assert [a.compressed for a in back] == [a.compressed for a in assets], name


def test_raw_form():
    arena = b'\x89RW4' + bytes(0x200)
    s = sfil.raw_stream(0x1234, [sfil.Asset(7, arena)], tail=1 << 24)
    assert len(s) == 0x100 + 0x100 + 0x300
    h, back = sfil.read(s)
    assert h['align'] == 0x100 and h['platform'] == 0 and back[0].arena == arena and back[0].id == 7


def test_refpack_roundtrip_synthetic():
    import random
    rnd = random.Random(5)
    for n in (0, 1, 3, 4, 5, 113, 2048, 70000):
        src = bytes(rnd.choice(b'\0\0\0\1AB') for _ in range(n))
        assert refpack.decompress(refpack.compress(src)) == src
    big_run = bytes(200000) + bytes(range(256)) * 600
    assert refpack.decompress(refpack.compress(big_run)) == big_run


if __name__ == '__main__':
    maps = sys.argv[1:] or SMALL + LARGE
    for m in maps:
        p = m if os.path.exists(m) else world_path(m)
        n, bad = check(p)
        secs = exact = 0
        for name, d in streams(p):
            if secs >= 300:
                break
            for a in sfil.read(d)[1]:
                if a.form != 'xbox2':
                    continue
                for raw, st, c in zip((a.arena[:struct.unpack_from('>I', a.arena, 0x44)[0]],
                                       a.arena[struct.unpack_from('>I', a.arena, 0x44)[0]:]),
                                      a.stored, a.compressed):
                    if c and len(raw) < 200000:
                        secs += 1
                        exact += refpack.compress(raw) == st
        print(f'{os.path.basename(p)}: streams {n - len(bad)}/{n} exact; '
              f'RefPack sections reproduced {exact}/{secs}')
