"""sk3/texture_write rebuilds every retail world texture arena byte for byte from decoded fields
(fetch-constant words, name-table id, arena id) plus the linear (untiled) block data of each level.
Also: the from-image path decodes back to the source image within BC1/BC3 error.

Small worlds by default; SK3_TEX_ALL=1 adds DownTown / Industrial / University.
`python tests/test_texture_roundtrip.py [maps]` prints pass counts per archive.
"""
import collections, os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import big, sfil, rw4, texture, texture_write

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')
SMALL = ['BlackBoxPark', 'MegaPark', 'StartPark', 'IndustrialSkatePark', 'MaloofMoneyCup',
         'DownTownSkatePark', 'SkateSchool']
LARGE = ['DownTown', 'Industrial', 'University']
WORLDS = SMALL + (LARGE if os.environ.get('SK3_TEX_ALL') else [])


def world_path(m):
    return os.path.join(DATA, 'content', f'worldDIST_{m}.big')


def texture_arenas(m):
    a = big.read(world_path(m))
    for e in a.entries:
        if e.name.endswith('.xsf'):
            _h, assets = sfil.read(a.read(e))
            for i, x in enumerate(assets):
                if b'\x00\x02\x00\xe8' in x.arena[:0x200] and rw4.parse(x.arena).of_type(texture.T_TEXTURE):
                    yield f'{os.path.basename(e.name)}#{i}', x.arena


def roundtrip(d):
    return texture_write.build(texture_write.extract(d)) == d


@pytest.mark.parametrize('m', WORLDS)
def test_world_textures_roundtrip(m):
    if not os.path.exists(world_path(m)):
        pytest.skip('no retail Skate 3 data')
    n, bad = 0, []
    for name, d in texture_arenas(m):
        n += 1
        if not roundtrip(d):
            bad.append(name)
    assert n and not bad, f'{len(bad)}/{n} differ, e.g. {bad[:5]}'


def _test_image(w, h, alpha):
    """Smooth colour fields (what BC1 is built for), plus an alpha ramp."""
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w]).reshape(2, 1, 1)
    img = np.stack([128 + 100 * np.sin(6 * x), 128 + 100 * np.cos(5 * y), 128 + 90 * np.sin(4 * (x + y)),
                    255 * (x if alpha else np.ones_like(x))], -1)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


@pytest.mark.parametrize('w,h,alpha', [(256, 256, False), (512, 128, True), (64, 256, False), (16, 16, True)])
def test_from_image_decodes_back(w, h, alpha):
    img = _test_image(w, h, alpha)
    spec = texture_write.from_image(img, 0x1234567800000001, 0xABCD0001)
    arena = texture_write.build(spec)
    assert texture_write.build(texture_write.extract(arena)) == arena
    t = texture.from_arena(arena)
    assert t.format_name == ('DXT5' if alpha else 'DXT1') and (t.fc.width, t.fc.height) == (w, h)
    assert t.name == '0x1234567800000001.Texture'
    chain = texture_write.mip_chain(img, t.fc.levels)
    for lv in range(t.fc.levels):
        if min(chain[lv].shape[:2]) < 32:     # small levels: steep 2-D gradients exceed one BC1 line
            continue
        got = texture.decode_level(t, lv).astype(int)
        err = np.abs(got - chain[lv].astype(int)).mean()
        assert err < 5, (lv, err)


def test_encoder_reencodes_retail_blocks():
    """Retail DXT1 images are exactly BC1-representable: our encoder must nearly reproduce them."""
    if not os.path.exists(world_path('BlackBoxPark')):
        pytest.skip('no retail Skate 3 data')
    errs = []
    for _n, d in texture_arenas('BlackBoxPark'):
        t = texture.from_arena(d)
        if t.fc.format == 18 and t.fc.width >= 256:
            img = texture.decode_level(t, 0)
            enc = texture_write.encode_level(img, 18)
            dec = texture.blocks_to_rgba(enc, 18, img.shape[1] // 4, img.shape[0] // 4)
            errs.append(np.abs(dec.astype(int) - img).mean())
    assert errs and np.mean(errs) < 0.5 and max(errs) < 1.5, (np.mean(errs), max(errs))


if __name__ == '__main__':
    total = collections.Counter()
    for m in sys.argv[1:] or SMALL + LARGE:
        c = collections.Counter(roundtrip(d) for _n, d in texture_arenas(m))
        total.update(c)
        print(f'{m:20s} pass {c[True]:5d}  fail {c[False]}', flush=True)
    print(f'{"total":20s} pass {total[True]:5d}  fail {total[False]}')
