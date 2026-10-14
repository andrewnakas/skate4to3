"""sk3/render_write.py rebuilds every retail render arena (cPres_*.xsf, cPres_Global.xsf) byte for
byte from the semantic decode of sk3/render.py: vertex attributes (positions, uv0/uv2, lightmap uv,
normal, tangent, binormal sign), index lists, declarations, meshes, models, instances, materials,
export tables and light probes are re-encoded, not copied (texture headers/texels are carried raw).

Small worlds run by default; SK3_RENDER_ALL=1 adds DownTown / Industrial / University.
`python tests/test_render_roundtrip.py [maps]` prints pass counts per archive.
"""
import collections, os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import big, sfil, render, render_write

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')
SMALL = ['BlackBoxPark', 'MegaPark', 'StartPark', 'IndustrialSkatePark', 'MaloofMoneyCup',
         'DownTownSkatePark', 'SkateSchool']
LARGE = ['DownTown', 'Industrial', 'University']
WORLDS = SMALL + (LARGE if os.environ.get('SK3_RENDER_ALL') else [])


def world_path(m):
    return os.path.join(DATA, 'content', f'worldDIST_{m}.big')


def arenas(m):
    a = big.read(world_path(m))
    for e in a.entries:
        n = os.path.basename(e.name)
        if n.startswith('cPres') and n.endswith('.xsf'):
            _h, assets = sfil.read(a.read(e))
            for i, x in enumerate(assets):
                yield f'{n}#{i}', x.arena


def roundtrip(d):
    return render_write.rebuild(d) == d


@pytest.mark.parametrize('m', WORLDS)
def test_render_arenas_roundtrip(m):
    if not os.path.exists(world_path(m)):
        pytest.skip('no retail Skate 3 data')
    n, bad = 0, []
    for name, d in arenas(m):
        n += 1
        if not roundtrip(d):
            bad.append(name)
    assert n and not bad, f'{len(bad)}/{n} differ, e.g. {bad[:5]}'


def _quad_part(x0, z0, size=4.0, shader='environmentsimple.default'):
    pos = np.array([[x0, 0, z0], [x0 + size, 0, z0], [x0 + size, 0, z0 + size], [x0, 0, z0 + size]], float)
    return dict(positions=pos, normals=np.tile([0.0, 1.0, 0.0], (4, 1)),
                uv0=pos[:, [0, 2]] / 2, uv1=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) * 0.5,
                indices=np.array([[0, 2, 1], [0, 3, 2]]),
                material=dict(name='test_concrete', shader=shader,
                              textures={'diffuse': 0x1111, 'normal': 0x2222, 'specular': 0x3333,
                                        'lightmap': 0x4444}))


def test_builder_decodes_back():
    parts = [_quad_part(0, 0), _quad_part(10, 0, 2.0)]
    d = render_write.build_render_arena(parts, arena_id=0x12345678, name='TestModel')
    ra = render.decode(d)
    out = list(ra.parts())
    assert len(out) == 2
    assert render_write.rebuild(d) == d
    for p, q in zip(parts, out):
        np.testing.assert_allclose(q.positions[q.indices].reshape(-1, 3),
                                   p['positions'][p['indices']].reshape(-1, 3), atol=1e-6)
        assert render.shader_name(q.material) == 'environmentsimple.default'
        assert render.texture_refs(q.material)['diffuse'] == 0x1111
        np.testing.assert_allclose(q.attrs['normal'][:, 1], 1, atol=1e-4)


if __name__ == '__main__':
    total = collections.Counter()
    for m in sys.argv[1:] or SMALL + LARGE:
        c = collections.Counter(roundtrip(d) for _n, d in arenas(m))
        total.update(c)
        print(f'{m:20s} pass {c[True]:5d}  fail {c[False]}', flush=True)
    print(f'{"total":20s} pass {total[True]:5d}  fail {total[False]}')
