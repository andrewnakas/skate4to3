"""Our ShaderBlockDepot parse vs the Studio-dumped depots and Studio's `shader-params` listing."""
import os, re, struct, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sk4'))
import depot

D = os.path.join(ROOT, 'work', 'oracle', 'depots')
SP = os.path.join(ROOT, 'work', 'oracle', 'shader_params.txt')
KEYS = os.path.join(ROOT, 'work', 'oracle', 'sp_keys.txt')


def load(i):
    return depot.parse(open(os.path.join(D, f'{i}.bin'), 'rb').read(), open(os.path.join(D, f'{i}.meta'), 'rb').read())


@pytest.mark.skipif(not os.path.exists(os.path.join(D, 'index.tsv')), reason='no depot dumps')
def test_every_dumped_depot_parses():
    n = 0
    for line in open(os.path.join(D, 'index.tsv'), encoding='utf-8'):
        load(line.split('\t')[0]); n += 1
    assert n > 2000


@pytest.mark.skipif(not os.path.exists(SP), reason='no shader-params oracle')
def test_params_match_studio_listing():
    studio, cur = {}, None
    for line in open(SP, encoding='utf-8'):
        if line.startswith('=== '):
            cur = int(line.split()[1], 16); studio[cur] = []
        elif line.startswith('0x') and cur is not None:
            h, _, t, *rest = line.split()
            studio[cur].append((int(h, 16), int(t, 16), rest))
    checked = 0
    for row in open(KEYS).read().split('\n'):
        i, key = row.split('\t'); key = int(key, 16)
        got = load(i)[key]
        want = studio[key]
        assert [(p.hash, p.type) for p in got] == [(h, t) for h, t, _ in want]
        for p, (_, t, rest) in zip(got, want):
            if t in (depot.VEC3, depot.FLOAT):
                assert p.floats == pytest.approx([float(x) for x in rest], rel=1e-4, abs=1e-30)
        checked += 1
    assert checked > 100
