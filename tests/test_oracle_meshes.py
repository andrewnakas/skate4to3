"""Every section's ShaderBlockDepot key we parse == ReSkate Studio's `mesh-sections` key."""
import os, re, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sk4'))
import meshset

D = os.path.join(ROOT, 'work', 'oracle', 'meshes')


def studio():
    cur, lod, rows = None, None, {}
    for line in open(os.path.join(D, 'mesh_sections.txt'), encoding='utf-8'):
        if not line.startswith(' '):
            cur = line.strip(); rows[cur] = []
        elif line.strip().startswith('lod'):
            lod = int(line.split()[0][3:])
        else:
            m = re.match(r'\s*\[(\d+)\] (\S+) key=(0x[0-9a-f]+)', line)
            if m:
                rows[cur].append((lod, m.group(2), int(m.group(3), 16)))
    return rows


@pytest.mark.skipif(not os.path.exists(os.path.join(D, 'index.tsv')), reason='run tests/make_mesh_oracle.py')
def test_depot_keys_match_studio():
    want = studio()
    n = 0
    for line in open(os.path.join(D, 'index.tsv'), encoding='utf-8'):
        i, name = line.split('\t')[:2]
        if name not in want:
            continue
        M = meshset.parse(open(os.path.join(D, f'{i}.bin'), 'rb').read())
        got = [(li, s.material, s.depot_key) for li, l in enumerate(M.lods) for s in l.sections]
        assert got == want[name], name
        n += 1
    assert n >= 50, n
