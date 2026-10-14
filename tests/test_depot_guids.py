"""Both guids of a depot texture record belong to one TextureAsset EBX: the partition guid names
it (work/guid_to_texture.json) and the instance guid occurs inside that same EBX's bytes."""
import json, os, random, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sk4'))
import depot

W = os.path.join(ROOT, 'work')
D = os.path.join(W, 'oracle', 'depots')


@pytest.mark.skipif(not os.path.exists(os.path.join(W, 'guid_to_texture.json')), reason='run sk4/guid_index.py')
def test_instance_guid_inside_named_ebx():
    from cas import decompress_many
    from guid_index import LAYOUT
    G = json.load(open(os.path.join(W, 'guid_to_texture.json')))
    EBX = json.load(open(os.path.join(W, 'global_assets.json')))['ebx']
    rows = [l.split('\t')[0] for l in open(os.path.join(D, 'index.tsv'), encoding='utf-8')]
    rng = random.Random(7)
    pairs = []
    for i in rng.sample(rows, 120):
        for params in depot.parse(open(os.path.join(D, f'{i}.bin'), 'rb').read(),
                                  open(os.path.join(D, f'{i}.meta'), 'rb').read()).values():
            for p in params:
                for inst, part in p.textures:
                    if part in G:
                        pairs.append((inst, G[part]))
    pairs = rng.sample(sorted(set(pairs)), 400)
    raw = []
    for _, name in pairs:
        pkg, ci, off, size = EBX[name][1:5]
        with open(os.path.join(LAYOUT, pkg, f'cas_{ci:02d}.cas'), 'rb') as f:
            f.seek(off); raw.append(f.read(size))
    blobs = decompress_many(raw)
    missing = [n for (inst, n), b in zip(pairs, blobs) if bytes.fromhex(inst) not in b]
    assert not missing, (len(missing), missing[:3])
