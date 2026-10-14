"""TOC chunk table vs ReSkate Studio: encoded size per sampled chunk, decoded size after Oodle."""
import json, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sk4'))
import superbundle as sb

GAME = os.environ.get('SKATE_ROOT', r'C:\Program Files (x86)\Steam\steamapps\common\Skate')
ORACLE = os.path.join(ROOT, 'work', 'oracle', 'chunks.json')


@pytest.mark.skipif(not os.path.exists(ORACLE), reason='run tests/make_chunk_oracle.py')
def test_chunk_sizes_and_decode_match_studio():
    import cas
    rows = [r for r in json.load(open(ORACLE)) if r['studio_encoded'] is not None]
    assert len(rows) >= 30
    tocs, raw, want = {}, [], []
    for r in rows:
        S = tocs.get(r['toc']) or tocs.setdefault(r['toc'], sb.SuperBundle(os.path.join(GAME, r['toc']), GAME))
        c = S.chunk(r['i'])
        assert c.guid == r['guid']
        assert c.size == r['studio_encoded'], r
        raw.append(S.read(c)); want.append(r['studio_decoded'])
    got = [len(b) for b in cas.decompress_many(raw)]
    assert got == want
