"""sk3/big.py reads every retail world archive with every record named and hash-verified."""
import glob, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import big

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')
WORLDS = sorted(glob.glob(os.path.join(DATA, 'content', 'worldDIST_*.big')))


@pytest.mark.skipif(not WORLDS, reason='no retail Skate 3 data')
@pytest.mark.parametrize('path', WORLDS, ids=os.path.basename)
def test_world_archive_fully_named(path):
    a = big.read(path)
    assert a.entries and all(not e.name.startswith('~') for e in a.entries)
    assert all(big.djb2(e.path) == e.id for e in a.entries)
    assert [e.id for e in a.entries] == sorted(e.id for e in a.entries)


class Compare:
    """File-like sink that checks written bytes against an existing file as they arrive."""
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.pos = 0
        self.first_diff = None

    def write(self, b):
        if self.first_diff is None:
            ref = self.f.read(len(b))
            if ref != b:
                k = next((i for i in range(min(len(ref), len(b))) if ref[i] != b[i]), min(len(ref), len(b)))
                self.first_diff = self.pos + k
        else:
            self.f.seek(len(b), 1)
        self.pos += len(b)

    def done(self):
        rest = self.f.read(1)
        self.f.close()
        return self.first_diff is None and not rest


def rebuild_matches(path, generate_directory):
    a = big.read(path)
    sink = Compare(path)
    if generate_directory:
        flavour = big.read_directory(a.directory)[0]
        big.write(sink, big.sources_of(a), a.flags, None, None, flavour)
    else:
        big.rebuild(a, sink)
    return sink.done(), sink.first_diff


@pytest.mark.skipif(not WORLDS, reason='no retail Skate 3 data')
@pytest.mark.parametrize('path', WORLDS, ids=os.path.basename)
def test_world_archive_rebuild_exact(path):
    ok, at = rebuild_matches(path, False)
    assert ok, f'first difference at {at}'


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sfil_roundtrip import dlc_packs  # noqa: E402
PACKS = dlc_packs(int(os.environ.get('SK3_BIG_MAX', 1 << 40)))


@pytest.mark.parametrize('path', PACKS, ids=os.path.basename)
def test_dlc_pack_rebuild_exact_with_generated_directory(path):
    """Data order and path directory both derived from the path list alone."""
    ok, at = rebuild_matches(path, True)
    assert ok, f'first difference at {at}'


def test_directory_roundtrip():
    paths = ['data/a/B.txt', 'data/a/c/d.bin', 'data/Z.xml', 'data/a/a.txt']
    nodes = big.plan_directory(paths)
    d = big.build_directory(nodes)
    assert big.read_directory(d) == (1, nodes)
    assert [p for p, f in nodes if f] == ['data/Z.xml', 'data/a/a.txt', 'data/a/B.txt', 'data/a/c/d.bin']
