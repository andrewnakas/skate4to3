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
