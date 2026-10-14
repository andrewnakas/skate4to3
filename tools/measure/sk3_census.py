"""Census of every retail world arena: object types per stream kind (cPres/cSim/cTex/global)."""
import collections, glob, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from sk3 import big, sfil, rw4
DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')
maps = sys.argv[1:] or ['BlackBoxPark', 'MegaPark', 'StartPark', 'IndustrialSkatePark']
types = collections.defaultdict(collections.Counter); arenas = collections.Counter(); hdr = collections.defaultdict(collections.Counter)
for m in maps:
    a = big.read(os.path.join(DATA, 'content', f'worldDIST_{m}.big'))
    for e in a.entries:
        if not e.name.endswith('.xsf'):
            continue
        kind = re.match(r'(c[A-Za-z]+)', e.name).group(1) + ('_Global' if 'Global' in e.name else '')
        _h, assets = sfil.read(a.read(e))
        for x in assets:
            ar = rw4.parse(x.arena)
            arenas[kind] += 1
            for o in ar.objects:
                types[kind][o.type_id] += 1
            for k in ('h08', 'h0c', 'h10', 'h14', 'h18', 'h20', 'h28', 'h2c', 'h38', 'h3c', 'h40', 'h48', 'h4c', 'h50', 'h58'):
                hdr[k][ar.header[k]] += 1
for k, c in types.items():
    print(f'{k:12s} arenas {arenas[k]:4d}:', ', '.join(f'{t:#x}x{n}' for t, n in c.most_common()))
print('header constants:', {k: dict(v.most_common(3)) for k, v in hdr.items() if len(v) <= 3})
