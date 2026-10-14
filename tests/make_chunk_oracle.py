#!/usr/bin/env python3
"""Sample chunks from every level TOC and record ReSkate Studio's encoded size for each.

    python tests/make_chunk_oracle.py [per_toc=6]   -> work/oracle/chunks.json
"""
import glob, json, os, random, re, subprocess, sys, uuid
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sk4'))
import superbundle as sb

GAME = os.environ.get('SKATE_ROOT', r'C:\Program Files (x86)\Steam\steamapps\common\Skate')
CLI = os.environ.get('RESKATE_CLI', r'D:\ReSkate Studio\reskate_cli.exe')
per = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rng = random.Random(4)
rows = []
for toc in sorted(glob.glob(os.path.join(GAME, 'Data', 'Win32', '**', '*.toc'), recursive=True)):
    S = sb.SuperBundle(toc, GAME)
    if S.chunks_count < 10 or os.sep + 'loc' + os.sep in toc:
        continue
    for i in rng.sample(range(S.chunks_count), per):
        c = S.chunk(i)
        ms = str(uuid.UUID(bytes_le=bytes.fromhex(c.guid)))
        out = subprocess.run([CLI, 'read', GAME, 'chunk', ms], capture_output=True, text=True).stdout
        m = re.search(r'^encoded=(\d+)', out, re.M)
        rows.append({'toc': os.path.relpath(toc, GAME), 'i': i, 'guid': c.guid,
                     'studio_encoded': int(m.group(1)) if m else None,
                     'studio_decoded': int(re.search(r'^decoded=(\d+)', out, re.M).group(1)) if m else None})
        print(rows[-1], flush=True)
os.makedirs(os.path.join(ROOT, 'work', 'oracle'), exist_ok=True)
json.dump(rows, open(os.path.join(ROOT, 'work', 'oracle', 'chunks.json'), 'w'), indent=0)
print(f'{len(rows)} oracle chunks')
