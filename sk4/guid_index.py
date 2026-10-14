"""partition GUID -> texture asset name, for resolving ShaderBlockDepot bindings.

A depot block names each bound texture as `(instanceGuid, partitionGuid)`, the same pair form
EBX imports use. The partition guid lives in the EBX's EFIX chunk, so this walks every texture
EBX once and caches the mapping. Only texture-named EBX are indexed: that is all a depot
binding can point at, and it keeps the pass to ~23k assets instead of 171k.
"""
from __future__ import annotations
import json, os, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas import decompress_many

GAME = os.environ.get('SKATE_ROOT',
                      ('C:/Program Files (x86)/Steam/steamapps/common/Skate' if os.name == 'nt' else
                       '/run/media/nakas/Windows/Program Files (x86)/Steam/steamapps/common/Skate'))
LAYOUT = os.path.join(GAME, 'Data', 'Win32', 'configurations', 'layout')
WORK = os.environ.get('SKATE_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work'))
OUT = os.path.join(WORK, 'guid_to_texture.json')


def partition_guid(blob: bytes):
    """The partition guid out of a RIFF EBX, without a full parse."""
    if blob[:4] != b'RIFF':
        return None
    size = struct.unpack_from('<I', blob, 4)[0]
    p = 12
    while p + 8 <= min(len(blob), 8 + size):
        cid = blob[p:p + 4]
        csz = struct.unpack_from('<I', blob, p + 4)[0]
        if cid == b'EFIX':
            return blob[p + 8:p + 24]
        p += 8 + csz
        p = (p + 1) & ~1
    return None


def main():
    if os.path.exists(OUT):
        print(f"  cached: {len(json.load(open(OUT))):,} guids")
        return
    G = json.load(open(os.path.join(WORK, 'global_assets.json')))
    EBX = G['ebx']
    del G
    texres = set(json.load(open(os.path.join(WORK, 'tex_res.json'))))
    names = sorted(n for n in texres if n in EBX)
    print(f"  texture EBX to index: {len(names):,} of {len(texres):,} textures")
    handles = {}

    def rd(pkg, ci, off, size):
        h = handles.get((pkg, ci))
        if h is None:
            h = handles[(pkg, ci)] = open(os.path.join(LAYOUT, pkg, f'cas_{ci:02d}.cas'), 'rb')
        h.seek(off)
        return h.read(size)

    out, bad, B = {}, 0, 400
    for s in range(0, len(names), B):
        chunk = names[s:s + B]
        try:
            blobs = decompress_many([rd(*EBX[n][1:5]) for n in chunk])
        except Exception:
            bad += len(chunk)
            continue
        for n, b in zip(chunk, blobs):
            g = partition_guid(b)
            if g:
                out[g.hex()] = n
            else:
                bad += 1
        if s % 4000 == 0:
            print(f"    {s:,}/{len(names):,} ... {len(out):,} guids", flush=True)
    json.dump(out, open(OUT, 'w'))
    print(f"  wrote {len(out):,} guid -> texture entries ({bad} unreadable)")


if __name__ == '__main__':
    main()
