"""Per-mesh textures for name-consistent bindings: verified mips only, proxies excluded."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textures
from texcache import TexCache, WORK


def main():
    tc = TexCache()
    bind = json.load(open(os.path.join(WORK, 'texture_bindings.json')))
    names = sorted({b['base'] for b in bind.values()
                    if '_proxy' not in b['base'].rsplit('/', 1)[-1].lower()})
    out = os.path.join(WORK, 'textures')
    os.makedirs(out, exist_ok=True)
    man, hist, fails = {}, collections.Counter(), collections.Counter()
    for n in names:
        g = tc.fetch(n)
        if not g:
            fails['no verified mip'] += 1
            continue
        i, w, h, im, fmt = g
        if textures.looks_like_normal_map(im):
            fails['decodes as a normal map'] += 1
            continue
        p = os.path.join(out, n.rsplit('/', 1)[-1] + '.png')
        im.save(p)
        man[n] = {'png': os.path.relpath(p, WORK), 'mip': i, 'w': w, 'h': h, 'format': fmt}
        hist[f'{w}x{h}'] += 1
    json.dump(man, open(os.path.join(WORK, 'texture_manifest.json'), 'w'), indent=1)
    print(f"per-mesh: {len(man)} verified of {len(names)} non-proxy bindings")
    for k, c in hist.most_common(6):
        print(f"    {k:<10} {c}")
    for k, c in fails.most_common():
        print(f"    skipped: {k} x{c}")


if __name__ == '__main__':
    main()
