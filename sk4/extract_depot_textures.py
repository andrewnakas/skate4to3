"""Extract the textures the ShaderBlockDepot binds, into the exporter's material map.

Input  work/depot_textures.json   material -> {c, ny, msk} texture asset names
Output work/material_textures.json  material -> {png, normal, w, h}  (what export_glb reads)

This replaces the earlier basename-guessing binder: the depot is the game's own binding, so a
material gets the texture skate. actually draws it with rather than a per-class stand-in.
"""
from __future__ import annotations
import collections, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textures
from texcache import TexCache, WORK


def main():
    # Decoding all 608 textures costs ~20 min: every block goes through one wine spawn for
    # Oodle. A re-run after a binder change needs the same PNGs, so --reuse keeps the ones
    # already on disk and only decodes what is new.
    reuse = '--reuse' in sys.argv[1:]
    tc = TexCache()
    depot = json.load(open(os.path.join(WORK, 'depot_textures.json')))
    tdir = os.path.join(WORK, 'textures')
    os.makedirs(tdir, exist_ok=True)
    if not reuse:
        for f in os.listdir(tdir):
            if f.startswith('MAT_'):
                os.remove(os.path.join(tdir, f))
    on_disk = {f for f in os.listdir(tdir) if f.startswith('MAT_')} if reuse else set()

    wanted = collections.OrderedDict()          # texture name -> None
    for v in depot.values():
        for ch in ('c', 'ny'):
            if v.get(ch):
                wanted.setdefault(v[ch], None)
    print(f"  distinct depot-bound textures to extract: {len(wanted):,}", flush=True)

    cache, res_hist, fails = {}, collections.Counter(), collections.Counter()
    reused = 0
    for i, name in enumerate(wanted):
        slug = 'MAT_' + re.sub(r'[^A-Za-z0-9_.-]', '_', name)[-80:] + '.png'
        if slug in on_disk:
            try:
                from PIL import Image
                with Image.open(os.path.join(tdir, slug)) as im:
                    im = im.convert('RGB')
                    cache[name] = (f'textures/{slug}', im.size[0], im.size[1],
                                   textures.looks_like_normal_map(im),
                                   textures.valid_tangent_normal(im))
                res_hist[f'{im.size[0]}x{im.size[1]}'] += 1
                reused += 1
                continue
            except Exception:  # noqa: BLE001
                pass
        g = tc.fetch(name, min_px=16)
        if not g:
            fails['no verified mip'] += 1
            continue
        mip, w, h, im, fmt = g
        im.save(os.path.join(tdir, slug))
        cache[name] = (f'textures/{slug}', w, h,
                       textures.looks_like_normal_map(im),
                       textures.valid_tangent_normal(im))
        im.close()
        del im
        res_hist[f'{w}x{h}'] += 1
        if i and i % 100 == 0:
            print(f"    {i:,}/{len(wanted):,} ... {len(cache):,} ok", flush=True)
    print(f"  extracted {len(cache):,} of {len(wanted):,} ({reused:,} reused from disk)")
    for k, c in res_hist.most_common(8):
        print(f"    {k:<10} {c}")
    for k, c in fails.most_common():
        print(f"    {k}: {c}")

    out, skipped = {}, collections.Counter()
    for v in depot.values():
        col = cache.get(v.get('c') or '')
        if not col:
            skipped['colour not extractable'] += 1
            continue
        png, w, h, is_nrm, _ = col
        if is_nrm:
            skipped['colour decodes as a normal map'] += 1
            continue
        e = {'base': v['c'], 'png': png, 'w': w, 'h': h, 'mip': 0}
        if v.get('tint'):
            e['tint'] = v['tint']
        nrm = cache.get(v.get('ny') or '')
        if nrm and nrm[4]:
            e['normal'] = nrm[0]
            e['nw'], e['nh'] = nrm[1], nrm[2]
        for m in v['materials']:
            out[m] = e
    # Written through a temp file and read back: an earlier run printed 588 materials while
    # the file on disk held 581, and a silently short material map is the kind of fault that
    # only shows up as untextured geometry three build stages later.
    dst = os.path.join(WORK, 'material_textures.json')
    tmp = dst + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dst)
    back = json.load(open(dst))
    if len(back) != len(out):
        raise SystemExit(f"material_textures.json wrote {len(back)} of {len(out)} entries")
    n_nrm = sum(1 for e in out.values() if 'normal' in e)
    print(f"\n  materials with an exact game texture: {len(out):,}  ({n_nrm:,} with normal maps)")
    for k, c in skipped.most_common():
        print(f"    skipped: {k} x{c}")


if __name__ == '__main__':
    main()
