"""Shared, memory-lean texture fetch built on the slim on-disk indices."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textures
from cas import decompress_many

GAME = os.environ.get('SKATE_ROOT',
                      ('C:/Program Files (x86)/Steam/steamapps/common/Skate' if os.name == 'nt' else
                       '/run/media/nakas/Windows/Program Files (x86)/Steam/steamapps/common/Skate'))
LAYOUT = os.path.join(GAME, 'Data', 'Win32', 'configurations', 'layout')
WORK = os.environ.get('SKATE_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work'))


class TexCache:
    def __init__(self):
        self.res = json.load(open(os.path.join(WORK, 'tex_res.json')))
        self.cat = json.load(open(os.path.join(WORK, 'chunk_catalog.json')))
        self.loc = json.load(open(os.path.join(WORK, 'chunk_ranges.json')))
        self.h = {}

    def rd(self, pkg, ci, off, size):
        h = self.h.get((pkg, ci))
        if h is None:
            h = self.h[(pkg, ci)] = open(os.path.join(LAYOUT, pkg, f'cas_{ci:02d}.cas'), 'rb')
        h.seek(off)
        return h.read(size)

    def fetch(self, name, min_px=32):
        """(mip, w, h, PIL image, format) for the best VERIFIED mip, or None."""
        v = self.res.get(name)
        if not v:
            return None
        try:
            d = textures.parse_descriptor(decompress_many([self.rd(*v)])[0])
        except Exception:
            return None
        if not d:
            return None
        # One Oodle call for every block this texture needs. Each call is a wine spawn, and
        # doing them one at a time turned a 562-texture pass into 25+ minutes; batching is
        # what tools/oodle.py was built for and what its docstring asks for.
        spans, seen = [], set()
        for pkg, ci, off, size, lo, ls in self.loc.get(d['guid'], []):
            if (lo, ls) in seen:
                continue
            seen.add((lo, ls))
            spans.append((lo, ls, self.rd(pkg, ci, off, size)))
            if len(spans) >= 3:
                break
        # the TOC chunk table is keyed in descriptor order since superbundle.py was fixed
        c = self.cat.get(d['guid']) or self.cat.get(d['guid_rev'])
        catraw = self.rd(*c) if c else None
        try:
            got = decompress_many([sp[2] for sp in spans]
                                  + ([catraw] if catraw is not None else []))
        except Exception:
            got = []
        br = [(lo, ls, got[i]) for i, (lo, ls, _) in enumerate(spans) if i < len(got)]
        cb = ([got[len(spans)]] if catraw is not None and len(got) > len(spans) else [])
        i, px = textures.verified_mip(d, br, cb)
        if i is None:
            return None
        w, h = d['w'] >> i, d['h'] >> i
        if w < min_px or h < min_px:
            return None
        try:
            return i, w, h, textures.decode(px, w, h, d['format']), d['format']
        except Exception:
            return None
