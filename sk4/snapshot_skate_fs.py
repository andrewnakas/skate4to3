"""Snapshot (or diff) the files skate. can write to, to see what playing it downloads.

    python tools/snapshot_skate_fs.py snap work/skate_fs_before.json
    python tools/snapshot_skate_fs.py diff work/skate_fs_before.json work/skate_fs_after.json

The high-detail render LODs are not in the install (see PLAN.md), but the game has a
content-pack download path (`/persistent/contentpacks/` = AppData/Local/Skate/data/contentpacks).
Snapshot before a session on Windows, snapshot after, and diff: anything new or grown is what
the game fetched. Read-only; run it against the read-only mount.
"""
import json, os, sys

WIN = os.environ.get('WIN_ROOT', '/run/media/nakas/Windows')
ROOTS = [
    'Program Files (x86)/Steam/steamapps/common/Skate',
    'Program Files (x86)/Steam/steamapps/appmanifest_3354750.acf',
    'Users/andre/AppData/Local/Skate',
]


def snap(out):
    files = {}
    for r in ROOTS:
        base = os.path.join(WIN, r)
        if os.path.isfile(base):
            st = os.stat(base)
            files[r] = [st.st_size, int(st.st_mtime)]
            continue
        for dp, _, fs in os.walk(base):
            for f in fs:
                p = os.path.join(dp, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                files[os.path.relpath(p, WIN)] = [st.st_size, int(st.st_mtime)]
    json.dump(files, open(out, 'w'), indent=0)
    print(f"{len(files):,} files, {sum(v[0] for v in files.values()) / 1e9:.2f} GB -> {out}")


def diff(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    new = sorted(set(B) - set(A))
    gone = sorted(set(A) - set(B))
    changed = sorted(k for k in set(A) & set(B) if A[k] != B[k])
    for label, ks, src in (('NEW', new, B), ('CHANGED', changed, B), ('GONE', gone, A)):
        print(f"{label}: {len(ks)} files, {sum(src[k][0] for k in ks) / 1e6:.1f} MB")
        for k in sorted(ks, key=lambda k: -src[k][0])[:40]:
            extra = f"  (was {A[k][0]:,})" if label == 'CHANGED' else ''
            print(f"  {src[k][0]:>14,}  {k}{extra}")


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'snap':
        snap(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == 'diff':
        diff(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
