#!/usr/bin/env python3
"""Fail if any tracked file mentions a source we must not use (see CLEANROOM.md).

    python tools/taint_scan.py          # scans `git ls-files` + untracked, prints hits, exit 1 on any
"""
import base64, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# kept encoded so this file does not match itself
_ENC = ('cmlwcGVy bWFwcmlw c3VuamF5 d3JhdGg= YXJlbmFidWlsZGVy ZGxjYnVpbGRlcg== ZHVtYmFk dHV1a2thcw== '
        'YmlnZmlsZVwuZXhl c3RyZWFtIGZpbGUgdG9vbA== Z2xidG9yeDI= ZXRoYW53MDU= d3JAMTExdGg= '
        'Y2hhem1t ZHVja3lpbm5pdA==')
PATTERNS = [re.compile(re.escape(base64.b64decode(t).decode()).replace(r'\\.', r'\.'), re.I)
            for t in _ENC.split()]
SKIP = {'tools/taint_scan.py'}


def files():
    out = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f not in SKIP]


def main():
    hits = 0
    for f in files():
        p = os.path.join(ROOT, f)
        try:
            text = open(p, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for rx in PATTERNS:
                if rx.search(line):
                    hits += 1
                    if hits <= 40:
                        print(f'{f}:{n}: {line.strip()[:120]}')
                    break
    print(f'taint scan: {hits} hit(s) in {len(files())} files')
    sys.exit(1 if hits else 0)


if __name__ == '__main__':
    main()
