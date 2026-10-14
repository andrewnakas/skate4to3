#!/usr/bin/env bash
# Cross-compile the Oodle shim for Win64. Needs a mingw-w64 toolchain; if you have
# no root, fetch one without it:
#   apt-get download gcc-mingw-w64-x86-64-win32 binutils-mingw-w64-x86-64 \
#                    mingw-w64-x86-64-dev gcc-mingw-w64-base mingw-w64-common
#   for f in *.deb; do dpkg-deb -x "$f" x/; done && cp -a x/usr/. ~/.local/mingw-w64/
# mingw-w64-common is easy to miss: without it every header is a dangling symlink.
set -euo pipefail
MINGW="${MINGW:-$HOME/.local/mingw-w64}"
CC="$MINGW/bin/x86_64-w64-mingw32-gcc-win32"
[ -x "$CC" ] || { echo "no mingw at $MINGW (set MINGW=)" >&2; exit 1; }
cd "$(dirname "$0")"
"$CC" -O2 -s oodle_batch.c -o oodle_batch.exe
echo "built $(pwd)/oodle_batch.exe"
