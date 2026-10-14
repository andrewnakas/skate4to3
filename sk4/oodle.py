"""Oodle Kraken decompression via the game's own oo2core_9_win64.dll under wine.

skate. block-compresses CAS chunks with Oodle Kraken (every compressed block's
payload starts 0x8c). Oodle is proprietary and the open-source reimplementations
have been taken down, so we call the DLL the game itself ships -- which is also
the exact codec version the data was packed with.

The DLL is a Windows PE, so the call goes through tools/oodle_batch.exe (natively on
Windows, under wine elsewhere). Blocks are batched into one invocation because wine startup (~0.3 s) would
otherwise dominate: a level has tens of thousands of blocks.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OODLE_DIR = os.path.join(os.path.dirname(HERE), 'external', 'oodle')
EXE = os.path.join(OODLE_DIR, 'oodle_batch.exe')
DLL = 'oo2core_9_win64.dll'      # resolved next to the exe by LoadLibraryA


class OodleError(RuntimeError):
    pass


def decompress_blocks(blocks: list[tuple[bytes, int]]) -> list[bytes]:
    """blocks = [(compressed, raw_size), ...] -> [raw, ...], order preserved."""
    if not blocks:
        return []
    if not os.path.exists(EXE):
        raise OodleError(f"{EXE} missing - run tools/build_oodle_batch.sh")
    if not os.path.exists(os.path.join(OODLE_DIR, DLL)):
        raise OodleError(f"{DLL} missing from {OODLE_DIR} - copy it from the game install")

    with tempfile.TemporaryDirectory() as td:
        job, out = os.path.join(td, 'job.bin'), os.path.join(td, 'out.bin')
        with open(job, 'wb') as f:
            f.write(struct.pack('<II', 0x424F4A4F, len(blocks)))
            for comp, raw in blocks:
                f.write(struct.pack('<QQ', len(comp), raw))
            for comp, _ in blocks:
                f.write(comp)
        env = dict(os.environ, WINEDEBUG='-all')
        env.setdefault('WINEPREFIX', os.path.expanduser('~/.wine-d2s3'))
        # On Windows the shim runs natively; everywhere else it goes through wine.
        argv = ([EXE] if sys.platform == 'win32' else ['wine', EXE]) + [DLL, job, out]
        p = subprocess.run(argv, cwd=OODLE_DIR, env=env,
                           capture_output=True, text=True, timeout=1800)
        if not os.path.exists(out):
            raise OodleError(f"no output (rc={p.returncode}): {p.stderr.strip()[:400]}")
        data = open(out, 'rb').read()
        if p.returncode != 0:
            raise OodleError(f"{p.stderr.strip()[:400]}")

    res, pos = [], 0
    for _, raw in blocks:
        res.append(data[pos:pos + raw])
        pos += raw
    return res
