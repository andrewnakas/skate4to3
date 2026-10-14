#!/usr/bin/env python3
"""Dump N MeshSet resources and Studio's section listing for the same name filter.

    python tests/make_mesh_oracle.py <name-substring> <limit>  -> work/oracle/meshes/
"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.environ.get('SKATE_ROOT', r'C:\Program Files (x86)\Steam\steamapps\common\Skate')
CLI = os.environ.get('RESKATE_CLI', r'D:\ReSkate Studio\reskate_cli.exe')
sub, lim = sys.argv[1], sys.argv[2]
out = os.path.join(ROOT, 'work', 'oracle', 'meshes')
os.makedirs(out, exist_ok=True)
subprocess.run([CLI, 'dump-res', GAME, '0x49B156D4', sub, out, lim], check=True)
txt = subprocess.run([CLI, 'mesh-sections', GAME, sub, lim], capture_output=True, text=True, check=True).stdout
open(os.path.join(out, 'mesh_sections.txt'), 'w', encoding='utf-8').write(txt)
print('done', len(txt.splitlines()), 'lines')
