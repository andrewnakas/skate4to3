#!/bin/bash
# Unattended windowed boot of a built pack on the owner's portable test rig (never the main install).
#   tools/smoke_boot.sh <pack folder (CONTENTID dir)> [seconds=90] [extra --cvar args]
# Stages the pack under the rig's dlc/, boots it via --skate3_dlc_select + demo path, waits, takes a
# screenshot to work/smoke/<id>_<t>.png every 30 s, then closes the game and prints the log tail.
set -u
RIG=${RIG:-/d/n64work/sanvan/colldiag/run}
R=$(cd "$(dirname "$0")/.." && pwd)
PACK=$(cd "$1" && pwd); ID=$(basename "$PACK"); SECS=${2:-90}; shift; [ $# -gt 0 ] && shift
mkdir -p "$R/work/smoke"
kill_rig() { powershell -NoProfile -Command "Get-Process colldiag_game -ErrorAction SilentlyContinue | Where-Object { \$_.Path -like 'D:\n64work\sanvan\colldiag\run\*' } | Stop-Process -Force" >/dev/null 2>&1; }
kill_rig; sleep 2
rm -rf "$RIG/dlc/$ID"; mkdir -p "$RIG/dlc/$ID"
for f in "$PACK"/*; do ln "$f" "$RIG/dlc/$ID/" 2>/dev/null || cp "$f" "$RIG/dlc/$ID/"; done
find "$RIG/0000000000000000" -name "*.big" -delete 2>/dev/null
n0=$(ls "$RIG"/logs/*.log 2>/dev/null | wc -l)
args="'--skate3_dlc_select=$ID','--skate3_demo_path=true','--skate3_demo_path_signed_in=true'"
for a in "$@"; do args="$args,'$a'"; done
powershell -NoProfile -Command "Start-Process -FilePath 'D:\n64work\sanvan\colldiag\run\colldiag_game.exe' -WorkingDirectory 'D:\n64work\sanvan\colldiag\run' -ArgumentList $args"
for i in $(seq 1 20); do [ $(ls "$RIG"/logs/*.log 2>/dev/null | wc -l) -gt $n0 ] && break; sleep 1; done
LOG=$(ls -t "$RIG"/logs/*.log | head -1); echo "log: $LOG"
t=0
while [ $t -lt $SECS ]; do
  sleep 30; t=$((t+30))
  powershell -NoProfile -Command "if (-not (Get-Process colldiag_game -ErrorAction SilentlyContinue)) { exit 1 }" || { echo "game exited at ~${t}s"; break; }
  python /d/n64work/sanvan/colldiag/shot.py "$R/work/smoke/${ID}_${t}.png" >/dev/null 2>&1 && echo "shot ${t}s"
done
kill_rig
cp "$LOG" "$R/work/smoke/${ID}.log"
grep -aiE "add-on|error|fatal|crash|exception|stream.*(fail|missing)|dlc" "$LOG" | tail -25
