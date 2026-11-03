#!/usr/bin/env python3
"""Fold a pcprof.txt (from libpcprof) into instructions per function.

    python3 tools/pcprof.py FIRMWARE.elf pcprof.txt [TOP]
"""
import bisect
import subprocess
import sys

elf, prof = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
syms = []
for line in subprocess.run(["arm-none-eabi-nm", "-n", "-C", elf], capture_output=True,
                           text=True, check=True).stdout.splitlines():
    parts = line.split(maxsplit=2)
    if len(parts) == 3 and parts[1] in "tTwW":
        syms.append((int(parts[0], 16) & ~1, parts[2]))
addrs = [a for a, _ in syms]
per = {}
total = 0
for line in open(prof):
    pc, n = line.split()
    pc, n = int(pc, 16), int(n)
    i = bisect.bisect_right(addrs, pc) - 1
    name = syms[i][1] if i >= 0 else "?"
    per[name] = per.get(name, 0) + n
    total += n
print(f"{total / 1e9:.3f} G instructions")
for name, n in sorted(per.items(), key=lambda kv: -kv[1])[:top]:
    print(f"{100 * n / total:6.2f}%  {n / 1e6:10.3f} M  {name[:110]}")
