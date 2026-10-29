"""Summarise eval_policy.py --json outputs: success with 95% Wilson interval, failure reasons, hover stalls.

  python summarize.py ../results/L4_*exec*.json

Hover stall = failed episode where the peg tip got within 10 mm of the hole axis at some point, but the peg
never entered, and the right hand barely moved over the last 100 policy steps (10 s).
"""
import json
import math
import sys
from collections import Counter


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def stalled(ep):
    tr = ep.get("trace")      # [t, hand height above block, finger closure, peg z, tip-to-hole offset]
    if tr is None:
        return None           # older result file without traces
    if ep["success"] or len(tr) < 100:
        return False
    aligned = min(r[4] for r in tr) < 0.010
    tail = [r[1] for r in tr[-100:]]
    return aligned and ep["entry"] is None and (max(tail) - min(tail)) < 0.01


for fn in sorted(sys.argv[1:]):
    r = json.load(open(fn))
    eps = r["results"]
    n, k = len(eps), sum(e["success"] for e in eps)
    lo, hi = wilson(k, n)
    reasons = Counter(e["reason"] for e in eps if not e["success"])
    flags = [stalled(e) for e in eps]
    st = "n/a" if any(f is None for f in flags) else sum(flags)
    name = fn.split("/")[-1].replace(".json", "")
    print(f"{name:22s} exec {r['exec']:2d}  {k:2d}/{n}  {100 * k / n:3.0f}%  95% CI [{100 * lo:.0f}, {100 * hi:.0f}]  hover-stalls {st}")
    for why, c in reasons.most_common():
        print(f"      {c:2d} x {why}")