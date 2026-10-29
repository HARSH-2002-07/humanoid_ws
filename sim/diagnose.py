"""Two diagnostics for the history study.

  python diagnose.py --data ../data_L4 --results ../results/L4_hist_exec5.json ../results/L4_hist_exec1.json

A) How long do the demonstrations pause? (If pauses are longer than the 1.5 s history window, history cannot
   tell "just arrived" from "waiting long enough to move on".)
B) For failed rollouts: was the peg dropped (fingers open?), or did the hand freeze above the hole?
"""
import argparse
import glob
import json

import h5py
import numpy as np


def run_lengths(x):
    out, start = [], 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[start]:
            out.append((x[start], i - start))
            start = i
    return out


def demo_dwell(data, n_files=60):
    files = sorted(glob.glob(f"{data}/ep_*.hdf5"))[:n_files]
    with h5py.File(files[0], "r") as h:
        print("hdf5 keys:", list(h.keys()))
    phase_key = None
    pauses, per_phase = [], {}
    for fn in files:
        with h5py.File(fn, "r") as h:
            ctrl = h["ctrl"][:]
            phase_key = phase_key or next((k for k in h.keys() if "phase" in k.lower()), None)
            ph = h[phase_key][:] if phase_key else None
        speed = np.abs(np.diff(ctrl, axis=0)).max(1)          # largest per-joint target change per 0.1 s
        for still, n in run_lengths(list(speed < 0.01)):
            if still and n >= 3:
                pauses.append(n / 10)
        if ph is not None:
            for lab, n in run_lengths([x.decode() if isinstance(x, bytes) else x for x in ph.tolist()]):
                per_phase.setdefault(lab, []).append(n / 10)
    print(f"\nA) demo pauses (all joint targets nearly constant, >= 0.3 s), {len(files)} episodes")
    if pauses:
        p = np.array(pauses)
        print(f"   {len(p)} pauses: median {np.median(p):.1f} s, 90th pct {np.percentile(p, 90):.1f} s, max {p.max():.1f} s; "
              f"{(p > 1.5).mean() * 100:.0f}% longer than the 1.5 s history window")
    else:
        print("   no pauses found")
    if per_phase:
        print(f"   phase durations (key '{phase_key}'), seconds:")
        for lab, v in per_phase.items():
            print(f"     {str(lab):>14}: mean {np.mean(v):5.1f}  min {np.min(v):5.1f}  max {np.max(v):5.1f}")


def failures(fn, show=8):
    r = json.load(open(fn))
    eps = [e for e in r["results"] if not e["success"]]
    print(f"\nB) {fn.split('/')[-1]}: {len(eps)} failures of {len(r['results'])}")
    lost = [e for e in eps if e.get("lost")]
    if lost:
        opened = sum(e["lost"]["right_fingers_closure"] <= 0.15 for e in lost)
        print(f"   peg lost in {len(lost)}: fingers OPEN at the drop in {opened}, CLOSED in {len(lost) - opened}; "
              f"median step {np.median([e['lost']['step'] for e in lost]):.0f}, "
              f"median hand height above block {np.median([e['lost']['hand_above_block'] for e in lost]) * 1000:.0f} mm")
    stuck = [e for e in eps if not e.get("lost") and e.get("trace")]
    if stuck:
        fin = np.array([e["trace"][-1] for e in stuck])      # t, hand height, finger closure, peg z, tip offset
        print(f"   no drop, {len(stuck)} episodes, final state: hand {np.median(fin[:, 1]) * 1000:.0f} mm above block, "
              f"finger closure {np.median(fin[:, 2]):.2f}, tip offset {np.median(fin[:, 4]) * 1000:.1f} mm, "
              f"depth {np.median([e['depth'] for e in stuck]) * 1000:.0f} mm")
    for e in eps[:show]:
        tr = e.get("trace") or [[0, 0, 0, 0, 0]]
        print(f"     seed {e['seed']}: {e['reason'][:38]:38s} depth {e['depth'] * 1000:6.1f} mm  "
              f"final hand {tr[-1][1] * 1000:5.0f} mm, closure {tr[-1][2]:.2f}, tip {tr[-1][4] * 1000:5.1f} mm, "
              f"best tip {min(x[4] for x in tr) * 1000:5.1f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", nargs="*", default=[])
    a = ap.parse_args()
    if a.data:
        demo_dwell(a.data)
    for f in a.results:
        failures(f)
