"""How hard can the task get before the scripted expert itself fails?

  python sweep_expert.py --episodes 20

Runs the expert (no rendering, so it's fast) on a grid of difficulty settings and prints a table. A setting is only
useful for imitation learning if the expert solves it reliably, so this decides which levels we can train on.
"""
import argparse
import contextlib
import io
import time
from dataclasses import replace

import mujoco
import numpy as np

from difficulty import LEVELS, Difficulty, randomise_episode
from expert import Expert
from robot_utils import READY, load, set_pose

L1 = LEVELS["L1"]
GRID = [
    ("L0  5 mm, baseline", LEVELS["L0"]),
    ("5 mm, random grip only", Difficulty(grasp_dx=0.01, grasp_dy=0.005, grasp_dz_lo=0.03, grasp_dz_hi=0.04)),
    ("5 mm, block yaw only", Difficulty(block_yaw=0.2)),
    ("5 mm, spread 4 cm", Difficulty(jitter=0.04)),
    ("L1  5 mm, all of the above", L1),
    ("L2  3 mm", LEVELS["L2"]),
    ("L3  2 mm", LEVELS["L3"]),
    ("L4  1 mm", LEVELS["L4"]),
]


def run_config(diff, n, seed0):
    m = load(diff.clearance)
    out = {"ok": 0, "tip": [], "held_depth": [], "reasons": {}}
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        d = mujoco.MjData(m)
        set_pose(m, d, READY)
        grip = randomise_episode(m, d, rng, diff)
        mujoco.mj_forward(m, d)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = Expert(m, d, verbose=False, grip=grip).run()
        except Exception as ex:
            out["reasons"][f"exception: {type(ex).__name__}"] = out["reasons"].get(f"exception: {type(ex).__name__}", 0) + 1
            continue
        out["tip"].append(res["final"]["lateral"] * 1000)
        out["held_depth"].append(res["held"]["depth"] * 1000)
        if res["success"]:
            out["ok"] += 1
        else:
            if res["held"]["depth"] < 0.03:
                why = "peg did not enter the hole (jammed on the rim?)"
            elif res["final"]["lateral"] >= 0.01 or res["final"]["depth"] < 0.045:
                why = "entered but ended up out of the hole after release"
            else:
                why = "tilted in the hole"
            out["reasons"][why] = out["reasons"].get(why, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    print(f"{'setting':32s} {'success':>9s} {'tip offset (mm)':>16s} {'depth while held (mm)':>22s}   time")
    for name, diff in GRID:
        t0 = time.time()
        r = run_config(diff, args.episodes, args.seed0)
        tip = f"{np.mean(r['tip']):4.1f} (max {np.max(r['tip']):4.1f})" if r["tip"] else "-"
        dep = f"{np.mean(r['held_depth']):4.1f} (min {np.min(r['held_depth']):4.1f})" if r["held_depth"] else "-"
        print(f"{name:32s} {r['ok']:3d}/{args.episodes:<5d} {tip:>16s} {dep:>22s}   {time.time() - t0:3.0f} s")
        for why, c in sorted(r["reasons"].items()):
            print(f"{'':34s}{c} x {why}")


if __name__ == "__main__":
    main()
