"""Collect demonstrations, train and evaluate a policy for one difficulty level, in one command. Safe to re-run:
steps whose outputs already exist are skipped.

  python run_level.py L1
  python run_level.py L4 --episodes 300 --steps 15000 --eval-episodes 50

Outputs (relative to the project root, i.e. one level above sim/):
  data_<level>/ep_*.hdf5     demonstrations
  ckpt/bc_<level>.pt         trained policy
  results/<level>.json       per-episode evaluation results (success, failure reason, grasp diagnostics)
"""
import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("level", help="L0 .. L4 (see difficulty.py)")
    ap.add_argument("--episodes", type=int, default=300, help="demonstrations to collect")
    ap.add_argument("--steps", type=int, default=15000, help="training steps")
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--exec", type=int, default=5, help="actions executed per re-plan during evaluation")
    ap.add_argument("--retrain", action="store_true", help="train again even if a checkpoint exists")
    args = ap.parse_args()

    L = args.level
    data, ckpt, res = ROOT / f"data_{L}", ROOT / "ckpt" / f"bc_{L}.pt", ROOT / "results" / f"{L}_exec{args.exec}.json"
    py, t0 = sys.executable, time.time()

    n_have = len(glob.glob(str(data / "ep_*.hdf5")))
    if n_have >= args.episodes:
        print(f"[1/3] {n_have} demonstrations already in {data}, skipping collection")
    else:
        run([py, "collect.py", "--level", L, "--episodes", args.episodes, "--out", data])

    if ckpt.exists() and not args.retrain:
        print(f"[2/3] checkpoint {ckpt} exists, skipping training (use --retrain to redo)")
    else:
        run([py, "train_bc.py", "--data", data, "--out", ckpt, "--steps", args.steps])

    run([py, "eval_policy.py", "--ckpt", ckpt, "--level", L, "--episodes", args.eval_episodes,
         "--exec", args.exec, "--json", res])
    print(f"\n{L} finished in {(time.time() - t0) / 60:.0f} min; results in {res}")


if __name__ == "__main__":
    main()
