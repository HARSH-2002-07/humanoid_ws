"""Collect expert demonstrations with randomised object positions.

  python collect.py --episodes 20 --jitter 0.01                  # quick test: prints the success rate
  python collect.py --episodes 300 --jitter 0.015 --out ../data  # real collection

One HDF5 file per successful episode (ep_<seed>.hdf5):
  images  (T, 144, 192, 3) uint8   head camera, recorded at --hz (default 10 Hz)
  qpos    (T, 22) float32          joint position of every actuated joint (order = attrs['actuators'])
  ctrl    (T, 22) float32          the expert's joint-position targets in effect at that frame
                                   (policy target: ctrl[t+1 : t+1+H] given the observation at t)
  obj     (T, 14) float32          peg pos+quat, block pos+quat (privileged; for analysis, not for the policy)
  phase   (T,)    int8             expert phase 1..7 (grasp, lift, carry, insert, ...)
"""
import argparse
import contextlib
from dataclasses import asdict
import io
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
import mujoco
import numpy as np

from difficulty import add_args, from_args, randomise_episode
from expert import Expert
from robot_utils import OUT, READY, load, set_pose

IMG_H, IMG_W = 144, 192


def save_episode(path, buf, attrs):
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=np.stack(buf["img"]), chunks=(1, IMG_H, IMG_W, 3),
                         compression="gzip", compression_opts=4)
        f.create_dataset("qpos", data=np.stack(buf["qpos"]))
        f.create_dataset("ctrl", data=np.stack(buf["ctrl"]))
        f.create_dataset("obj", data=np.stack(buf["obj"]))
        f.create_dataset("phase", data=np.array(buf["phase"], dtype=np.int8))
        for k, v in attrs.items():
            f.attrs[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--hz", type=float, default=10.0, help="recording rate")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--save-failed", action="store_true", help="also save failed episodes to <out>/failed")
    add_args(ap)
    args = ap.parse_args()
    diff = from_args(args)
    print("difficulty:", diff)

    m = load(diff.clearance)
    act_qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    act_names = [m.actuator(i).name for i in range(m.nu)]
    renderer = mujoco.Renderer(m, IMG_H, IMG_W)
    rec_every = int(round(1.0 / (args.hz * m.opt.timestep)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "failed").mkdir(exist_ok=True)

    n_ok, n_total, t_start, preview_done = 0, 0, time.time(), False
    for i in range(args.episodes):
        seed = args.seed0 + i
        rng = np.random.default_rng(seed)
        d = mujoco.MjData(m)
        set_pose(m, d, READY)
        grip = randomise_episode(m, d, rng, diff)
        mujoco.mj_forward(m, d)

        buf = {"img": [], "qpos": [], "ctrl": [], "obj": [], "phase": []}

        def hook(e):
            if e.nstep % rec_every:
                return
            renderer.update_scene(d, camera="head_cam")
            buf["img"].append(renderer.render().copy())
            buf["qpos"].append(d.qpos[act_qadr].astype(np.float32))
            buf["ctrl"].append(d.ctrl.astype(np.float32))
            buf["obj"].append(np.concatenate([d.body("peg").xpos, d.body("peg").xquat,
                                              d.body("block").xpos, d.body("block").xquat]).astype(np.float32))
            buf["phase"].append(e.phase)

        t0 = time.time()
        res, err = None, ""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = Expert(m, d, hook=hook, verbose=False, grip=grip).run()
        except Exception as ex:  # an IK / physics problem counts as a failed episode
            err = f"{type(ex).__name__}: {ex}"
        n_total += 1
        ok = bool(res and res["success"])
        n_ok += ok
        if res:
            f = res["final"]
            msg = f"depth {f['depth'] * 1000:5.1f} mm, tip offset {f['lateral'] * 1000:5.1f} mm, tilt {np.degrees(f['tilt']):4.1f} deg"
        else:
            msg = err
        print(f"ep {i:4d} seed {seed:5d}: {'SUCCESS' if ok else 'FAIL   '}  frames {len(buf['img']):3d}  {msg}  ({time.time() - t0:4.1f} s)")

        if res and (ok or args.save_failed):
            attrs = {"success": ok, "seed": seed, "level": args.level, "hz": args.hz, "grip": grip,
                     **{f"diff_{k}": v for k, v in asdict(diff).items()},
                     "depth": res["final"]["depth"], "lateral": res["final"]["lateral"], "tilt": res["final"]["tilt"],
                     "actuators": np.array(act_names, dtype="S")}
            save_episode((out if ok else out / "failed") / f"ep_{seed:05d}.hdf5", buf, attrs)
            if ok and not preview_done:  # contact sheet of the first good episode
                idx = np.linspace(0, len(buf["img"]) - 1, 8).astype(int)
                tiles = [cv2.cvtColor(buf["img"][k], cv2.COLOR_RGB2BGR) for k in idx]
                sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
                cv2.imwrite(str(OUT / "collect_preview.png"), sheet)
                preview_done = True

    renderer.close()
    print(f"\n{n_ok}/{n_total} episodes succeeded ({100 * n_ok / max(n_total, 1):.0f}%) in {time.time() - t_start:.0f} s; "
          f"saved to {out}")
    if preview_done:
        print(f"contact sheet of the first good episode: {OUT}/collect_preview.png")


if __name__ == "__main__":
    main()