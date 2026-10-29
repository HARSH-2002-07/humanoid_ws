"""Sanity-check the upper-body robot.

  python check_robot.py            # headless: load, settle in ready pose, report, save images to ./out/
  python check_robot.py --view     # same, then open the interactive MuJoCo viewer
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--view", action="store_true", help="open the interactive viewer afterwards")
args = ap.parse_args()
if not args.view:
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; the viewer needs the default GLFW

import cv2
import mujoco
import numpy as np

from robot_utils import OUT, READY, XML, set_pose


def contact_pairs(m, d):
    pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.body(m.geom_bodyid[c.geom1]).name
        b2 = m.body(m.geom_bodyid[c.geom2]).name
        pairs.add(tuple(sorted((b1, b2))))
    return sorted(pairs)


def main():
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    print(f"loaded: nq={m.nq} nv={m.nv} nu={m.nu} ncam={m.ncam} timestep={m.opt.timestep}")
    print("actuators:", [m.actuator(i).name for i in range(m.nu)])

    set_pose(m, d, READY)
    mujoco.mj_forward(m, d)
    print("\ncontacts at t=0 (expected: peg/block with table only):")
    for p in contact_pairs(m, d):
        print("  ", p)

    # settle for 2 s of sim time
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    if not np.isfinite(d.qpos).all():
        print("\nFAIL: simulation went unstable (NaN/inf).")
        sys.exit(1)

    err = {n: abs(d.qpos[m.jnt_qposadr[m.actuator_trnid[m.actuator(n).id, 0]]] - v) for n, v in READY.items()}
    worst = max(err, key=err.get)
    print(f"\nafter 2 s: worst joint tracking error {err[worst]:.3f} rad ({worst})")
    for obj in ("peg", "block"):
        print(f"{obj} position:", np.round(d.body(obj).xpos, 3))
    for h in ("l_hand", "r_hand"):
        print(f"{h} (wrist) position:", np.round(d.body(h).xpos, 3))
    print("contacts after settling:")
    for p in contact_pairs(m, d):
        print("  ", p)

    # render both cameras
    r = mujoco.Renderer(m, 480, 640)
    for cam in ("head_cam", "overview"):
        r.update_scene(d, camera=cam)
        img = r.render()
        cv2.imwrite(str(OUT / f"{cam}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    r.close()  # avoids the EGL cleanup traceback at exit
    print(f"\nsaved images to {OUT}/head_cam.png and overview.png")

    if args.view:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(m, d) as v:
            while v.is_running():
                t0 = time.time()
                mujoco.mj_step(m, d)
                v.sync()
                time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    main()
