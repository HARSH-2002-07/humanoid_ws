"""Scripted bimanual peg-in-hole expert.

  python expert.py                  # nominal episode, writes out/expert.mp4 + stills, prints metrics
  python expert.py --jitter 0.01    # randomise object xy by +-1 cm
  python expert.py --level L2 --seed 3   # harder: rotated block, random grip on the peg, 3 mm clearance (see difficulty.py)

Task: the LEFT hand picks up the block (square hole) and holds it in the air; the RIGHT hand picks up the
peg, carries it across (hand yawed inward), aligns it over the hole, lowers it in and lets go.
The expert uses ground-truth object poses from the simulator (privileged information). Its job is to produce
good demonstrations; the learned policy will only see camera images and joint states.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from difficulty import add_args, from_args, randomise_episode
from ik import ArmIK
from robot_utils import FINGERS, OUT, READY, ARM_JOINTS, load, set_pose

CLOSE = {"index": -0.6, "middle": -0.6, "thumb": 0.6}   # finger commands when gripping
YAW_R = 0.9          # right hand yaw (rad) while carrying the peg; the peg is round so yaw is free
PEG_HALF = 0.06      # peg half length
BLOCK_HALF = 0.03    # block wall half height (hole depth = 2 * BLOCK_HALF)
PEG_GRIP_ABOVE_CENTRE = 0.04   # the peg is gripped near its top so it hangs 0.10 m below the pinch point
PEG_BELOW_PINCH = PEG_HALF + PEG_GRIP_ABOVE_CENTRE


def smoothstep(x):
    return x * x * (3 - 2 * x)


class Expert:
    def __init__(self, m, d, hook=None, verbose=True, grip=(0.0, 0.0, PEG_GRIP_ABOVE_CENTRE)):
        self.m, self.d, self.hook, self.verbose = m, d, hook, verbose
        self.grip = np.asarray(grip, float)   # peg centre -> pinch point offset, world frame (hand yaw is 0 at the grasp)
        self.snapshot = None
        self.dt = m.opt.timestep
        self.ik = {s: ArmIK(m, s) for s in ("l_", "r_")}
        self.q = {s: np.array([READY[s + n] for n in ARM_JOINTS]) for s in self.ik}
        self.pos = {s: self.ik[s].site_pos(self.q[s]) for s in self.ik}
        self.yaw = {"l_": 0.0, "r_": 0.0}
        self.nstep = 0
        self.phase = 0

    # ------------------------------------------------------------------ low level
    def step(self):
        mujoco.mj_step(self.m, self.d)
        self.nstep += 1
        if self.hook:
            self.hook(self)

    def hold(self, T):
        for _ in range(int(T / self.dt)):
            self.step()

    def fingers(self, sides, mode, T):
        targets = {self.m.actuator(s + f).id: (CLOSE[f] if mode == "close" else 0.0)
                   for s in sides for f in FINGERS}
        start = {i: self.d.ctrl[i] for i in targets}
        n = max(1, int(T / self.dt))
        for k in range(1, n + 1):
            a = smoothstep(k / n)
            for i, v in targets.items():
                self.d.ctrl[i] = start[i] + a * (v - start[i])
            self.step()

    def plan(self, side, pos, yaw, n=12, label=""):
        """Cartesian straight-line path (position + yaw) from the current hand pose, solved with IK per waypoint."""
        ik = self.ik[side]
        pos = np.asarray(pos, float)
        p0, y0 = self.pos[side].copy(), self.yaw[side]
        qs = [self.q[side].copy()]
        q = qs[0]
        worst_p = worst_o = 0.0
        hit = False
        for k in range(1, n + 1):
            a = k / n
            q, ep, eo, h = ik.solve(q, p0 + a * (pos - p0), y0 + a * (yaw - y0), iters=100)
            worst_p, worst_o, hit = max(worst_p, ep), max(worst_o, eo), hit or h
            qs.append(q)
        self.pos[side], self.yaw[side], self.q[side] = pos.copy(), yaw, qs[-1]
        if self.verbose:
            flag = ""
            if worst_p > 2e-3 or worst_o > 0.05:
                flag += "  <-- IK NOT CONVERGED"
            if hit:
                flag += "  <-- ARM HITS TORSO/TABLE"
            print(f"  plan {label:22s} worst pos err {worst_p * 1000:5.1f} mm, ori err {worst_o:.3f}{flag}")
        return qs

    def play(self, plans, T):
        """Execute joint-space plans (dict side -> list of q) over T seconds with a smooth time profile."""
        n = max(1, int(T / self.dt))
        for k in range(1, n + 1):
            s = smoothstep(k / n)
            for side, qs in plans.items():
                idx = s * (len(qs) - 1)
                i = min(int(idx), len(qs) - 2)
                f = idx - i
                self.d.ctrl[self.ik[side].act] = qs[i] * (1 - f) + qs[i + 1] * f
            self.step()

    # ------------------------------------------------------------------ task helpers
    def peg_bottom(self):
        b = self.d.body("peg")
        ax = b.xmat.reshape(3, 3)[:, 2]
        if ax[2] < 0:
            ax = -ax
        return b.xpos - ax * PEG_HALF, ax

    def metrics(self):
        bottom, ax = self.peg_bottom()
        blk = self.d.body("block").xpos
        return {"depth": (blk[2] + BLOCK_HALF) - bottom[2],
                "lateral": float(np.linalg.norm((bottom - blk)[:2])),
                "tilt": float(np.arccos(np.clip(ax[2], -1, 1)))}

    def align_over_block(self, label):
        """Shift the right hand so the peg tip is centred over the hole (uses ground-truth poses)."""
        bottom, _ = self.peg_bottom()
        delta = self.d.body("block").xpos[:2] - bottom[:2]
        target = self.pos["r_"].copy()
        target[:2] += delta
        self.play({"r_": self.plan("r_", target, YAW_R, n=6, label=label)}, 1.0)
        self.hold(0.4)
        bottom, _ = self.peg_bottom()
        err = np.linalg.norm(self.d.body("block").xpos[:2] - bottom[:2])
        if self.verbose:
            print(f"  aligned over hole ({label}): peg-tip offset was {np.linalg.norm(delta) * 1000:.1f} mm, "
                  f"now {err * 1000:.1f} mm")

    # ------------------------------------------------------------------ the task
    def run(self):
        d, snap = self.d, self.snapshot
        self.hold(0.5)  # let everything settle
        blk0, peg0 = d.body("block").xpos.copy(), d.body("peg").xpos.copy()
        up = lambda z: np.array([0.0, 0.0, z])
        bx = d.body("block").xmat
        yaw_b = float(np.arctan2(bx[3], bx[0]))     # block yaw: the left hand turns to match it
        L_grasp = blk0.copy()
        R_grasp = peg0 + self.grip

        self.phase = 1

        print("1. reach above both objects")
        self.play({"l_": self.plan("l_", L_grasp + up(0.10), yaw_b, 10, "L above block"),
                   "r_": self.plan("r_", R_grasp + up(0.10), 0.0, 10, "R above peg")}, 3.0)
        self.phase = 2
        print("2. descend")
        self.play({"l_": self.plan("l_", L_grasp, yaw_b, 8, "L descend"),
                   "r_": self.plan("r_", R_grasp, 0.0, 8, "R descend")}, 1.5)
        self.phase = 3
        print("3. grip")
        self.fingers(["l_", "r_"], "close", 1.0)
        self.hold(0.5)

        self.phase = 4

        print("4. lift: left holds the block up, right lifts the peg and turns the hand inward")
        l_hold = self.pos["l_"] + up(0.05)
        hover_z = l_hold[2] + 0.16
        r_up = self.pos["r_"].copy()
        r_up[2] = hover_z
        self.play({"l_": self.plan("l_", l_hold, yaw_b, 8, "L lift block"),
                   "r_": self.plan("r_", r_up, YAW_R, 14, "R lift peg + yaw")}, 3.0)
        self.hold(0.5)

        self.phase = 5

        print("5. carry the peg across, above the block")
        r_over = self.d.body("block").xpos.copy()
        r_over[2] = hover_z
        r_over[:2] += self.pos["r_"][:2] - self.peg_bottom()[0][:2]   # compensate peg offset in the grip
        self.play({"r_": self.plan("r_", r_over, YAW_R, 12, "R carry over block")}, 2.5)
        self.hold(0.3)
        self.align_over_block("hover")
        if snap:
            snap("hover")

        self.phase = 6

        print("6. lower the peg toward the hole, re-align, then insert")
        bottom, _ = self.peg_bottom()
        block_top = d.body("block").xpos[2] + BLOCK_HALF
        mid = self.pos["r_"].copy()
        mid[2] += (block_top + 0.015) - bottom[2]          # peg tip 1.5 cm above the hole
        self.play({"r_": self.plan("r_", mid, YAW_R, 6, "R lower to hole")}, 1.5)
        self.hold(0.3)
        self.align_over_block("above hole")
        final = self.pos["r_"].copy()
        final[2] = d.body("block").xpos[2] + 0.08          # keeps the right hand 8 cm above the left hand
        self.play({"r_": self.plan("r_", final, YAW_R, 10, "R insert")}, 3.0)
        self.hold(0.5)
        held = self.metrics()
        if snap:
            snap("inserted_held")

        self.phase = 7

        print("7. release and retract")
        self.fingers(["r_"], "open", 1.0)
        self.hold(0.5)
        retract = self.pos["r_"].copy()
        retract[2] += 0.10
        self.play({"r_": self.plan("r_", retract, YAW_R, 6, "R retract")}, 1.5)
        self.hold(1.0)
        final_m = self.metrics()
        if snap:
            snap("final")

        ok = final_m["depth"] >= 0.045 and final_m["lateral"] < 0.01 and final_m["tilt"] < 0.15
        return {"success": bool(ok), "held": held, "final": final_m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    add_args(ap)
    args = ap.parse_args()
    diff = from_args(args)
    print("difficulty:", diff)

    m = load(diff.clearance)
    d = mujoco.MjData(m)
    set_pose(m, d, READY)
    grip = randomise_episode(m, d, np.random.default_rng(args.seed), diff)
    print(f"peg grip offset (dx, dy, dz): {np.round(grip * 1000, 1)} mm")
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, 480, 640)
    video = None if args.no_video else cv2.VideoWriter(
        str(OUT / "expert.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 480))
    seen = set()

    def hook(e):
        if e.nstep % 10 == 0:
            for i in range(d.ncon):
                c = d.contact[i]
                seen.add(tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name))))
        if video is not None and e.nstep % 17 == 0:
            frames = []
            for cam in ("head_cam", "overview"):
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render())
            video.write(cv2.cvtColor(np.hstack(frames), cv2.COLOR_RGB2BGR))

    def snapshot(name):
        renderer.update_scene(d, camera="head_cam")
        cv2.imwrite(str(OUT / f"expert_{name}_head.png"), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        renderer.update_scene(d, camera="overview")
        cv2.imwrite(str(OUT / f"expert_{name}_overview.png"), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    e = Expert(m, d, hook=hook, grip=grip)
    e.snapshot = snapshot
    res = e.run()

    print("\nresult")
    for k in ("held", "final"):
        r = res[k]
        print(f"  {k:5s}: insertion depth {r['depth'] * 1000:5.1f} mm (hole is 60), tip offset {r['lateral'] * 1000:5.1f} mm, "
              f"tilt {np.degrees(r['tilt']):4.1f} deg")
    print("  =>", "SUCCESS" if res["success"] else "FAIL")
    print("\ncontact pairs seen:")
    for p in sorted(seen):
        print("  ", p)
    if video is not None:
        video.release()
    renderer.close()
    print(f"\nsaved video and stills to {OUT}/ (expert.mp4, expert_*_head.png, expert_*_overview.png)")


if __name__ == "__main__":
    main()