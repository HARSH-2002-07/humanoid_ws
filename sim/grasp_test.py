"""Both hands: IK reach -> descend onto object -> close fingers -> lift. Left hand takes the block, right hand the peg.

  python grasp_test.py

Writes out/grasp_test.mp4 (head camera | overview) and prints a PASS/FAIL per hand.
Hands point forward with the pincer gap along world y; they come down on the object from above.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from robot_utils import ARM_JOINTS, FINGERS, OUT, READY, load, set_pose

# desired hand orientation: fingers point +x (forward), pincer gap along +y, hand "x" axis up
R_DES = np.array([[0, 0, -1],
                  [0, 1, 0],
                  [1, 0, 0]], float)
W_ORI = 0.5

OBJECT = {"l_": "block", "r_": "peg"}
CLOSE = {"index": -0.6, "middle": -0.6, "thumb": 0.6}   # symmetric: the thumb actuator is 2x as stiff


class ArmIK:
    """Damped-least-squares IK for the 7 arm joints, driving the hand's `pinch` site.

    The redundant 7th joint is used to stay close to `qrest` (the ready pose) so the arm keeps a natural,
    torso-clearing posture and stays away from joint limits."""

    def __init__(self, m, side):
        self.m, self.side = m, side
        self.d = mujoco.MjData(m)
        self.site = m.site(side + "pinch").id
        jids = [m.joint(side + n).id for n in ARM_JOINTS]
        self.qadr = [m.jnt_qposadr[j] for j in jids]
        self.dadr = [m.jnt_dofadr[j] for j in jids]
        self.lo, self.hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
        self.act = [m.actuator(side + n).id for n in ARM_JOINTS]
        self.qrest = np.array([READY[side + n] for n in ARM_JOINTS])

    def collides(self):
        """True if the arm (as posed in self.d) touches the torso or table."""
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
            if {b1, b2} & {"torso", "table"} and any(b.startswith(("l_", "r_")) for b in (b1, b2)):
                return True
        return False

    def solve(self, seed, pos, iters=400):
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        d.qpos[self.qadr] = seed
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_fwdPosition(m, d)
            Rc = d.site_xmat[self.site].reshape(3, 3)
            ep = pos - d.site_xpos[self.site]
            eo = 0.5 * sum(np.cross(Rc[:, i], R_DES[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(eo) < 0.02:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], W_ORI * jacr[:, self.dadr]])
            e = np.concatenate([ep, W_ORI * eo])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            q = d.qpos[self.qadr]
            dq = Jp @ e + (np.eye(len(q)) - Jp @ J) @ (0.3 * (self.qrest - q))   # null-space pull to ready pose
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.lo, self.hi)
        mujoco.mj_fwdPosition(m, d)
        ep = pos - d.site_xpos[self.site]
        Rc = d.site_xmat[self.site].reshape(3, 3)
        eo = 0.5 * sum(np.cross(Rc[:, i], R_DES[:, i]) for i in range(3))
        return d.qpos[self.qadr].copy(), np.linalg.norm(ep), np.linalg.norm(eo), self.collides()


def main():
    m = load()
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    set_pose(m, d, READY)
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, 480, 640)
    video = cv2.VideoWriter(str(OUT / "grasp_test.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 480))
    seen_pairs, counter = set(), [0]

    def hook():
        counter[0] += 1
        if counter[0] % 10 == 0:
            for i in range(d.ncon):
                c = d.contact[i]
                pair = tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)))
                seen_pairs.add(pair)
        if counter[0] % 17 == 0:  # ~30 fps
            frames = []
            for cam in ("head_cam", "overview"):
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render())
            video.write(cv2.cvtColor(np.hstack(frames), cv2.COLOR_RGB2BGR))

    def snapshot(name):
        renderer.update_scene(d, camera="head_cam")
        cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    def ramp(target, T):
        ids = list(target)
        start = {i: d.ctrl[i] for i in ids}
        n = max(1, int(T / dt))
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)  # smoothstep
            for i in ids:
                d.ctrl[i] = start[i] + a * (target[i] - start[i])
            mujoco.mj_step(m, d)
            hook()

    ik = {s: ArmIK(m, s) for s in OBJECT}
    obj0 = {s: d.body(OBJECT[s]).xpos.copy() for s in OBJECT}
    seed = {s: np.array([d.ctrl[a] for a in ik[s].act]) for s in OBJECT}

    def reach(label, offsets, T):
        """Solve IK for both hands (object centre + offset) and ramp the arm controls there."""
        target = {}
        for s in OBJECT:
            q, ep, eo, hit = ik[s].solve(seed[s], obj0[s] + np.array(offsets))
            flag = "" if (ep < 2e-3 and eo < 0.05) else "   <-- IK NOT CONVERGED"
            flag += "   <-- ARM HITS TORSO/TABLE" if hit else ""
            print(f"  {label:9s} {s}: pos err {ep * 1000:5.1f} mm, ori err {eo:.3f}{flag}")
            seed[s] = q
            target.update({a: v for a, v in zip(ik[s].act, q)})
        ramp(target, T)

    print("planning + executing")
    ramp({}, 0.5)
    reach("pre-grasp", (0, 0, 0.10), 3.0)
    snapshot("pregrasp_head")
    reach("grasp", (0, 0, 0.0), 1.5)
    snapshot("grasp_head")

    close = {}
    for s in OBJECT:
        for f in FINGERS:
            close[m.actuator(s + f).id] = CLOSE[f]
    ramp(close, 1.0)
    ramp({}, 0.5)

    gripped = {s: d.body(OBJECT[s]).xpos.copy() for s in OBJECT}
    reach("lift", (0, 0, 0.12), 1.5)
    ramp({}, 1.0)
    snapshot("lift_head")

    print("\nresults")
    for s in OBJECT:
        pos = d.body(OBJECT[s]).xpos
        gain = pos[2] - obj0[s][2]
        dist = np.linalg.norm(pos - d.site_xpos[ik[s].site])
        ok = gain > 0.08 and dist < 0.03
        print(f"  {OBJECT[s]:5s} ({s}hand): lifted {gain * 100:5.1f} cm, "
              f"offset from pinch site {dist * 1000:5.1f} mm, drift while gripping "
              f"{np.linalg.norm(gripped[s] - obj0[s]) * 1000:4.1f} mm -> {'PASS' if ok else 'FAIL'}")
    print("\ncontact pairs seen during the run:")
    for p in sorted(seen_pairs):
        print("  ", p)

    video.release()
    renderer.close()
    print(f"\nsaved {OUT}/grasp_test.mp4 and pregrasp_head.png, grasp_head.png, lift_head.png")


if __name__ == "__main__":
    main()