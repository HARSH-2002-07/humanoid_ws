"""Damped-least-squares IK for one 7-joint arm, driving the hand's `pinch` site.

The hand orientation target is "fingers forward, pincer gap along y, hand x-axis up", optionally rotated by
`yaw` about the world z axis (positive = fingers turn toward +y). The redundant 7th joint is used to stay close
to the ready pose, which keeps the arm in a natural, torso-clearing posture and away from joint limits.
"""
import mujoco
import numpy as np

from robot_utils import ARM_JOINTS, READY

R_FWD = np.array([[0, 0, -1],
                  [0, 1, 0],
                  [1, 0, 0]], float)
W_ORI = 0.5


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class ArmIK:
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

    def site_pos(self, q):
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        d.qpos[self.qadr] = q
        mujoco.mj_fwdPosition(m, d)
        return d.site_xpos[self.site].copy()

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

    def solve(self, seed, pos, yaw=0.0, iters=150):
        """Returns (q, position error [m], orientation error, collides_with_torso_or_table)."""
        m, d = self.m, self.d
        R_des = rot_z(yaw) @ R_FWD
        d.qpos[:] = m.qpos0
        d.qpos[self.qadr] = seed
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

        def errors():
            Rc = d.site_xmat[self.site].reshape(3, 3)
            ep = pos - d.site_xpos[self.site]
            eo = 0.5 * sum(np.cross(Rc[:, i], R_des[:, i]) for i in range(3))
            return ep, eo

        for _ in range(iters):
            mujoco.mj_fwdPosition(m, d)
            ep, eo = errors()
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(eo) < 0.02:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], W_ORI * jacr[:, self.dadr]])
            e = np.concatenate([ep, W_ORI * eo])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            q = d.qpos[self.qadr]
            dq = Jp @ e + (np.eye(len(q)) - Jp @ J) @ (0.3 * (self.qrest - q))
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            d.qpos[self.qadr] = np.clip(q + dq, self.lo, self.hi)
        mujoco.mj_fwdPosition(m, d)
        ep, eo = errors()
        return d.qpos[self.qadr].copy(), np.linalg.norm(ep), np.linalg.norm(eo), self.collides()
