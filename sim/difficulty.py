"""Difficulty knobs for the bimanual peg-in-hole task, and named levels L0..L4.

  clearance   gap between peg and hole wall, per side (m).      5 mm = easy, 1 mm = hard
  jitter      object xy position spread, +- (m)
  block_yaw   initial block rotation about the vertical, +- (rad); the left hand has to match it
  grasp_dx/dy/dz_lo/dz_hi
              how the peg is gripped: offset along the pads (dx), sideways (dy, the grip re-centres it),
              and height above the peg centre (dz in [lo, hi]). A random grip means the peg tip ends up
              somewhere new each time, so the right hand has to correct for it (the expert measures it;
              the learned policy has to see it).
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Difficulty:
    clearance: float = 0.005
    jitter: float = 0.02
    block_yaw: float = 0.0
    grasp_dx: float = 0.0
    grasp_dy: float = 0.0
    grasp_dz_lo: float = 0.04
    grasp_dz_hi: float = 0.04


_L1 = Difficulty(jitter=0.03, block_yaw=0.2, grasp_dx=0.01, grasp_dy=0.005, grasp_dz_lo=0.03, grasp_dz_hi=0.04)
LEVELS = {
    "L0": Difficulty(),                              # the original setup (all 100% so far)
    "L1": _L1,                                       # wider spread, rotated block, random grip, 5 mm clearance
    "L2": replace(_L1, clearance=0.003),
    "L3": replace(_L1, clearance=0.002),
    "L4": replace(_L1, clearance=0.001),
}
_OVERRIDES = ("clearance", "jitter", "block_yaw", "grasp_dx", "grasp_dy", "grasp_dz_lo", "grasp_dz_hi")


def add_args(ap):
    ap.add_argument("--level", default="L0", choices=sorted(LEVELS), help="named difficulty level (see difficulty.py)")
    for k in _OVERRIDES:
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=None, help=f"override {k} of the level")


def from_args(args):
    over = {k: getattr(args, k) for k in _OVERRIDES if getattr(args, k, None) is not None}
    return replace(LEVELS[args.level], **over)


def randomise_episode(m, d, rng, diff):
    """Place the objects for one episode (xy jitter, block yaw) and draw the expert's grip offset on the peg.

    The random numbers are drawn in a fixed order, so for the default level the object positions are identical
    to the ones used for the first dataset. Returns the grip offset (dx, dy, dz) from the peg centre to the pinch point.
    """
    for name in ("peg", "block"):
        adr = m.jnt_qposadr[m.joint(name + "_free").id]
        d.qpos[adr:adr + 2] += rng.uniform(-diff.jitter, diff.jitter, 2)
    yaw = rng.uniform(-diff.block_yaw, diff.block_yaw)
    adr = m.jnt_qposadr[m.joint("block_free").id]
    d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    grip = np.array([rng.uniform(-diff.grasp_dx, diff.grasp_dx),
                     rng.uniform(-diff.grasp_dy, diff.grasp_dy),
                     rng.uniform(diff.grasp_dz_lo, diff.grasp_dz_hi)])
    return grip
