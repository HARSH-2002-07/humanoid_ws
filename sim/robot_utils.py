"""Shared helpers for the upper-body robot scripts."""
from pathlib import Path

import mujoco

XML = Path(__file__).resolve().parent.parent / "assets" / "upper_body.xml"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

ARM_JOINTS = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_dev"]
FINGERS = ["index", "middle", "thumb"]

# ready pose (found with a collision-aware numeric search): elbows forward of the chest, hands above
# and slightly behind the objects, fingers open. Right arm is the mirror image of the left.
READY_LEFT = [-0.234, 0.045, -0.645, -2.067, 0.22, 0.646, 0.759]   # order of ARM_JOINTS
READY_RIGHT = [-0.234, -0.045, 0.645, -2.067, -0.22, 0.646, -0.759]
READY = {"head_pan": 0.0, "head_tilt": 0.0}
for _side, _q in (("l_", READY_LEFT), ("r_", READY_RIGHT)):
    READY.update({_side + n: v for n, v in zip(ARM_JOINTS, _q)})
    READY.update({_side + f: 0.0 for f in FINGERS})


def load(clearance=None):
    """Load the robot. `clearance` (m, per side) rebuilds the block's hole on the fly; None/0.005 uses the shipped XML."""
    if clearance is None or abs(clearance - 0.005) < 1e-9:
        return mujoco.MjModel.from_xml_path(str(XML))
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_robot", XML.parent / "build_robot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mujoco.MjModel.from_xml_string(mod.build(clearance=clearance))


def set_pose(m, d, pose):
    """Teleport joints to `pose` (actuator-name -> target) and set matching controls."""
    for name, val in pose.items():
        a = m.actuator(name)
        d.ctrl[a.id] = val
        jid = m.actuator_trnid[a.id, 0]
        d.qpos[m.jnt_qposadr[jid]] = val
        if name.endswith(tuple(FINGERS)):  # coupled distal joint
            j2 = m.joint(m.joint(jid).name.replace("_j1", "_j2")).id
            d.qpos[m.jnt_qposadr[j2]] = -val   # distal joint counter-rotates (parallel pads)