"""Generate upper_body.xml (MuJoCo MJCF) for the humanoid upper-body bimanual robot.

Run:  python build_robot.py      -> writes upper_body.xml next to this file.

Robot layout (world frame: +x forward from the chest, +y to the robot's left, +z up):
  * torso  : fixed box on a pedestal
  * head   : sphere on a neck, 2 DOF (pan/tilt), RGB camera "head_cam" in the front
  * arms   : 7 DOF each  (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/deviation)
  * hands  : 3 fingers each (index, middle, thumb), 1 actuator per finger
             (the distal joint counter-rotates against the proximal one, so the fingertip pads stay parallel). Pincer closes along the hand's local y axis.
  * scene  : table, a vertical peg (right side) and a block with a square vertical hole (left side)
"""
from pathlib import Path

# ----------------------------------------------------------------------------- parameters
TORSO_HALF = (0.09, 0.15, 0.25)      # half sizes x, y, z
TORSO_Z = 1.0                        # torso centre height
SHOULDER_Y = 0.24                    # |y| of shoulder joints
SHOULDER_DZ = 0.10                   # shoulder height above torso centre
UPPER = 0.33                         # upper-arm length
FORE = 0.30                          # forearm length
TABLE_TOP = 0.80

# camera tilted 58 deg below horizontal, narrow field of view (work area fills more of the image)
CAM_XYAXES = "0 -1 0 0.848 0 0.530"


def finger(prefix, name, x, y, curl_sign):
    """One 2-link finger. curl_sign=-1: curls toward -y (fingers on +y side), +1: toward +y (thumb)."""
    lo, hi = (-1.6, 0.0) if curl_sign < 0 else (0.0, 1.6)
    return f"""
          <body name="{prefix}{name}_prox" pos="{x} {y} -0.06" gravcomp="1">
            <joint name="{prefix}{name}_j1" class="finger_joint" axis="1 0 0" range="{lo} {hi}"/>
            <geom class="finger_geom" fromto="0 0 0 0 0 -0.05" size="0.008"/>
            <body name="{prefix}{name}_dist" pos="0 0 -0.05" gravcomp="1">
              <joint name="{prefix}{name}_j2" class="finger_joint" axis="1 0 0" range="{0.0 - hi} {0.0 - lo}"/>
              <geom class="finger_geom" fromto="0 0 0 0 0 -0.045" size="0.0075"/>
            </body>
          </body>"""


def arm(prefix, s):
    """s = +1 for the left arm (y>0), -1 for the right arm (y<0)."""
    roll_lo, roll_hi = (-1.2, 2.0) if s > 0 else (-2.0, 1.2)
    return f"""
      <body name="{prefix}upper_arm" pos="0 {s * SHOULDER_Y} {SHOULDER_DZ}" gravcomp="1">
        <joint name="{prefix}sh_pitch" class="arm_joint" axis="0 1 0" range="-2.8 1.0"/>
        <joint name="{prefix}sh_roll" class="arm_joint" axis="1 0 0" range="{roll_lo} {roll_hi}"/>
        <joint name="{prefix}sh_yaw" class="arm_joint" axis="0 0 1" range="-1.5 1.5"/>
        <geom type="sphere" size="0.045" contype="0" conaffinity="0" rgba="0.35 0.38 0.45 1"/>
        <geom class="link_geom" fromto="0 0 0 0 0 -{UPPER}" size="0.035"/>
        <body name="{prefix}forearm" pos="0 0 -{UPPER}" gravcomp="1">
          <joint name="{prefix}elbow" class="arm_joint" axis="0 1 0" range="-2.4 0.05"/>
          <geom type="sphere" size="0.038" contype="0" conaffinity="0" rgba="0.35 0.38 0.45 1"/>
          <geom class="link_geom" fromto="0 0 0 0 0 -{FORE}" size="0.03"/>
          <body name="{prefix}hand" pos="0 0 -{FORE}" gravcomp="1">
            <joint name="{prefix}wrist_roll" class="wrist_joint" axis="0 0 1" range="-1.6 1.6"/>
            <joint name="{prefix}wrist_pitch" class="wrist_joint" axis="0 1 0" range="-1.0 1.0"/>
            <joint name="{prefix}wrist_dev" class="wrist_joint" axis="1 0 0" range="-1.2 1.2"/>
            <geom name="{prefix}palm" class="link_geom" type="box" pos="0 0 -0.03" size="0.02 0.045 0.03" rgba="0.85 0.7 0.6 1"/>
            {finger(prefix, "index", 0.012, 0.045, -1)}
            {finger(prefix, "middle", -0.012, 0.045, -1)}
            {finger(prefix, "thumb", 0.0, -0.045, +1)}
            <site name="{prefix}pinch" pos="0 0 -0.125" size="0.008" rgba="1 0.9 0 0.6"/>
          </body>
        </body>
      </body>"""


def actuators(prefix):
    out = []
    for j, cls in (("sh_pitch", "shoulder"), ("sh_roll", "shoulder"), ("sh_yaw", "shoulder"),
                   ("elbow", "elbow"), ("wrist_roll", "wrist"), ("wrist_pitch", "wrist"), ("wrist_dev", "wrist")):
        out.append(f'    <position name="{prefix}{j}" joint="{prefix}{j}" class="{cls}"/>')
    out.append(f'    <position name="{prefix}index" joint="{prefix}index_j1" class="finger_act" ctrlrange="-1.6 0"/>')
    out.append(f'    <position name="{prefix}middle" joint="{prefix}middle_j1" class="finger_act" ctrlrange="-1.6 0"/>')
    out.append(f'    <position name="{prefix}thumb" joint="{prefix}thumb_j1" class="thumb_act" ctrlrange="0 1.6"/>')
    return "\n".join(out)


def equalities(prefix):
    return "\n".join(
        f'    <joint joint1="{prefix}{f}_j2" joint2="{prefix}{f}_j1" polycoef="0 -1 0 0 0"/>'
        for f in ("index", "middle", "thumb"))


def block_geoms(clearance):
    """Four walls + floor plate. Outer size is fixed (0.06 x 0.06 x 0.06 plus plate); the hole shrinks with `clearance`.
    clearance = gap between the 0.015 m peg radius and the hole wall, per side."""
    h = 0.015 + clearance           # inner half width of the square hole
    t = (0.03 - h) / 2              # wall half thickness (outer half width stays 0.03)
    c = h + t                       # wall centre offset
    common = 'mass="0.0375" rgba="0.2 0.4 0.9 1" friction="1.5 0.05 0.001" condim="4"'
    return f"""      <geom name="wall_xp" type="box" pos="{c:.5f} 0 0" size="{t:.5f} 0.03 0.03" {common}/>
      <geom name="wall_xn" type="box" pos="{-c:.5f} 0 0" size="{t:.5f} 0.03 0.03" {common}/>
      <geom name="plate" type="box" pos="0 0 -0.035" size="0.03 0.03 0.005" mass="0.03" rgba="0.2 0.4 0.9 1" friction="1.5 0.05 0.001" condim="4"/>
      <geom name="wall_yp" type="box" pos="0 {c:.5f} 0" size="{h:.5f} {t:.5f} 0.03" {common}/>
      <geom name="wall_yn" type="box" pos="0 {-c:.5f} 0" size="{h:.5f} {t:.5f} 0.03" {common}/>"""


def build(clearance=0.005):
    tx, ty, tz = TORSO_HALF
    return f"""<mujoco model="upper_body">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.75 0.9" rgb2="0.1 0.12 0.2" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.27 0.3" rgb2="0.3 0.32 0.35" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
  </asset>

  <default>
    <default class="link_geom">
      <geom type="capsule" rgba="0.6 0.65 0.75 1" density="500"/>
    </default>
    <default class="arm_joint">
      <joint damping="1.0" armature="0.01"/>
    </default>
    <default class="wrist_joint">
      <joint damping="0.2" armature="0.005"/>
    </default>
    <default class="finger_joint">
      <joint damping="0.01" armature="0.002"/>
    </default>
    <default class="finger_geom">
      <geom type="capsule" rgba="0.85 0.7 0.6 1" density="800" friction="2 0.1 0.01" condim="4"/>
    </default>
    <default class="shoulder"><position kp="150" kv="15" forcerange="-80 80" ctrlrange="-2.8 2.8"/></default>
    <default class="elbow"><position kp="100" kv="10" forcerange="-60 60" ctrlrange="-2.4 0.05"/></default>
    <default class="wrist"><position kp="20" kv="2" forcerange="-20 20" ctrlrange="-1.6 1.6"/></default>
    <default class="finger_act"><position kp="3" kv="0.2" forcerange="-6 6"/></default>
    <default class="thumb_act"><position kp="6" kv="0.4" forcerange="-12 12"/></default>  <!-- 2x a finger: balances two fingers -->
  </default>

  <worldbody>
    <light pos="0.5 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>
    <camera name="overview" pos="1.4 0 1.5" mode="targetbody" target="torso"/>

    <!-- table -->
    <body name="table" pos="0.55 0 {TABLE_TOP / 2}">
      <geom name="table_top" type="box" size="0.25 0.45 {TABLE_TOP / 2}" rgba="0.55 0.45 0.35 1" friction="1 0.005 0.0001"/>
    </body>

    <!-- pedestal (visual only) -->
    <geom type="box" pos="0 0 {(TORSO_Z - tz) / 2}" size="0.07 0.1 {(TORSO_Z - tz) / 2}" rgba="0.3 0.3 0.35 1" contype="0" conaffinity="0"/>

    <!-- torso -->
    <body name="torso" pos="0 0 {TORSO_Z}">
      <geom name="torso_box" type="box" size="{tx} {ty} {tz}" rgba="0.25 0.35 0.6 1"/>

      <!-- head: pan/tilt, camera in the front. positive tilt = look down -->
      <body name="head" pos="0 0 {tz + 0.02}" gravcomp="1">
        <joint name="head_pan" axis="0 0 1" range="-1.2 1.2" damping="0.5" armature="0.005"/>
        <joint name="head_tilt" axis="0 1 0" range="-0.6 0.8" damping="0.5" armature="0.005"/>
        <geom type="capsule" fromto="0 0 -0.02 0 0 0.04" size="0.03" rgba="0.8 0.7 0.6 1" contype="0" conaffinity="0"/>
        <geom type="sphere" pos="0 0 0.12" size="0.09" rgba="0.9 0.8 0.7 1" contype="0" conaffinity="0"/>
        <geom type="sphere" pos="0.08 0.035 0.13" size="0.012" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
        <geom type="sphere" pos="0.08 -0.035 0.13" size="0.012" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
        <camera name="head_cam" pos="0.095 0 0.12" xyaxes="{CAM_XYAXES}" fovy="50"/>
      </body>
{arm("l_", +1)}
{arm("r_", -1)}
    </body>

    <!-- vertical peg on the right side of the table -->
    <body name="peg" pos="0.40 -0.08 {TABLE_TOP + 0.061}">
      <freejoint name="peg_free"/>
      <geom name="peg_geom" type="cylinder" size="0.015 0.06" mass="0.06" rgba="0.9 0.2 0.2 1"
            friction="1.5 0.05 0.001" condim="4"/>
    </body>

    <!-- block with a square vertical hole (4 walls, 0.04 m inner width, 0.03 m peg diameter) on the left -->
    <body name="block" pos="0.40 0.08 {TABLE_TOP + 0.041}">
      <freejoint name="block_free"/>
{block_geoms(clearance)}
    </body>
  </worldbody>

  <equality>
{equalities("l_")}
{equalities("r_")}
  </equality>

  <actuator>
    <position name="head_pan" joint="head_pan" kp="10" kv="1" ctrlrange="-1.2 1.2" forcerange="-10 10"/>
    <position name="head_tilt" joint="head_tilt" kp="10" kv="1" ctrlrange="-0.6 0.8" forcerange="-10 10"/>
{actuators("l_")}
{actuators("r_")}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    out = Path(__file__).with_name("upper_body.xml")
    out.write_text(build())
    print("wrote", out)