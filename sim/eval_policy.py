"""Roll out a trained policy in the simulator (camera + joint positions in, joint targets out) and score it.

  python eval_policy.py --ckpt ../ckpt/bc.pt --episodes 20 --jitter 0.01 --video

The policy never sees object poses or the expert; it only gets the head-camera image and its own joint positions.
For each failed episode the script says how far it got (block lifted? peg lifted? inserted?).
"""
import argparse
import json
import os
import time
from dataclasses import asdict

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import torch

from difficulty import add_args, from_args, randomise_episode
from expert import BLOCK_HALF, PEG_HALF
from policy import load_ckpt
from robot_utils import OUT, READY, load, set_pose

IMG_H, IMG_W = 144, 192      # must match the recorder
STEP_HZ = 10.0


def insertion_metrics(d):
    b = d.body("peg")
    ax = b.xmat.reshape(3, 3)[:, 2]
    if ax[2] < 0:
        ax = -ax
    bottom = b.xpos - ax * PEG_HALF
    blk = d.body("block").xpos
    return {"depth": (blk[2] + BLOCK_HALF) - bottom[2],
            "lateral": float(np.linalg.norm((bottom - blk)[:2])),
            "tilt": float(np.arccos(np.clip(ax[2], -1, 1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../ckpt/bc.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=10000, help="use seeds different from the training data")
    ap.add_argument("--max-steps", type=int, default=300, help="policy steps per episode (10 Hz)")
    ap.add_argument("--exec", type=int, default=5,
                    help="predicted actions (0.1 s each) executed before re-planning. 2 -> 10%% success, 5 and 10 -> 100%% "
                         "(re-planning too often from ambiguous hover states makes the policy skip ahead)")
    ap.add_argument("--video", action="store_true", help="record videos of the first --video-eps episodes (out/policy_ep<N>_seed<S>_SUCCESS|FAIL.mp4)")
    ap.add_argument("--video-eps", type=int, default=3)
    ap.add_argument("--json", default=None, help="write per-episode results to this JSON file")
    add_args(ap)
    args = ap.parse_args()
    diff = from_args(args)
    print("difficulty:", diff)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True    # same seed -> same rollout (as far as the GPU allows)
    torch.backends.cudnn.benchmark = False
    model, stats, cfg = load_ckpt(args.ckpt, dev)
    H, W, K = cfg["img_hw"][0], cfg["img_hw"][1], cfg["chunk"]
    lags = cfg.get("hist", [])
    print(f"policy: chunk {K}, history lags {lags} frames")
    m = load(diff.clearance)
    act_qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    sub = int(round(1.0 / (STEP_HZ * m.opt.timestep)))
    renderer = mujoco.Renderer(m, IMG_H, IMG_W)
    big = mujoco.Renderer(m, 480, 640) if args.video else None
    video = None

    # grasp diagnostics: where is each hand's pinch point relative to its object, and where were the fingers when they closed?
    sites = {"l_": m.site("l_pinch").id, "r_": m.site("r_pinch").id}
    grasp_target = {"l_": ("block", np.zeros(3)), "r_": ("peg", np.array([0.0, 0.0, 0.04]))}   # expert grips the peg 4 cm above its centre
    fing = {s: [m.actuator(s + f).id for f in ("index", "middle", "thumb")] for s in sites}

    peg_id, blk_id = m.body("peg").id, m.body("block").id
    n_ok, reasons, records = 0, [], []
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        d = mujoco.MjData(m)
        set_pose(m, d, READY)
        randomise_episode(m, d, np.random.default_rng(seed), diff)
        mujoco.mj_forward(m, d)
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        blk0, peg0 = d.body("block").xpos.copy(), d.body("peg").xpos.copy()
        qn = lambda: ((d.qpos[act_qadr] - stats["qpos_mean"]) / stats["qpos_std"]).astype(np.float32)
        qhist = [qn()]      # one entry per 0.1 s, like the recorded demonstrations
        max_blk, max_peg = 0.0, 0.0
        tr = {s: {"min": 1e9, "min_h": 0.0, "min_v": 0.0, "t_close": None, "close_h": 0.0, "close_v": 0.0} for s in sites}
        video, vpath = None, None
        if args.video and ep < args.video_eps:
            vpath = OUT / f"policy_ep{ep}_seed{seed}.mp4"
            video = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 480))
        trace = []   # per policy step: [t, right hand height above block (m), right finger closure, peg z, peg tip to hole axis (m)]
        peg_speed, lost, entry = 0.0, None, None   # peak peg speed; when/how the peg left the table; peg-tip offset when it reached the hole
        t0 = time.time()

        for t in range(args.max_steps):
            renderer.update_scene(d, camera="head_cam")
            img = cv2.resize(renderer.render(), (W, H), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(img).to(dev).permute(2, 0, 1).float().div(255.0)[None]
            q = torch.from_numpy(np.concatenate([qhist[-1]] + [qhist[max(len(qhist) - 1 - L, 0)] for L in lags]))[None].to(dev)
            with torch.no_grad():
                acts = model(x, q)[0].cpu().numpy() * stats["ctrl_std"] + stats["ctrl_mean"]
            for j in range(min(args.exec, K)):
                d.ctrl[:] = acts[j]
                for _ in range(sub):
                    mujoco.mj_step(m, d)
                    sp = float(np.linalg.norm(d.cvel[peg_id, 3:]))
                    if sp > peg_speed:
                        peg_speed = sp
                    if lost is None and d.xpos[peg_id, 2] < 0.75:          # below the table top: it has fallen off
                        lost = {"step": t, "speed": sp,
                                "right_fingers_closure": float(np.mean([abs(d.qpos[act_qadr[a]]) for a in fing["r_"]])),
                                "hand_above_block": float(d.site_xpos[sites["r_"]][2] - d.xpos[blk_id][2])}
                qhist.append(qn())
                if entry is None:
                    b_, ax_ = d.xpos[peg_id].copy(), d.xmat[peg_id].reshape(3, 3)[:, 2]
                    ax_ = ax_ if ax_[2] > 0 else -ax_
                    tip_ = b_ - ax_ * PEG_HALF
                    if tip_[2] < d.xpos[blk_id][2] + BLOCK_HALF + 0.003 and np.linalg.norm((tip_ - d.xpos[blk_id])[:2]) < 0.05:
                        entry = {"offset": float(np.linalg.norm((tip_ - d.xpos[blk_id])[:2])),
                                 "tilt": float(np.arccos(np.clip(ax_[2], -1, 1))), "step": t}
                if video is not None:
                    frames = []
                    for cam, r in (("head_cam", big), ("overview", big)):
                        r.update_scene(d, camera=cam)
                        frames.append(r.render())
                    video.write(cv2.cvtColor(np.hstack(frames), cv2.COLOR_RGB2BGR))
            max_blk = max(max_blk, d.body("block").xpos[2] - blk0[2])
            max_peg = max(max_peg, d.body("peg").xpos[2] - peg0[2])
            ax_t = d.xmat[peg_id].reshape(3, 3)[:, 2]
            ax_t = ax_t if ax_t[2] > 0 else -ax_t
            tip_t = d.xpos[peg_id] - ax_t * PEG_HALF
            trace.append([t, round(float(d.site_xpos[sites["r_"]][2] - d.xpos[blk_id][2]), 4),
                          round(float(np.mean([abs(d.qpos[act_qadr[a]]) for a in fing["r_"]])), 3),
                          round(float(d.xpos[peg_id][2]), 4), round(float(np.linalg.norm((tip_t - d.xpos[blk_id])[:2])), 4)])
            for s_ in sites:
                delta = d.site_xpos[sites[s_]] - (d.body(grasp_target[s_][0]).xpos + grasp_target[s_][1])
                dh, dv, d3 = float(np.linalg.norm(delta[:2])), float(delta[2]), float(np.linalg.norm(delta))
                closure = float(np.mean([abs(d.qpos[act_qadr[a]]) for a in fing[s_]]))
                r_ = tr[s_]
                if d3 < r_["min"]:
                    r_.update(min=d3, min_h=dh, min_v=dv)
                if r_["t_close"] is None and closure > 0.15:
                    r_.update(t_close=t, close_h=dh, close_v=dv)

        r = insertion_metrics(d)
        ok = r["depth"] >= 0.045 and r["lateral"] < 0.01 and r["tilt"] < 0.15
        n_ok += ok
        if ok:
            why = ""
        elif max_blk < 0.03:
            why = "grasp failed: block never lifted"
        elif max_peg < 0.05:
            why = "grasp failed: peg never lifted"
        elif lost is not None or r["lateral"] > 0.30:
            why = "peg lost (fell off the table / flung away)"
        elif r["depth"] < 0.045 and r["lateral"] < 0.06:
            why = "peg stuck on the block's rim (not entered)"
        else:
            why = "other (peg near the block but not in the hole)"
        reasons.append(why)
        if video is not None:
            video.release()
            vpath.rename(vpath.with_name(vpath.stem + ("_SUCCESS" if ok else "_FAIL") + ".mp4"))
            video = None
        records.append({"seed": seed, "success": bool(ok), "depth": float(r["depth"]), "lateral": float(r["lateral"]),
                        "tilt": float(r["tilt"]), "max_block_lift": float(max_blk), "max_peg_lift": float(max_peg),
                        "reason": why, "peg_max_speed": peg_speed, "lost": lost, "entry": entry, "trace": trace,
                        "grasp": {k: {kk: (None if vv is None else float(vv)) for kk, vv in tr[k].items()} for k in tr}})
        print(f"ep {ep:3d} seed {seed}: {'SUCCESS' if ok else 'FAIL   '}  depth {r['depth'] * 1000:6.1f} mm  "
              f"tip {r['lateral'] * 1000:5.1f} mm  max lift block {max_blk * 100:4.1f} cm peg {max_peg * 100:4.1f} cm"
              f"{'  -> ' + why if why else ''}  ({time.time() - t0:.0f} s)")
        if not ok:
            ent = f"{entry['offset'] * 1000:.1f} mm off and {np.degrees(entry['tilt']):.0f} deg tilted when the tip reached the hole rim" if entry else "never reached the hole rim"
            los = (f"; peg left the table at step {lost['step']} (right fingers {'CLOSED' if lost['right_fingers_closure'] > 0.15 else 'OPEN'}, "
                   f"speed {lost['speed']:.1f} m/s, hand {lost['hand_above_block'] * 1000:.0f} mm above the block)") if lost else ""
            print(f"        peg: {ent}{los}; peak peg speed {peg_speed:.1f} m/s")
        for s_, name in (("l_", "L"), ("r_", "R")):
            r_ = tr[s_]
            closed = ("never closed the fingers" if r_["t_close"] is None else
                      f"fingers closed at step {r_['t_close']} with the pinch point {r_['close_h'] * 1000:.0f} mm off horizontally, "
                      f"{r_['close_v'] * 1000:+.0f} mm vertically")
            print(f"        {name}: closest approach {r_['min'] * 1000:.0f} mm (horiz {r_['min_h'] * 1000:.0f}, "
                  f"vert {r_['min_v'] * 1000:+.0f}); {closed}")

    renderer.close()
    if big is not None:
        big.close()
    if video is not None:
        video.release()
    print(f"\n{n_ok}/{args.episodes} successful ({100 * n_ok / args.episodes:.0f}%)")
    for why in sorted(set(r for r in reasons if r)):
        print(f"  {reasons.count(why):3d} x {why}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"checkpoint": args.ckpt, "exec": args.exec, "difficulty": asdict(diff), "episodes": len(records),
                       "success_rate": n_ok / max(len(records), 1), "results": records}, f, indent=1)
        print(f"results written to {args.json}")


if __name__ == "__main__":
    main()