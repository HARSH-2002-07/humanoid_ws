"""Train the image + joint-state -> action-chunk policy on collected demonstrations.

  python train_bc.py --data ../data --steps 15000
  python train_bc.py --data ../data --steps 300 --batch 16      # quick smoke test

Every 10th episode is held out for validation. The best checkpoint (lowest validation loss) goes to --out.
"""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from policy import Policy, get_batch, load_demos, make_stats, normalise, save_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../ckpt/bc.pt")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=10, help="number of future joint targets predicted (10 = 1 s at 10 Hz)")
    ap.add_argument("--img", type=int, nargs=2, default=[96, 128], metavar=("H", "W"))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--hist", type=int, nargs="*", default=[], help="history lags in frames (10 frames = 1 s), e.g. --hist 5 10 15")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(str(Path(args.data) / "ep_*.hdf5")))
    assert files, f"no ep_*.hdf5 files in {args.data}"
    val_files = files[::10]
    train_files = [f for f in files if f not in val_files]
    print(f"{len(train_files)} train / {len(val_files)} val episodes, device {dev}")

    print("loading training data")
    train = load_demos(train_files, tuple(args.img))
    print("loading validation data")
    val = load_demos(val_files, tuple(args.img))
    stats = make_stats(train)
    normalise(train, stats)
    normalise(val, stats)
    n_act = train["qpos"].shape[1]

    model = Policy(n_act, args.chunk, tuple(args.img), pretrained=not args.no_pretrained, n_hist=len(args.hist)).to(dev)
    cfg = {"n_act": n_act, "chunk": args.chunk, "img_hw": list(args.img), "hist": list(args.hist)}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def validate(n_batches=40):
        model.eval()
        tot = 0.0
        rng = np.random.default_rng(123)
        with torch.no_grad():
            for _ in range(n_batches):
                ids = rng.integers(0, len(val["qpos"]), args.batch)
                img, q, y = get_batch(val, ids, args.chunk, dev, augment=False, lags=args.hist)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    pred = model(img, q)
                tot += F.l1_loss(pred.float(), y).item()
        model.train()
        return tot / n_batches

    best, run_loss, t0 = 1e9, 0.0, time.time()
    model.train()
    n_train = len(train["qpos"])
    for step in range(1, args.steps + 1):
        ids = np.random.randint(0, n_train, args.batch)
        img, q, y = get_batch(train, ids, args.chunk, dev, augment=True, lags=args.hist)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            pred = model(img, q)
        loss = F.l1_loss(pred.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            v = validate()
            n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
            print(f"step {step:6d}/{args.steps}  train L1 {run_loss / n:.4f}  val L1 {v:.4f}  "
                  f"({step / (time.time() - t0):.1f} it/s)")
            run_loss = 0.0
            if v < best:
                best = v
                save_ckpt(out, model, stats, cfg, step, v)
                print(f"   saved {out} (val {v:.4f})")
    print(f"done. best val L1 {best:.4f} -> {out}")


if __name__ == "__main__":
    main()