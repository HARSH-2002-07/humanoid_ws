"""Behaviour-cloning policy: head-camera image + joint positions -> a chunk of future joint-position targets.

Shared by train_bc.py and eval_policy.py.
"""
import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Policy(nn.Module):
    """ResNet18 trunk (spatial features kept, not globally pooled) + joint-state embedding -> MLP -> K x n_act."""

    def __init__(self, n_act=22, chunk=10, img_hw=(96, 128), pretrained=True, n_hist=0):
        super().__init__()
        self.n_act, self.chunk = n_act, chunk
        self.n_hist = n_hist      # number of extra past joint-position vectors the policy sees (0 = memoryless)
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            r = torchvision.models.resnet18(weights=weights)
        except Exception as ex:  # e.g. no internet for the weight download
            print(f"could not load pretrained weights ({ex}); training from scratch")
            r = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(r.children())[:-2])      # (B, 512, H/32, W/32)
        self.reduce = nn.Sequential(nn.Conv2d(512, 64, 1), nn.ReLU())
        was_training = self.training
        self.eval()
        with torch.no_grad():
            n_feat = self.reduce(self.backbone(torch.zeros(1, 3, *img_hw))).numel()
        self.train(was_training)
        self.proprio = nn.Sequential(nn.Linear(n_act * (1 + n_hist), 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(n_feat + 128, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(),
                                  nn.Linear(512, chunk * n_act))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img, qpos):
        """img: (B,3,H,W) float in [0,1]; qpos: (B, n_act*(1+n_hist)) normalised = [now, lag1, lag2, ...].
        Returns (B, chunk, n_act) normalised."""
        f = self.reduce(self.backbone((img - self.mean) / self.std)).flatten(1)
        h = torch.cat([f, self.proprio(qpos)], dim=1)
        return self.head(h).view(-1, self.chunk, self.n_act)


def random_shift(x, pad=4):
    """Per-sample random translation (+-pad pixels) with replicate padding."""
    n, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    oy = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    return torch.stack([xp[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w] for i in range(n)])


def load_demos(files, img_hw):
    """Load HDF5 episodes into memory. Images are resized to img_hw = (H, W)."""
    H, W = img_hw
    imgs, qpos, ctrl, last, first, n = [], [], [], [], [], 0
    for k, fn in enumerate(files):
        with h5py.File(fn, "r") as h:
            im, q, c = h["images"][:], h["qpos"][:], h["ctrl"][:]
        im = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA) for x in im])
        T = len(q)
        imgs.append(im)
        qpos.append(q)
        ctrl.append(c)
        last.append(np.full(T, n + T - 1))
        first.append(np.full(T, n))
        n += T
        if (k + 1) % 50 == 0 or k + 1 == len(files):
            print(f"  loaded {k + 1}/{len(files)} episodes ({n} frames)")
    return {"img": torch.from_numpy(np.concatenate(imgs)), "qpos": np.concatenate(qpos),
            "ctrl": np.concatenate(ctrl), "last": np.concatenate(last), "first": np.concatenate(first)}


def make_stats(D):
    return {"qpos_mean": D["qpos"].mean(0), "qpos_std": np.maximum(D["qpos"].std(0), 1e-2),
            "ctrl_mean": D["ctrl"].mean(0), "ctrl_std": np.maximum(D["ctrl"].std(0), 1e-2)}


def normalise(D, stats):
    D["qpos_n"] = torch.from_numpy(((D["qpos"] - stats["qpos_mean"]) / stats["qpos_std"]).astype(np.float32))
    D["ctrl_n"] = torch.from_numpy(((D["ctrl"] - stats["ctrl_mean"]) / stats["ctrl_std"]).astype(np.float32))


def get_batch(D, ids, chunk, device, augment, lags=()):
    """ids: numpy int array of frame indices. Target = the next `chunk` joint targets (ctrl[t+1 .. t+chunk]).
    lags: how many frames back (10 frames = 1 s) to also feed the joint positions from; clamped at the episode start."""
    tgt = np.minimum(ids[:, None] + 1 + np.arange(chunk)[None], D["last"][ids][:, None])
    t_ids = torch.from_numpy(ids)
    img = D["img"][t_ids].to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    if augment:
        img = random_shift(img)
        img = (img * (1 + 0.1 * torch.randn(img.shape[0], 1, 1, 1, device=device))).clamp(0, 1)
    cols = [D["qpos_n"][t_ids]] + [D["qpos_n"][torch.from_numpy(np.maximum(ids - L, D["first"][ids]))] for L in lags]
    q = torch.cat(cols, dim=1).to(device)
    y = D["ctrl_n"][torch.from_numpy(tgt)].to(device)
    return img, q, y


def save_ckpt(path, model, stats, cfg, step, val):
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, "val": val,
                "stats": {k: torch.from_numpy(v.astype(np.float32)) for k, v in stats.items()}}, path)


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = ck["cfg"]
    model = Policy(cfg["n_act"], cfg["chunk"], tuple(cfg["img_hw"]), pretrained=False, n_hist=len(cfg.get("hist", []))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = {k: v.cpu().numpy() for k, v in ck["stats"].items()}
    return model, stats, cfg