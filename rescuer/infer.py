"""Run the restoration network, with a small geometric self-ensemble.

The image is restored several times under flips; each result is flipped back.
The mean is the restored image. The individual results ("members") are kept:
where they disagree, the network is not sure what was on the paper.
"""
import cv2
import numpy as np
import torch

from .model import RescueNet

# name -> dims to flip (the same flip undoes itself)
MEMBERS = {"identity": (), "hflip": (-1,), "vflip": (-2,), "rot180": (-2, -1)}


def load_model(path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(path, map_location=device, weights_only=True)
    model = RescueNet(width=state.get("width", 32)).to(device).eval()
    model.load_state_dict(state["model"])
    info = {"path": str(path), "iter": state.get("iter"), "val_psnr": state.get("val_psnr")}
    return model, device, info


def prepare(img, max_side=1600):
    """Any uint8 image (gray or BGR) -> float32 gray in [0, 1], longest side capped."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


@torch.no_grad()
def _tiled(model, x, tile=512, overlap=32):
    """Run `model` over a 1x1xHxW tensor in overlapping tiles and blend."""
    _, _, h, w = x.shape
    if h <= tile and w <= tile:
        return model(x).float()
    out = torch.zeros_like(x, dtype=torch.float32)
    weight = torch.zeros_like(out)
    step = tile - overlap
    ys = sorted({min(y, max(h - tile, 0)) for y in range(0, h, step)})
    xs = sorted({min(v, max(w - tile, 0)) for v in range(0, w, step)})
    for y in ys:
        for v in xs:
            part = x[..., y:y + tile, v:v + tile]
            out[..., y:y + tile, v:v + tile] += model(part).float()
            weight[..., y:y + tile, v:v + tile] += 1
    return out / weight


@torch.no_grad()
def restore(img, model, device, members=tuple(MEMBERS)):
    """img: float32 gray HxW. Returns dict(mean, members, std), all float32 HxW."""
    x = torch.from_numpy(img)[None, None].to(device)
    outs = []
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        for name in members:
            dims = MEMBERS[name]
            y = _tiled(model, torch.flip(x, dims) if dims else x)
            outs.append((torch.flip(y, dims) if dims else y).clamp(0, 1))
    stack = torch.cat(outs, 0)[:, 0]
    return {
        "mean": stack.mean(0).cpu().numpy(),
        "members": [m.cpu().numpy() for m in stack],
        "std": stack.std(0).cpu().numpy() if len(outs) > 1 else np.zeros_like(img),
    }
