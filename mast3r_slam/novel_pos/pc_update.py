"""Point-cloud update (ADR-001).

Registers the AI completion view onto the degraded-GS point cloud and appends new
points. Single-shot, NO dedup / surface-association in v1 (overlap is double-represented;
accepted per the ADR). Operating pose = the render pose (used directly).

WARNING: imports the CUDA backend (via mast3r_utils). GPU-only.
"""
import numpy as np
import torch

from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_inference_mono


def estimate_global_scale(Xc_world_s1, existing_xyz, anchor) -> float:
    """v1 heuristic: single global scale from the ratio of median radial distances about
    the shared camera center (anchor = render-pose translation == novel-view camera
    center). Rotation-insensitive, correspondence-free. Fallback 1.0 on degenerate input.

    TODO(v2): correspondence-based scale (pair matched 3D distances between the completion
    and render pointmaps).
    """
    if existing_xyz.numel() == 0 or Xc_world_s1.numel() == 0:
        return 1.0
    dc = torch.linalg.norm(Xc_world_s1 - anchor, dim=-1)
    de = torch.linalg.norm(existing_xyz - anchor, dim=-1)
    rc = torch.median(dc)
    re = torch.median(de)
    if not torch.isfinite(rc) or not torch.isfinite(re) or rc <= 1e-8:
        return 1.0
    s = float(re / rc)
    if not np.isfinite(s) or s <= 0.0:
        return 1.0
    return s


def update_point_cloud(model, completion_frame, T_render, existing_xyz, existing_rgb,
                       conf_percentile: float = 50.0, device: str = "cuda:0"):
    """Run mono inference on the completion frame, scale-align to the world point cloud,
    transform by the render pose, confidence-filter, and append.

    Returns (out_xyz (N,3) f32, out_rgb (N,3) u8, stats dict).
    """
    assert config["dataset"]["img_downsample"] == 1, (
        "novel_pos v1 requires dataset.img_downsample == 1 so uimg and the pointmap "
        "share the same pixel grid for the confidence mask."
    )

    Xc, Cc = mast3r_inference_mono(model, completion_frame)   # (HW,3), (HW,1), camera frame

    ex = torch.as_tensor(existing_xyz, device=Xc.device, dtype=Xc.dtype)
    anchor = T_render.data.detach().to(device=Xc.device, dtype=Xc.dtype).reshape(-1)[0:3]

    # provisional s=1 world transform to estimate a single global scale, then apply it
    Xc_world_s1 = T_render.act(Xc)
    s = estimate_global_scale(Xc_world_s1, ex, anchor)
    Xc_world = T_render.act(Xc * s)

    # confidence filter (percentile of this single-frame pointmap's confidence)
    c = Cc.reshape(-1).float()
    thresh = torch.quantile(c, conf_percentile / 100.0)
    keep = c >= thresh
    keep_np = keep.cpu().numpy()

    new_xyz = Xc_world[keep].detach().cpu().numpy().astype(np.float32)
    uimg = completion_frame.uimg.reshape(-1, 3).cpu().numpy()  # (HW,3) in [0,1]
    new_rgb = (uimg * 255.0).round().astype(np.uint8)[keep_np]

    out_xyz = np.concatenate([np.asarray(existing_xyz, np.float32), new_xyz], axis=0)
    out_rgb = np.concatenate([np.asarray(existing_rgb, np.uint8), new_rgb], axis=0)
    stats = {
        "scale": s,
        "n_new": int(keep_np.sum()),
        "n_existing": int(len(existing_xyz)),
        "n_total": int(len(out_xyz)),
    }
    return out_xyz, out_rgb, stats
