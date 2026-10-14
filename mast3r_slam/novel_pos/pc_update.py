"""Point-cloud update (ADR-001 修订2): place the AI completion into GS world via the
fitted similarity transform C, then append to the degraded-GS cloud.

C = (s, R, t) maps MASt3R camera frame -> GS world: ``X_GS = s * (x @ R.T) + t``
(R may contain a reflection, det R = -1). C is fit in the pipeline from the render frame;
here we only apply it. Single-shot, NO dedup in v1.

WARNING: imports the CUDA backend (via mast3r_utils). GPU-only.
"""
import numpy as np
import torch

from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_inference_mono


def update_point_cloud(model, completion_frame, C_s, C_R, C_t,
                       existing_xyz, existing_rgb, conf_percentile: float = 50.0,
                       device: str = "cuda:0"):
    """Mono-infer the completion, apply C, confidence-filter, append.

    Args:
        C_s: float scale; C_R: (3,3) rotation (possibly reflected); C_t: (3,) translation.
    Returns (out_xyz (N,3) f32, out_rgb (N,3) u8, stats dict).
    """
    assert config["dataset"]["img_downsample"] == 1, (
        "novel_pos v1 requires dataset.img_downsample == 1 so uimg and the pointmap "
        "share the same pixel grid for the confidence mask."
    )

    # completion pointmap in MASt3R camera frame (mono — same convention C was fit on)
    Xc, Cc = mast3r_inference_mono(model, completion_frame)   # (HW,3), (HW,1)

    R = torch.as_tensor(C_R, device=Xc.device, dtype=Xc.dtype)
    t = torch.as_tensor(C_t, device=Xc.device, dtype=Xc.dtype).reshape(1, 3)
    Xc_world = C_s * (Xc @ R.T) + t                          # X_GS = s (x R^T) + t

    # confidence filter (percentile of this single-frame pointmap's confidence)
    c = Cc.reshape(-1).float()
    thresh = torch.quantile(c, conf_percentile / 100.0)
    keep = c >= thresh
    keep_np = keep.cpu().numpy()

    new_xyz = Xc_world[keep].detach().cpu().numpy().astype(np.float32)
    uimg = completion_frame.uimg.reshape(-1, 3).cpu().numpy()  # (HW,3) in [0,1], same grid
    new_rgb = (uimg * 255.0).round().astype(np.uint8)[keep_np]

    out_xyz = np.concatenate([np.asarray(existing_xyz, np.float32), new_xyz], axis=0)
    out_rgb = np.concatenate([np.asarray(existing_rgb, np.uint8), new_rgb], axis=0)
    stats = {
        "scale": float(C_s),
        "det_R": float(np.linalg.det(np.asarray(C_R, dtype=np.float64))),
        "n_new": int(keep_np.sum()),
        "n_existing": int(len(existing_xyz)),
        "n_total": int(len(out_xyz)),
    }
    return out_xyz, out_rgb, stats
