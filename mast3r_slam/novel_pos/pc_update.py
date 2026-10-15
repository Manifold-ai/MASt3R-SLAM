"""Point-cloud update (ADR-001): place the AI completion into GS world via the fitted
similarity transform C, then append to the degraded-GS cloud.

Backend-free: numpy only. The completion pointmap, confidence and match validity are
supplied by the pipeline from the single ``mast3r_match_asymmetric(render, completion)``
call (completion expressed in the render frame) -- nothing is inferred here.

Selection:
  - default: a geometric-confidence percentile of the whole completion grid;
  - ``new_only``: only pixels that FAILED to match the render (``vmatch == False``), i.e.
    M_fill / genuinely new content (the GS has nothing there).
"""
import numpy as np


def update_point_cloud(X_comp, C_comp, uimg, vmatch, C_s, C_R, C_t,
                       existing_xyz, existing_rgb, conf_percentile=50.0, new_only=False):
    """Apply C to the completion pointmap (render frame -> GS world), select, append.

    Args:
        X_comp: (HW,3) completion pointmap in the RENDER coordinate frame.
        C_comp: (HW,) completion geometric confidence.
        uimg:   (H,W,3) or (HW,3) completion RGB in [0,1], same grid as X_comp.
        vmatch: (HW,) bool, True where the completion pixel matched the render (anchor).
        C_s, C_R, C_t: similarity transform (s, R(3,3), t(3,)), MASt3R cam -> GS world.
    Returns (out_xyz (N,3) f32, out_rgb (N,3) u8, stats dict).
    """
    X_comp = np.asarray(X_comp, dtype=np.float64)
    R = np.asarray(C_R, dtype=np.float64)
    t = np.asarray(C_t, dtype=np.float64).reshape(1, 3)
    Xw = float(C_s) * (X_comp @ R.T) + t                      # (HW,3) GS world

    C_comp = np.asarray(C_comp).reshape(-1)
    vmatch = np.asarray(vmatch).reshape(-1).astype(bool)
    finite = np.isfinite(Xw).all(1)
    if new_only:
        keep = (~vmatch) & finite                             # M_fill only
    else:
        c_fin = C_comp[finite]
        thresh = np.quantile(c_fin, conf_percentile / 100.0) if c_fin.size else np.inf
        keep = (C_comp >= thresh) & finite

    new_xyz = Xw[keep].astype(np.float32)
    rgb = np.asarray(uimg).reshape(-1, 3)
    new_rgb = (rgb * 255.0).round().astype(np.uint8)[keep]

    out_xyz = np.concatenate([np.asarray(existing_xyz, np.float32), new_xyz], axis=0)
    out_rgb = np.concatenate([np.asarray(existing_rgb, np.uint8), new_rgb], axis=0)
    stats = {
        "scale": float(C_s),
        "det_R": float(np.linalg.det(R)),
        "n_new": int(keep.sum()),
        "n_existing": int(len(existing_xyz)),
        "n_total": int(len(out_xyz)),
        "new_only": bool(new_only),
    }
    return out_xyz, out_rgb, stats
