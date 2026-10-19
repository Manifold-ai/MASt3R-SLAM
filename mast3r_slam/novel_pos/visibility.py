"""Render visibility map (SLAM-C-002): mode resolution, validation, visible set, and the
C-fit / gate corrections it drives.

Backend-free: numpy only. The map is the render's accumulated opacity, 8-bit, 255 = fully
opaque (the pixel has GS content), 0 = transparent background; ``alpha = v / 255``. The
visible set is ``V = {p : alpha_p >= tau}``. Modes:

- ``off``: no effect (identical to running without a map).
- ``mask``: fit C only on (coverage & finite & V); gate anchors restricted to V.
- ``weight``: fit C on (coverage & finite) with weights multiplied by alpha; gate anchors
  restricted to V (the gate is an unweighted median, so it always uses the binary V).
- ``auto``: ``mask`` if the manifest provides a map, else ``off``.

Loading the image and aligning it to the MASt3R grid lives in ``frames.load_visibility_alpha``
(it needs ``resize_img``, which pulls in the CUDA backend).
"""
import numpy as np

VIS_MODES = ("auto", "off", "mask", "weight")


def resolve_vis_mode(vis_mode, has_map) -> str:
    """Resolve ``auto`` and reject a mode the manifest cannot support. Returns off/mask/weight."""
    if vis_mode not in VIS_MODES:
        raise ValueError(f"vis_mode must be one of {VIS_MODES}, got {vis_mode!r}")
    if vis_mode == "auto":
        return "mask" if has_map else "off"
    if vis_mode in ("mask", "weight") and not has_map:
        raise ValueError(f"--vis-mode {vis_mode} needs novel_view.visibility_map in the manifest")
    return vis_mode


def check_threshold(tau) -> float:
    """tau must lie in (0, 1]; written as a negated range check so NaN is rejected too."""
    tau = float(tau)
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"--vis-threshold must be in (0, 1], got {tau}")
    return tau


def as_gray_u8(img, expect_hw) -> np.ndarray:
    """Validate a raw (``cv2.IMREAD_UNCHANGED``) visibility image and return it as (H,W) uint8.

    Accepts 8-bit single-channel, or 3 identical channels. Rejects other bit depths, 4-channel
    images, non-identical channels, and any size other than ``expect_hw`` (the render image's
    (H, W)) -- the map is never rescaled to fit.
    """
    img = np.asarray(img)
    if img.dtype != np.uint8:
        raise ValueError(f"visibility_map must be 8-bit (uint8), got dtype {img.dtype}")
    if img.ndim == 3 and img.shape[2] == 3:
        if not ((img[..., 0] == img[..., 1]).all() and (img[..., 0] == img[..., 2]).all()):
            raise ValueError("visibility_map has 3 channels that differ; save the alpha as a "
                             "single-channel 8-bit image")
        img = img[..., 0]
    elif img.ndim != 2:
        raise ValueError(f"visibility_map must be single-channel or 3 identical channels, got "
                         f"shape {img.shape}; save the alpha as a single-channel 8-bit image")
    if tuple(img.shape) != tuple(expect_hw):
        raise ValueError(f"visibility_map size {img.shape[1]}x{img.shape[0]} != render_image "
                         f"{expect_hw[1]}x{expect_hw[0]}")
    return img


def visible_set(alpha, tau) -> np.ndarray:
    """(HW,) bool: alpha >= tau."""
    return np.asarray(alpha).reshape(-1) >= tau


def restrict_fit_mask(mask, vis, mode):
    """mask mode: intersect the C-fit set with V. Other modes return it unchanged.

    Returns (mask, n_dropped).
    """
    if mode != "mask":
        return mask, 0
    out = mask & vis
    return out, int(mask.sum() - out.sum())


def alpha_weights(fit_w, alpha_sel):
    """weight mode: multiply the C-fit weights (None = uniform) by alpha on the fit set.

    Returns (weights, kept_ratio) where kept_ratio = sum(new) / sum(old), for the log.
    """
    alpha_sel = np.asarray(alpha_sel, dtype=np.float64)
    if fit_w is None:
        w, old = alpha_sel, float(alpha_sel.size)
    else:
        w, old = np.asarray(fit_w, dtype=np.float64) * alpha_sel, float(np.sum(fit_w))
    return w, (float(w.sum()) / old if old > 0 else 0.0)
