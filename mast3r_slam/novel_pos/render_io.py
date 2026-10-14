"""Render world-position pointmap + coverage loading, and resize-gather (ADR-001 修订2).

Backend-free: numpy + cv2 only (no mast3r_slam / no CUDA backend). The resize transform is
NOT computed here — the pipeline (which already imports mast3r_utils) passes the
``(scale_w, scale_h, half_crop_w, half_crop_h)`` tuple and ``true_shape = (H, W)`` down,
so this module stays importable without a GPU for smoke tests.
"""
import os

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # must precede `import cv2`
import cv2  # noqa: E402


def load_pointmap(path) -> np.ndarray:
    """Load an (H0,W0,3) float32 per-pixel GS-world XYZ map. Supports ``.npy`` and ``.exr``.

    NOTE: cv2 reads a 3-channel EXR as BGR; we reverse to XYZ (assume the renderer wrote
    X,Y,Z into channels R,G,B). If the output cloud comes out mirrored, flip this reversal.
    """
    p = str(path)
    low = p.lower()
    if low.endswith(".npy"):
        arr = np.load(p)
    elif low.endswith((".exr", ".pfm", ".tiff", ".tif", ".hdr")):
        arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f"could not read pointmap: {p}")
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # BGR -> XYZ
    else:
        raise ValueError(f"unsupported pointmap format: {p} (use .npy or .exr)")
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"pointmap must be (H,W,3); got {arr.shape}")
    return arr


def load_coverage(path) -> np.ndarray:
    """Load an (H0,W0) bool coverage mask. ``.npy`` -> (>0); image -> grayscale (>127)."""
    p = str(path)
    if p.lower().endswith(".npy"):
        cov = np.asarray(np.load(p))
        if cov.ndim == 3:
            cov = cov[..., 0]
        return cov > 0
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"could not read coverage: {p}")
    return img > 127


def gather_to_grid(arr_hwc, true_shape, transform) -> np.ndarray:
    """Nearest-gather an (H0,W0,C) original-resolution map onto the MASt3R output grid.

    Returns (H*W, C), row-major over (h, w) to match einops ``'h w c -> (h w) c'``.
    Inverse mapping (authoritative, from dataloader.Intrinsics ``K_frame = K/scale - crop``):
        u_orig = (u_out + half_crop_w) * scale_w,  v_orig = (v_out + half_crop_h) * scale_h
    """
    arr = np.asarray(arr_hwc)
    if arr.ndim == 2:
        arr = arr[..., None]
    H0, W0 = arr.shape[:2]
    H, W = int(true_shape[0]), int(true_shape[1])
    sw, sh, hcw, hch = (float(x) for x in transform)
    us, vs = np.meshgrid(np.arange(W), np.arange(H))   # (H,W), row-major
    u = np.clip(np.rint((us + hcw) * sw).astype(np.int64), 0, W0 - 1)
    v = np.clip(np.rint((vs + hch) * sh).astype(np.int64), 0, H0 - 1)
    out = arr[v, u]                                     # (H,W,C)
    return out.reshape(H * W, -1)
