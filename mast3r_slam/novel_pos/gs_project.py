"""Project the degraded GS point cloud into the render view to get a GS-world pointmap
(ADR-001 修订3). Backend-free: numpy only.

Replaces the external render-pointmap input (修订2): using the render camera pose + K we
rasterize `ex_xyz` into the MASt3R output grid (z-buffer, nearest point) and store each
hit pixel's GS-WORLD coordinate -> `X_r_GS`, which is then fit against the render image's
MASt3R pointmap to recover the similarity transform C.

CAMERA CONVENTION (contract §2, mandatory): pose is camera-to-world, quaternion wxyz,
OpenCV axes (z forward). We project with an OpenCV pinhole and keep Z > 0.
"""
import numpy as np


def quat_wxyz_to_matrix(q_wxyz) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` (normalized first) -> 3x3 rotation matrix."""
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("quat_wxyz_to_matrix: zero quaternion")
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def intrinsics_to_K_frame(fx, fy, cx, cy, transform) -> np.ndarray:
    """Original-resolution pinhole intrinsics -> MASt3R-grid ``K_frame``.

    ``transform = (scale_w, scale_h, half_crop_w, half_crop_h)`` from
    ``resize_img(..., return_transformation=True)``; mirrors dataloader.Intrinsics:289-293.
    """
    sw, sh, hcw, hch = (float(v) for v in transform)
    return np.array([
        [fx / sw, 0.0,     cx / sw - hcw],
        [0.0,     fy / sh, cy / sh - hch],
        [0.0,     0.0,     1.0],
    ], dtype=np.float64)


def project_gs_to_grid(xyz, R_c2w, t, K_frame, true_shape, z_eps=1e-6):
    """Rasterize GS world points into the render view (OpenCV pinhole, z-buffer nearest).

    world -> camera: ``P_cam = R_c2w^T (P_w - t)`` == ``(P_w - t) @ R_c2w``.
    Project ``u = fx*X/Z + cx``, ``v = fy*Y/Z + cy``; keep ``Z > z_eps`` and in-bounds;
    per pixel keep the nearest point and store its GS-WORLD coordinate.

    Returns:
        X_r_GS (H*W, 3) float32 (NaN where uncovered), coverage (H*W,) bool.
        Row-major over (h, w) to match the MASt3R pointmap ordering.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    R = np.asarray(R_c2w, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    H, W = int(true_shape[0]), int(true_shape[1])

    P = (xyz - t) @ R                                   # (M,3) camera frame
    Z = P[:, 2]
    fx, fy = K_frame[0, 0], K_frame[1, 1]
    cx, cy = K_frame[0, 2], K_frame[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * P[:, 0] / Z + cx
        v = fy * P[:, 1] / Z + cy
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    valid = (
        (Z > z_eps) & np.isfinite(u) & np.isfinite(v)
        & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    )

    X = np.full((H * W, 3), np.nan, dtype=np.float32)
    cov = np.zeros(H * W, dtype=bool)
    idx = np.where(valid)[0]
    if idx.size:
        order = idx[np.argsort(-Z[idx])]               # far -> near
        lin = vi[order] * W + ui[order]
        X[lin] = xyz[order].astype(np.float32)         # nearest written last -> wins
        cov[lin] = True
    return X, cov
