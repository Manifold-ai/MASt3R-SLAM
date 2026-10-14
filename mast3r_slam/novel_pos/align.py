"""Weighted similarity (Sim3) fit WITH reflection allowed — Umeyama closed form.

Backend-free: numpy only. Fits C such that ``X_r_GS ≈ C·X_r_cam``, where C absorbs the
coordinate-convention / scale / handedness difference between MASt3R's camera frame and
the GS world. Reflection is ALLOWED (``det(R)`` may be -1): a plain SO(3) / roma solver
cannot represent a handedness flip (e.g. the ``diag(1,-1,1)`` the scheme flagged), so we
deliberately omit the ``det(R)=+1`` correction.
"""
import numpy as np


def umeyama_with_reflection(src, tgt, weights=None, eps=1e-8):
    """Fit (s, R, t) minimizing ``sum_i w_i || s R x_i + t - y_i ||^2`` over s>0,
    R in O(3) (det R = +/-1, reflection allowed), t in R^3.

    Args:
        src: (N,3) source points X_cam (MASt3R camera frame).
        tgt: (N,3) target points X_GS (GS world).
        weights: (N,) non-negative weights, or None for uniform.
        eps: numerical floor.

    Returns:
        (s: float, R: (3,3), t: (3,), info: dict{det_R, n, rms, degenerate}).
        Apply as ``X_GS = s * (x @ R.T) + t``.

    Raises:
        ValueError if N < 3, total weight is zero, or the source has (near) zero variance.
    """
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    assert src.shape == tgt.shape and src.ndim == 2 and src.shape[1] == 3, (
        f"src/tgt must be (N,3) and equal shape; got {src.shape}, {tgt.shape}"
    )
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"umeyama: need >= 3 points, got {n}")
    assert np.isfinite(src).all() and np.isfinite(tgt).all(), (
        "umeyama: non-finite inputs (mask them out before calling)"
    )

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        assert w.shape[0] == n and (w >= 0).all(), "weights must be (N,) non-negative"
    W = float(w.sum())
    if W < eps:
        raise ValueError("umeyama: total weight is zero")

    mu_x = (w[:, None] * src).sum(0) / W
    mu_y = (w[:, None] * tgt).sum(0) / W
    Xc = src - mu_x
    Yc = tgt - mu_y

    sigma2_x = float((w * (Xc ** 2).sum(1)).sum() / W)
    if sigma2_x < eps:
        raise ValueError("umeyama: degenerate source (zero variance)")

    Sigma = (Yc * w[:, None]).T @ Xc / W            # (3,3) target(x)source
    U, D, Vt = np.linalg.svd(Sigma)
    R = U @ Vt                                       # reflection allowed: NO det correction
    s = float(D.sum() / sigma2_x)                    # tr(D) because S = I (reflection allowed)
    t = mu_y - s * (R @ mu_x)

    det_R = float(np.linalg.det(R))
    degenerate = bool(D[1] < eps * max(D[0], eps))   # near planar/collinear -> ill-scaled

    resid = s * (src @ R.T) + t - tgt
    rms = float(np.sqrt((w * (resid ** 2).sum(1)).sum() / W))

    return s, R, t, {"det_R": det_R, "n": int(n), "rms": rms, "degenerate": degenerate}
