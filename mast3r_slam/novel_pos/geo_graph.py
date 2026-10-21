"""Geometric factor graph + Sim(3) Gauss-Newton for the ADR-002 A merge (SLAM-C-003).

Backend-free: numpy only. Nodes are the new keyframes' sealed local clouds; P0 is the fixed
world (T0 = I) and known cameras are fixed observations, never nodes. Factors:

- edge (i <- j): MASt3R-SLAM calibrated residual ``[u - u_t, v - v_t, log z - log z_t]`` of
  ``y = T_i^-1 T_j X_j(q)`` projected with K_i against the matched pixel p of i (mirrors
  ``gauss_newton_calib`` / ``calib_proj_kernel``: Q gate, sqrt(q)/sigma weights, Huber 1.345).
- anchor (i): log-depth of ``T_i X_i(q)`` vs a trusted P0 point, both measured along the
  known source camera's z axis (fixes scale; P0 is world, so this ties i to the anchor).
- fixed (i, camera v): pixel residual ``pi_Kv(C_v^-1 T_i X_i(q)) - m`` against a fixed
  observation (the keyframe's own camera with m = q, or a fixed view with matched m).
- prior (i): weak numerical regulariser towards the initial T (NOT geometric evidence).

No RGB residual enters anywhere.
"""
import dataclasses
from typing import List, Optional

import numpy as np

from mast3r_slam.novel_pos.sim3 import (
    Sim3, point_jacobian, rotation_angle_deg, so3_left_jacobian_inv, so3_log, skew,
)


@dataclasses.dataclass
class GeoParams:
    # mirrored from config["local_opt"] (MASt3R-SLAM calibrated GN) unless overridden
    Q_conf: float = 1.5
    C_conf: float = 0.0
    pixel_border: float = -10.0
    depth_eps: float = 1e-6
    sigma_pixel: float = 1.0
    sigma_depth: float = 10.0          # log-depth, edges
    max_iters: int = 10
    delta_norm: float = 1e-8
    min_match_frac: float = 0.1
    # ADR-002 A additions
    huber_k: float = 1.345
    sigma_anchor: float = 0.1          # log-depth, anchors to P0
    sigma_fixed: float = 1.0           # pixels, fixed-camera factors
    sigma_prior_t: float = 100.0       # weak regulariser (world units / rad / log-scale)
    sigma_prior_r: float = 1.0
    sigma_prior_s: float = 1.0
    fixed_stride: int = 4              # subsample the source-view fixed factor
    edge_stride: int = 1               # subsample edge points (1 = all, as MASt3R-SLAM)
    # edge selection / quality gate
    min_overlap: float = 0.1
    max_edges_per_kf: int = 4
    min_edge_matches: int = 500
    min_edge_cells: int = 4
    grid_cells: int = 4
    # fixed views
    max_fixed_views_per_kf: int = 2
    min_fixed_matches: int = 500
    # admission gates
    min_anchor: int = 100
    anchor_inlier: float = 0.1         # |log-depth| counted as anchor inlier
    max_anchor_logdepth: float = 0.15  # median |anchor log-depth residual|
    max_fixed_px: float = 3.0          # median source-view pixel residual
    max_edge_px: float = 4.0           # median edge pixel residual
    max_view_px: float = 4.0           # median fixed-view pixel residual
    max_rot_dev_deg: float = 1.0       # T rotation vs the known camera
    max_center_shift_rel: float = 0.03 # |t - camera centre| / median scene depth
    min_positive_depth: float = 0.99

    @classmethod
    def from_config(cls, local_opt: Optional[dict] = None, overrides: Optional[dict] = None):
        p = cls()
        src = dict(local_opt or {})
        for k in ("Q_conf", "C_conf", "pixel_border", "sigma_pixel", "sigma_depth", "max_iters",
                  "delta_norm", "min_match_frac"):
            if k in src:
                setattr(p, k, type(getattr(p, k))(src[k]))
        if "depth_eps" in src:
            p.depth_eps = float(src["depth_eps"])
        for k, v in (overrides or {}).items():
            if not hasattr(p, k):
                raise ValueError(f"geo: unknown parameter '{k}'")
            setattr(p, k, type(getattr(p, k))(v))
        return p


@dataclasses.dataclass
class LocalCloud:
    """A sealed keyframe cloud as the solver sees it (local frame = its camera frame)."""
    X: np.ndarray            # (HW,3) ray-constrained points, camera frame, arbitrary scale
    conf: np.ndarray         # (HW,)
    valid: np.ndarray        # (HW,) bool
    K: np.ndarray            # (3,3) K_frame on the MASt3R grid
    shape: tuple             # (H, W)
    cam_R: np.ndarray        # known camera-to-world rotation (OpenCV)
    cam_t: np.ndarray        # known camera center


@dataclasses.dataclass
class EdgeObs:
    """Residuals of j's points against their matched pixels in i (one direction)."""
    i: int
    j: int
    idx: np.ndarray          # (HW_j,) pixel index in i matched to each pixel of j
    valid: np.ndarray        # (HW_j,) bool valid_match
    q: np.ndarray            # (HW_j,) match confidence sqrt(Q_i[idx] * Q_ji)


@dataclasses.dataclass
class AnchorObs:
    i: int
    pix: np.ndarray          # (M,) point index in keyframe i
    target: np.ndarray       # (M,3) trusted P0 world points
    q: np.ndarray            # (M,)


@dataclasses.dataclass
class FixedObs:
    i: int
    pix: np.ndarray          # (M,) point index in keyframe i
    uv: np.ndarray           # (M,2) observed pixel in camera v
    q: np.ndarray            # (M,)
    R: np.ndarray            # camera v camera-to-world rotation
    t: np.ndarray            # camera v center
    K: np.ndarray            # camera v K_frame
    shape: tuple             # camera v (H, W)
    view: str = "source"     # report label (the manifest id for fixed views)
    is_source: bool = False  # the keyframe's own camera (m = q, no match confidence)


def _huber_weight(e, k):
    a = np.abs(e)
    return np.where(a < k, 1.0, k / np.maximum(a, 1e-12))


def _huber_cost(e, k):
    a = np.abs(e)
    return np.where(a < k, 0.5 * a * a, k * (a - 0.5 * k))


def _project(y, K):
    """(N,3) camera points -> (N,2) pixels and (N,2,3) d(uv)/dy."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = y[:, 2]
    zi = 1.0 / z
    uv = np.stack([fx * y[:, 0] * zi + cx, fy * y[:, 1] * zi + cy], -1)
    D = np.zeros((y.shape[0], 2, 3))
    D[:, 0, 0] = fx * zi
    D[:, 0, 2] = -fx * y[:, 0] * zi * zi
    D[:, 1, 1] = fy * zi
    D[:, 1, 2] = -fy * y[:, 1] * zi * zi
    return uv, D


def _in_image(uv, shape, border):
    H, W = shape
    return ((uv[:, 0] > border) & (uv[:, 0] < W - 1 - border)
            & (uv[:, 1] > border) & (uv[:, 1] < H - 1 - border))


def _pack(ok, r, sqrt_w, nodes, jac_fn, jac, compact):
    """Finish a factor: keep only valid rows (compact) or zero the invalid ones, and build the
    Jacobian only when asked. ``jac_fn(rows)`` returns (M,d,7) for the given row indices."""
    rows = np.nonzero(ok)[0] if compact else np.arange(len(ok))
    r, sqrt_w = r[rows], sqrt_w[rows]
    if not compact:                      # invalid rows must not leak NaN/inf into H or g
        r = np.where(ok[:, None], r, 0.0)
        sqrt_w = np.where(ok[:, None], sqrt_w, 0.0)
    J = None
    if jac:
        J = jac_fn(rows)
        if not compact:
            J = np.where(ok[:, None, None], J, 0.0)
    return r, sqrt_w, [(n, sign, J) for n, sign in nodes]


# ------------------------------------------------------------------------- factors
# Each returns (r (M,d), sqrt_w (M,d), blocks [(node, sign, J (M,d,7) or None)]). A block's
# Jacobian is sign * J, so an edge shares one J between its two nodes. With compact=True only
# valid rows are returned; with jac=False no Jacobian is built (line-search cost evaluation).

def edge_factor(e: EdgeObs, kfs, T, p: GeoParams, stride=None, jac=True, compact=True):
    ki, kj = kfs[e.i], kfs[e.j]
    stride = p.edge_stride if stride is None else stride
    q_ix = np.arange(0, kj.X.shape[0], max(1, int(stride)))
    idx = e.idx[q_ix].astype(np.int64)
    w = T[e.j].act(kj.X[q_ix])
    y = T[e.i].inv_act(w)
    Xi = ki.X[idx]
    ok = (e.valid[q_ix] & (e.q[q_ix] > p.Q_conf) & kj.valid[q_ix] & ki.valid[idx]
          & (kj.conf[q_ix] > p.C_conf) & (ki.conf[idx] > p.C_conf)
          & (y[:, 2] > p.depth_eps) & (Xi[:, 2] > p.depth_eps))
    y_safe = np.where(ok[:, None], y, np.array([0.0, 0.0, 1.0]))
    uv, D = _project(y_safe, ki.K)
    ok &= _in_image(uv, ki.shape, p.pixel_border)
    W_i = ki.shape[1]
    tgt = np.stack([idx % W_i, idx // W_i], -1).astype(np.float64)
    zt = np.where(ok, Xi[:, 2], 1.0)
    r = np.concatenate([uv - tgt, (np.log(y_safe[:, 2]) - np.log(zt))[:, None]], 1)
    sq = np.sqrt(np.where(ok, e.q[q_ix], 0.0))
    sqrt_w = np.stack([sq / p.sigma_pixel, sq / p.sigma_pixel, sq / p.sigma_depth], 1)

    def jac_fn(rows):
        dr_dy = np.concatenate([D[rows], np.zeros((len(rows), 1, 3))], 1)
        dr_dy[:, 2, 2] = 1.0 / y_safe[rows, 2]
        dy_dw = T[e.i].R.T / T[e.i].s                           # y = R_i^T (w - t_i) / s_i
        return dr_dy @ dy_dw @ point_jacobian(w[rows])            # d/d xi_j

    # left perturbation of T_i enters y through T_i^-1: its Jacobian is -d/d xi_j
    return _pack(ok, r, sqrt_w, [(e.j, 1.0), (e.i, -1.0)], jac_fn, jac, compact)


def anchor_factor(a: AnchorObs, kfs, T, p: GeoParams, jac=True, compact=True):
    kf = kfs[a.i]
    w = T[a.i].act(kf.X[a.pix])
    zax = kf.cam_R[:, 2]                                          # camera z axis in world
    z = (w - kf.cam_t) @ zax
    zt = (a.target - kf.cam_t) @ zax
    ok = (z > p.depth_eps) & (zt > p.depth_eps) & (a.q > p.Q_conf) & kf.valid[a.pix]
    zs, zts = np.where(ok, z, 1.0), np.where(ok, zt, 1.0)
    r = (np.log(zs) - np.log(zts))[:, None]
    sqrt_w = (np.sqrt(np.where(ok, a.q, 0.0)) / p.sigma_anchor)[:, None]

    def jac_fn(rows):
        return (zax[None, None, :] @ point_jacobian(w[rows])) / zs[rows, None, None]

    return _pack(ok, r, sqrt_w, [(a.i, 1.0)], jac_fn, jac, compact)


def fixed_factor(f: FixedObs, kfs, T, p: GeoParams, jac=True, compact=True):
    kf = kfs[f.i]
    w = T[f.i].act(kf.X[f.pix])
    y = (w - f.t) @ f.R                                           # C_v^-1 w
    # the source view has no match confidence (q = 1): only matched fixed views use the Q gate
    ok = (y[:, 2] > p.depth_eps) & kf.valid[f.pix] & ((f.q > p.Q_conf) | f.is_source)
    y_safe = np.where(ok[:, None], y, np.array([0.0, 0.0, 1.0]))
    uv, D = _project(y_safe, f.K)
    ok &= _in_image(uv, f.shape, p.pixel_border)
    r = uv - f.uv
    sq = np.sqrt(np.where(ok, f.q, 0.0)) / p.sigma_fixed

    def jac_fn(rows):
        return D[rows] @ f.R.T @ point_jacobian(w[rows])

    return _pack(ok, r, np.stack([sq, sq], 1), [(f.i, 1.0)], jac_fn, jac, compact)


def prior_factor(i, T0: Sim3, Ti: Sim3, p: GeoParams, jac=True):
    dphi = so3_log(Ti.R @ T0.R.T)
    r = np.concatenate([Ti.t - T0.t, dphi, [np.log(Ti.s / T0.s)]])[None]
    sw = np.array([[1 / p.sigma_prior_t] * 3 + [1 / p.sigma_prior_r] * 3 + [1 / p.sigma_prior_s]])
    if not jac:
        return r, sw, [(i, 1.0, None)]
    J = np.zeros((1, 7, 7))
    J[0, 0:3, 0:3] = np.eye(3)
    J[0, 0:3, 3:6] = -skew(Ti.t)
    J[0, 0:3, 6] = Ti.t
    J[0, 3:6, 3:6] = so3_left_jacobian_inv(dphi)
    J[0, 6, 6] = 1.0
    return r, sw, [(i, 1.0, J)]


# -------------------------------------------------------------------------- solver

def _iter_factors(kfs, T, T0, edges, anchors, fixeds, p, free, jac):
    """Yield factors one at a time so their Jacobians never all live in memory together."""
    for e in edges:
        if e.i in free or e.j in free:
            yield edge_factor(e, kfs, T, p, jac=jac)
    for a in anchors:
        if a.i in free:
            yield anchor_factor(a, kfs, T, p, jac=jac)
    for f in fixeds:
        if f.i in free:
            yield fixed_factor(f, kfs, T, p, jac=jac)
    for i in sorted(free):
        yield prior_factor(i, T0[i], T[i], p, jac=jac)


def _cost(kfs, T, T0, edges, anchors, fixeds, p, free):
    return float(sum(_huber_cost(sw * r, p.huber_k).sum()
                     for r, sw, _ in _iter_factors(kfs, T, T0, edges, anchors, fixeds, p, free, False)))


def _linearize(kfs, T, T0, edges, anchors, fixeds, p, free, col):
    """Accumulate the IRLS normal equations factor by factor; returns (H, g, cost)."""
    n = 7 * len(col)
    H, g, cost = np.zeros((n, n)), np.zeros(n), 0.0
    for r, sw, blocks in _iter_factors(kfs, T, T0, edges, anchors, fixeds, p, free, True):
        e = sw * r
        cost += float(_huber_cost(e, p.huber_k).sum())
        Wt = _huber_weight(e, p.huber_k) * sw * sw                   # (M,d), as the CUDA kernel
        WJ, JWJ = {}, {}                                              # cached per distinct J
        for (a, sa, Ja) in blocks:
            if a not in col:
                continue
            if id(Ja) not in WJ:
                WJ[id(Ja)] = Wt[:, :, None] * Ja
            ca = 7 * col[a]
            g[ca:ca + 7] += sa * np.einsum("mdk,md->k", WJ[id(Ja)], r)
            for (b, sb, Jb) in blocks:
                if b not in col:
                    continue
                key = (id(Ja), id(Jb))
                if key not in JWJ:
                    JWJ[key] = np.einsum("mdk,mdl->kl", WJ[id(Ja)], Jb)
                cb = 7 * col[b]
                H[ca:ca + 7, cb:cb + 7] += sa * sb * JWJ[key]
    return H, g, cost


def solve(kfs: List[LocalCloud], T_init: List[Sim3], edges, anchors, fixeds, p: GeoParams,
          free: Optional[List[int]] = None):
    """IRLS Gauss-Newton over the free nodes; returns (T list, info).

    info["status"]: "converged" (step norm < delta_norm), "no_descent" (no backtracking step
    lowered the robust cost) or "max_iters".
    """
    free = sorted(range(len(kfs)) if free is None else free)
    fset = set(free)
    col = {n: k for k, n in enumerate(free)}
    T = [Sim3(t.s, t.R, t.t) for t in T_init]
    T0 = [Sim3(t.s, t.R, t.t) for t in T_init]
    args = (edges, anchors, fixeds, p, fset)
    if not free:
        cost = _cost(kfs, T, T0, *args)
        return T, {"iterations": 0, "cost": [cost], "status": "converged", "converged": True,
                   "free": free}
    H, g, cost = _linearize(kfs, T, T0, *args, col)
    hist, status, iters = [cost], "max_iters", 0
    n = 7 * len(free)
    for iters in range(1, p.max_iters + 1):
        H[np.diag_indices(n)] += 1e-9 + 1e-9 * np.abs(np.diag(H))
        try:
            dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            status = "singular"
            break
        step, accepted = 1.0, False
        for _ in range(6):                                           # backtracking on robust cost
            Tn = list(T)
            for nd in free:
                Tn[nd] = T[nd].retract(step * dx[7 * col[nd]:7 * col[nd] + 7])
            cost_n = _cost(kfs, Tn, T0, *args)
            if cost_n <= cost:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            status = "converged" if np.linalg.norm(dx) < p.delta_norm else "no_descent"
            break
        T, cost = Tn, cost_n
        hist.append(cost)
        if np.linalg.norm(step * dx) < p.delta_norm:
            status = "converged"
            break
        if iters < p.max_iters:
            H, g, cost = _linearize(kfs, T, T0, *args, col)
    return T, {"iterations": iters, "cost": hist, "status": status,
               "converged": status == "converged", "free": free}


# ------------------------------------------------------------------ residual stats

def residual_stats(kfs, T, edges, anchors, fixeds, p: GeoParams, nodes=None):
    """Unweighted post-solve residuals used by the admission gates (C-003 §六)."""
    nodes = range(len(kfs)) if nodes is None else nodes
    anchor_r = {i: [] for i in nodes}
    source_r = {i: [] for i in nodes}
    for a in anchors:
        if a.i in anchor_r:
            r, _, _ = anchor_factor(a, kfs, T, p, jac=False)
            anchor_r[a.i].append(np.abs(r[:, 0]))
    views = []
    for f in fixeds:
        if f.i not in source_r:
            continue
        r, _, _ = fixed_factor(f, kfs, T, p, jac=False)
        px = np.linalg.norm(r, axis=1)
        if f.is_source:
            source_r[f.i].append(px)
        else:
            views.append({"i": f.i, "view": f.view, "n": int(px.size),
                          "med_px": float(np.median(px)) if px.size else None})
    node = {}
    for i in nodes:
        kf = kfs[i]
        ra = np.concatenate(anchor_r[i]) if anchor_r[i] else np.zeros(0)
        rf = np.concatenate(source_r[i]) if source_r[i] else np.zeros(0)
        z = (T[i].act(kf.X[kf.valid]) - kf.cam_t) @ kf.cam_R[:, 2]
        depth = float(np.median(z)) if z.size else 0.0
        node[i] = {
            "anchor_n": int(ra.size),
            "anchor_inliers": int((ra < p.anchor_inlier).sum()),
            "anchor_med": float(np.median(ra)) if ra.size else None,
            "fixed_n": int(rf.size),
            "fixed_med_px": float(np.median(rf)) if rf.size else None,
            "positive_depth": float((z > p.depth_eps).mean()) if z.size else 0.0,
            "rot_dev_deg": rotation_angle_deg(T[i].R @ kf.cam_R.T),
            "center_shift_rel": (float(np.linalg.norm(T[i].t - kf.cam_t) / depth)
                                 if depth > p.depth_eps else float("inf")),
        }
    edge = []
    for e in edges:
        r, _, _ = edge_factor(e, kfs, T, p, jac=False)
        px = np.linalg.norm(r[:, :2], axis=1)
        edge.append({"i": e.i, "j": e.j, "n": int(px.size),
                     "med_px": float(np.median(px)) if px.size else None,
                     "med_logdepth": float(np.median(np.abs(r[:, 2]))) if px.size else None})
    return node, edge, views
