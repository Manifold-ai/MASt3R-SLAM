"""Graph construction helpers for the ADR-002 A merge (SLAM-C-003). Backend-free: numpy only.

Turns per-keyframe MASt3R outputs into solver inputs: ray-constrained local clouds, P0 depth
anchors, initial transforms from the known cameras, candidate edges from known-pose overlap,
the edge quality gate, and scale propagation for keyframes without anchors.
"""
from collections import deque

import numpy as np

from mast3r_slam.novel_pos.geo_graph import (
    AnchorObs, EdgeObs, FixedObs, GeoParams, LocalCloud, edge_factor, solve,
)
from mast3r_slam.novel_pos.sim3 import Sim3


def pixel_grid(shape) -> np.ndarray:
    """(H*W, 2) float pixel coords (u, v), row-major like the MASt3R pointmaps."""
    H, W = shape
    v, u = np.mgrid[0:H, 0:W]
    return np.stack([u.reshape(-1), v.reshape(-1)], -1).astype(np.float64)


def constrain_to_rays(X, K, shape) -> np.ndarray:
    """Keep each point's depth z and put it on its pixel's ray: ``z * K^-1 [u, v, 1]``.

    numpy mirror of ``geometry.constrain_points_to_ray`` (the calibrated MASt3R-SLAM path).
    """
    X = np.asarray(X, dtype=np.float64)
    uv = pixel_grid(shape)
    z = X[:, 2]
    return np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * z, (uv[:, 1] - K[1, 2]) / K[1, 1] * z, z], -1)


def build_anchors(i, idx, valid_match, q, X_r_GS, cov, vis, kf_valid, p: GeoParams) -> AnchorObs:
    """P0 depth anchors for keyframe i from its render<->completion asymmetric match.

    Completion pixel c anchors to the P0 point projected at its matched render pixel idx[c]
    when the match is valid, its confidence passes Q_conf, the render pixel is covered by the
    GS projection (finite) and, if a visibility map is given, visible there (C-002).
    """
    idx = np.asarray(idx).reshape(-1).astype(np.int64)
    ok = (np.asarray(valid_match).reshape(-1).astype(bool) & (np.asarray(q).reshape(-1) > p.Q_conf)
          & np.asarray(kf_valid, bool) & cov[idx] & np.isfinite(X_r_GS[idx]).all(1))
    if vis is not None:
        ok &= np.asarray(vis, bool)[idx]
    pix = np.nonzero(ok)[0]
    return AnchorObs(i=i, pix=pix, target=np.asarray(X_r_GS[idx[pix]], np.float64),
                     q=np.asarray(q).reshape(-1)[pix].astype(np.float64))


def source_fixed_obs(i, kf: LocalCloud, p: GeoParams) -> FixedObs:
    """The keyframe's own known camera as a fixed observation: point q must stay on pixel q."""
    pix = np.nonzero(kf.valid)[0][::max(1, int(p.fixed_stride))]
    return FixedObs(i=i, pix=pix, uv=pixel_grid(kf.shape)[pix], q=np.ones(len(pix)),
                    R=kf.cam_R, t=kf.cam_t, K=kf.K, shape=kf.shape, view="source", is_source=True)


def fixed_view_obs(i, idx, valid_match, q, view_R, view_t, view_K, view_shape, view_id,
                   kf_valid, p: GeoParams) -> FixedObs:
    """Fixed view v: keyframe pixel c matched to view pixel idx[c] (asymmetric(view, kf))."""
    idx = np.asarray(idx).reshape(-1).astype(np.int64)
    q = np.asarray(q).reshape(-1).astype(np.float64)
    ok = np.asarray(valid_match).reshape(-1).astype(bool) & (q > p.Q_conf) & np.asarray(kf_valid, bool)
    pix = np.nonzero(ok)[0]
    W = view_shape[1]
    uv = np.stack([idx[pix] % W, idx[pix] // W], -1).astype(np.float64)
    return FixedObs(i=i, pix=pix, uv=uv, q=q[pix], R=np.asarray(view_R, np.float64),
                    t=np.asarray(view_t, np.float64), K=np.asarray(view_K, np.float64),
                    shape=tuple(view_shape), view=view_id)


def init_transform(kf: LocalCloud, anchor: AnchorObs, p: GeoParams):
    """T_i^0 = (s, R_c2w, camera center); s = median P0/local depth ratio over anchors.

    Returns (Sim3, n_used). n_used < p.min_anchor means the scale is a placeholder (1.0).
    """
    z_loc = kf.X[anchor.pix, 2]
    z_p0 = (anchor.target - kf.cam_t) @ kf.cam_R[:, 2]
    ok = (z_loc > p.depth_eps) & (z_p0 > p.depth_eps) & (anchor.q > p.Q_conf)
    n = int(ok.sum())
    s = float(np.median(z_p0[ok] / z_loc[ok])) if n >= p.min_anchor else 1.0
    return Sim3(s, kf.cam_R, kf.cam_t), n


def overlap(kf_j: LocalCloud, T_j: Sim3, kf_i: LocalCloud, stride=8, eps=1e-6) -> float:
    """Fraction of j's valid points (subsampled) that land inside camera i in front of it."""
    pts = kf_j.X[np.nonzero(kf_j.valid)[0][::stride]]
    if len(pts) == 0:
        return 0.0
    y = (T_j.act(pts) - kf_i.cam_t) @ kf_i.cam_R
    z = y[:, 2]
    front = z > eps
    zs = np.where(front, z, 1.0)
    u = kf_i.K[0, 0] * y[:, 0] / zs + kf_i.K[0, 2]
    v = kf_i.K[1, 1] * y[:, 1] / zs + kf_i.K[1, 2]
    H, W = kf_i.shape
    inside = front & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    return float(inside.mean())


def candidate_pairs(kfs, T, p: GeoParams):
    """Manifest-adjacent pairs plus, per keyframe, the best-overlapping others.

    Returns sorted [(i, j, overlap)] with i < j. Adjacent pairs are only CANDIDATES: they
    still have to pass the edge quality gate (ADR-002 A.2).
    """
    n = len(kfs)
    ov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ov[i, j] = overlap(kfs[j], T[j], kfs[i])
    sym = np.maximum(ov, ov.T)
    pairs = {(i, i + 1) for i in range(n - 1)}
    for i in range(n):
        order = [j for j in np.argsort(-sym[i]) if j != i and sym[i, j] >= p.min_overlap]
        for j in order[:p.max_edges_per_kf]:
            pairs.add((min(i, j), max(i, j)))
    return sorted((i, j, float(sym[i, j])) for i, j in pairs)


def overlap_mask(kf_j: LocalCloud, T_j: Sim3, kf_i: LocalCloud, eps=1e-6) -> np.ndarray:
    """(HW_j,) bool: valid points of j that project inside camera i (the overlap region)."""
    y = (T_j.act(kf_j.X) - kf_i.cam_t) @ kf_i.cam_R
    z = y[:, 2]
    front = kf_j.valid & (z > eps)
    zs = np.where(front, z, 1.0)
    u = kf_i.K[0, 0] * y[:, 0] / zs + kf_i.K[0, 2]
    v = kf_i.K[1, 1] * y[:, 1] / zs + kf_i.K[1, 2]
    H, W = kf_i.shape
    return front & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)


def edge_gate(e: EdgeObs, kfs, T, p: GeoParams, check_frac=True) -> dict:
    """Quality gate for one direction: absolute count, fraction within the overlap region,
    and spatial spread over a grid of cells (ADR-002 A.2: no exemption for adjacency).

    The overlap region places j's cloud with T[e.j]; pass check_frac=False while j's scale is
    still a placeholder (the fraction would be measured against a wrongly scaled region).
    """
    kj = kfs[e.j]
    good = np.asarray(e.valid, bool) & (np.asarray(e.q) > p.Q_conf) & kj.valid
    n = int(good.sum())
    H, W = kj.shape
    g = p.grid_cells
    uv = pixel_grid(kj.shape)[good]
    cells = np.unique((np.minimum(uv[:, 1] * g // H, g - 1) * g
                       + np.minimum(uv[:, 0] * g // W, g - 1)).astype(np.int64)) if n else []
    out = {"i": e.i, "j": e.j, "n": n, "cells": int(len(cells)), "frac": None,
           "overlap_px": None, "frac_checked": bool(check_frac)}
    ok = n >= p.min_edge_matches and len(cells) >= p.min_edge_cells
    if check_frac:
        ovm = overlap_mask(kj, T[e.j], kfs[e.i])
        n_ov = int(ovm.sum())
        out["overlap_px"] = n_ov
        out["frac"] = float((good & ovm).sum() / n_ov) if n_ov else 0.0
        ok = ok and out["frac"] >= p.min_match_frac
    out["pass"] = bool(ok)
    return out


def independent_placements(kfs, T_init, anchors, fixeds, p: GeoParams, nodes):
    """Solve each node alone from its own P0 anchors + known camera (no edges).

    Used to check an edge against evidence that does not contain it: a coherent but wrong
    edge can otherwise be absorbed by the joint solve and then look consistent.
    """
    T = list(T_init)
    for i in nodes:
        Ti, _ = solve(kfs, T_init, [], [a for a in anchors if a.i == i],
                      [f for f in fixeds if f.i == i and f.is_source], p, free=[i])
        T[i] = Ti[i]
    return T


def edge_consistency(e: EdgeObs, kfs, T, p: GeoParams):
    """Median pixel residual of one edge direction at placements T (None if no valid rows)."""
    r, _, _ = edge_factor(e, kfs, T, p, jac=False)
    return float(np.median(np.linalg.norm(r[:, :2], axis=1))) if len(r) else None


def propagate_scale(kfs, T, edges, has_scale, p: GeoParams):
    """BFS over gated edges: give a keyframe without anchors the median depth ratio implied
    by an already-scaled neighbour's matched points. Returns (T, set of scaled nodes)."""
    T = list(T)
    scaled = set(has_scale)
    adj = {}
    for e in edges:
        adj.setdefault(e.j, []).append(e)               # e: points of j matched into i
    queue = deque(sorted(scaled))
    while queue:
        a = queue.popleft()
        for e in [x for es in adj.values() for x in es if x.i == a and x.j not in scaled]:
            kj, ka = kfs[e.j], kfs[a]
            good = (np.asarray(e.valid, bool) & (np.asarray(e.q) > p.Q_conf) & kj.valid
                    & ka.valid[e.idx])
            if good.sum() < p.min_edge_matches:
                continue
            w = T[a].act(ka.X[e.idx[good]])                       # matched points of a, world
            z_w = (w - kj.cam_t) @ kj.cam_R[:, 2]                 # their depth in camera j
            z_l = kj.X[good, 2]
            ok = (z_w > p.depth_eps) & (z_l > p.depth_eps)
            if ok.sum() < p.min_edge_matches:
                continue
            T[e.j] = Sim3(float(np.median(z_w[ok] / z_l[ok])), T[e.j].R, T[e.j].t)
            scaled.add(e.j)
            queue.append(e.j)
    return T, scaled
