"""Admission gates and the duplicate-preserving merge of ADR-002 A (SLAM-C-003 §六, §七).

Backend-free: numpy + plyfile. A keyframe cloud is ADMITTED when it is connected to the world
anchor (directly through P0 anchors, or through admitted edges to an anchored keyframe) and
passes the geometric gates; otherwise it stays PROVISIONAL and is not merged. The merge is a
plain concatenation that keeps duplicates and provenance: no dedup, no averaging, no recolor.
"""
from collections import deque

import numpy as np
from plyfile import PlyData, PlyElement


def admit(node_stats, edge_stats, view_stats, T, p):
    """Judge one solve. Returns (node status, admitted edge pairs, rejected fixed views).

    Edge pairs and fixed views come in already gated; here they are checked on residuals.
    """
    by_pair = {}
    for es in edge_stats:
        key = (min(es["i"], es["j"]), max(es["i"], es["j"]))
        ok = es["med_px"] is not None and es["med_px"] <= p.max_edge_px
        by_pair[key] = by_pair.get(key, True) and ok           # both directions must pass
    edges_ok = sorted(k for k, v in by_pair.items() if v)
    views_bad = sorted((v["i"], v["view"]) for v in view_stats
                       if v["med_px"] is None or v["med_px"] > p.max_view_px)

    status = {}
    for i, st in node_stats.items():
        reasons = []
        anchored = (st["anchor_inliers"] >= p.min_anchor and st["anchor_med"] is not None
                    and st["anchor_med"] <= p.max_anchor_logdepth)
        if st["anchor_n"] >= p.min_anchor and not anchored:     # enough anchors, but they disagree
            reasons.append(f"P0 anchors disagree: median |log-depth| {st['anchor_med']} > "
                           f"{p.max_anchor_logdepth} or inliers {st['anchor_inliers']} < {p.min_anchor}")
        if st["fixed_med_px"] is None or st["fixed_med_px"] > p.max_fixed_px:
            reasons.append(f"source-view residual {st['fixed_med_px']} px > {p.max_fixed_px}")
        if st["rot_dev_deg"] > p.max_rot_dev_deg:
            reasons.append(f"rotation {st['rot_dev_deg']:.3f} deg off the known camera "
                           f"(> {p.max_rot_dev_deg})")
        if st["center_shift_rel"] > p.max_center_shift_rel:
            reasons.append(f"origin {st['center_shift_rel']:.4f} x depth off the camera centre "
                           f"(> {p.max_center_shift_rel})")
        if st["positive_depth"] < p.min_positive_depth:
            reasons.append(f"positive depth {st['positive_depth']:.3f} < {p.min_positive_depth}")
        if not (np.isfinite(T[i].s) and T[i].s > 0):
            reasons.append(f"invalid scale {T[i].s}")
        status[i] = {"anchored": bool(anchored), "gates_ok": not reasons, "reasons": reasons}

    # connectivity to the world anchor through admitted edges between gate-passing nodes
    adj = {i: [] for i in status}
    for a, b in edges_ok:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    seen = {i for i, s in status.items() if s["anchored"] and s["gates_ok"]}
    queue = deque(sorted(seen))
    while queue:
        a = queue.popleft()
        for b in adj[a]:
            if b not in seen and status[b]["gates_ok"]:
                seen.add(b)
                queue.append(b)
    for i, s in status.items():
        s["admitted"] = i in seen
        if not s["admitted"] and s["gates_ok"]:
            s["reasons"].append("not connected to the P0 anchor (no anchors and no admitted "
                                "edge to an anchored keyframe): scale/placement unobservable")
    return status, edges_ok, views_bad


def merge(p0_xyz, p0_rgb, clouds, T, admitted):
    """P_A = P0 (+) every admitted cloud's valid points mapped by T*; duplicates kept.

    clouds: list of dicts with X (HW,3), rgb (HW,3) uint8, valid (HW,) bool.
    Returns xyz float32, rgb uint8, source_id int32 (0 = P0, i+1 = keyframe i), point_id int32
    (P0 vertex index, or the keyframe pixel index).
    """
    xs = [np.asarray(p0_xyz, np.float32)]
    cs = [np.asarray(p0_rgb, np.uint8)]
    sid = [np.zeros(len(p0_xyz), np.int32)]
    pid = [np.arange(len(p0_xyz), dtype=np.int32)]
    for i in sorted(admitted):
        c = clouds[i]
        keep = np.nonzero(c["valid"])[0]
        xs.append(T[i].act(c["X"][keep]).astype(np.float32))
        cs.append(np.asarray(c["rgb"], np.uint8)[keep])
        sid.append(np.full(len(keep), i + 1, np.int32))
        pid.append(keep.astype(np.int32))
    out = (np.concatenate(xs), np.concatenate(cs), np.concatenate(sid), np.concatenate(pid))
    expected = len(p0_xyz) + sum(int(np.count_nonzero(clouds[i]["valid"])) for i in admitted)
    if len(out[0]) != expected:                                  # point-count conservation
        raise RuntimeError(f"merge: {len(out[0])} points != expected {expected}")
    return out


def write_ply_provenance(path, xyz, rgb, source_id, point_id):
    """Binary PLY: x y z (float32), red green blue (uint8), source_id point_id (int32)."""
    v = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"),
                                  ("green", "u1"), ("blue", "u1"), ("source_id", "i4"),
                                  ("point_id", "i4")])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    v["source_id"], v["point_id"] = source_id, point_id
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))
