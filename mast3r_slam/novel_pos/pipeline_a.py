"""ADR-002 A orchestration: seal keyframe clouds -> geometric graph -> Sim(3) GN -> admission ->
duplicate-preserving merge (SLAM-C-003).

Flow: load manifest -> degrade GS to P0 (world, T0 = I) -> per keyframe: one
mast3r_match_asymmetric(render, completion) gives the completion's local cloud (ray-constrained
with the known K) and P0 depth anchors (GS projected into the known camera, as ADR-001) ->
seal it (or reuse a sealed asset with identical inputs) -> T^0 from the known camera + anchor
scale -> candidate pairs from known-pose overlap -> mast3r_match_symmetric per pair + quality
gate (scale propagation in between) -> edges between anchored keyframes checked against their
independent placements -> optional fixed views -> GN (no RGB residual) -> rounds that first
drop residual-rejected edges/views, then provisional keyframes -> merge all admitted clouds
with P0 (duplicates kept) -> merged.ply + transforms.json + report.json.

WARNING: imports the CUDA backend (via the submodules below). GPU-only.
"""
import json
import os
import time

import numpy as np
import torch
import lietorch

from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import (
    load_mast3r, mast3r_match_asymmetric, mast3r_match_symmetric, resize_img,
)

from mast3r_slam.novel_pos import cloud_assets, geo_build, geo_graph, gs_project, merge_a, visibility
from mast3r_slam.novel_pos.frames import build_frame, load_rgb01, load_visibility_alpha
from mast3r_slam.novel_pos.gs_io import degrade_gaussians_to_points, read_gaussian_ply
from mast3r_slam.novel_pos.manifest_a import load_manifest_a

ASSET_SCHEMA = "C-003/v1"
CHECKPOINT = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"  # load_mast3r default


class _Timer:
    def __init__(self):
        self.t = {}

    def add(self, key, t0):
        self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0


class _Cam:
    """Camera-only stand-in for geo_build.overlap's target argument."""

    def __init__(self, R, t, K, shape):
        self.cam_R, self.cam_t, self.K, self.shape = R, t, K, shape


def _np(x):
    return x.detach().cpu().numpy()


@torch.inference_mode()
def _ensure_feat(model, frame):
    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)


def _grid(frame):
    return tuple(int(x) for x in frame.img_true_shape.view(-1).tolist())


def _camera(pose, intr, rgb):
    """Known camera (R_c2w, center, K_frame) on the MASt3R grid of an image."""
    _, transform = resize_img(rgb, 512, return_transformation=True)
    h, w = rgb.shape[:2]
    if (intr.width, intr.height) != (w, h):
        print(f"[merge_a] WARNING: intrinsics {intr.width}x{intr.height} != image {w}x{h}; "
              f"K_frame assumes the image resolution")
    R = gs_project.quat_wxyz_to_matrix(pose.quaternion_wxyz)
    t = np.asarray(pose.translation, dtype=np.float64)
    K = gs_project.intrinsics_to_K_frame(intr.fx, intr.fy, intr.cx, intr.cy, transform)
    return R, t, K


def _checkpoint_id():
    try:
        st = os.stat(CHECKPOINT)
        return {"path": os.path.abspath(CHECKPOINT), "size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {"path": CHECKPOINT, "missing": True}


def _build_asset(model, kf_spec, i, render_frame, comp_frame, render_rgb, R, t, K, shape,
                 p0_xyz, tau, p, tm):
    """One asymmetric inference -> sealed cloud fields + P0 anchors + match/missing records."""
    t0 = time.perf_counter()
    (idx, valid_match, X_r_t, C_r_t, Q_r_t,
     X_c_t, C_c_t, Q_c_t) = mast3r_match_asymmetric(model, render_frame, comp_frame)
    X_comp = _np(X_c_t).reshape(-1, 3).astype(np.float64)
    conf = _np(C_c_t).reshape(-1).astype(np.float32)
    idx = _np(idx).reshape(-1).astype(np.int64)
    valid_match = _np(valid_match).reshape(-1).astype(bool)
    q = np.sqrt(_np(Q_r_t).reshape(-1)[idx] * _np(Q_c_t).reshape(-1))
    tm.add("kf_mast3r", t0)

    t0 = time.perf_counter()
    X = geo_build.constrain_to_rays(X_comp, K, shape)
    valid = np.isfinite(X).all(1) & (X[:, 2] > p.depth_eps) & (conf > p.C_conf)
    X = np.where(valid[:, None], X, 0.0)
    rgb = (_np(comp_frame.uimg).reshape(-1, 3) * 255).round().astype(np.uint8)
    X_r_GS, cov = gs_project.project_gs_to_grid(p0_xyz, R, t, K, shape)
    vis, alpha = None, np.zeros(0, np.float32)
    if kf_spec.visibility_map is not None:
        rh, rw = render_rgb.shape[:2]
        a_grid, a_shape = load_visibility_alpha(kf_spec.visibility_map, (rh, rw))
        if a_shape != tuple(shape):
            raise RuntimeError(f"{kf_spec.id}: visibility grid {a_shape} != {shape}")
        alpha = a_grid.reshape(-1).astype(np.float32)
        vis = visibility.visible_set(alpha, tau)
    anchor = geo_build.build_anchors(i, idx, valid_match, q, X_r_GS, cov, vis, valid, p)
    tm.add("kf_anchor_build", t0)
    return {
        "X": X.astype(np.float32), "rgb": rgb, "conf": conf, "valid": valid,
        "shape": np.asarray(shape, np.int64),
        "anchor_pix": anchor.pix.astype(np.int64), "anchor_target": anchor.target,
        "anchor_q": anchor.q, "K": K, "cam_R": R, "cam_t": t, "vis_alpha": alpha,
        # records for later stages (ADR-002 B reads A's missing/match state); never drop points
        "render_coverage": cov, "match_valid": valid_match, "match_idx": idx.astype(np.int32),
    }


def _edge_obs(model, frames, i, j):
    """Symmetric match of keyframes i, j -> (EdgeObs j->i, EdgeObs i->j)."""
    fi, fj = frames[i], frames[j]
    _ensure_feat(model, fi)
    _ensure_feat(model, fj)
    (idx_i2j, idx_j2i, valid_j, valid_i, Qii, Qjj, Qji, Qij) = mast3r_match_symmetric(
        model, fi.feat, fi.pos, fj.feat, fj.pos, [fi.img_true_shape], [fj.img_true_shape])
    idx_i2j, idx_j2i = _np(idx_i2j)[0].astype(np.int64), _np(idx_j2i)[0].astype(np.int64)
    Qii, Qjj = _np(Qii)[0].reshape(-1), _np(Qjj)[0].reshape(-1)
    q_j = np.sqrt(Qii[idx_i2j] * _np(Qji)[0].reshape(-1))
    q_i = np.sqrt(Qjj[idx_j2i] * _np(Qij)[0].reshape(-1))
    e_ji = geo_graph.EdgeObs(i=i, j=j, idx=idx_i2j, valid=_np(valid_j)[0].reshape(-1).astype(bool), q=q_j)
    e_ij = geo_graph.EdgeObs(i=j, j=i, idx=idx_j2i, valid=_np(valid_i)[0].reshape(-1).astype(bool), q=q_i)
    return e_ji, e_ij


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None                                  # strict JSON has no inf/nan
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _pair(e):
    return (min(e.i, e.j), max(e.i, e.j))


def run_merge_a(manifest_path, device="cuda:0", output_dir=None, vis_threshold=0.5) -> str:
    tm = _Timer()
    t_all = time.perf_counter()
    m = load_manifest_a(manifest_path)
    out_dir = str(output_dir or m.output_dir)
    p = geo_graph.GeoParams.from_config(config.get("local_opt"), m.geo)
    tau = visibility.check_threshold(vis_threshold)
    if config["dataset"]["img_downsample"] != 1:
        raise RuntimeError("merge A requires dataset.img_downsample == 1 (C-003 §三)")
    os.makedirs(out_dir, exist_ok=True)
    assets_dir = os.path.join(out_dir, "assets")

    # (1) P0 = degraded GS, the world (T0 = I); hashed to prove it is never modified
    t0 = time.perf_counter()
    gs_hash = cloud_assets.file_sha256(m.gs_ply)
    gs = read_gaussian_ply(m.gs_ply)
    p0_xyz, p0_rgb = degrade_gaussians_to_points(gs, m.opacity_threshold)
    p0_hash = cloud_assets.arrays_hash(p0_xyz, p0_rgb)
    tm.add("p0_load", t0)
    print(f"[merge_a] P0: {len(p0_xyz)} points from {m.gs_ply}")

    # (2) model
    t0 = time.perf_counter()
    model = load_mast3r(path=CHECKPOINT, device=device)
    model.eval()
    tm.add("model_load", t0)

    # (3) sealed keyframe assets (create once, or reuse when the inputs are identical)
    T_build = lietorch.Sim3.Identity(1, device=device)
    kfs, clouds, anchors, frames, kf_report, sealed = [], [], [], [], [], []
    grid = None
    fp_common = {"schema": ASSET_SCHEMA, "gs": gs_hash, "opacity_threshold": m.opacity_threshold,
                 "img_size": 512, "matching": dict(config.get("matching", {})),
                 "checkpoint": _checkpoint_id(),
                 "Q_conf": p.Q_conf, "C_conf": p.C_conf, "depth_eps": p.depth_eps}
    for i, k in enumerate(m.keyframes):
        t0 = time.perf_counter()
        render_rgb, comp_rgb = load_rgb01(k.render_image), load_rgb01(k.completion_image)
        if render_rgb.shape != comp_rgb.shape:
            raise RuntimeError(f"{k.id}: render {render_rgb.shape} and completion "
                               f"{comp_rgb.shape} must have the same size")
        comp_frame = build_frame(i, comp_rgb, T_build, device=device)
        shape = _grid(comp_frame)
        if grid is None:
            grid = shape
        elif shape != grid:
            raise RuntimeError(f"{k.id}: MASt3R grid {shape} != {grid}; all keyframes must share "
                               f"one grid (symmetric matching, C-003 §三)")
        R, t, K = _camera(k.pose, k.intrinsics, render_rgb)
        fp = cloud_assets.fingerprint(dict(
            fp_common,
            render=cloud_assets.file_sha256(k.render_image),
            completion=cloud_assets.file_sha256(k.completion_image),
            visibility=(cloud_assets.file_sha256(k.visibility_map)
                        if k.visibility_map is not None else None),
            tau=tau if k.visibility_map is not None else None,
            pose=[k.pose.translation, k.pose.quaternion_wxyz], intrinsics=vars(k.intrinsics),
            grid=shape))
        path = cloud_assets.asset_path(assets_dir, k.id)
        tm.add("kf_prepare", t0)
        t0 = time.perf_counter()
        reused = cloud_assets.load_or_none(path, fp)
        tm.add("asset_io", t0)
        if reused is not None:
            asset, meta = reused
            if not (np.allclose(asset["K"], K) and np.allclose(asset["cam_R"], R)
                    and np.allclose(asset["cam_t"], t)):
                raise RuntimeError(f"sealed asset {path}: stored camera differs from the manifest")
            print(f"[merge_a] {k.id}: reusing sealed asset {meta['content_hash'][:12]}")
        else:
            render_frame = build_frame(i, render_rgb, T_build, device=device)
            asset = _build_asset(model, k, i, render_frame, comp_frame, render_rgb, R, t, K,
                                 shape, p0_xyz, tau, p, tm)
            t0 = time.perf_counter()
            digest = cloud_assets.seal(path, asset, {"id": k.id, "fingerprint": fp,
                                                     "n_valid": int(asset["valid"].sum())})
            tm.add("asset_io", t0)
            meta = {"content_hash": digest}
            print(f"[merge_a] {k.id}: sealed {int(asset['valid'].sum())} valid points, "
                  f"{len(asset['anchor_pix'])} P0 anchors -> {path}")
        sealed.append(asset)
        kfs.append(geo_graph.LocalCloud(
            X=asset["X"].astype(np.float64), conf=asset["conf"].astype(np.float64),
            valid=asset["valid"].astype(bool), K=asset["K"],
            shape=tuple(int(x) for x in asset["shape"]), cam_R=asset["cam_R"], cam_t=asset["cam_t"]))
        anchors.append(geo_graph.AnchorObs(i=i, pix=asset["anchor_pix"],
                                           target=asset["anchor_target"], q=asset["anchor_q"]))
        clouds.append({"X": asset["X"], "rgb": asset["rgb"], "valid": asset["valid"]})
        frames.append(comp_frame)
        cov_c = asset["render_coverage"][asset["match_idx"].astype(np.int64)]
        kf_report.append({
            "id": k.id, "asset": path, "content_hash": meta["content_hash"],
            "reused": reused is not None, "n_valid": int(asset["valid"].sum()),
            "n_anchor_candidates": int(len(asset["anchor_pix"])),
            "n_match_render": int(asset["match_valid"].sum()),
            "n_missing_in_render": int((~cov_c).sum()),     # M_missing: recorded, not used to drop
        })

    # (4) initial T from the known cameras; scale from the P0 anchors
    T_init, has_scale = [], set()
    for i, kf in enumerate(kfs):
        Ti, n_used = geo_build.init_transform(kf, anchors[i], p)
        T_init.append(Ti)
        if n_used >= p.min_anchor:
            has_scale.add(i)

    # (5) edges: overlap candidates -> symmetric match -> count/spread gate -> scale
    #     propagation -> full gate (incl. overlap fraction, now with real scales)
    t0 = time.perf_counter()
    pairs = geo_build.candidate_pairs(kfs, T_init, p)
    obs = {(i, j): _edge_obs(model, frames, i, j) for i, j, _ in pairs}
    tm.add("edge_matching", t0)
    t0 = time.perf_counter()
    T_anchor = list(T_init)                          # anchor-only scales (placeholders elsewhere)
    overlap_of = {(i, j): ov for i, j, ov in pairs}
    cur = {k for k, es in obs.items()                # count/spread only: scale-free
           if all(geo_build.edge_gate(x, kfs, T_anchor, p, check_frac=False)["pass"] for x in es)}
    for _ in range(4):                               # propagate over the gated set until stable
        T_init, scaled = geo_build.propagate_scale(kfs, T_anchor, [e for k in sorted(cur) for e in obs[k]],
                                                   has_scale, p)
        gates, nxt = [], set()
        for (i, j), es in obs.items():
            gs_ = [geo_build.edge_gate(x, kfs, T_init, p, check_frac=(x.j in scaled)) for x in es]
            for g in gs_:
                g["overlap"] = overlap_of[(i, j)]
            gates += gs_
            if all(g["pass"] for g in gs_):
                nxt.add((i, j))
        if nxt == cur:
            break
        cur = nxt
    edges = [e for k in sorted(cur) for e in obs[k]]
    fixeds = [geo_build.source_fixed_obs(i, kf, p) for i, kf in enumerate(kfs)]
    tm.add("edge_gating", t0)
    # every edge whose two ends have a scale must agree with placements that do not use it:
    # anchored keyframes solved alone (own anchors + known camera), others at known camera +
    # propagated scale. A coherent but wrong edge is otherwise absorbed by the joint solve.
    t0 = time.perf_counter()
    T_ind = geo_build.independent_placements(kfs, T_init, anchors, fixeds, p, sorted(has_scale))
    tm.add("gn_independent", t0)
    edge_rejects = {}
    for e in edges:
        if e.i in scaled and e.j in scaled:
            med = geo_build.edge_consistency(e, kfs, T_ind, p)
            if med is None or med > p.max_edge_px:
                edge_rejects[_pair(e)] = (f"inconsistent with the independent placements "
                                          f"(median {med} px > {p.max_edge_px})")
    edges = [e for e in edges if _pair(e) not in edge_rejects]
    print(f"[merge_a] edges: {len({_pair(e) for e in edges})}/{len(pairs)} candidate pairs passed "
          f"the gate ({len(edge_rejects)} rejected by the independent-placement check)")

    # (6) optional fixed views (the keyframes' own cameras are already in `fixeds`)
    t0 = time.perf_counter()
    view_report, cand = [], {i: [] for i in range(len(kfs))}
    for v in m.fixed_views:
        v_rgb = load_rgb01(v.image)
        v_frame = build_frame(len(kfs), v_rgb, T_build, device=device)
        if _grid(v_frame) != grid:
            view_report.append({"id": v.id, "matched": [], "reason": f"grid {_grid(v_frame)} != {grid}"})
            continue
        vR, vt, vK = _camera(v.pose, v.intrinsics, v_rgb)
        cam = _Cam(vR, vt, vK, grid)
        matched = []
        for i, kf in enumerate(kfs):
            if geo_build.overlap(kf, T_init[i], cam) < p.min_overlap:
                continue
            idx, vm, _, _, Qv, _, _, Qc = mast3r_match_asymmetric(model, v_frame, frames[i])
            idx = _np(idx).reshape(-1).astype(np.int64)
            q = np.sqrt(_np(Qv).reshape(-1)[idx] * _np(Qc).reshape(-1))
            f = geo_build.fixed_view_obs(i, idx, _np(vm).reshape(-1), q, vR, vt, vK, grid, v.id,
                                         kf.valid, p)
            if len(f.pix) >= p.min_fixed_matches:
                cand[i].append(f)
                matched.append(m.keyframes[i].id)
        view_report.append({"id": v.id, "matched": matched})
    used, view_rejects = {}, {}
    for i, fs in cand.items():                       # per keyframe: the best-matched fixed views
        for f in sorted(fs, key=lambda f: -len(f.pix))[:p.max_fixed_views_per_kf]:
            # like edges: a view must agree with the placement that does not use it, else a
            # strongly weighted view (e.g. a mis-posed cut) drags the keyframe off its camera
            r_v, _, _ = geo_graph.fixed_factor(f, kfs, T_ind, p, jac=False)
            med = float(np.median(np.linalg.norm(r_v, axis=1))) if len(r_v) else None
            if i not in scaled or med is None or med > p.max_view_px:
                view_rejects.setdefault(f.view, []).append(
                    {"keyframe": m.keyframes[i].id, "median_px": med,
                     "reason": "no scale" if i not in scaled else "inconsistent with the independent placement"})
                continue
            fixeds.append(f)
            used.setdefault(f.view, []).append(m.keyframes[i].id)
    for r in view_report:
        r["used_for"] = used.get(r["id"], [])
        r["rejected"] = view_rejects.get(r["id"], [])
    tm.add("fixed_view_matching", t0)

    # (7) GN rounds: first drop residual-rejected edges / fixed views and re-solve, then drop
    #     provisional keyframes and re-solve from T^0; stop when nothing changes
    active = list(range(len(kfs)))
    act_edges = {_pair(e) for e in edges}
    act_views = {(f.i, f.view) for f in fixeds if not f.is_source}
    T = list(T_init)
    rounds, status, views_dropped = [], {}, []
    n_items = len(kfs) + len(act_edges) + len(act_views)
    for rnd in range(1, n_items + 2):                # every non-final round removes >= 1 item
        act = set(active)
        e_act = [e for e in edges if e.i in act and e.j in act and _pair(e) in act_edges]
        a_act = [a for a in anchors if a.i in act]
        f_act = [f for f in fixeds if f.i in act and (f.is_source or (f.i, f.view) in act_views)]
        t0 = time.perf_counter()
        T_round, info = geo_graph.solve(kfs, T_init, e_act, a_act, f_act, p, free=active)
        tm.add("gn_solve", t0)
        t0 = time.perf_counter()
        node_st, edge_st, view_st = geo_graph.residual_stats(kfs, T_round, e_act, a_act, f_act, p,
                                                             nodes=active)
        st, edges_ok, views_bad = merge_a.admit(node_st, edge_st, view_st, T_round, p)
        tm.add("admission", t0)
        for i in active:
            T[i] = T_round[i]
            status[i] = dict(st[i], round=rnd, stats=node_st[i])
        bad_edges = {_pair(e) for e in e_act} - set(edges_ok)
        worst = None                                 # drop ONE item per round: the worst offender
        for es in edge_st:
            k = (min(es["i"], es["j"]), max(es["i"], es["j"]))
            if k in bad_edges:
                r = float("inf") if es["med_px"] is None else es["med_px"] / p.max_edge_px
                if worst is None or r > worst[0]:
                    worst = (r, "edge", k)
        for vs in view_st:
            if (vs["i"], vs["view"]) in set(views_bad):
                r = float("inf") if vs["med_px"] is None else vs["med_px"] / p.max_view_px
                if worst is None or r > worst[0]:
                    worst = (r, "view", (vs["i"], vs["view"]))
        rounds.append({"round": rnd, "active": list(active), "solver": info,
                       "edge_residuals": edge_st, "view_residuals": view_st,
                       "dropped": None if worst is None else [worst[1], worst[2]]})
        print(f"[merge_a] round {rnd}: GN {info['status']} after {info['iterations']} iters, "
              f"cost {info['cost'][0]:.4g}->{info['cost'][-1]:.4g}")
        if worst is not None:                        # edge/view first: it may sink good nodes
            if worst[1] == "edge":
                act_edges.discard(worst[2])
                edge_rejects[worst[2]] = "post-solve median pixel residual above max_edge_px"
            else:
                act_views.discard(worst[2])
                views_dropped.append(worst[2])
            print(f"[merge_a]   dropping {worst[1]} {worst[2]} (residual {worst[0]:.2f}x the "
                  f"threshold); re-solving")
            continue
        nxt = [i for i in active if st[i]["admitted"]]
        print(f"[merge_a]   {len(nxt)}/{len(active)} keyframes admitted")
        if nxt == active or not nxt:
            active = nxt
            break
        active = nxt
    admitted = sorted(active)
    for r in view_report:
        r["dropped_for"] = [m.keyframes[i].id for i, v in views_dropped if v == r["id"]]

    # (8) integrity: sealed assets (file + the arrays actually merged) and P0 unchanged
    t0 = time.perf_counter()
    for r, asset in zip(kf_report, sealed):
        _, meta = cloud_assets.load(r["asset"])
        if (meta["content_hash"] != r["content_hash"]
                or cloud_assets.content_hash(asset) != r["content_hash"]):
            raise RuntimeError(f"sealed asset {r['asset']} changed during the run")
    if cloud_assets.file_sha256(m.gs_ply) != gs_hash:
        raise RuntimeError("P0 gs_ply changed during the run")
    if cloud_assets.arrays_hash(p0_xyz, p0_rgb) != p0_hash:
        raise RuntimeError("P0 points were modified in memory")
    tm.add("integrity", t0)

    # (9) merge (duplicates kept) and export
    t0 = time.perf_counter()
    xyz, rgb, sid, pid = merge_a.merge(p0_xyz, p0_rgb, clouds, T, admitted)
    tm.add("merge", t0)
    t0 = time.perf_counter()
    ply = os.path.join(out_dir, "merged.ply")
    merge_a.write_ply_provenance(ply, xyz, rgb, sid, pid)
    tm.add("export", t0)

    transforms = []
    for i, k in enumerate(m.keyframes):
        s = status.get(i, {"reasons": ["not solved"]})
        ok = i in admitted
        transforms.append({
            "id": k.id, "source_id": i + 1, "status": "admitted" if ok else "provisional",
            "reasons": [] if ok else s.get("reasons", []), "T": T[i].to_dict(),
            "T_init": T_init[i].to_dict(),
            "scale_init_from": ("anchors" if i in has_scale else
                                "edge propagation" if i in scaled else "placeholder"),
            "stats": s.get("stats"), "decided_in_round": s.get("round"),
        })
    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(_jsonable({"world": "P0 (T0 = identity)", "keyframes": transforms}), f, indent=1)
    tm.add("total", t_all)
    n_expected = len(p0_xyz) + sum(int(clouds[i]["valid"].sum()) for i in admitted)
    report = {
        "contract": "SLAM-C-003", "output_ply": ply,
        "counts": {"p0": len(p0_xyz), "admitted": len(admitted), "keyframes": len(kfs),
                   "per_keyframe_valid": {k.id: int(c["valid"].sum()) for k, c in zip(m.keyframes, clouds)},
                   "output": int(len(xyz)), "expected": n_expected,
                   "conserved": int(len(xyz)) == n_expected},
        "hashes": {"gs_ply": gs_hash, "p0_points": p0_hash, "p0_unchanged": True,
                   "assets": kf_report, "assets_unchanged": True},   # (8) raises otherwise
        "edges": {"candidates": [list(x) for x in pairs], "gates": gates,
                  "rejected": {f"{a}-{b}": r for (a, b), r in sorted(edge_rejects.items())},
                  "admitted": sorted(e for e in act_edges if e[0] in admitted and e[1] in admitted)},
        "fixed_views": view_report, "rounds": rounds,
        "params": vars(p), "vis_threshold": tau, "timings_s": tm.t,
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(_jsonable(report), f, indent=1)
    print(f"[merge_a] admitted {len(admitted)}/{len(kfs)} keyframes; wrote {ply} "
          f"({len(xyz)} points = P0 {len(p0_xyz)} + {len(xyz) - len(p0_xyz)} keyframe points)")
    for r in transforms:
        if r["status"] != "admitted":
            print(f"[merge_a] provisional {r['id']}: {'; '.join(r['reasons'])}")
    return ply
