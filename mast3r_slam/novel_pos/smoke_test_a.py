"""Backend-free smoke test for the ADR-002 A geometric merge (SLAM-C-003).

Runs WITHOUT a GPU, the MASt3R model, or the compiled CUDA backend. Needs numpy, plyfile
and pyyaml.

    python -m mast3r_slam.novel_pos.smoke_test_a
"""
import os
import pathlib
import tempfile

import numpy as np
import yaml

from mast3r_slam.novel_pos import cloud_assets, geo_build, merge_a
from mast3r_slam.novel_pos import geo_graph as G
from mast3r_slam.novel_pos.manifest_a import load_manifest_a
from mast3r_slam.novel_pos.sim3 import Sim3, matrix_to_quat_wxyz, point_jacobian, so3_exp, so3_log
from mast3r_slam.novel_pos.gs_project import quat_wxyz_to_matrix

H, W = 48, 64
K = np.array([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1.0]])


# ------------------------------------------------------------------ synthetic scene
# World (OpenCV axes): a wall z = 5 and a floor y = 1.5. Keyframe i sees it from a known
# camera; its MASt3R cloud is the true camera-frame cloud divided by an unknown scale s_i.

def _true_cam_points(R, c):
    v, u = np.mgrid[0:H, 0:W]
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    ray = ray.reshape(-1, 3)
    d = ray @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_wall = (5.0 - c[2]) / d[:, 2]
        lam_floor = np.where(d[:, 1] > 1e-9, (1.5 - c[1]) / d[:, 1], np.inf)
    lam = np.minimum(np.where(lam_wall > 0, lam_wall, np.inf), lam_floor)
    return ray * lam[:, None]


def _scene(n=3, seed=0):
    rng = np.random.RandomState(seed)
    cams = [(so3_exp([0.05 * k, 0.15 * k - 0.15, 0.02 * k]), np.array([0.4 * k - 0.4, -0.1 * k, 0.1 * k]))
            for k in range(n)]
    scales = [0.4, 1.7, 0.9, 2.5][:n]
    kfs, T_true = [], []
    for (R, c), s in zip(cams, scales):
        Xc = _true_cam_points(R, c)
        kfs.append(G.LocalCloud(X=Xc / s, conf=np.full(H * W, 3.0), valid=np.isfinite(Xc).all(1),
                                K=K, shape=(H, W), cam_R=R, cam_t=c))
        T_true.append(Sim3(s, R, c))
    return kfs, T_true, rng


def _anchor(i, kfs, T_true, rng, region=lambda w: w[:, 0] < 0.5):
    w = T_true[i].act(kfs[i].X)
    pix = np.nonzero(kfs[i].valid & region(w) & (np.abs(w[:, 2] - 5) < 1e-6))[0]
    return G.AnchorObs(i=i, pix=pix, target=w[pix] * (1 + 0.001 * rng.randn(len(pix), 1)),
                       q=np.full(len(pix), 3.0))


def _edge(i, j, kfs, T_true):
    """Points of j matched to the pixel of i where they project (rounded), as MASt3R would."""
    y = T_true[i].inv_act(T_true[j].act(kfs[j].X))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(K[0, 0] * y[:, 0] / y[:, 2] + K[0, 2])
        v = np.round(K[1, 1] * y[:, 1] / y[:, 2] + K[1, 2])
    ok = kfs[j].valid & (y[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.where(ok, v * W + u, 0).astype(np.int64)
    return G.EdgeObs(i=i, j=j, idx=idx, valid=ok, q=np.full(H * W, 3.0))


# --------------------------------------------------------------------------- tests

def test_sim3_basics():
    rng = np.random.RandomState(0)
    for _ in range(20):
        R = so3_exp(rng.randn(3) * rng.uniform(0, 3))
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-9)
        assert np.allclose(quat_wxyz_to_matrix(matrix_to_quat_wxyz(R)), R, atol=1e-9)
    T = Sim3(2.0, so3_exp(rng.randn(3)), rng.randn(3))
    X = rng.randn(6, 3)
    assert np.allclose(T.inv_act(T.act(X)), X)
    J = point_jacobian(T.act(X))
    for k in range(7):
        d = np.zeros(7)
        d[k] = 1e-6
        num = (T.retract(d).act(X) - T.retract(-d).act(X)) / 2e-6
        assert np.allclose(num, J[:, :, k], atol=1e-6), k
    print("OK: sim3 exp/log/quat/act + left-perturbation point Jacobian")


def test_factor_jacobians():
    kfs, T_true, rng = _scene(2)
    T = [t.retract(rng.randn(7) * 0.02) for t in T_true]
    p = G.GeoParams(pixel_border=-1e6)
    view = G.FixedObs(i=1, pix=np.arange(0, H * W, 7), uv=rng.rand(len(range(0, H * W, 7)), 2) * 40,
                      q=np.full(len(range(0, H * W, 7)), 3.0), R=so3_exp([0.02, 0, 0]),
                      t=np.array([0.1, 0, 0]), K=K, shape=(H, W), view="cut")
    T0 = T_true[0].retract(np.r_[0, 0, 0, 0.3, -0.2, 0.1, 0.2])
    e, a, src = _edge(0, 1, kfs, T_true), _anchor(0, kfs, T_true, rng), geo_build.source_fixed_obs(1, kfs[1], p)
    facs = {
        "edge": lambda T: G.edge_factor(e, kfs, T, p, compact=False),
        "anchor": lambda T: G.anchor_factor(a, kfs, T, p, compact=False),
        "source": lambda T: G.fixed_factor(src, kfs, T, p, compact=False),
        "view": lambda T: G.fixed_factor(view, kfs, T, p, compact=False),
        "prior": lambda T: G.prior_factor(0, T0, T[0], p),
    }
    for name, fn in facs.items():
        r, sw, blocks = fn(T)
        ok = sw[:, 0] > 0
        assert ok.any(), name
        for node, sign, J in blocks:
            J = sign * J
            for k in range(7):
                d = np.zeros(7)
                d[k] = 1e-6
                Tp, Tm = list(T), list(T)
                Tp[node], Tm[node] = T[node].retract(d), T[node].retract(-d)
                num = (fn(Tp)[0] - fn(Tm)[0]) / 2e-6
                err = np.abs(num[ok] - J[ok][:, :, k]).max()
                assert err < 1e-4 * max(1.0, np.abs(J[ok][:, :, k]).max()), (name, node, k, err)
    print("OK: factor Jacobians (edge, anchor, source view, fixed view, prior) vs finite diff")


def test_constrain_to_rays():
    rng = np.random.RandomState(1)
    X = rng.randn(H * W, 3) + np.array([0, 0, 4.0])
    Xr = geo_build.constrain_to_rays(X, K, (H, W))
    uv = geo_build.pixel_grid((H, W))
    assert np.allclose(Xr[:, 2], X[:, 2])
    assert np.allclose(K[0, 0] * Xr[:, 0] / Xr[:, 2] + K[0, 2], uv[:, 0])
    assert np.allclose(K[1, 1] * Xr[:, 1] / Xr[:, 2] + K[1, 2], uv[:, 1])
    print("OK: constrain_to_rays keeps z and puts points on their pixel rays")


def _graph(kfs, T_true, rng, anchored=(0, 1)):
    p = G.GeoParams(min_edge_matches=300)
    anchors = [_anchor(i, kfs, T_true, rng) if i in anchored
               else G.AnchorObs(i, np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0))
               for i in range(len(kfs))]
    T_init, has_scale = [], set()
    for i, kf in enumerate(kfs):
        Ti, n = geo_build.init_transform(kf, anchors[i], p)
        T_init.append(Ti)
        if n >= p.min_anchor:
            has_scale.add(i)
    return p, anchors, T_init, has_scale


def test_init_and_gn_recover_truth():
    kfs, T_true, rng = _scene(3)
    p, anchors, T_init, has_scale = _graph(kfs, T_true, rng, anchored=(0, 1))
    assert has_scale == {0, 1}
    for i in (0, 1):                                        # anchor scale init ~ truth
        assert abs(T_init[i].s / T_true[i].s - 1) < 0.01, (i, T_init[i].s, T_true[i].s)
    edges, gates = [], []
    for i, j, _ in geo_build.candidate_pairs(kfs, T_init, p):
        e1, e2 = _edge(i, j, kfs, T_true), _edge(j, i, kfs, T_true)
        g1, g2 = geo_build.edge_gate(e1, kfs, T_init, p), geo_build.edge_gate(e2, kfs, T_init, p)
        gates += [g1, g2]
        if g1["pass"] and g2["pass"]:
            edges += [e1, e2]
    assert edges, gates
    T_init, scaled = geo_build.propagate_scale(kfs, T_init, edges, has_scale, p)
    assert 2 in scaled and abs(T_init[2].s / T_true[2].s - 1) < 0.05, T_init[2].s
    # perturb the known-camera init to show GN pulls it back
    T_start = [t.retract(np.r_[0.03, -0.02, 0.02, 0.01, -0.01, 0.008, 0.05]) for t in T_init]
    fixeds = [geo_build.source_fixed_obs(i, kf, p) for i, kf in enumerate(kfs)]
    T, info = G.solve(kfs, T_start, edges, anchors, fixeds, p)
    T_ref, info_ref = G.solve(kfs, T_true, edges, anchors, fixeds, p)
    assert info["converged"] and info["cost"][-1] < info["cost"][0], info["cost"]
    assert abs(info["cost"][-1] / info_ref["cost"][-1] - 1) < 1e-3   # same optimum from both starts
    for i in range(3):
        rot = np.degrees(np.linalg.norm(so3_log(T[i].R @ T_true[i].R.T)))
        # tolerances reflect rounded-pixel matches on a 64x48 grid (1 px ~ 1.1 deg here)
        assert abs(T[i].s / T_true[i].s - 1) < 0.01 and rot < 0.3, (i, T[i].s, T_true[i].s, rot)
        assert np.linalg.norm(T[i].t - T_true[i].t) < 0.03, (i, T[i].t, T_true[i].t)
        assert np.degrees(np.linalg.norm(so3_log(T[i].R @ T_ref[i].R.T))) < 0.01
    node, edge_st, view_st = G.residual_stats(kfs, T, edges, anchors, fixeds, p)
    status, edges_ok, views_bad = merge_a.admit(node, edge_st, view_st, T, p)
    assert all(status[i]["admitted"] for i in range(3)), status
    assert len(edges_ok) == len(edges) // 2 and not views_bad
    assert status[0]["anchored"] and not status[2]["anchored"]   # 2 admitted via edges only
    print("OK: anchor scale init + edge scale propagation + GN recover the true Sim(3)s")


def test_admission_isolated_and_bad_anchor():
    kfs, T_true, rng = _scene(4)
    p, anchors, T_init, _ = _graph(kfs, T_true, rng, anchored=(0, 1, 2))
    a2 = anchors[2]                                          # keyframe 2: P0 anchors 30% too far
    anchors[2] = G.AnchorObs(2, a2.pix, kfs[2].cam_t + (a2.target - kfs[2].cam_t) * 1.3, a2.q)
    T_init[2] = Sim3(T_true[2].s, T_true[2].R, T_true[2].t)  # isolate the anchor effect
    fixeds = [geo_build.source_fixed_obs(i, kf, p) for i, kf in enumerate(kfs)]
    T, _ = G.solve(kfs, T_init, [], anchors, fixeds, p)      # no edges: 3 is isolated
    node, edge_st, view_st = G.residual_stats(kfs, T, [], anchors, fixeds, p)
    status, _, _ = merge_a.admit(node, edge_st, view_st, T, p)
    assert status[0]["admitted"] and status[1]["admitted"], status
    assert status[2]["admitted"]                              # consistent anchors: scale absorbs 1.3
    assert not status[3]["admitted"] and "not connected" in status[3]["reasons"][0], status[3]
    # now inconsistent anchors: half pushed 30% further -> anchors disagree -> provisional
    tgt = anchors[2].target.copy()
    half = np.arange(len(tgt)) % 2 == 0
    tgt[half] = kfs[2].cam_t + (tgt[half] - kfs[2].cam_t) * 1.3
    anchors[2] = G.AnchorObs(2, anchors[2].pix, tgt, anchors[2].q)
    T, _ = G.solve(kfs, T_init, [], anchors, fixeds, p)
    node, edge_st, view_st = G.residual_stats(kfs, T, [], anchors, fixeds, p)
    status, _, _ = merge_a.admit(node, edge_st, view_st, T, p)
    assert not status[2]["admitted"] and "disagree" in status[2]["reasons"][0], status[2]
    print("OK: admission (isolated keyframe and disagreeing P0 anchors stay provisional)")


def test_gate_placeholder_scale_and_bad_edge_precheck():
    # (a) keyframe 2 has no anchors and a true scale 20x off the placeholder: the count/spread
    # gate must not depend on scale; after propagation the full gate (with fraction) passes.
    kfs, T_true, rng = _scene(3)
    kfs[2] = G.LocalCloud(X=kfs[2].X * T_true[2].s / 20.0, conf=kfs[2].conf, valid=kfs[2].valid,
                          K=K, shape=(H, W), cam_R=kfs[2].cam_R, cam_t=kfs[2].cam_t)
    T_true[2] = Sim3(20.0, T_true[2].R, T_true[2].t)
    p, anchors, T_init, has_scale = _graph(kfs, T_true, rng, anchored=(0, 1))
    es = [_edge(i, j, kfs, T_true) for i, j in ((0, 2), (2, 0), (1, 2), (2, 1))]
    assert all(geo_build.edge_gate(e, kfs, T_init, p, check_frac=False)["pass"] for e in es)
    T_prop, scaled = geo_build.propagate_scale(kfs, T_init, es, has_scale, p)
    assert 2 in scaled and abs(T_prop[2].s / 20.0 - 1) < 0.05, T_prop[2].s
    assert all(geo_build.edge_gate(e, kfs, T_prop, p)["pass"] for e in es)
    # (b) a coherent but wrong edge between two anchored keyframes (matches implying an 8 deg
    # rotated relative pose, ~7 px on this 64x48 grid) is caught against their independent placements
    kfs, T_true, rng = _scene(2)
    p, anchors, T_init, _ = _graph(kfs, T_true, rng, anchored=(0, 1))
    fixeds = [geo_build.source_fixed_obs(i, kf, p) for i, kf in enumerate(kfs)]
    T_ind = geo_build.independent_placements(kfs, T_init, anchors, fixeds, p, [0, 1])
    good = _edge(0, 1, kfs, T_true)
    T_wrong = list(T_true)
    T_wrong[1] = Sim3(T_true[1].s, so3_exp([0, np.radians(8), 0]) @ T_true[1].R, T_true[1].t)
    bad = _edge(0, 1, kfs, T_wrong)
    assert geo_build.edge_consistency(good, kfs, T_ind, p) < 1.0
    assert geo_build.edge_consistency(bad, kfs, T_ind, p) > p.max_edge_px
    print("OK: edge gate without placeholder scale + anchored-placement check rejects a wrong edge")


def test_fixed_view_named_source_and_solver_status():
    kfs, T_true, rng = _scene(2)
    p, anchors, T_init, _ = _graph(kfs, T_true, rng, anchored=(0, 1))
    src = geo_build.source_fixed_obs(0, kfs[0], p)
    w = T_true[0].act(kfs[0].X[src.pix])
    y = (w - kfs[1].cam_t) @ kfs[1].cam_R
    uv = np.stack([K[0, 0] * y[:, 0] / y[:, 2] + K[0, 2], K[1, 1] * y[:, 1] / y[:, 2] + K[1, 2]], -1)
    view = G.FixedObs(i=0, pix=src.pix, uv=uv + 5.0, q=np.full(len(uv), 3.0), R=kfs[1].cam_R,
                      t=kfs[1].cam_t, K=K, shape=(H, W), view="source")   # a view NAMED "source"
    node, _, views = G.residual_stats(kfs, T_true, [], anchors, [src, view], p)
    assert node[0]["fixed_med_px"] < 0.01, node[0]           # own camera only
    assert len(views) == 1 and abs(views[0]["med_px"] - np.hypot(5, 5)) < 0.5, views
    _, _, views_bad = merge_a.admit(node, [], views, T_true, p)
    assert views_bad == [(0, "source")]
    _, info = G.solve(kfs, T_init, [], anchors, [src], p)
    assert info["status"] in ("converged", "no_descent", "max_iters") and info["iterations"] >= 1
    print("OK: fixed views are told apart by is_source, judged on residual; solver reports status")


def test_merge_conservation_and_ply():
    kfs, T_true, _ = _scene(3)
    clouds = [{"X": kf.X.astype(np.float32), "rgb": np.full((H * W, 3), 7 * i, np.uint8),
               "valid": kf.valid} for i, kf in enumerate(kfs)]
    p0 = np.random.RandomState(2).randn(100, 3).astype(np.float32)
    p0c = np.zeros((100, 3), np.uint8)
    xyz, rgb, sid, pid = merge_a.merge(p0, p0c, clouds, T_true, admitted=[0, 2])
    n_expect = 100 + int(kfs[0].valid.sum()) + int(kfs[2].valid.sum())
    assert len(xyz) == n_expect and set(np.unique(sid)) == {0, 1, 3}
    assert np.array_equal(xyz[:100], p0)                     # P0 untouched
    k = np.nonzero(sid == 3)[0][5]
    assert np.allclose(xyz[k], T_true[2].act(kfs[2].X[pid[k]][None])[0], atol=1e-4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.ply")
        merge_a.write_ply_provenance(path, xyz, rgb, sid, pid)
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"].data
        assert len(v) == n_expect and np.array_equal(v["source_id"], sid)
    print("OK: merge keeps P0 + admitted clouds with duplicates and provenance; PLY round-trip")


def test_sealed_assets():
    a = {"X": np.ones((4, 3), np.float32), "rgb": np.zeros((4, 3), np.uint8),
         "conf": np.ones(4, np.float32), "valid": np.ones(4, bool), "shape": np.array([2, 2]),
         "anchor_pix": np.arange(2)}
    with tempfile.TemporaryDirectory() as d:
        path = cloud_assets.asset_path(d, "kf")
        h = cloud_assets.seal(path, a, {"fingerprint": "f1"})
        got, meta = cloud_assets.load(path)
        assert meta["content_hash"] == h and np.array_equal(got["X"], a["X"])
        assert cloud_assets.load_or_none(path, "f1") is not None
        assert sorted(os.listdir(d)) == ["kf.npz"]             # atomic publish leaves no temp file
        assert cloud_assets.load_or_none(os.path.join(d, "none.npz"), "f1") is None
        for bad in (lambda: cloud_assets.seal(path, a, {}),
                    lambda: cloud_assets.load_or_none(path, "f2")):
            try:
                bad()
            except (FileExistsError, RuntimeError):
                continue
            raise AssertionError("expected refusal")
        z = dict(np.load(path))                              # tamper: rewrite X, keep the hash
        z["X"] = z["X"] + 1
        np.savez(path, **z)
        try:
            cloud_assets.load(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered asset not detected")
    print("OK: sealed assets (write-once, hash check, fingerprint reuse, tamper detection)")


def test_edge_gate_and_anchors():
    kfs, T_true, rng = _scene(2)
    p = G.GeoParams(min_edge_matches=300)
    e = _edge(0, 1, kfs, T_true)
    g = geo_build.edge_gate(e, kfs, T_true, p)
    assert g["pass"] and g["n"] > 300 and g["frac"] > 0.9 and g["cells"] >= 4, g
    few = G.EdgeObs(0, 1, e.idx, e.valid & (np.arange(H * W) < 5 * W), e.q)   # top rows only
    g2 = geo_build.edge_gate(few, kfs, T_true, p)
    assert not g2["pass"], g2
    uv = geo_build.pixel_grid((H, W))                        # many matches, but in one corner
    corner = G.EdgeObs(0, 1, e.idx, e.valid & (uv[:, 0] < W // 4) & (uv[:, 1] < H // 4), e.q)
    p_relaxed = G.GeoParams(min_edge_matches=50, min_match_frac=0.0)
    g3 = geo_build.edge_gate(corner, kfs, T_true, p_relaxed)
    assert g3["n"] >= 50 and g3["cells"] < p.min_edge_cells and not g3["pass"], g3
    hw = H * W
    X_r_GS = np.where((np.arange(hw) % 2 == 0)[:, None], 1.0, np.nan) * np.ones((hw, 3))
    cov = np.isfinite(X_r_GS).all(1)
    vis = np.arange(hw) < hw // 2
    q = np.full(hw, 3.0)
    q[:10] = 1.0                                             # below Q_conf
    a = geo_build.build_anchors(0, np.arange(hw), np.ones(hw, bool), q, X_r_GS, cov, vis,
                                np.ones(hw, bool), p)
    expect = np.nonzero(cov & vis & (q > p.Q_conf))[0]
    assert np.array_equal(a.pix, expect), (len(a.pix), len(expect))
    print("OK: edge quality gate (count/fraction/spread) + anchor gating (Q, coverage, visibility)")


def test_manifest_a():
    with tempfile.TemporaryDirectory() as d:
        dp = pathlib.Path(d)
        for f in ("gs.ply", "r.png", "c.png", "cut.png"):
            (dp / f).write_bytes(b"x")
        kf = {"id": "A", "render_image": "r.png", "completion_image": "c.png",
              "pose": {"translation": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
              "intrinsics": {"width": 4, "height": 4, "fx": 1, "fy": 1, "cx": 2, "cy": 2}}
        m = {"gs_ply": "gs.ply", "keyframes": [kf, dict(kf, id="B")], "output_dir": "out",
             "fixed_views": [{"id": "cut", "image": "cut.png", "pose": kf["pose"],
                              "intrinsics": kf["intrinsics"]}], "geo": {"min_anchor": 50}}
        (dp / "m.yaml").write_text(yaml.safe_dump(m))
        ma = load_manifest_a(str(dp / "m.yaml"))
        assert [k.id for k in ma.keyframes] == ["A", "B"] and ma.fixed_views[0].id == "cut"
        assert ma.output_dir == dp / "out" and G.GeoParams.from_config({}, ma.geo).min_anchor == 50
        for bad in ({**m, "keyframes": [kf, kf]},                           # duplicate id
                    {**m, "keyframes": [dict(kf, completion_image="missing.png")]},
                    {**m, "keyframes": []}, {**m, "geo": {"nope": 1}}):
            (dp / "bad.yaml").write_text(yaml.safe_dump(bad))
            try:
                mb = load_manifest_a(str(dp / "bad.yaml"))
                G.GeoParams.from_config({}, mb.geo)
            except (ValueError, FileNotFoundError):
                continue
            raise AssertionError(f"expected manifest error for {bad}")
    print("OK: multi-keyframe manifest (ids, fixed views, overrides, error cases)")


if __name__ == "__main__":
    test_sim3_basics()
    test_factor_jacobians()
    test_constrain_to_rays()
    test_init_and_gn_recover_truth()
    test_admission_isolated_and_bad_anchor()
    test_gate_placeholder_scale_and_bad_edge_precheck()
    test_fixed_view_named_source_and_solver_status()
    test_merge_conservation_and_ply()
    test_sealed_assets()
    test_edge_gate_and_anchors()
    test_manifest_a()
    print("smoke test A done")
