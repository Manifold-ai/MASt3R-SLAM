"""Validation gate (ADR-001): geometric residual between the render and completion
pointmaps over anchor pixels.

Backend-free: numpy only. Both pointmaps come from one ``mast3r_match_asymmetric(render,
completion)`` call, so they are in the SAME (render) coordinate frame on the same pixel
grid; at an anchor pixel (``vmatch == True``) they should coincide if the completion
preserved the scene. A large residual means the completion moved/distorted geometry.

This replaces the earlier FrameTracker/Sim3 relative-pose solve, which was degenerate in
translation for a zero-baseline same-viewpoint pair. Report-only; the pipeline may abort
only under ``--strict-gate``.
"""
import numpy as np


def anchor_set(X_render, X_comp, vmatch) -> np.ndarray:
    """(HW,) bool anchor pixels: matched (``vmatch``) and finite in both pointmaps."""
    return (
        np.asarray(vmatch).reshape(-1).astype(bool)
        & np.isfinite(np.asarray(X_render, dtype=np.float64)).all(1)
        & np.isfinite(np.asarray(X_comp, dtype=np.float64)).all(1)
    )


def residual_gate(X_render, X_comp, vmatch, thresholds, min_anchor=100, anchor_mask=None) -> dict:
    """Median geometric residual over anchor pixels, normalized by scene depth.

    Args:
        X_render, X_comp: (HW,3) pointmaps in the render frame (same grid).
        vmatch: (HW,) bool, True where the completion pixel matched the render (anchor).
        thresholds: object with ``.max_rel_residual``.
        anchor_mask: optional (HW,) bool further restricting the anchors (e.g. the render
            visible set V, C-002); None keeps every matched finite pixel.
    Returns {ok, [median_residual, rel_residual, frac_anchor, within_tolerance] | reason}.
    """
    X_render = np.asarray(X_render, dtype=np.float64)
    X_comp = np.asarray(X_comp, dtype=np.float64)
    v = anchor_set(X_render, X_comp, vmatch)
    if anchor_mask is not None:
        v &= np.asarray(anchor_mask).reshape(-1).astype(bool)
    frac = float(v.mean()) if v.size else 0.0
    if int(v.sum()) < min_anchor:
        return {"ok": False, "reason": "insufficient_anchor",
                "frac_anchor": frac, "within_tolerance": False}

    resid = np.linalg.norm(X_comp[v] - X_render[v], axis=1)
    depth = float(np.median(np.linalg.norm(X_render[v], axis=1)))
    med = float(np.median(resid))
    rel = float(med / max(depth, 1e-9))
    return {
        "ok": True,
        "median_residual": med,
        "rel_residual": rel,
        "frac_anchor": frac,
        "within_tolerance": bool(rel <= thresholds.max_rel_residual),
    }
