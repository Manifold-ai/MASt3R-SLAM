"""novel_pos: single-shot novel-view completion -> point-cloud update pipeline (ADR-001).

Keep this package __init__ side-effect-free. Backend-free modules — importable without a
GPU or the compiled `mast3r_slam_backends` extension — are `manifest`, `gs_io`, `align`,
`gs_project`, `pc_update`, `pose_experiment`, `visibility`, and `smoke_test` (pure numpy).
Only `frames` and `pipeline` pull in the CUDA backend and are GPU-only. Do NOT import
submodules here.

ADR-002 A (SLAM-C-003, multi-keyframe geometric merge) adds the backend-free `sim3`,
`geo_graph`, `geo_build`, `merge_a`, `cloud_assets`, `manifest_a`; only `pipeline_a` is GPU-only.
"""
