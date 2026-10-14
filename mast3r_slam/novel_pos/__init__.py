"""novel_pos: single-shot novel-view completion -> point-cloud update pipeline (ADR-001).

Keep this package __init__ side-effect-free. Backend-free modules — importable without a
GPU or the compiled `mast3r_slam_backends` extension — are `manifest`, `gs_io`,
`pose_utils`, `align`, `gs_project`, and `smoke_test`. The modules `frames`,
`pose_experiment`, `pc_update`, and `pipeline` pull in the CUDA backend and are only safe
on a GPU machine. Do NOT import submodules here.
"""
