"""novel_pos: single-shot novel-view completion -> point-cloud update pipeline (ADR-001).

Keep this package __init__ side-effect-free: the backend-free modules
(`manifest`, `gs_io`, `pose_utils`) must be importable without a GPU or the compiled
`mast3r_slam_backends` extension, while `frames`/`pose_experiment`/`pc_update`/`pipeline`
pull in the CUDA backend and are only safe on a GPU machine. Do NOT import submodules here.
"""
