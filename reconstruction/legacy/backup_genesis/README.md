# Genesis pipeline, as it stood before the SuperDex port

Kept so the SuperDex move stays a test rather than a one-way door. These are copies,
not symlinks: editing them changes nothing that runs.

    scripts/replay_wuji_genesis.py   replay + --physics + --no-object-physics
    scripts/refine_grasp.py          two-stage grasp refinement (Genesis FK)
    scripts/solve_franka_ik.py       Franka arm trajectory by IK
    run_reconstruction_pipeline.sh                 CoTracker/Kabsch variant
    run_reconstruction_pipeline_foundationpose.sh  FoundationPose variant
    launch.sh                                      per-take re-runner
    config/paths.sh                                env names at the time of the copy

To go back: copy these over the working ones and use ENV_GENESIS (conda env `genesis`).
The measured Genesis results they produced, on grasping_cup/take_006, were:
grasp window 48-131 refined to 5.85 cm of slip with 75 contact frames.
