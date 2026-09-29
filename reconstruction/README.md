# Reconstruction: human video → Wuji Hand + object in simulation

Turns a single human demonstration video (ZED stereo or plain RGB) into a metric
object mesh, a 6-DoF object trajectory, and a Wuji Hand trajectory. It then refines
the grasp and replays it in Genesis.

Each stage runs in its own conda env, because the upstream models (SAM3, MoGe,
RecGen, HaWoR, Lightning Grasp, ...) pin incompatible Python/torch/CUDA versions.
The pipeline scripts activate the right env for each stage.

## Layout

```
reconstruction/
├── run_v2s2r_pipeline.sh                    # main pipeline (Video2Sim2Real)
├── run_reconstruction_pipeline_cotracker.sh # CoTracker/Kabsch variant + grasp refinement
├── launch.sh                                # re-run one take with chosen stages skipped
├── config/
│   ├── paths.sh                 # env names, external checkouts, checkpoints, GPU
│   └── wuji_mano_{left,right}.yaml
├── envs/                        # one conda yml per stage env + create_envs.sh
├── scripts/                     # every pipeline stage and tool (Python)
├── assets/                      # Franka + Wuji URDF
├── human_videos/                # input clips and all outputs, per take (git-ignored)
├── legacy/                      # superseded pipelines (FoundationPose, SuperDex, Genesis backup)
├── co-tracker/  moge/  recgen/  lightning-grasp/  HOPformer/   # upstream code
├── ec-fit/  epic-contact/  touchanything/                     # contact datasets / fitting
└── modules/Fast-SAM3D/
```

## Setup

### 1. Conda envs

```bash
cd reconstruction
./envs/create_envs.sh              # creates every env in envs/ that does not exist yet
./envs/create_envs.sh genesis sam3 # or only some
```

| Env | Used for |
|---|---|
| `sam3` | frame extraction, SAM3 video segmentation, CoTracker point tracking |
| `sam3d` | semantic parsing (Gemini), SAM 3D point maps |
| `moge` | camera intrinsics, MoGe monocular depth |
| `recgen` | object mesh from RGB + depth + mask |
| `hawor` | HaWoR hand reconstruction (MANO) |
| `wuji_retargeting` | MANO → Wuji Hand retargeting |
| `genesis` | ground plane, trajectory optimization, pose fitting, keyframes, replay, contact refinement |
| `tapnet` | BootsTAPIR point tracking (v2s2r default tracker) |
| `lygra` | Lightning Grasp grasp synthesis |
| `superdex` | grasp refinement (`refine_grasp.py`, CoTracker pipeline) |
| `ecfit` | EC-Fit (`ec-fit/`), used by `run_ecfit.py` |
| `foundationpose` | legacy FoundationPose pipeline only |

The ymls hold the conda packages that were explicitly requested plus pinned pip
packages. Git-installed packages are pinned to a commit. Packages installed from a
local checkout (e.g. `sam3`, `tapnet`, `recgen`, `cotracker`) cannot be installed from
an index, so each yml's header lists them for you to `pip install` by hand, and
`create_envs.sh` prints them after creating the env.

The ZED depth stage (`extract_zed_depth.py`) runs outside these envs, with the conda
base interpreter where the ZED SDK installed `pyzed` (`PYTHON_ZED` in `paths.sh`).
Without an `.svo2` next to the video, the pipelines fall back to MoGe depth.

To refresh a yml after changing an env, re-export it the same way: `conda env export
-n NAME --from-history` for the conda part, and the pip section of `conda env export
-n NAME --no-builds`.

### 2. External checkouts and models

`config/paths.sh` holds every path, and each one can be overridden from the environment:

| Variable | Default | What |
|---|---|---|
| `RECGEN_DIR` | `~/recgen` | RecGen inference code |
| `MOGE_REPO` | `~/MoGe` | MoGe |
| `HAWOR_DIR` | `~/do-as-i-do/reconstruction/modules/HaWoR` | HaWoR + weights |
| `WUJI_RETARGETING_DIR` | `~/wuji-ego-mint/eval/simulate/wuji-retargeting` | Wuji URDF + retargeter |
| `TAPNET_CKPT` | `~/do-as-i-do/.../bootstapir_checkpoint_v2.pt` | BootsTAPIR weights |
| `LIGHTNING_GRASP_DIR` | `reconstruction/lightning-grasp` | vendored (Wuji config registered there) |
| `PANDA_WUJI_URDF_DIR` | `~/genesis-world/robots/panda_wuji/urdf` | calibrated Panda + Wuji URDFs |
| `FOUNDATIONPOSE_DIR` | `~/FoundationPose` | legacy only |

Also in `paths.sh`: `CUDA_VISIBLE_DEVICES`, `SAM3_DISPLAY` (the X display for
click-based segmentation), and the checkpoints `RECGEN_CHECKPOINT` / `MOGE_CHECKPOINT`.

Semantic parsing (v2s2r stage 1) needs `GEMINI_API_KEY`, set in the environment or in
`reconstruction/.env` (git-ignored). Without the key, the stage is skipped and the
`OBJECT` argument is used instead.

The contact tools expect the MANO models in `~/.cache/mano_models/MANO_{LEFT,RIGHT}.pkl`.

## Running

Both pipelines take the same arguments. `FRAME_N` is the reference frame: pick one
where the object is fully visible and unoccluded.

```bash
./run_v2s2r_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
./run_v2s2r_pipeline.sh human_videos/hrdexdb/spray_bottle/23029839.mp4 0 "spray bottle" right

./run_reconstruction_pipeline_cotracker.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
./run_reconstruction_pipeline_cotracker.sh human_videos/pouring_a_cup/take001_pouring_a_cup_rgb.mp4 49 cup left
```

All outputs go next to the video. The stage list, with the env of each stage, is at
the top of each script.

### `run_v2s2r_pipeline.sh` (main)

Follows *Video2Sim2Real* (arXiv 2606.08828):

semantic parsing → SAM3 → intrinsics → depth → ground plane → RecGen mesh → HaWoR →
Wuji retargeting → trajectory optimization → point tracking + Kabsch → keyframes
(contact / detachment) → Lightning Grasp at keyframes → keyframe refinement → Genesis replay.

Departures from the paper: RecGen replaces SAM 3D for the mesh; physical property
estimation is not implemented; the hand is free-floating (no arm IK). The header of
the script covers each of these.

Useful knobs: `TRACKER=cotracker` (default `tapir`), `DEPTH_SOURCE=moge`,
`SAM3_OFFLOAD=1` (keeps frames in CPU RAM when SAM3 runs out of GPU memory),
`SEMANTIC_OVERRIDE=0`.

### `run_reconstruction_pipeline_cotracker.sh`

Uses CoTracker points lifted with depth for the object trajectory, then Lightning
Grasp at the grasp windows, grasp refinement under SuperDex, and a Genesis replay. It
also computes SAM 3D point maps checked against ZED depth. `launch.sh` re-runs it on
one take with the slow stages skipped.

### Resuming and skipping stages

Every stage has a `SKIP_*=1` flag, so a run can resume without redoing the slow
parts. For example:

```bash
SKIP_SEMANTIC=1 SKIP_EXTRACT_AND_SAM3=1 SKIP_DEPTH=1 SKIP_MESH=1 SKIP_HAWOR=1 \
  ./run_v2s2r_pipeline.sh human_videos/grasping_cup/take_006v3/take006_grasping_cup_rgb.mp4 49 cup left
```

The full flag list of each pipeline is in its header.

## Contact-constrained refinement (EPIC-Contact)

These tools use hand-object contact labels to refine the reconstructed object pose
against the HaWoR hand. The labels come either from EPIC-Contact or from our own
annotation. Run them in the `genesis` env from `scripts/`:

| Script | What |
|---|---|
| `prepare_epic_contact_clip.py` | cut an EPIC-Kitchens clip (use `--align-central`) + `epic_frames.json` |
| `annotate_contact.py` (+ `annotate_contact_ui.html`) | paint EPIC-format contact on your own clip in a browser |
| `contact_transfer.py` | MANO fitting and contact-patch transfer, shared by the two above |
| `refine_object_contact.py` | refine the object trajectory so the labelled hand region touches its surface |
| `eval_contact_reconstruction.py` | before/after metrics |
| `run_ecfit.py` | run EC-Fit on an annotated frame |

The typical flow runs the pipeline up to tracking, then annotates, refines and evaluates:

```bash
V=human_videos/grasping_cup/take_006v3
SKIP_KEYFRAMES=1 SKIP_KEYFRAME_GRASPS=1 SKIP_REFINE=1 SKIP_REPLAY=1 \
  ./run_v2s2r_pipeline.sh $V/take006_grasping_cup_rgb.mp4 49 cup left
conda activate genesis && cd scripts
python annotate_contact.py --video-dir ../$V --object cup --side left --frame 170 --contact-frames 114 233
python refine_object_contact.py --hand-meshes ../$V/take006_grasping_cup_rgb/all_hand_meshes.npz --side left \
  --object-mesh ../$V/recgen_out_zed/cup/posed_mesh.obj --object-poses ../$V/cup_traj_v2s2r.npz \
  --epic-clip-dir ../$V/contact_annotation/central_frame --epic-frames ../$V/contact_annotation/epic_frames.json \
  --intrinsics ../$V/intrinsics.yaml --out-dir ../$V/contact_refined
```

Caveats:

- EPIC contact pairs are accurate only region to region (~28 mm), not point to point.
- EPIC object meshes are category templates and can have the wrong shape.
- To test `annotate_contact.py`, pass `--out-dir` pointing at a scratch copy. The page
  reloads `annotation.json`, so a test run writes into the real annotation.

Results and per-clip notes: `human_videos/epic_contact/NOTES.md`.

## Other tools in `scripts/`

| Script | What |
|---|---|
| `render_reconstruction.py`, `compare_kalman_reconstruction.py` | render a reconstruction; compare smoothers |
| `compare_point_lifts.py` | same point tracks lifted with ZED vs MoGe depth (`SKIP_COMPARE_LIFTS=0`) |
| `overlay_object_traj.py`, `overlay_pointmap.py` | draw a trajectory / point map on the video |
| `show_keyframe_grasps.py` | render Lightning Grasp candidates at the keyframes |
| `verify_wuji_lygra.py` | re-derive and check the Wuji entries in Lightning Grasp's config (`lygra` env) |
| `make_franka_wuji_urdf.py` | join an arm and a hand URDF (other arms/hands; the Panda+Wuji one is calibrated) |

## Legacy

`legacy/` holds superseded variants, kept so their results can be reproduced:

- `run_reconstruction_pipeline.sh`: the original point-tracking/Kabsch pipeline with Franka IK.
- `run_reconstruction_pipeline_foundationpose.sh`: FoundationPose object tracking.
- `run_reconstruction_pipeline_superdex.sh`: SuperDex as the simulator.
- `scripts/`: stages only these pipelines use (FoundationPose tracking, hand anchoring,
  Franka IK).
- `backup_genesis/`: a frozen copy of the Genesis-only pipeline from before the SuperDex port.

The three pipelines read the shared `config/paths.sh` and `scripts/`;
`backup_genesis/` is a self-contained copy and is not meant to be run in place.
