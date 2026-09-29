#!/usr/bin/env bash
# Human demo video -> Wuji Hand + object trajectory replayed in Genesis, following
# Video2Sim2Real end to end.
#
# From "Video2Sim2Real: Full-Stack Autonomous Dexterous Skill Acquisition from a Single
# Human Video" (arXiv 2606.08828). This runs the paper's two modules that this repo has
# the pieces for -- digital-twin reconstruction from video, and keyframe refinement --
# rather than the hand-contact grasp-window refinement the other pipelines here use.
#
# The one deliberate departure from the paper:
#
#   * the object mesh comes from RecGen, not sam-3d-objects. SAM 3D does not run on this
#     machine. RecGen reconstructs from the same inputs (reference RGB + metric depth +
#     SAM3 mask) and writes the same posed_mesh.obj every later stage already reads, so
#     the substitution is confined to stage 6 and nothing downstream changes.
#
# What the paper specifies and this does NOT do, said plainly rather than faked:
#
#   * physical-property estimation (the paper's stage 4: material -> mass, friction)
#     has no implementation in this repo. Genesis gets its defaults. semantic_parsing.py
#     does emit a material label per object, so scene.json already carries the input a
#     stage 4 would consume.
#   * the interaction keyframe needs a second, target object. These clips are
#     single-object, so extract_keyframes.py reports it as null and refinement anchors
#     on contact and detachment only.
#   * the paper solves exact IK at each keyframe and applies a damped-least-squares
#     task-space correction so an arm can hold the palm on target. This replays a
#     free-floating hand, so the palm is set directly and there is no arm residual.
#
#   1  semantic parsing: Gemini reads the clip, names the objects, the         [sam3d]
#      manipulated one, the task and a material per object  (paper stage 1)
#   2  frames + SAM3 segmentation: object (+ part) and hand                    [sam3]
#   3  camera intrinsics from the recording's calibration                      [moge]
#   4  metric depth, every frame (ZED stereo, else MoGe)          [zed python / moge]
#   5  supporting-plane fit (real up direction)                                [genesis]
#   6  object mesh from RGB + depth + mask -- RecGen, substituting SAM 3D      [recgen]
#   6b carry a part's mask onto the mesh as a per-vertex label                 [genesis]
#   7  HaWoR hand reconstruction                                               [hawor]
#   8  retarget the hand onto the Wuji Hand                          [wuji_retargeting]
#   8b trajectory optimization (DemoMimic A.1) over the retargeted hand        [genesis]
#   9  object pose tracking: BootsTAPIR point tracks lifted with depth and     [tapnet]
#      solved frame-to-reference with Kabsch under RANSAC
#  10  keyframe identification -- contact / interaction / detachment,          [genesis]
#      read off the OBJECT's motion, not the hand's  (paper A.2.1)
#  11  grasp candidates at those keyframes, from Lightning Grasp               [lygra]
#  12  keyframe refinement: clearance offset, candidate filtering,             [genesis]
#      simulation-based selection, quintic blend  (paper A.2.2)
#  13  replay the refined trajectory in Genesis                                [genesis]
#
# SuperDex is not used anywhere here; refinement is the paper's keyframe method, which
# runs in the genesis env.
#
# Usage:  ./run_v2s2r_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
# Example: ./run_v2s2r_pipeline.sh human_videos/hrdexdb/spray_bottle/23029839.mp4 0 "spray bottle" right
#
# Stage 1 names the object itself, so OBJECT is only a fallback for when semantic
# parsing is skipped -- it needs GEMINI_API_KEY, and is skipped automatically without
# one. Set SEMANTIC_OVERRIDE=0 to keep the OBJECT argument even when stage 1 ran.
#
# Stages are individually skippable so a run can resume rather than repeat the slow
# parts (MoGe over every frame, HaWoR and the point tracker):
#   SKIP_SEMANTIC=1  SKIP_EXTRACT_AND_SAM3=1  SKIP_DEPTH=1  SKIP_MESH=1  SKIP_HAWOR=1
#   SKIP_RETARGETING=1  SKIP_TRAJOPT=1  SKIP_TRACK=1  SKIP_KEYFRAMES=1
#   SKIP_KEYFRAME_GRASPS=1  SKIP_REFINE=1  SKIP_REPLAY=1
#
# TRACKER=cotracker swaps stage 9's BootsTAPIR for CoTracker + pose_from_points, which
# is the same Kabsch solve behind a different point tracker. BootsTAPIR is the paper's,
# and is the default.

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config/paths.sh"

# ──────────────────────────── Per-run inputs (args) ────────────────────────────
VIDEO_PATH="${1:?usage: run_v2s2r_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]}"
VIDEO_PATH="$(realpath "$VIDEO_PATH")"
n="${2:-49}"
IFS=',' read -ra OBJECT_NAMES <<< "${3:-cup}"
OBJECT_NAMES=("${OBJECT_NAMES[@]# }")
IFS=',' read -ra PART_NAMES <<< "${PART_NAMES:-}"
PART_NAMES=("${PART_NAMES[@]# }")
ANCHOR_HAND="${4:-left}"

VIDEO_DIR="$(dirname "$VIDEO_PATH")"
VIDEO_BASENAME="$(basename "$VIDEO_PATH")"
VIDEO_NAME="${VIDEO_BASENAME%%.*}"

FRAMES_DIR="$VIDEO_DIR/all_frames"
FRAME_PATH="$FRAMES_DIR/$(printf "%06d.png" "$n")"
DEPTH_SOURCE="${DEPTH_SOURCE:-zed}"
SVO_PATH="${SVO_PATH:-${VIDEO_PATH%_rgb.*}.svo2}"
if [ "$DEPTH_SOURCE" = "zed" ] && [ ! -f "$SVO_PATH" ]; then
    echo "No SVO at $SVO_PATH; falling back to monocular depth." >&2
    DEPTH_SOURCE=moge
fi
DEPTH_DIR="$VIDEO_DIR/depth_$DEPTH_SOURCE"
DEPTH_DIR_MOGE="$VIDEO_DIR/depth_moge"
if [ "$DEPTH_SOURCE" = "zed" ]; then DEPTH_TAG=""; else DEPTH_TAG="_$DEPTH_SOURCE"; fi
REF_DEPTH_PATH="$DEPTH_DIR/$(printf "%06d.png" "$n")"
REF_DEPTH_PATH_MOGE="$DEPTH_DIR_MOGE/$(printf "%06d.png" "$n")"
INTRINSICS_YAML="$VIDEO_DIR/intrinsics.yaml"
GROUND_PLANE="$VIDEO_DIR/ground_plane$DEPTH_TAG.json"
MASKS_DIR="$VIDEO_DIR/video_segmentation/masks/frame_$(printf "%06d" "$n")_masks"
VIDEO_MASKS_DIR="$VIDEO_DIR/video_segmentation/masks"
META_PATH="${META_PATH:-${VIDEO_PATH%_rgb.*}_meta.json}"

HAND_ID="${ANCHOR_HAND}_hand_0"
HAND_MESHES_PATH="$VIDEO_DIR/$VIDEO_NAME/all_hand_meshes.npz"
WUJI_TRAJ="$VIDEO_DIR/wuji_traj.npz"
WUJI_TRAJ_OPT="$VIDEO_DIR/wuji_traj_optimized.npz"

# Every V2S2R-specific output is tagged so it never overwrites another pipeline's.
SCENE_JSON="$VIDEO_DIR/scene.json"
KEYFRAMES_JSON="$VIDEO_DIR/keyframes_v2s2r$DEPTH_TAG.json"
KEYFRAMES_STRIP="$VIDEO_DIR/keyframes_v2s2r$DEPTH_TAG.png"
KEYFRAMES_PLOT="$VIDEO_DIR/keyframes_v2s2r${DEPTH_TAG}_signals.png"
KEYFRAME_GRASPS="$VIDEO_DIR/keyframe_grasps_v2s2r$DEPTH_TAG.npz"
REFINED_TRAJ="$VIDEO_DIR/wuji_traj_v2s2r$DEPTH_TAG.npz"
REPLAY_PATH="$VIDEO_DIR/wuji_replay_v2s2r$DEPTH_TAG.mp4"
GRASP_RENDER="$VIDEO_DIR/keyframe_grasps_v2s2r$DEPTH_TAG.png"

SKIP_SEMANTIC="${SKIP_SEMANTIC:-0}"
SKIP_EXTRACT_AND_SAM3="${SKIP_EXTRACT_AND_SAM3:-0}"
SKIP_DEPTH="${SKIP_DEPTH:-0}"
SKIP_MESH="${SKIP_MESH:-0}"
SKIP_HAWOR="${SKIP_HAWOR:-0}"
SKIP_RETARGETING="${SKIP_RETARGETING:-0}"
SKIP_TRAJOPT="${SKIP_TRAJOPT:-0}"
SKIP_TRACK="${SKIP_TRACK:-0}"
SKIP_KEYFRAMES="${SKIP_KEYFRAMES:-0}"
SKIP_KEYFRAME_GRASPS="${SKIP_KEYFRAME_GRASPS:-0}"
SKIP_REFINE="${SKIP_REFINE:-0}"
SKIP_REPLAY="${SKIP_REPLAY:-0}"

OBJECT_PROMPT="${OBJECT_PROMPT:-text}"
SEMANTIC_OVERRIDE="${SEMANTIC_OVERRIDE:-1}"
SEMANTIC_MODEL="${SEMANTIC_MODEL:-gemini-flash-lite-latest}"
TRACKER="${TRACKER:-tapir}"

# Grasp synthesis, as the other pipelines expose it.
GRASP_PART="${GRASP_PART:-}"
GRASP_PART_SHARE="${GRASP_PART_SHARE:-0.5}"
# Lightning Grasp's GPU batch. The 128x256 default fills a 16 GB card on some meshes;
# the pass then raises out-of-memory and the stage keeps whatever it had, which on the
# first pass is nothing.
LYGRA_BATCH_OUTER="${LYGRA_BATCH_OUTER:-128}"
LYGRA_BATCH_INNER="${LYGRA_BATCH_INNER:-256}"
LYGRA_MAX_PASSES="${LYGRA_MAX_PASSES:-12}"

# Trajectory-optimization weights (DemoMimic, Appendix A.1).
TRAJOPT_VELOCITY="${TRAJOPT_VELOCITY:-0.5}"
TRAJOPT_ACCELERATION="${TRAJOPT_ACCELERATION:-2.0}"
TRAJOPT_JERK="${TRAJOPT_JERK:-8.0}"

# Keyframe identification (paper A.2.1).
#
# delta_c is the paper's 4 px. It is a threshold on a PIXEL displacement, so it scales
# with the resolution the clip was shot at and does not transfer on its own. On
# hrdexdb/spray_bottle, 2048 px wide, it put contact at frame 10 -- while the hand was
# still reaching -- and the grasp candidates there came out 378 mm from the
# demonstrated wrist, which stage 12 then correctly rejected. Worse, sweeping it walks
# the answer smoothly from frame 10 (4 px) to frame 115 (120 px) with no plateau,
# because the tracked centroid drifts from the first frame rather than starting to move
# at a contact. Read the *_signals.png before trusting the contact frame on new footage.
CONTACT_DISPLACEMENT="${CONTACT_DISPLACEMENT:-4.0}"
CONTACT_STREAK="${CONTACT_STREAK:-50}"
# The paper's 33 cm is an absolute height for its own rig. extract_keyframes.py says so
# itself and offers 'auto', which takes the threshold from the object's own lift; that
# is the default here because 33 cm is simply unreachable on this footage, where the
# object peaks at 25.8 cm and the literal rule returns no detachment frame at all.
DETACH_HEIGHT="${DETACH_HEIGHT:-auto}"

# Keyframe refinement (paper A.2.2).
LIFT_THRESHOLD="${LIFT_THRESHOLD:-0.06}"
BLEND_FRAMES="${BLEND_FRAMES:-30}"
MAX_CANDIDATES="${MAX_CANDIDATES:-8}"

# Pose smoothing behind the Kabsch solve. Measured on hrdexdb/spray_bottle: a centred
# window beat every constant-velocity Kalman setting on mask reprojection (35.1% of
# mesh points inside the SAM3 mask and 54 px of centroid error, against 31.2% and
# 70 px), because it is zero-lag for constant-velocity motion where the filter's
# process model lags. Window size is flat in accuracy and monotone in smoothness, so
# it is raised well past the 9 the scripts default to.
SMOOTH_WINDOW="${SMOOTH_WINDOW:-21}"

source "$(conda info --base)/etc/profile.d/conda.sh"

# ──────────────── Stage 1: semantic parsing (paper stage 1) ────────────────
# Gemini reads three frames of the clip and returns a structured scene: every tabletop
# object, which one is manipulated, the task, and a material per object. The paper uses
# it to drive segmentation, so a run that has it should segment the description it
# returns rather than a name typed at the command line.
if [ "$SKIP_SEMANTIC" != "1" ] && [ -z "$GEMINI_API_KEY" ] && [ ! -f "$HERE/.env" ]; then
    echo "=== No GEMINI_API_KEY and no $HERE/.env: skipping semantic parsing ===" >&2
    echo "    Falling back to the OBJECT argument: ${OBJECT_NAMES[*]}" >&2
    SKIP_SEMANTIC=1
fi
if [ "$SKIP_SEMANTIC" != "1" ]; then
    conda activate "$ENV_SAM3D"
    cd "$SCRIPTS_DIR"
    echo "=== Semantic parsing (Gemini) ==="
    # Needs the frames, and stage 2 has not run yet on a cold start.
    if [ ! -f "$FRAME_PATH" ]; then
        conda activate "$ENV_SAM3"
        mkdir -p "$FRAMES_DIR"
        ffmpeg -i "$VIDEO_PATH" -vsync 0 -start_number 0 "$FRAMES_DIR/%06d.png"
        conda activate "$ENV_SAM3D"
    fi
    python semantic_parsing.py \
        --frames-dir "$FRAMES_DIR" \
        --reference-frame "$FRAME_PATH" \
        --model "$SEMANTIC_MODEL" \
        --out "$SCENE_JSON"
else
    echo "=== SKIP_SEMANTIC=1: skipping semantic parsing ==="
fi

# The manipulated object's name drives every later filename, and its visually grounded
# description is what SAM3 actually grounds well -- "white plastic spray bottle with a
# black trigger" beats "spray bottle". Both come from stage 1 when it ran.
OBJECT_PROMPT_TEXT="${OBJECT_NAMES[0]}"
if [ -f "$SCENE_JSON" ] && [ "$SEMANTIC_OVERRIDE" = "1" ]; then
    conda activate "$ENV_GENESIS"
    SEMANTIC_INFO="$(python - "$SCENE_JSON" <<'PY'
import json, sys
scene = json.load(open(sys.argv[1]))
objects = scene.get("objects", [])
manipulated = next((o for o in objects if o.get("is_manipulated")), None)
if manipulated:
    print("NAME=" + manipulated["name"])
    print("DESCRIPTION=" + manipulated.get("description", manipulated["name"]))
    part = manipulated.get("manipulated_part") or ""
    print("PART=" + part)
    print("MATERIAL=" + manipulated.get("material", ""))
print("TASK=" + scene.get("task_type", ""))
PY
)"
    SEM_NAME="$(sed -n 's/^NAME=//p' <<< "$SEMANTIC_INFO")"
    SEM_DESC="$(sed -n 's/^DESCRIPTION=//p' <<< "$SEMANTIC_INFO")"
    SEM_PART="$(sed -n 's/^PART=//p' <<< "$SEMANTIC_INFO")"
    echo "=== Scene: $(sed -n 's/^TASK=//p' <<< "$SEMANTIC_INFO") ==="
    if [ -n "$SEM_NAME" ]; then
        echo "    manipulated object: $SEM_NAME  ($(sed -n 's/^MATERIAL=//p' <<< "$SEMANTIC_INFO"))"
        echo "    segmentation prompt: $SEM_DESC"
        OBJECT_NAMES=("$SEM_NAME")
        OBJECT_PROMPT_TEXT="$SEM_DESC"
        # A part named by stage 1 is only adopted when the run did not ask for its own.
        if [ -n "$SEM_PART" ] && [ -z "${PART_NAMES[*]}" ]; then
            PART_NAMES=("$SEM_PART")
            [ -z "$GRASP_PART" ] && GRASP_PART="$SEM_PART"
        fi
    fi
fi

FIRST_OBJECT="${OBJECT_NAMES[0]// /_}"
FIRST_MESH="$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$FIRST_OBJECT/posed_mesh.obj"
OBJ_TRAJ="$VIDEO_DIR/${FIRST_OBJECT}_traj_v2s2r$DEPTH_TAG.npz"
POINTS_TRACK="$VIDEO_DIR/points_cotracker$DEPTH_TAG.npz"
POINTS_OVERLAY="$VIDEO_DIR/overlay_points$DEPTH_TAG.mp4"

# ──────────────── Stage 2: frames + SAM3 segmentation ────────────────
if [ "$SKIP_EXTRACT_AND_SAM3" != "1" ]; then
    conda activate "$ENV_SAM3"

    echo "=== Extracting all frames ==="
    mkdir -p "$FRAMES_DIR"
    ffmpeg -i "$VIDEO_PATH" -vsync 0 -start_number 0 "$FRAMES_DIR/%06d.png"

    cd "$SCRIPTS_DIR"
    echo "=== Running SAM3 video segmentation (object, $OBJECT_PROMPT mode) ==="
    if [ "$OBJECT_PROMPT" = "text" ]; then
        python run_sam3_video.py ${SAM3_OFFLOAD:+--offload-video} \
            --video "$VIDEO_PATH" \
            --text "$OBJECT_PROMPT_TEXT" \
            --obj_id "$FIRST_OBJECT" \
            --frame_idx "$n"
    else
        DISPLAY="$SAM3_DISPLAY" python run_sam3_video.py ${SAM3_OFFLOAD:+--offload-video} \
            --video "$VIDEO_PATH" \
            --click \
            --obj_id "$FIRST_OBJECT" \
            --frame_idx "$n"
    fi

    for PART_NAME in "${PART_NAMES[@]}"; do
        [ -z "$PART_NAME" ] && continue
        PART_ID="${FIRST_OBJECT}_${PART_NAME// /_}"
        echo "=== Running SAM3 video segmentation (part: $PART_NAME) ==="
        python run_sam3_video.py ${SAM3_OFFLOAD:+--offload-video} \
            --video "$VIDEO_PATH" \
            --text "${OBJECT_NAMES[0]} $PART_NAME" \
            --obj_id "$PART_ID" \
            --frame_idx "$n"
    done

    echo "=== Running SAM3 video segmentation (hand, text-based) ==="
    # SAM3 does not reason about laterality: "right hand" scores below its detection
    # threshold and yields an empty mask, so prompt "hand" and keep the side in the name.
    python run_sam3_video.py ${SAM3_OFFLOAD:+--offload-video} \
        --video "$VIDEO_PATH" \
        --text "hand" \
        --obj_id "$HAND_ID" \
        --frame_idx "$n"
else
    echo "=== SKIP_EXTRACT_AND_SAM3=1: skipping frame extraction + SAM3 ==="
fi

# ──────────────── Stage 3: camera intrinsics ────────────────
conda activate "$ENV_MOGE"
cd "$SCRIPTS_DIR"
echo "=== Writing camera intrinsics ==="
if [ -n "$CAM_PARAM_JSON" ]; then
    INTRINSICS_INFO="$(python make_intrinsics.py \
        --intrinsics-json "$CAM_PARAM_JSON" \
        --camera "${CAMERA_ID:-$VIDEO_NAME}" \
        ${FRAMES_UNDISTORTED:+--undistorted} \
        --out "$INTRINSICS_YAML")"
elif [ -f "$META_PATH" ]; then
    INTRINSICS_INFO="$(python make_intrinsics.py --meta "$META_PATH" --out "$INTRINSICS_YAML")"
else
    echo "No metadata at $META_PATH; estimating intrinsics from the reference frame." >&2
    INTRINSICS_INFO="$(python make_intrinsics.py \
        --from-moge "$FRAME_PATH" \
        --checkpoint "$MOGE_CHECKPOINT" \
        --moge-repo "$MOGE_REPO" \
        --out "$INTRINSICS_YAML")"
fi
echo "$INTRINSICS_INFO"
FOV_X="${FOV_X:-$(sed -n 's/^FOV_X=//p' <<< "$INTRINSICS_INFO")}"
FOCAL="${FOCAL:-$(sed -n 's/^FOCAL=//p' <<< "$INTRINSICS_INFO")}"
: "${FOV_X:?could not determine horizontal FOV}"
: "${FOCAL:?could not determine focal length}"

# ──────────────── Stage 4: metric depth, every frame ────────────────
if [ "$SKIP_DEPTH" != "1" ]; then
    # Only the ZED path has an SVO to read; with DEPTH_SOURCE=moge there is none and
    # the MoGe pass below is the whole stage.
    if [ "$DEPTH_SOURCE" = "zed" ]; then
        echo "=== Extracting ZED stereo depth ==="
        # pyzed lives in the system python, not a conda env.
        conda deactivate
        "$PYTHON_ZED" "$SCRIPTS_DIR/extract_zed_depth.py" \
            --svo "$SVO_PATH" \
            --out-dir "$DEPTH_DIR" \
            --depth-mode "$ZED_DEPTH_MODE"
    fi
    conda activate "$ENV_MOGE"
    # RecGen is fed MoGe depth in every pipeline here, so it is produced either way.
    echo "=== Running MoGe depth over all frames ==="
    python run_moge_depth_batch.py \
        --frames-dir "$FRAMES_DIR" \
        --out-dir "$DEPTH_DIR_MOGE" \
        --fov-x "$FOV_X" \
        --checkpoint "$MOGE_CHECKPOINT" \
        --moge-repo "$MOGE_REPO"
else
    echo "=== SKIP_DEPTH=1: reusing $DEPTH_DIR ==="
fi

# ──────────────── Stage 5: supporting-plane fit ────────────────
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"
echo "=== Fitting the supporting plane ==="
PLANE_EXCLUDE=("$MASKS_DIR/$HAND_ID.png" "$MASKS_DIR/$FIRST_OBJECT.png")
python fit_ground_plane.py \
    --depth "$REF_DEPTH_PATH" \
    --intrinsics "$INTRINSICS_YAML" \
    --exclude "${PLANE_EXCLUDE[@]}" \
    --out "$GROUND_PLANE"

# ──────────────── Stage 6: object mesh -- RecGen, substituting SAM 3D ────────────────
# The paper reconstructs the manipulated object with sam-3d-objects. That does not run
# here, so RecGen does it from the same reference RGB + metric depth + mask and writes
# the same posed_mesh.obj. This is the only intentional difference from the paper.
if [ "$SKIP_MESH" != "1" ]; then
    conda activate "$ENV_RECGEN"
    cd "$RECGEN_DIR"
    echo "=== Reconstructing object mesh (RecGen): $FIRST_OBJECT ==="
    python scripts/run_inference.py \
        --rgb "$FRAME_PATH" \
        --depth "$REF_DEPTH_PATH_MOGE" \
        --mask "$MASKS_DIR/${FIRST_OBJECT}.png" \
        --intrinsics "$INTRINSICS_YAML" \
        --checkpoint "$RECGEN_CHECKPOINT" \
        --out "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$FIRST_OBJECT" \
        --name "$FIRST_OBJECT"
else
    echo "=== SKIP_MESH=1: reusing $FIRST_MESH ==="
fi

# ──────────── Stage 6b: carry a part's mask onto the mesh ────────────
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"
for PART_NAME in "${PART_NAMES[@]}"; do
    [ -z "$PART_NAME" ] && continue
    PART_ID="${FIRST_OBJECT}_${PART_NAME// /_}"
    echo "=== Labelling the mesh with part: $PART_NAME ==="
    python label_mesh_part.py \
        --mesh "$FIRST_MESH" \
        --mask "$MASKS_DIR/${PART_ID}.png" \
        --intrinsics "$INTRINSICS_YAML" \
        --frame "$n" \
        --name "$PART_NAME" \
        --out "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$FIRST_OBJECT/part_${PART_NAME// /_}.npy" \
        --preview "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$FIRST_OBJECT/part_${PART_NAME// /_}.png"
done

# ──────────────── Stage 7: HaWoR hand reconstruction ────────────────
if [ "$SKIP_HAWOR" != "1" ]; then
    conda activate "$ENV_HAWOR"
    cd "$SCRIPTS_DIR"
    echo "=== Running HaWoR ==="
    python run_hawor.py \
        --video-path "$VIDEO_PATH" \
        --img-focal "$FOCAL" \
        --hawor-dir "$HAWOR_DIR"
else
    echo "=== SKIP_HAWOR=1: reusing $HAND_MESHES_PATH ==="
fi

# ──────────────── Stage 8: retarget onto the Wuji Hand ────────────────
if [ "$SKIP_RETARGETING" != "1" ]; then
    conda activate "$ENV_WUJI"
    cd "$SCRIPTS_DIR"
    echo "=== Retargeting hand onto the Wuji Hand ==="
    python hawor_to_wuji_qpos.py \
        --hand-meshes "$HAND_MESHES_PATH" \
        --side "$ANCHOR_HAND" \
        --wuji-dir "$WUJI_RETARGETING_DIR" \
        --out "$WUJI_TRAJ"
else
    echo "=== SKIP_RETARGETING=1: skipping the retargeting stage ==="
fi

# ──────────── Stage 8b: trajectory optimization over the retargeted hand ────────────
REFINE_SOURCE="$WUJI_TRAJ"
if [ "$SKIP_TRAJOPT" != "1" ]; then
    conda activate "$ENV_GENESIS"
    cd "$SCRIPTS_DIR"
    echo "=== Optimizing the retargeted trajectory ==="
    python optimize_trajectory.py \
        --traj "$WUJI_TRAJ" \
        --urdf "$WUJI_RETARGETING_DIR/wuji_retargeting/wuji-description/hand/body/urdf/${ANCHOR_HAND}.urdf" \
        --velocity "$TRAJOPT_VELOCITY" \
        --acceleration "$TRAJOPT_ACCELERATION" \
        --jerk "$TRAJOPT_JERK" \
        --out "$WUJI_TRAJ_OPT"
    REFINE_SOURCE="$WUJI_TRAJ_OPT"
else
    echo "=== SKIP_TRAJOPT=1: refining the trajectory as retargeted ==="
    [ -f "$WUJI_TRAJ_OPT" ] && REFINE_SOURCE="$WUJI_TRAJ_OPT"
fi

# ──────────────── Stage 9: object pose tracking, Video2Sim2Real ────────────────
# Point tracks lifted to 3D with per-frame metric depth, then the frame-to-reference
# rigid transform solved with Kabsch under RANSAC. The tracker writes its 2D tracks
# alongside the poses, so stage 10 reads the keyframes off the same tracking the poses
# came from rather than a second pass that would disagree with it.
KEYFRAME_POINTS="$OBJ_TRAJ"
if [ "$SKIP_TRACK" != "1" ]; then
    if [ "$TRACKER" = "cotracker" ]; then
        echo "=== Tracking surface points with CoTracker ==="
        conda activate "$ENV_SAM3"
        cd "$SCRIPTS_DIR"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python track_points_cotracker.py \
            --frames-dir "$FRAMES_DIR" \
            --depth-dir "$DEPTH_DIR" \
            --masks-root "$VIDEO_MASKS_DIR" \
            --object "$FIRST_OBJECT" \
            --ref-frame "$n" \
            --intrinsics "$INTRINSICS_YAML" \
            --out "$POINTS_TRACK" \
            --overlay "$POINTS_OVERLAY"
        conda activate "$ENV_GENESIS"
        echo "=== Fitting an object trajectory to the lifted points ==="
        python pose_from_points.py \
            --points "$POINTS_TRACK" \
            --mesh "$FIRST_MESH" \
            --smoother window \
            --smooth-window "$SMOOTH_WINDOW" \
            --out "$OBJ_TRAJ"
    else
        echo "=== Tracking the object pose (BootsTAPIR + Kabsch, Video2Sim2Real) ==="
        conda activate "$ENV_TAPNET"
        cd "$SCRIPTS_DIR"
        python track_object_v2s2r.py \
            --frames-dir "$FRAMES_DIR" \
            --masks-root "$VIDEO_MASKS_DIR" \
            --depth-dir "$DEPTH_DIR" \
            --object "$FIRST_OBJECT" \
            --ref-frame "$n" \
            --intrinsics "$INTRINSICS_YAML" \
            --checkpoint "$TAPNET_CKPT" \
            --smooth-window "$SMOOTH_WINDOW" \
            --out "$OBJ_TRAJ"
    fi
else
    echo "=== SKIP_TRACK=1: reusing $OBJ_TRAJ ==="
fi
[ "$TRACKER" = "cotracker" ] && KEYFRAME_POINTS="$POINTS_TRACK"

# ──────────────── Stage 10: keyframe identification (paper A.2.1) ────────────────
# Contact, interaction and detachment, read off the object's motion rather than the
# hand's. A hand-contact window says when the reconstruction put the hand near the
# surface; this says when the demonstration started changing the world, which is what
# the refinement wants to anchor on.
if [ "$SKIP_KEYFRAMES" != "1" ]; then
    conda activate "$ENV_GENESIS"
    cd "$SCRIPTS_DIR"
    echo "=== Identifying the object-centric keyframes ==="
    python extract_keyframes.py \
        --points "$KEYFRAME_POINTS" \
        --object-poses "$OBJ_TRAJ" \
        --mesh "$FIRST_MESH" \
        --ground-plane "$GROUND_PLANE" \
        --frames-dir "$FRAMES_DIR" \
        --contact-displacement "$CONTACT_DISPLACEMENT" \
        --contact-streak "$CONTACT_STREAK" \
        --detach-height "$DETACH_HEIGHT" \
        --strip "$KEYFRAMES_STRIP" \
        --plot "$KEYFRAMES_PLOT" \
        --out "$KEYFRAMES_JSON"
else
    echo "=== SKIP_KEYFRAMES=1: reusing $KEYFRAMES_JSON ==="
fi

# ──────────── Stage 11: grasp candidates at those keyframes ────────────
# Lightning Grasp is the paper's generator. Driven from the keyframes.json above rather
# than from its own hand-contact windows, so the candidates are synthesized exactly
# where the refinement will use them.
if [ "$SKIP_KEYFRAME_GRASPS" != "1" ]; then
    echo "=== Synthesizing grasps at the Video2Sim2Real keyframes (Lightning Grasp) ==="
    conda activate "$ENV_LYGRA"
    python "$SCRIPTS_DIR/keyframe_grasps.py" \
        --object-poses "$OBJ_TRAJ" \
        --mesh "$FIRST_MESH" \
        --hand-traj "$REFINE_SOURCE" \
        --hand-meshes "$HAND_MESHES_PATH" \
        --side "$ANCHOR_HAND" \
        --ground-plane "$GROUND_PLANE" \
        --lygra-dir "$LIGHTNING_GRASP_DIR" \
        --keyframes-json "$KEYFRAMES_JSON" \
        --batch-size-outer "$LYGRA_BATCH_OUTER" \
        --batch-size-inner "$LYGRA_BATCH_INNER" \
        --max-passes "$LYGRA_MAX_PASSES" \
        ${GRASP_PART:+--part "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$FIRST_OBJECT/part_${GRASP_PART// /_}.npy"} \
        ${GRASP_PART:+--part-share "$GRASP_PART_SHARE"} \
        --out "$KEYFRAME_GRASPS"
else
    echo "=== SKIP_KEYFRAME_GRASPS=1: reusing $KEYFRAME_GRASPS ==="
fi

# ──────────────── Stage 12: keyframe refinement (paper A.2.2) ────────────────
# The correction is applied at the keyframes and blended out with the paper's quintic
# smoothstep, so the demonstration is untouched away from them. Candidate selection is
# by simulation: approach, pre-grasp, close, lift, shake on three axes, accept if the
# object stays lifted past tau_lift.
REPLAY_TRAJ="$REFINE_SOURCE"
if [ "$SKIP_REFINE" != "1" ]; then
    conda activate "$ENV_GENESIS"
    cd "$SCRIPTS_DIR"
    echo "=== Refining the keyframes (Video2Sim2Real A.2.2) ==="
    python refine_keyframes_v2s2r.py \
        --traj "$REFINE_SOURCE" \
        --keyframes "$KEYFRAMES_JSON" \
        --candidates "$KEYFRAME_GRASPS" \
        --object-poses "$OBJ_TRAJ" \
        --mesh "$FIRST_MESH" \
        --ground-plane "$GROUND_PLANE" \
        --wuji-dir "$WUJI_RETARGETING_DIR" \
        --lift-threshold "$LIFT_THRESHOLD" \
        --max-candidates "$MAX_CANDIDATES" \
        --blend "$BLEND_FRAMES" \
        --out "$REFINED_TRAJ"
    REPLAY_TRAJ="$REFINED_TRAJ"
else
    echo "=== SKIP_REFINE=1: replaying the trajectory as reconstructed ==="
    # Unlike the other pipelines here, an existing refined file is NOT picked up when
    # the stage is skipped: skipping means skipping.
fi

# ──────────────── Stage 13: replay in Genesis ────────────────
if [ "$SKIP_REPLAY" != "1" ]; then
    conda activate "$ENV_GENESIS"
    cd "$SCRIPTS_DIR"
    echo "=== Replaying the refined trajectory in Genesis ==="
    # Hand the object to the simulation at the contact keyframe: released earlier it
    # drifts off its tracked pose before the hand arrives and the grasp closes on
    # nothing. Contact is exactly the frame the paper says the interaction begins.
    RELEASE_ARG=()
    RELEASE_FRAME="$(python -c "
import json,sys
kf=json.load(open(sys.argv[1]))['keyframes']
print(kf.get('contact') if kf.get('contact') is not None else 0)
" "$KEYFRAMES_JSON" 2>/dev/null || true)"
    [ -n "$RELEASE_FRAME" ] && RELEASE_ARG=(--release-frame "$RELEASE_FRAME")
    python replay_wuji_genesis.py \
        --traj "$REPLAY_TRAJ" \
        --object-mesh "$FIRST_MESH" \
        --object-poses "$OBJ_TRAJ" \
        --ground-plane "$GROUND_PLANE" \
        --wuji-dir "$WUJI_RETARGETING_DIR" \
        --record "$REPLAY_PATH" \
        "${RELEASE_ARG[@]}"

    # The grasps the refinement chose between, drawn against the demonstrated hand.
    if [ -f "$KEYFRAME_GRASPS" ]; then
        echo "=== Rendering the keyframe grasps ==="
        REFINED_ARG=()
        [ -f "$REFINED_TRAJ" ] && [ "$SKIP_REFINE" != "1" ] && REFINED_ARG=(--refined "$REFINED_TRAJ")
        python show_keyframe_grasps.py \
            --candidates "$KEYFRAME_GRASPS" \
            --traj "$REFINE_SOURCE" \
            --object-poses "$OBJ_TRAJ" \
            --mesh "$FIRST_MESH" \
            --ground-plane "$GROUND_PLANE" \
            ${REFINED_ARG[@]+"${REFINED_ARG[@]}"} \
            --out "$GRASP_RENDER" || echo "[warn] grasp render failed; the pipeline's outputs are unaffected" >&2
    fi
else
    echo "=== SKIP_REPLAY=1: skipping the replay ==="
fi

echo
echo "=== Done ==="
[ -f "$SCENE_JSON" ]      && echo "  scene:           $SCENE_JSON"
echo "  object mesh:     $FIRST_MESH   (RecGen)"
echo "  object traj:     $OBJ_TRAJ   ($TRACKER + Kabsch)"
echo "  keyframes:       $KEYFRAMES_JSON"
[ -f "$KEYFRAMES_STRIP" ] && echo "  keyframe strip:  $KEYFRAMES_STRIP"
echo "  hand traj:       $REFINE_SOURCE"
[ -f "$KEYFRAME_GRASPS" ] && echo "  grasp candidates: $KEYFRAME_GRASPS"
[ -f "$REFINED_TRAJ" ]    && echo "  refined traj:    $REFINED_TRAJ"
[ -f "$REPLAY_PATH" ]     && echo "  replay:          $REPLAY_PATH"
[ -f "$GRASP_RENDER" ]    && echo "  grasp render:    $GRASP_RENDER"
# The summary is a chain of [ -f ] && echo; a missing optional output would
# otherwise become the script's exit status and read as a failed run.
exit 0
