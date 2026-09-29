#!/usr/bin/env bash
# Human demo video -> Wuji Hand + object trajectory replayed in Genesis.
#
# CoTracker variant: the object trajectory comes from CoTracker surface points lifted
# with depth and solved frame-to-reference with Kabsch (pose_from_points.py), then
# Kalman-smoothed. Depth comes from the ZED recording's stereo when there is an SVO
# next to the video, else from MoGe (set DEPTH_SOURCE=moge to force it).
#
#   0  extract frames                                                  [sam3]
#   1  SAM3 segmentation: object (+ parts) and hand                    [sam3]
#   2  camera intrinsics from the recording's calibration              [moge]
#   3  metric depth, every frame                        [zed python / moge]
#   4  supporting-plane fit (real up direction)                        [genesis]
#   5  object mesh from RGB + depth + mask (RecGen)                    [recgen]
#   5b carry a part's mask onto the mesh as a per-vertex label          [genesis]
#   6  HaWoR hand reconstruction                                       [hawor]
#   7  retarget the hand onto the Wuji Hand                  [wuji_retargeting]
#   7b trajectory optimization over the retargeted hand (velocity /     [genesis]
#      acceleration / jerk costs, DemoMimic Appendix A.1)
#   8  overlay a tracked pose on the video (reads <object>_traj_fp.npz, [genesis]
#      which only the legacy FoundationPose pipeline writes)
#   9  CoTracker surface points lifted with depth, drawn on the video   [sam3]
#   9b rigid trajectory fitted to the lifted points, Kalman-smoothed    [genesis]
#  10  Lightning Grasp synthesis at each grasp window's onset, middle   [lygra]
#      and release, ranked against the demonstrated hand
#  11  refine the grasp until it holds                                  [superdex]
#  12  replay hand + object in Genesis                                  [genesis]
#  13  SAM 3D point maps, compared against ZED depth                    [sam3d]
#
# With DEPTH_SOURCE=moge every depth-dependent output gets a _moge suffix (ground
# plane, trajectories, replays), so a MoGe run sits beside the ZED one.
#
# Usage:  ./run_reconstruction_pipeline_cotracker.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
# Example: ./run_reconstruction_pipeline_cotracker.sh human_videos/pouring_a_cup/take001_pouring_a_cup_rgb.mp4 49 cup left
#
# OBJECT may list several objects separated by commas; PART_NAMES="handle" segments
# parts of the first one. Set OBJECT_PROMPT=click to segment the object by clicking
# (needs a display) instead of by name, and CAM_PARAM_JSON (+ CAMERA_ID) when the
# calibration ships apart from the footage.
#
# Stages are individually skippable so a run can resume rather than repeat the slow
# parts (MoGe over every frame, HaWoR, CoTracker, Lightning Grasp):
#   SKIP_EXTRACT_AND_SAM3=1  SKIP_DEPTH=1  SKIP_MESH=1  SKIP_HAWOR=1  SKIP_RETARGETING=1
#   SKIP_TRAJOPT=1  SKIP_OVERLAY=1  SKIP_COTRACKER=1  SKIP_KEYFRAME_GRASPS=1
#   SKIP_REFINE=1  SKIP_WUJI=1  SKIP_POINT_MAP=1
# and SKIP_COMPARE_LIFTS=0 turns on the ZED-vs-MoGe comparison of the point lifts.

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config/paths.sh"

# ──────────────────────────── Per-run inputs (args) ────────────────────────────
VIDEO_PATH="${1:?usage: run_reconstruction_pipeline_cotracker.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]}"
VIDEO_PATH="$(realpath "$VIDEO_PATH")"
n="${2:-49}"
# Bash does not split on commas, so "cup, handle" would be a single object called
# "cup,_handle". Objects are separated here explicitly; a name may still contain spaces.
IFS=',' read -ra OBJECT_NAMES <<< "${3:-cup}"
OBJECT_NAMES=("${OBJECT_NAMES[@]# }")
# Parts of the first object: segmented like objects, but they are not separate bodies.
# They become a label on its mesh (label_mesh_part.py) that grasp synthesis can aim at.
IFS=',' read -ra PART_NAMES <<< "${PART_NAMES:-}"
PART_NAMES=("${PART_NAMES[@]# }")
ANCHOR_HAND="${4:-left}"

VIDEO_DIR="$(dirname "$VIDEO_PATH")"
VIDEO_BASENAME="$(basename "$VIDEO_PATH")"
VIDEO_NAME="${VIDEO_BASENAME%%.*}"

FRAMES_DIR="$VIDEO_DIR/all_frames"
# The reference frame is taken from all_frames rather than a separate ffmpeg select,
# so its index provably matches the masks and depth maps indexed the same way.
FRAME_PATH="$FRAMES_DIR/$(printf "%06d.png" "$n")"
DEPTH_SOURCE="${DEPTH_SOURCE:-zed}"
SVO_PATH="${SVO_PATH:-${VIDEO_PATH%_rgb.*}.svo2}"
if [ "$DEPTH_SOURCE" = "zed" ] && [ ! -f "$SVO_PATH" ]; then
    echo "No SVO at $SVO_PATH; falling back to monocular depth." >&2
    DEPTH_SOURCE=moge
fi
DEPTH_DIR="$VIDEO_DIR/depth_$DEPTH_SOURCE"
DEPTH_DIR_MOGE="$VIDEO_DIR/depth_moge"
# ZED is the reference depth and keeps the plain names; any other source is suffixed.
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
OVERLAY_PATH="$VIDEO_DIR/overlay_fp$DEPTH_TAG.mp4"
POINTS_TRACK="$VIDEO_DIR/points_cotracker$DEPTH_TAG.npz"
POINTS_OVERLAY="$VIDEO_DIR/overlay_points$DEPTH_TAG.mp4"
POINTS_TRAJ="$VIDEO_DIR/${OBJECT_NAMES[0]// /_}_traj_points$DEPTH_TAG.npz"
POINTS_REPLAY="$VIDEO_DIR/wuji_replay_points$DEPTH_TAG.mp4"
WUJI_TRAJ_OPT="$VIDEO_DIR/wuji_traj_optimized.npz"
KEYFRAME_GRASPS="$VIDEO_DIR/keyframe_grasps$DEPTH_TAG.npz"
REFINED_TRAJ="$VIDEO_DIR/wuji_traj_refined$DEPTH_TAG.npz"

SKIP_EXTRACT_AND_SAM3="${SKIP_EXTRACT_AND_SAM3:-0}"
SKIP_DEPTH="${SKIP_DEPTH:-0}"
SKIP_MESH="${SKIP_MESH:-0}"
SKIP_HAWOR="${SKIP_HAWOR:-0}"
SKIP_OVERLAY="${SKIP_OVERLAY:-0}"
SKIP_RETARGETING="${SKIP_RETARGETING:-0}"
SKIP_WUJI="${SKIP_WUJI:-0}"
SKIP_COTRACKER="${SKIP_COTRACKER:-0}"
SKIP_POINT_MAP="${SKIP_POINT_MAP:-0}"
SKIP_REFINE="${SKIP_REFINE:-0}"
SKIP_TRAJOPT="${SKIP_TRAJOPT:-0}"
SKIP_KEYFRAME_GRASPS="${SKIP_KEYFRAME_GRASPS:-0}"
SKIP_COMPARE_LIFTS="${SKIP_COMPARE_LIFTS:-1}"
# click needs a display; text runs headless.
OBJECT_PROMPT="${OBJECT_PROMPT:-text}"
# Aim grasp synthesis at a labelled part, e.g. GRASP_PART=handle.
GRASP_PART="${GRASP_PART:-}"
GRASP_PART_SHARE="${GRASP_PART_SHARE:-0.5}"
# Lightning Grasp's GPU batch, as outer x inner IK poses in flight. The default fills
# a 16 GB card on some meshes; the pass then raises out-of-memory, the stage keeps the
# grasps found so far (none, on the first pass) and the run stops there. Halving both
# quarters the peak, and more passes make the candidate target reachable anyway.
LYGRA_BATCH_OUTER="${LYGRA_BATCH_OUTER:-128}"
LYGRA_BATCH_INNER="${LYGRA_BATCH_INNER:-256}"
LYGRA_MAX_PASSES="${LYGRA_MAX_PASSES:-12}"
# Object-trajectory smoothing: kalman (constant velocity + RTS) or window.
SMOOTHER="${SMOOTHER:-kalman}"
KALMAN_POS_NOISE="${KALMAN_POS_NOISE:-0.010}"
KALMAN_ACCEL="${KALMAN_ACCEL:-0.005}"
KALMAN_ROT_NOISE="${KALMAN_ROT_NOISE:-4.0}"
KALMAN_ANG_ACCEL="${KALMAN_ANG_ACCEL:-2.0}"
# Trajectory-optimization weights (DemoMimic, Appendix A.1), read relative to a unit
# fidelity term. Acceleration and jerk do the smoothing; velocity is kept low because
# charging for speed shrinks the demonstration's own motion.
TRAJOPT_VELOCITY="${TRAJOPT_VELOCITY:-0.5}"
TRAJOPT_ACCELERATION="${TRAJOPT_ACCELERATION:-2.0}"
TRAJOPT_JERK="${TRAJOPT_JERK:-8.0}"

source "$(conda info --base)/etc/profile.d/conda.sh"

# ──────────────── Steps 0-1: frames + SAM3 segmentation ────────────────
if [ "$SKIP_EXTRACT_AND_SAM3" != "1" ]; then
    # ffmpeg comes from the sam3 env, so activate it before extracting: the pipeline
    # then needs no system ffmpeg and runs directly on a clip.
    conda activate "$ENV_SAM3"

    echo "=== Extracting all frames ==="
    mkdir -p "$FRAMES_DIR"
    ffmpeg -i "$VIDEO_PATH" -vsync 0 -start_number 0 "$FRAMES_DIR/%06d.png"

    cd "$SCRIPTS_DIR"
    echo "=== Running SAM3 video segmentation (objects, $OBJECT_PROMPT mode) ==="
    for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
        OBJ_ID="${OBJ_NAME// /_}"
        if [ "$OBJECT_PROMPT" = "text" ]; then
            # Headless alternative to clicking. Works when the object name is a simple
            # noun phrase, which is what SAM3 grounds; check the reported score before
            # trusting it, and fall back to clicking if the object is ambiguous.
            python run_sam3_video.py \
                --video "$VIDEO_PATH" \
                --text "$OBJ_NAME" \
                --obj_id "$OBJ_ID" \
                --frame_idx "$n"
        else
            DISPLAY="$SAM3_DISPLAY" python run_sam3_video.py \
                --video "$VIDEO_PATH" \
                --click \
                --obj_id "$OBJ_ID" \
                --frame_idx "$n"
        fi
    done

    for PART_NAME in "${PART_NAMES[@]}"; do
        [ -z "$PART_NAME" ] && continue
        PART_ID="${OBJECT_NAMES[0]// /_}_${PART_NAME// /_}"
        echo "=== Running SAM3 video segmentation (part: $PART_NAME) ==="
        # A part is prompted with the object included ("cup handle", not "handle"), which
        # is what SAM3 grounds, and keeps the mask named per object so two objects can
        # each have a handle without colliding.
        python run_sam3_video.py \
            --video "$VIDEO_PATH" \
            --text "${OBJECT_NAMES[0]} $PART_NAME" \
            --obj_id "$PART_ID" \
            --frame_idx "$n"
    done

    echo "=== Running SAM3 video segmentation (hand, text-based) ==="
    # SAM3 grounds simple noun phrases and does not reason about laterality: "left
    # hand" scores below its 0.5 detection threshold and yields an empty mask, so
    # prompt "hand" and keep the side only in the output name.
    python run_sam3_video.py \
        --video "$VIDEO_PATH" \
        --text "hand" \
        --obj_id "$HAND_ID" \
        --frame_idx "$n"
else
    echo "=== SKIP_EXTRACT_AND_SAM3=1: skipping frame extraction + SAM3 ==="
fi

# ──────────────── Step 2: camera intrinsics ────────────────
conda activate "$ENV_MOGE"
cd "$SCRIPTS_DIR"
echo "=== Writing camera intrinsics ==="
if [ -n "$CAM_PARAM_JSON" ]; then
    # Calibration shipped separately from the footage (HRDexDB keys it by camera
    # serial, which is also the video's basename).
    INTRINSICS_INFO="$(python make_intrinsics.py \
        --intrinsics-json "$CAM_PARAM_JSON" \
        --camera "${CAMERA_ID:-$VIDEO_NAME}" \
        ${FRAMES_UNDISTORTED:+--undistorted} \
        --out "$INTRINSICS_YAML")"
elif [ -f "$META_PATH" ]; then
    INTRINSICS_INFO="$(python make_intrinsics.py --meta "$META_PATH" --out "$INTRINSICS_YAML")"
else
    # No calibration shipped with the recording, so fall back to MoGe's own predicted
    # intrinsics. Less accurate than a real calibration, but it keeps uncalibrated
    # footage runnable instead of stopping the pipeline here.
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

# ──────────────── Step 3: metric depth, every frame (ZED or MoGe) ────────────────
if [ "$SKIP_DEPTH" != "1" ]; then

    # Only the ZED path has an SVO to read. With DEPTH_SOURCE=moge there is none, so
    # the MoGe pass below is the whole depth stage -- which is what the header promises
    # for footage that ships without a recording.
    if [ "$DEPTH_SOURCE" = "zed" ]; then
        echo "=== Extracting ZED stereo depth ==="
        # pyzed lives in the system python, not a conda env, so this one stage runs
        # outside the env chain.
        conda deactivate
        "$PYTHON_ZED" "$SCRIPTS_DIR/extract_zed_depth.py" \
            --svo "$SVO_PATH" \
            --out-dir "$DEPTH_DIR" \
            --depth-mode "$ZED_DEPTH_MODE"
    fi

    conda activate "$ENV_MOGE"
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

# ──────────────── Step 4: supporting-plane fit ────────────────
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"
echo "=== Fitting the supporting plane ==="
PLANE_EXCLUDE=("$MASKS_DIR/$HAND_ID.png")
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    PLANE_EXCLUDE+=("$MASKS_DIR/${OBJ_NAME// /_}.png")
done
python fit_ground_plane.py \
    --depth "$REF_DEPTH_PATH" \
    --intrinsics "$INTRINSICS_YAML" \
    --exclude "${PLANE_EXCLUDE[@]}" \
    --out "$GROUND_PLANE"

# ──────────────── Step 5: object mesh (RecGen) ────────────────
if [ "$SKIP_MESH" != "1" ]; then
    conda activate "$ENV_RECGEN"
    cd "$RECGEN_DIR"
    for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
        OBJECT_ID="${OBJ_NAME// /_}"
        echo "=== Reconstructing object mesh: $OBJECT_ID ==="
        python scripts/run_inference.py \
            --rgb "$FRAME_PATH" \
            --depth "$REF_DEPTH_PATH_MOGE" \
            --mask "$MASKS_DIR/${OBJECT_ID}.png" \
            --intrinsics "$INTRINSICS_YAML" \
            --checkpoint "$RECGEN_CHECKPOINT" \
            --out "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$OBJECT_ID" \
            --name "$OBJECT_ID"
    done
else
    echo "=== SKIP_MESH=1: reusing $VIDEO_DIR/recgen_out_$DEPTH_SOURCE ==="
fi

# ──────────── Step 5b: carry each part's mask onto the object's mesh ────────────
# A part moves with the object, so it stays one mesh and one rigid body; what it gains
# is a per-vertex label that grasp synthesis can aim placements at.
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"
for PART_NAME in "${PART_NAMES[@]}"; do
    [ -z "$PART_NAME" ] && continue
    PART_ID="${OBJECT_NAMES[0]// /_}_${PART_NAME// /_}"
    OBJECT_ID="${OBJECT_NAMES[0]// /_}"
    echo "=== Labelling the mesh with part: $PART_NAME ==="
    python label_mesh_part.py \
        --mesh "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$OBJECT_ID/posed_mesh.obj" \
        --mask "$MASKS_DIR/${PART_ID}.png" \
        --intrinsics "$INTRINSICS_YAML" \
        --frame "$n" \
        --name "$PART_NAME" \
        --out "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$OBJECT_ID/part_${PART_NAME// /_}.npy" \
        --preview "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$OBJECT_ID/part_${PART_NAME// /_}.png"
done

# ──────────────── Step 6: HaWoR hand reconstruction ────────────────
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

# ──────────────── Step 7: retarget onto the Wuji Hand ────────────────
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

# ──────────── Step 7b: trajectory optimization over the retargeted hand ────────────
# Retargeting solves each frame independently, so its output carries the per-frame
# solver's noise as velocity reversals and unbounded acceleration. This refits every
# channel as a whole against velocity, acceleration and jerk costs -- the
# post-retargeting step from DemoMimic (arXiv 2609.01938, Appendix A.1) -- and fills
# the frames retargeting dropped along the way. Everything downstream reads the
# optimized trajectory; set SKIP_TRAJOPT=1 to go back to the raw retargeted one.
REPLAY_SOURCE="$WUJI_TRAJ"
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
    REPLAY_SOURCE="$WUJI_TRAJ_OPT"
else
    echo "=== SKIP_TRAJOPT=1: replaying the trajectory as retargeted ==="
    [ -f "$WUJI_TRAJ_OPT" ] && REPLAY_SOURCE="$WUJI_TRAJ_OPT"
fi


FIRST_OBJECT="${OBJECT_NAMES[0]// /_}"
FIRST_MESH="$VIDEO_DIR/recgen_out_${DEPTH_SOURCE}/$FIRST_OBJECT/posed_mesh.obj"
FIRST_TRAJ_RAW="$VIDEO_DIR/${FIRST_OBJECT}_traj_fp${DEPTH_TAG}_raw.npz"
FIRST_TRAJ="$VIDEO_DIR/${FIRST_OBJECT}_traj_fp${DEPTH_TAG}.npz"
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"

# ──────────────── Step 8: overlay the tracked pose on the video ────────────────
# The only check against ground truth: the posed mesh drawn over the frames it came
# from, with the SAM3 mask outline. When anchoring ran, the raw track is drawn too so
# its effect is visible.
if [ "$SKIP_OVERLAY" != "1" ]; then
    echo "=== Rendering pose overlay: $OVERLAY_PATH ==="
    if cmp -s "$FIRST_TRAJ_RAW" "$FIRST_TRAJ"; then
        OVERLAY_TRAJS=("$FIRST_TRAJ");  OVERLAY_LABELS=("foundationpose$DEPTH_TAG")
    else
        OVERLAY_TRAJS=("$FIRST_TRAJ_RAW" "$FIRST_TRAJ")
        OVERLAY_LABELS=("foundationpose${DEPTH_TAG}_raw" "foundationpose${DEPTH_TAG}_anchored")
    fi
    python overlay_object_traj.py \
        --frames-dir "$FRAMES_DIR" \
        --mesh "$FIRST_MESH" \
        --intrinsics "$INTRINSICS_YAML" \
        --traj "${OVERLAY_TRAJS[@]}" \
        --labels "${OVERLAY_LABELS[@]}" \
        --masks-root "$VIDEO_MASKS_DIR" \
        --object "$FIRST_OBJECT" \
        --out "$OVERLAY_PATH"
else
    echo "=== SKIP_OVERLAY=1: skipping the overlay ==="
fi

# ──────────────── Step 9: CoTracker points, lifted with depth ────────────────
# An independent read on the object's motion: tracked surface points lifted straight
# to 3D, with no mesh and no pose model, drawn on the video they came from.
if [ "$SKIP_COTRACKER" != "1" ]; then
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

    # With both depth sources on disk, the same tracks lifted twice isolate the depth:
    # the tracking is identical, so every difference in 3D comes from it. Off by
    # default; set SKIP_COMPARE_LIFTS=0 to run it.
    OTHER_DEPTH="$VIDEO_DIR/depth_moge"
    [ "$DEPTH_SOURCE" = "moge" ] && OTHER_DEPTH="$VIDEO_DIR/depth_zed"
    if [ "$SKIP_COMPARE_LIFTS" != "1" ] && [ -d "$OTHER_DEPTH" ]; then
        echo "=== Comparing depth sources behind the point tracks ==="
        conda activate "$ENV_GENESIS"
        python compare_point_lifts.py \
            --points "$POINTS_TRACK" \
            --frames-dir "$FRAMES_DIR" \
            --depth-dirs "$DEPTH_DIR" "$OTHER_DEPTH" \
            --labels "$DEPTH_SOURCE" "$(basename "$OTHER_DEPTH" | sed 's/^depth_//')" \
            --intrinsics "$INTRINSICS_YAML" \
            --out "$VIDEO_DIR/compare_depth_lift.mp4" \
            --plot "$VIDEO_DIR/compare_depth_lift.png"
    fi

    # The lifted points give the object's trajectory: the rigid motion carrying the
    # reference frame's cloud onto each later one, with no pose model.
    conda activate "$ENV_GENESIS"
    echo "=== Fitting an object trajectory to the lifted points ==="
    # No ground constraint here: it rests the object on the table whenever it moves
    # slowly, and a slow lift then gets flattened onto it. Add --ground-plane and
    # --mesh if the object stays put for most of the clip.
    #
    # The poses are smoothed with a constant-velocity Kalman filter and an RTS pass
    # rather than the sliding window the other variants use. The per-frame Kabsch
    # solve is independent frame to frame, so it passes the depth map's jitter
    # straight through; the filter states the motion model instead, trusts each frame
    # in proportion to the inliers behind it, and rides through frames with no solve
    # by prediction rather than by interpolating across them. Set SMOOTHER=window to
    # fall back, and KALMAN_POS_NOISE / KALMAN_ACCEL (metres, metres/frame^2) to trade
    # smoothness against lag -- raising the measurement noise or lowering the
    # acceleration makes the trajectory stiffer and slower to follow a real move.
    python pose_from_points.py \
        --points "$POINTS_TRACK" \
        --mesh "$FIRST_MESH" \
        --smoother "$SMOOTHER" \
        --kalman-pos-noise "$KALMAN_POS_NOISE" \
        --kalman-accel "$KALMAN_ACCEL" \
        --kalman-rot-noise "$KALMAN_ROT_NOISE" \
        --kalman-ang-accel "$KALMAN_ANG_ACCEL" \
        --out "$POINTS_TRAJ"

    # ─────── Step 10: grasp synthesis at the video's grasp keyframes ───────
    # An independent read on the grasp, as the point tracks are on the pose: Lightning
    # Grasp is asked what the object's mesh affords at the onset, middle and release of
    # each grasp window, and the candidates are ranked against the hand the human
    # actually had there. The demonstration stays the reference -- these are corrections
    # available to it, not a replacement task.
    if [ "$SKIP_KEYFRAME_GRASPS" != "1" ]; then
        echo "=== Synthesizing grasps at the grasp keyframes (Lightning Grasp) ==="
        conda activate "$ENV_LYGRA"
        python "$SCRIPTS_DIR/keyframe_grasps.py" \
            --object-poses "$POINTS_TRAJ" \
            --mesh "$FIRST_MESH" \
            --hand-traj "$REPLAY_SOURCE" \
            --hand-meshes "$HAND_MESHES_PATH" \
            --side "$ANCHOR_HAND" \
            --ground-plane "$GROUND_PLANE" \
            --lygra-dir "$LIGHTNING_GRASP_DIR" \
            --batch-size-outer "$LYGRA_BATCH_OUTER" \
            --batch-size-inner "$LYGRA_BATCH_INNER" \
            --max-passes "$LYGRA_MAX_PASSES" \
            ${GRASP_PART:+--part "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$FIRST_OBJECT/part_${GRASP_PART// /_}.npy"} \
            ${GRASP_PART:+--part-share "$GRASP_PART_SHARE"} \
            --out "$KEYFRAME_GRASPS"
        conda activate "$ENV_GENESIS"
        cd "$SCRIPTS_DIR"
    else
        echo "=== SKIP_KEYFRAME_GRASPS=1: skipping Lightning Grasp ==="
    fi

    # A reconstructed grasp lands near the object rather than on it, and under --physics
    # that difference is the whole result: the fingers nudge the object instead of
    # closing on it. The refinement corrects the grasp against the object's own surface
    # and then against the simulation, inside the grasp windows only.
    REPLAY_TRAJ="$REPLAY_SOURCE"
    if [ "$SKIP_REFINE" != "1" ]; then
        echo "=== Refining the grasp ==="
        # refine_grasp.py builds its hand model on superdex.physics and polishes the
        # grasp by replaying it under SuperDex, so it runs in that env rather than
        # genesis; the trajectory it writes is read back by the Genesis replay below.
        conda activate "$ENV_SUPERDEX"
        python refine_grasp.py \
            --traj "$REPLAY_SOURCE" \
            --object-mesh "$FIRST_MESH" \
            --object-poses "$POINTS_TRAJ" \
            --ground-plane "$GROUND_PLANE" \
            --wuji-dir "$WUJI_RETARGETING_DIR" \
            --out "$REFINED_TRAJ"
        conda activate "$ENV_GENESIS"
        REPLAY_TRAJ="$REFINED_TRAJ"
    else
        echo "=== SKIP_REFINE=1: replaying the grasp as reconstructed ==="
        [ -f "$REFINED_TRAJ" ] && REPLAY_TRAJ="$REFINED_TRAJ"
    fi

    if [ "$SKIP_WUJI" != "1" ]; then
        echo "=== Replaying the point-based trajectory in Genesis ==="
        # The object is handed to the simulation when the grasp starts, not at the
        # tracker's reference frame: released earlier it drifts off its tracked pose
        # before the hand arrives, and the grasp then closes on empty space.
        RELEASE_ARG=()
        if [ -f "$REFINED_TRAJ" ]; then
            RELEASE_FRAME="$(python -c "import numpy,sys; w=numpy.load(sys.argv[1])['refine_windows']; print(int(w[0][0]) if len(w) else 0)" "$REFINED_TRAJ" 2>/dev/null || true)"
            [ -n "$RELEASE_FRAME" ] && RELEASE_ARG=(--release-frame "$RELEASE_FRAME")
        fi
        python replay_wuji_genesis.py \
            --traj "$REPLAY_TRAJ" \
            --object-mesh "$FIRST_MESH" \
            --object-poses "$POINTS_TRAJ" \
            --ground-plane "$GROUND_PLANE" \
            --wuji-dir "$WUJI_RETARGETING_DIR" \
            --record "$POINTS_REPLAY" \
            "${RELEASE_ARG[@]}" 
    fi
else
    echo "=== SKIP_COTRACKER=1: skipping the point tracks ==="
fi

if [ "$SKIP_POINT_MAP" != "1" ]; then
    echo "=== Getting point map ==="
    conda activate "$ENV_SAM3D"
    python get_pointmap_dir.py --image_dir "$VIDEO_DIR/all_frames"
    # A point map projects perfectly through its own intrinsics, so it is judged
    # against the measured depth and the real calibration instead.
    if [ -d "$VIDEO_DIR/depth_zed" ]; then
        echo "=== Comparing point maps with ZED depth ==="
        conda activate "$ENV_GENESIS"
        python overlay_pointmap.py \
            --frames-dir "$FRAMES_DIR" \
            --depth-dir "$VIDEO_DIR/depth_zed" \
            --intrinsics "$INTRINSICS_YAML" \
            --masks-root "$VIDEO_MASKS_DIR" \
            --object "$FIRST_OBJECT" \
            --out "$VIDEO_DIR/overlay_pointmap.mp4" \
            --plot "$VIDEO_DIR/pointmap_accuracy.png"
    fi
else
    echo "=== SKIP_POINT_MAP=1: skipping the point map stage ==="
fi

echo
echo "=== Done ==="
echo "  overlay:       $OVERLAY_PATH"
echo "  hand traj:     $WUJI_TRAJ"
echo "  optimized:     $WUJI_TRAJ_OPT"
echo "  object traj:   $FIRST_TRAJ"
echo "  object mesh:   $FIRST_MESH"
if [ "$SKIP_COTRACKER" != "1" ]; then
    echo "  point tracks:  $POINTS_TRACK"
    echo "  point overlay: $POINTS_OVERLAY"
    echo "  point traj:    $POINTS_TRAJ"
    if [ "$SKIP_WUJI" != "1" ]; then echo "  point replay:  $POINTS_REPLAY"; fi
    if [ "$SKIP_REFINE" != "1" ]; then echo "  refined traj:  $REFINED_TRAJ"; fi
    if [ "$SKIP_KEYFRAME_GRASPS" != "1" ]; then echo "  keyframe grasps: $KEYFRAME_GRASPS"; fi
fi