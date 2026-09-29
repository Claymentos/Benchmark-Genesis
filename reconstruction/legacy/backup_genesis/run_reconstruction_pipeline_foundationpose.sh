#!/usr/bin/env bash
# Human demo video -> Wuji Hand + object trajectory replayed in Genesis.
#
# FoundationPose variant. Two differences from run_reconstruction_pipeline.sh, which
# stays as the CoTracker/Kabsch (Video2Sim2Real) version:
#
#   * Depth comes from the ZED recording's own stereo measurement rather than MoGe's
#     monocular estimate. On this data MoGe is 5x noisier frame to frame (5.0 mm vs
#     1.0 mm of depth jitter) and reads 2-4 cm too far, worst while the object is
#     lifted -- which is most of the object-pose noise.
#   * Object pose comes from FoundationPose, which renders the mesh against RGB-D
#     instead of fitting sparse point correspondences. A textureless, near-cylindrical
#     object barely constrains those correspondences, so its own geometry does the
#     disambiguating instead.
#
# Set DEPTH_SOURCE=moge to fall back to monocular depth when there is no SVO.
#
#   0  extract frames                                              [sam3]
#   1  SAM3 segmentation: object (click) + hand (text)             [sam3]
#   2  camera intrinsics from the recording's calibration          [-]
#   3  metric depth, every frame                                   [system python / moge]
#   4  supporting-plane fit (real up direction)                    [genesis]
#   5  object mesh from RGB + depth + mask                         [recgen]
#   6  HaWoR hand reconstruction                                   [hawor]
#   7  retarget hand onto the Wuji Hand                            [wuji_retargeting]
#   8  object pose tracking with FoundationPose, with a spin check  [foundationpose]
#      about a symmetry axis found in the mesh (skipped when there is none)
#   9  re-anchor object depth to the hand (monocular depth only), then take
#      the object's spin from the wrist while it is held             [genesis]
#  10  overlay the tracked pose on the source video                [genesis]
#  11  replay the free-floating hand in Genesis                    [genesis]
#  12  Franka Panda arm trajectory by IK, replayed in Genesis      [genesis]
#  13  CoTracker surface points lifted with depth, drawn on the video [sam3],
#      then a rigid trajectory fitted to them and replayed in Genesis  [genesis]
#  14  refine the grasp until it holds, and replay that              [genesis]
#
# Every output here is tagged with the tracker (_fp) so this variant never overwrites
# the CoTracker/Kabsch results: <object>_traj_fp{_raw}.npz and overlay_fp.mp4. With
# DEPTH_SOURCE=moge every depth-dependent output also gets a _moge suffix (ground
# plane, trajectories, overlay, replays), so a MoGe run sits beside the ZED one.
#
# Usage:  ./run_reconstruction_pipeline_foundationpose.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
# Example: ./run_reconstruction_pipeline_foundationpose.sh human_videos/pouring_a_cup/take001_pouring_a_cup_rgb.mp4 49 cup left
#
# Stages are individually skippable so a run can resume rather than repeat the slow
# parts (MoGe over every frame and FoundationPose are the expensive ones):
# Set OBJECT_PROMPT=text to segment the object by name instead of by clicking, and
# CAM_PARAM_JSON (+ CAMERA_ID) when calibration ships apart from the footage.
#   SKIP_EXTRACT_AND_SAM3=1  SKIP_DEPTH=1  SKIP_MESH=1  SKIP_HAWOR=1  SKIP_RETARGETING=1  SKIP_TRACK=1  SKIP_HAND_LOCK=1  SKIP_OVERLAY=1  SKIP_WUJI=1  SKIP_FRANKA=1  SKIP_COTRACKER=1  SKIP_POINT_MAP=1  SKIP_REFINE=1

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config/paths.sh"

# ──────────────────────────── Per-run inputs (args) ────────────────────────────
VIDEO_PATH="${1:?usage: run_reconstruction_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]}"
VIDEO_PATH="$(realpath "$VIDEO_PATH")"
n="${2:-49}"
OBJECT_NAMES=("${3:-screwdriver}")
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
REPLAY_PATH="$VIDEO_DIR/wuji_replay$DEPTH_TAG.mp4"
OVERLAY_PATH="$VIDEO_DIR/overlay_fp$DEPTH_TAG.mp4"
POINTS_TRACK="$VIDEO_DIR/points_cotracker$DEPTH_TAG.npz"
POINTS_OVERLAY="$VIDEO_DIR/overlay_points$DEPTH_TAG.mp4"
POINTS_TRAJ="$VIDEO_DIR/${OBJECT_NAMES[0]// /_}_traj_points$DEPTH_TAG.npz"
POINTS_REPLAY="$VIDEO_DIR/wuji_replay_points$DEPTH_TAG.mp4"
REFINED_TRAJ="$VIDEO_DIR/wuji_traj_refined$DEPTH_TAG.npz"
FRANKA_TRAJ="$VIDEO_DIR/franka_traj$DEPTH_TAG.npz"
FRANKA_REPLAY_PATH="$VIDEO_DIR/franka_replay$DEPTH_TAG.mp4"
FRANKA_WUJI_URDF="${PANDA_WUJI_URDF:-$PANDA_WUJI_URDF_DIR/panda_wuji_${ANCHOR_HAND}.urdf}"

SKIP_EXTRACT_AND_SAM3="${SKIP_EXTRACT_AND_SAM3:-0}"
SKIP_DEPTH="${SKIP_DEPTH:-0}"
SKIP_MESH="${SKIP_MESH:-0}"
SKIP_HAWOR="${SKIP_HAWOR:-0}"
SKIP_TRACK="${SKIP_TRACK:-0}"
SKIP_OVERLAY="${SKIP_OVERLAY:-0}"
SKIP_FRANKA="${SKIP_FRANKA:-0}"
SKIP_RETARGETING="${SKIP_RETARGETING:-0}"
SKIP_HAND_LOCK="${SKIP_HAND_LOCK:-0}"
SKIP_WUJI="${SKIP_WUJI:-0}"
SKIP_COTRACKER="${SKIP_COTRACKER:-0}"
SKIP_POINT_MAP="${SKIP_POINT_MAP:-0}"
SKIP_REFINE="${SKIP_REFINE:-0}"
# click needs a display; text runs headless.
OBJECT_PROMPT="${OBJECT_PROMPT:-click}"
# Anchoring rebuilds the object's position relative to the hand. It was built to
# repair monocular depth drift and measurably helps there, but with ZED stereo depth
# the tracker's own position is already metric and anchoring adds the hand's error
# instead: on grasping_cup it took reprojection error from 15 px to 39 px. So it
# follows the depth source unless forced with yes/no.
ANCHOR_TO_HAND="${ANCHOR_TO_HAND:-auto}"

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

# ──────────────── Step 3: MoGe metric depth, every frame ────────────────
if [ "$SKIP_DEPTH" != "1" ]; then

    echo "=== Extracting ZED stereo depth ==="
    # pyzed lives in the system python, not a conda env, so this one stage runs
    # outside the env chain.
    conda deactivate
    "$PYTHON_ZED" "$SCRIPTS_DIR/extract_zed_depth.py" \
        --svo "$SVO_PATH" \
        --out-dir "$DEPTH_DIR" \
        --depth-mode "$ZED_DEPTH_MODE"

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

# ──────────────── Step 5: object mesh (RecGen on MoGe depth) ────────────────
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

# ──────────────── Steps 8-9: object tracking, then anchored to the hand ────────────────
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJECT_ID="${OBJ_NAME// /_}"
    OBJECT_MESH="$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$OBJECT_ID/posed_mesh.obj"
    OBJECT_TRAJ_RAW="$VIDEO_DIR/${OBJECT_ID}_traj_fp${DEPTH_TAG}_raw.npz"
    OBJECT_TRAJ="$VIDEO_DIR/${OBJECT_ID}_traj_fp${DEPTH_TAG}.npz"

    if [ "$SKIP_TRACK" != "1" ]; then
        conda activate "$ENV_FOUNDATIONPOSE"
        cd "$SCRIPTS_DIR"
        echo "=== Tracking object pose with FoundationPose: $OBJECT_ID ==="
        python track_object_foundationpose.py \
            --mesh "$OBJECT_MESH" \
            --frames-dir "$FRAMES_DIR" \
            --depth-dir "$DEPTH_DIR" \
            --masks-root "$VIDEO_MASKS_DIR" \
            --object "$OBJECT_ID" \
            --ref-frame "$n" \
            --intrinsics "$INTRINSICS_YAML" \
            --foundationpose-dir "$FOUNDATIONPOSE_DIR" \
            --out "$OBJECT_TRAJ_RAW"
    else
        echo "=== SKIP_TRACK=1: reusing $OBJECT_TRAJ_RAW ==="
    fi

    conda activate "$ENV_GENESIS"
    cd "$SCRIPTS_DIR"
    if [ "$ANCHOR_TO_HAND" = "yes" ] || { [ "$ANCHOR_TO_HAND" = "auto" ] && [ "$DEPTH_SOURCE" != "zed" ]; }; then
        echo "=== Anchoring $OBJECT_ID depth to the hand ==="
        # Kept separate from tracking so the anchoring can be re-tuned without paying
        # for another FoundationPose pass.
        python anchor_object_to_hand.py \
            --object-traj "$OBJECT_TRAJ_RAW" \
            --hand-meshes "$HAND_MESHES_PATH" \
            --side "$ANCHOR_HAND" \
            --mesh "$OBJECT_MESH" \
            --depth-dir "$DEPTH_DIR" \
            --masks-root "$VIDEO_MASKS_DIR" \
            --object "$OBJECT_ID" \
            --intrinsics "$INTRINSICS_YAML" \
            --ground-plane "$GROUND_PLANE" \
            --out "$OBJECT_TRAJ"
    else
        echo "=== Skipping hand anchoring (depth source: $DEPTH_SOURCE) ==="
        cp "$OBJECT_TRAJ_RAW" "$OBJECT_TRAJ"
    fi

    # While the object is held it moves rigidly with the hand, which is also what hides
    # the features the tracker needs. Whichever of the two explains the object's mask
    # better over the grasp is kept, ties going to the hand.
    if [ "$SKIP_HAND_LOCK" != "1" ]; then
        echo "=== Locking $OBJECT_ID rotation to the wrist while held ==="
        python lock_object_to_hand.py \
            --object-traj "$OBJECT_TRAJ" \
            --hand-traj "$WUJI_TRAJ" \
            --hand-meshes "$HAND_MESHES_PATH" \
            --side "$ANCHOR_HAND" \
            --mesh "$OBJECT_MESH" \
            --masks-root "$VIDEO_MASKS_DIR" \
            --object "$OBJECT_ID" \
            --intrinsics "$INTRINSICS_YAML" \
            --out "$OBJECT_TRAJ"
    else
        echo "=== SKIP_HAND_LOCK=1: keeping the tracked rotation ==="
    fi
done

FIRST_OBJECT="${OBJECT_NAMES[0]// /_}"
FIRST_MESH="$VIDEO_DIR/recgen_out_moge/$FIRST_OBJECT/posed_mesh.obj"
FIRST_TRAJ_RAW="$VIDEO_DIR/${FIRST_OBJECT}_traj_fp${DEPTH_TAG}_raw.npz"
FIRST_TRAJ="$VIDEO_DIR/${FIRST_OBJECT}_traj_fp${DEPTH_TAG}.npz"
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"

# ──────────────── Step 10: overlay the tracked pose on the video ────────────────
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

# # ──────────────── Step 11: replay in Genesis ────────────────
# if [ "$SKIP_WUJI" != "1" ]; then
#     echo "=== Replaying in Genesis ==="
#     python replay_wuji_genesis.py \
#         --traj "$WUJI_TRAJ" \
#         --object-mesh "$FIRST_MESH" \
#         --object-poses "$FIRST_TRAJ" \
#         --ground-plane "$GROUND_PLANE" \
#         --wuji-dir "$WUJI_RETARGETING_DIR" \
#         --record "$REPLAY_PATH"
# else
#     echo "=== SKIP_WUJI=1: skipping the hand stage ==="
# fi

# ──────────────── Step 12: Franka Panda arm trajectory ────────────────
# The retargeting leaves the wrist free-floating; mounting the hand on an arm turns
# that pose into a constraint, so the arm's joint angles are solved per frame.
if [ "$SKIP_FRANKA" != "1" ]; then
    echo "=== Solving Franka arm trajectory ==="
    python solve_franka_ik.py \
        --traj "$WUJI_TRAJ" \
        --urdf "$FRANKA_WUJI_URDF" \
        --object-mesh "$FIRST_MESH" \
        --object-poses "$FIRST_TRAJ" \
        --ground-plane "$GROUND_PLANE" \
        --out "$FRANKA_TRAJ" \
        --record "$FRANKA_REPLAY_PATH"
else
    echo "=== SKIP_FRANKA=1: skipping the arm stage ==="
fi
# ──────────────── CoTracker points, lifted with depth ────────────────
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
    # the tracking is identical, so every difference in 3D comes from it.
    OTHER_DEPTH="$VIDEO_DIR/depth_moge"
    [ "$DEPTH_SOURCE" = "moge" ] && OTHER_DEPTH="$VIDEO_DIR/depth_zed"
    if [ -d "$OTHER_DEPTH" ]; then
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

    # The lifted points also give a trajectory of their own: the rigid motion carrying
    # the reference frame's cloud onto each later one. Model-free, so it is a check on
    # the FoundationPose trajectory rather than a replacement.
    conda activate "$ENV_GENESIS"
    echo "=== Fitting an object trajectory to the lifted points ==="
    # No ground constraint here: it rests the object on the table whenever it moves
    # slowly, and a slow lift then gets flattened onto it. Add --ground-plane and
    # --mesh if the object stays put for most of the clip.
    python pose_from_points.py \
        --points "$POINTS_TRACK" \
        --mesh "$FIRST_MESH" \
        --out "$POINTS_TRAJ"

    # A reconstructed grasp lands near the object rather than on it, and under --physics
    # that difference is the whole result: the fingers nudge the object instead of
    # closing on it. The refinement corrects the grasp against the object's own surface
    # and then against the simulation, inside the grasp windows only.
    REPLAY_TRAJ="$WUJI_TRAJ"
    if [ "$SKIP_REFINE" != "1" ]; then
        echo "=== Refining the grasp ==="
        python refine_grasp.py \
            --traj "$WUJI_TRAJ" \
            --object-mesh "$FIRST_MESH" \
            --object-poses "$POINTS_TRAJ" \
            --ground-plane "$GROUND_PLANE" \
            --wuji-dir "$WUJI_RETARGETING_DIR" \
            --out "$REFINED_TRAJ"
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
            "${RELEASE_ARG[@]}" \
            --physics
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
echo "  replay:        $REPLAY_PATH"
echo "  hand traj:     $WUJI_TRAJ"
echo "  object traj:   $FIRST_TRAJ"
echo "  object mesh:   $FIRST_MESH"
echo "  arm traj:      $FRANKA_TRAJ"
echo "  arm replay:    $FRANKA_REPLAY_PATH"
if [ "$SKIP_COTRACKER" != "1" ]; then
    echo "  point tracks:  $POINTS_TRACK"
    echo "  point overlay: $POINTS_OVERLAY"
    echo "  point traj:    $POINTS_TRAJ"
    if [ "$SKIP_WUJI" != "1" ]; then echo "  point replay:  $POINTS_REPLAY"; fi
    if [ "$SKIP_REFINE" != "1" ]; then echo "  refined traj:  $REFINED_TRAJ"; fi
fi
