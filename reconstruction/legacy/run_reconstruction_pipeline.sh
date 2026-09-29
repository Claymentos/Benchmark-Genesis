#!/usr/bin/env bash
# Human demo video -> Wuji Hand + object trajectory replayed in Genesis.
#
#   0  extract frames                                              [sam3]
#   1  SAM3 segmentation: object (click) + hand (text)             [sam3]
#   2  camera intrinsics from the recording's calibration          [-]
#   3  metric depth, every frame (ZED stereo, else MoGe)          [system python / moge]
#   4  supporting-plane fit (real up direction)                    [genesis]
#   5  object mesh from RGB + depth + mask                         [recgen]
#   6  HaWoR hand reconstruction                                   [hawor]
#   7  retarget hand onto the Wuji Hand                            [wuji_retargeting]
#   8  object pose tracking, Video2Sim2Real style                  [tapnet]
#   9  re-anchor object depth to the hand                          [genesis]
#  10  replay the free-floating hand in Genesis                    [genesis]
#  11  Franka Panda arm trajectory by IK, replayed in Genesis      [genesis]
#
# Usage:  ./run_reconstruction_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
# Example: ./run_reconstruction_pipeline.sh human_videos/pouring_a_cup/take001_pouring_a_cup_rgb.mp4 49 cup left
#
# Stages are individually skippable so a run can resume rather than repeat the slow
# parts (MoGe over every frame and TAPIR are the expensive ones):
# Set OBJECT_PROMPT=text to segment the object by name instead of by clicking, and
# CAM_PARAM_JSON (+ CAMERA_ID) when calibration ships apart from the footage.
#   SKIP_EXTRACT_AND_SAM3=1  SKIP_DEPTH=1  SKIP_MESH=1  SKIP_HAWOR=1  SKIP_TRACK=1  SKIP_FRANKA=1

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../config/paths.sh"
# Scripts only these legacy pipelines use live in legacy/scripts; they still import
# shared modules (e.g. track_object_v2s2r) from $SCRIPTS_DIR.
LEGACY_SCRIPTS_DIR="$HERE/scripts"
export PYTHONPATH="$SCRIPTS_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ──────────────────────────── Per-run inputs (args) ────────────────────────────
VIDEO_PATH="${1:?usage: run_reconstruction_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]}"
VIDEO_PATH="$(realpath "$VIDEO_PATH")"
n="${2:-49}"
OBJECT_NAMES=("${3:-cup}")
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
REF_DEPTH_PATH="$DEPTH_DIR/$(printf "%06d.png" "$n")"
INTRINSICS_YAML="$VIDEO_DIR/intrinsics.yaml"
GROUND_PLANE="$VIDEO_DIR/ground_plane.json"
MASKS_DIR="$VIDEO_DIR/video_segmentation/masks/frame_$(printf "%06d" "$n")_masks"
VIDEO_MASKS_DIR="$VIDEO_DIR/video_segmentation/masks"
META_PATH="${META_PATH:-${VIDEO_PATH%_rgb.*}_meta.json}"

HAND_ID="${ANCHOR_HAND}_hand_0"
HAND_MESHES_PATH="$VIDEO_DIR/$VIDEO_NAME/all_hand_meshes.npz"
WUJI_TRAJ="$VIDEO_DIR/wuji_traj.npz"
REPLAY_PATH="$VIDEO_DIR/wuji_replay.mp4"
FRANKA_TRAJ="$VIDEO_DIR/franka_traj.npz"
FRANKA_REPLAY_PATH="$VIDEO_DIR/franka_replay.mp4"
FRANKA_WUJI_URDF="${PANDA_WUJI_URDF:-$PANDA_WUJI_URDF_DIR/panda_wuji_${ANCHOR_HAND}.urdf}"

SKIP_EXTRACT_AND_SAM3="${SKIP_EXTRACT_AND_SAM3:-0}"
SKIP_DEPTH="${SKIP_DEPTH:-0}"
SKIP_MESH="${SKIP_MESH:-0}"
SKIP_HAWOR="${SKIP_HAWOR:-0}"
SKIP_TRACK="${SKIP_TRACK:-0}"
SKIP_FRANKA="${SKIP_FRANKA:-0}"
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
    if [ "$DEPTH_SOURCE" = "zed" ]; then
        echo "=== Extracting ZED stereo depth ==="
        # pyzed lives in the system python, not a conda env, so this one stage runs
        # outside the env chain.
        conda deactivate
        "$PYTHON_ZED" "$SCRIPTS_DIR/extract_zed_depth.py" \
            --svo "$SVO_PATH" \
            --out-dir "$DEPTH_DIR" \
            --depth-mode "$ZED_DEPTH_MODE"
        conda activate "$ENV_MOGE"
    else
        echo "=== Running MoGe depth over all frames ==="
        python run_moge_depth_batch.py \
            --frames-dir "$FRAMES_DIR" \
            --out-dir "$DEPTH_DIR" \
            --fov-x "$FOV_X" \
            --checkpoint "$MOGE_CHECKPOINT" \
            --moge-repo "$MOGE_REPO"
    fi
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
            --depth "$REF_DEPTH_PATH" \
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
conda activate "$ENV_WUJI"
cd "$SCRIPTS_DIR"
echo "=== Retargeting hand onto the Wuji Hand ==="
python hawor_to_wuji_qpos.py \
    --hand-meshes "$HAND_MESHES_PATH" \
    --side "$ANCHOR_HAND" \
    --wuji-dir "$WUJI_RETARGETING_DIR" \
    --out "$WUJI_TRAJ"

# ──────────────── Steps 8-9: object tracking, then anchored to the hand ────────────────
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJECT_ID="${OBJ_NAME// /_}"
    OBJECT_MESH="$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$OBJECT_ID/posed_mesh.obj"
    OBJECT_TRAJ_RAW="$VIDEO_DIR/${OBJECT_ID}_traj_raw.npz"
    OBJECT_TRAJ="$VIDEO_DIR/${OBJECT_ID}_traj.npz"

    if [ "$SKIP_TRACK" != "1" ]; then
        conda activate "$ENV_TAPNET"
        cd "$SCRIPTS_DIR"
        echo "=== Tracking object pose: $OBJECT_ID ==="
        python track_object_v2s2r.py \
            --frames-dir "$FRAMES_DIR" \
            --masks-root "$VIDEO_MASKS_DIR" \
            --depth-dir "$DEPTH_DIR" \
            --object "$OBJECT_ID" \
            --ref-frame "$n" \
            --intrinsics "$INTRINSICS_YAML" \
            --ground-plane "$GROUND_PLANE" \
            --mesh "$OBJECT_MESH" \
            --checkpoint "$TAPNET_CKPT" \
            --out "$OBJECT_TRAJ_RAW"
    else
        echo "=== SKIP_TRACK=1: reusing $OBJECT_TRAJ_RAW ==="
    fi

    conda activate "$ENV_GENESIS"
    cd "$SCRIPTS_DIR"
    if [ "$ANCHOR_TO_HAND" = "yes" ] || { [ "$ANCHOR_TO_HAND" = "auto" ] && [ "$DEPTH_SOURCE" != "zed" ]; }; then
    echo "=== Anchoring $OBJECT_ID depth to the hand ==="
    # Kept separate from tracking so the anchoring can be re-tuned without paying for
    # another TAPIR pass.
    python "$LEGACY_SCRIPTS_DIR/anchor_object_to_hand.py" \
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
done

# ──────────────── Step 10: replay in Genesis ────────────────
FIRST_OBJECT="${OBJECT_NAMES[0]// /_}"
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"
echo "=== Replaying in Genesis ==="
python replay_wuji_genesis.py \
    --traj "$WUJI_TRAJ" \
    --object-mesh "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$FIRST_OBJECT/posed_mesh.obj" \
    --object-poses "$VIDEO_DIR/${FIRST_OBJECT}_traj.npz" \
    --ground-plane "$GROUND_PLANE" \
    --wuji-dir "$WUJI_RETARGETING_DIR" \
    --record "$REPLAY_PATH"
# ──────────────── Step 11: Franka Panda arm trajectory ────────────────
# The retargeting leaves the wrist free-floating; mounting the hand on an arm turns
# that pose into a constraint, so the arm's joint angles are solved per frame.
if [ "$SKIP_FRANKA" != "1" ]; then
    echo "=== Solving Franka arm trajectory ==="
    python "$LEGACY_SCRIPTS_DIR/solve_franka_ik.py" \
        --traj "$WUJI_TRAJ" \
        --urdf "$FRANKA_WUJI_URDF" \
        --object-mesh "$VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$FIRST_OBJECT/posed_mesh.obj" \
        --object-poses "$VIDEO_DIR/${FIRST_OBJECT}_traj.npz" \
        --ground-plane "$GROUND_PLANE" \
        --out "$FRANKA_TRAJ" \
        --record "$FRANKA_REPLAY_PATH"
else
    echo "=== SKIP_FRANKA=1: skipping the arm stage ==="
fi

echo
echo "=== Done ==="
echo "  replay:        $REPLAY_PATH"
echo "  hand traj:     $WUJI_TRAJ"
echo "  object traj:   $VIDEO_DIR/${FIRST_OBJECT}_traj.npz"
echo "  object mesh:   $VIDEO_DIR/recgen_out_$DEPTH_SOURCE/$FIRST_OBJECT/posed_mesh.obj"
echo "  arm traj:      $FRANKA_TRAJ"
echo "  arm replay:    $FRANKA_REPLAY_PATH"
