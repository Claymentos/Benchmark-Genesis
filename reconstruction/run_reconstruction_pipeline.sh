#!/usr/bin/env bash
# Human demo video -> Wuji Hand + object trajectory replayed in Genesis.
#
#   0  extract frames                                              [sam3]
#   1  SAM3 segmentation: object (click) + hand (text)             [sam3]
#   2  camera intrinsics from the recording's calibration          [-]
#   3  MoGe metric depth, every frame                              [moge]
#   4  supporting-plane fit (real up direction)                    [genesis]
#   5  object mesh from RGB + depth + mask                         [recgen]
#   6  HaWoR hand reconstruction                                   [hawor]
#   7  retarget hand onto the Wuji Hand                            [wuji_retargeting]
#   8  object pose tracking, Video2Sim2Real style                  [tapnet]
#   9  re-anchor object depth to the hand                          [genesis]
#  10  replay in Genesis                                           [genesis]
#
# Usage:  ./run_reconstruction_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
# Example: ./run_reconstruction_pipeline.sh human_videos/pouring_a_cup/take001_pouring_a_cup_rgb.mp4 49 cup left
#
# Stages are individually skippable so a run can resume rather than repeat the slow
# parts (MoGe over every frame and TAPIR are the expensive ones):
#   SKIP_EXTRACT_AND_SAM3=1  SKIP_DEPTH=1  SKIP_MESH=1  SKIP_HAWOR=1  SKIP_TRACK=1

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config/paths.sh"

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
DEPTH_DIR="$VIDEO_DIR/depth_moge"
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

SKIP_EXTRACT_AND_SAM3="${SKIP_EXTRACT_AND_SAM3:-0}"
SKIP_DEPTH="${SKIP_DEPTH:-0}"
SKIP_MESH="${SKIP_MESH:-0}"
SKIP_HAWOR="${SKIP_HAWOR:-0}"
SKIP_TRACK="${SKIP_TRACK:-0}"

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
    echo "=== Running SAM3 video segmentation (objects, click-based) ==="
    for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
        OBJ_ID="${OBJ_NAME// /_}"
        DISPLAY="$SAM3_DISPLAY" python run_sam3_video.py \
            --video "$VIDEO_PATH" \
            --click \
            --obj_id "$OBJ_ID" \
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
if [ -f "$META_PATH" ]; then
    INTRINSICS_INFO="$(python make_intrinsics.py --meta "$META_PATH" --out "$INTRINSICS_YAML")"
else
    echo "No metadata at $META_PATH; set META_PATH, or FOV_X and FOCAL, explicitly." >&2
    INTRINSICS_INFO=""
fi
echo "$INTRINSICS_INFO"
FOV_X="${FOV_X:-$(sed -n 's/^FOV_X=//p' <<< "$INTRINSICS_INFO")}"
FOCAL="${FOCAL:-$(sed -n 's/^FOCAL=//p' <<< "$INTRINSICS_INFO")}"
: "${FOV_X:?could not determine horizontal FOV}"
: "${FOCAL:?could not determine focal length}"

# ──────────────── Step 3: MoGe metric depth, every frame ────────────────
if [ "$SKIP_DEPTH" != "1" ]; then
    echo "=== Running MoGe depth over all frames ==="
    python run_moge_depth_batch.py \
        --frames-dir "$FRAMES_DIR" \
        --out-dir "$DEPTH_DIR" \
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
            --depth "$REF_DEPTH_PATH" \
            --mask "$MASKS_DIR/${OBJECT_ID}.png" \
            --intrinsics "$INTRINSICS_YAML" \
            --checkpoint "$RECGEN_CHECKPOINT" \
            --out "$VIDEO_DIR/recgen_out/$OBJECT_ID" \
            --name "$OBJECT_ID"
    done
else
    echo "=== SKIP_MESH=1: reusing $VIDEO_DIR/recgen_out ==="
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
    OBJECT_MESH="$VIDEO_DIR/recgen_out/$OBJECT_ID/posed_mesh.obj"
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
    echo "=== Anchoring $OBJECT_ID depth to the hand ==="
    # Kept separate from tracking so the anchoring can be re-tuned without paying for
    # another TAPIR pass.
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
done

# ──────────────── Step 10: replay in Genesis ────────────────
FIRST_OBJECT="${OBJECT_NAMES[0]// /_}"
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"
echo "=== Replaying in Genesis ==="
python replay_wuji_genesis.py \
    --traj "$WUJI_TRAJ" \
    --object-mesh "$VIDEO_DIR/recgen_out/$FIRST_OBJECT/posed_mesh.obj" \
    --object-poses "$VIDEO_DIR/${FIRST_OBJECT}_traj.npz" \
    --ground-plane "$GROUND_PLANE" \
    --wuji-dir "$WUJI_RETARGETING_DIR" \
    --record "$REPLAY_PATH"

echo
echo "=== Done ==="
echo "  replay:        $REPLAY_PATH"
echo "  hand traj:     $WUJI_TRAJ"
echo "  object traj:   $VIDEO_DIR/${FIRST_OBJECT}_traj.npz"
echo "  object mesh:   $VIDEO_DIR/recgen_out/$FIRST_OBJECT/posed_mesh.obj"
