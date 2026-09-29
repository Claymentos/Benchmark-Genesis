#!/usr/bin/env bash
# TouchAnything episode -> both MANO hands + the object, reconstructed and replayed in
# Genesis.
#
# TouchAnything records the hands with Rokoko gloves on Vive trackers, so the hand pose
# is measured, not estimated: no HaWoR, no HaMeR, no retargeting. The camera
# (chest.mp4, head-mounted in fact) moves and has no calibration. So the camera's
# motion comes from the static table, the mocap is placed in the camera from the
# video, and the object is tracked with CoTracker points lifted by depth -- with no
# keyframes, grasp synthesis or refinement anywhere.
#
# Everything is expressed in the reference camera (the camera at frame FRAME_N):
#
#   1  frames + SAM3 segmentation: object, and the static table              [sam3]
#   2  camera intrinsics, from MoGe on the reference frame                    [moge]
#   3  MoGe metric depth, every frame                                          [moge]
#   4  camera motion: CoTracker points on the table, lifted with depth,      [sam3 +
#      Kabsch to the reference frame -> T_ct<-cref                         genesis]
#   5  mocap joints (both hands) placed in the reference camera: one          [genesis]
#      similarity for the whole clip, from WiLoR 2D + depth
#   6  metric: MoGe's scale is off by a few to tens of percent; when the     [genesis]
#      fitted scale says so, the depth and the camera track are rescaled
#      into the mocap's true metres and the hands refitted at scale 1
#   7  supporting plane (table) from the reference depth                      [genesis]
#   8  object mesh from RGB + depth + mask (RecGen)                            [recgen]
#   9  object trajectory: CoTracker points on the object, lifted with        [sam3 +
#      depth, Kabsch -> camera-relative pose, composed with the camera     genesis]
#      motion into the reference camera, then a constant-velocity Kalman
#      filter + RTS smoother on the object's centre and rotation
#  9b  optional (HELD_WINDOW=A-B): the held object's depth from the hand      [genesis]
#  10  MANO fitted to the mocap joints, both hands, cut into 16 rigid parts    [ecfit]
#  11  kinematic replay in Genesis: both hands + object, table at z = 0,     [genesis]
#      rendered from the real moving camera next to the input
#
# Usage:   ./run_touchanything_pipeline.sh EPISODE_DIR OUT_DIR OBJECT [FRAME_N]
# Example: ./run_touchanything_pipeline.sh \
#            touchanything/Workbench/pick_up_screwdriver/20260321_103227_213 \
#            human_videos/touchanything/pick_up_screwdriver_103227 screwdriver 0
#
# FRAME_N is the reference frame: the object fully visible and unoccluded, and the
# table visible. Knobs:
#   TABLE_POINTS="x,y;x,y;..."  prompt SAM3 for the table with points on FRAME_N when
#                               the text prompt misses it (the mask must cover FRAME_N)
#   STATIC_PROMPT=table         the static scene element the camera is tracked from
#   SCALE_TOLERANCE=0.02        rescale the depth when |s - 1| exceeds this
#   RECGEN_EROSION=1            let RecGen erode the mask (off: it deletes thin parts)
#   HAND_RENDER=parts           replay the 16 rigid parts instead of the skinned mesh
#   OBJECT_SMOOTHER=kalman      kalman (after composition) | window (per frame, camera-relative)
#   HELD_WINDOW=A-B             see stage 9b; off by default (HELD_SIDE=left|right, default left)
#   SAM3_OFFLOAD=1              keep SAM3's frames in CPU RAM
# Stages are skippable to resume:
#   SKIP_EXTRACT_AND_SAM3=1 SKIP_DEPTH=1 SKIP_CAMERA=1 SKIP_HANDS=1 SKIP_MESH=1
#   SKIP_OBJECT=1 SKIP_MANO=1 SKIP_REPLAY=1

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config/paths.sh"

EPISODE_DIR="$(realpath "${1:?usage: run_touchanything_pipeline.sh EPISODE_DIR OUT_DIR OBJECT [FRAME_N]}")"
mkdir -p "${2:?usage: run_touchanything_pipeline.sh EPISODE_DIR OUT_DIR OBJECT [FRAME_N]}"
OUT_DIR="$(realpath "$2")"
OBJECT="${3:?usage: run_touchanything_pipeline.sh EPISODE_DIR OUT_DIR OBJECT [FRAME_N]}"
n="${4:-0}"

OBJ_ID="${OBJECT// /_}"
STATIC_PROMPT="${STATIC_PROMPT:-table}"
STATIC_ID="${STATIC_PROMPT// /_}"
TASK="$(basename "$(dirname "$EPISODE_DIR")")"
VIDEO_PATH="$OUT_DIR/${TASK}_rgb.mp4"
FRAMES_DIR="$OUT_DIR/all_frames"
FRAME_PATH="$FRAMES_DIR/$(printf "%06d.png" "$n")"
MASKS_ROOT="$OUT_DIR/video_segmentation/masks"
REF_MASKS="$MASKS_ROOT/frame_$(printf "%06d" "$n")_masks"
DEPTH_DIR="$OUT_DIR/depth_moge"
INTRINSICS_YAML="$OUT_DIR/intrinsics.yaml"
GROUND_PLANE="$OUT_DIR/ground_plane_moge.json"
MESH="$OUT_DIR/recgen_out_moge/$OBJ_ID/posed_mesh.obj"

# Outputs of this pipeline are tagged _cotracker so they sit beside run_v2s2r's.
TABLE_POINTS_NPZ="$OUT_DIR/${STATIC_ID}_points_cotracker.npz"
CAMERA_TRAJ="$OUT_DIR/camera_traj_cotracker.npz"
MOCAP_JOINTS="$OUT_DIR/mocap_joints_cotracker.npz"
MOCAP_FIT="$OUT_DIR/mocap_to_camera_cotracker.json"
OBJ_POINTS="$OUT_DIR/${OBJ_ID}_points_cotracker.npz"
OBJ_TRAJ_CAM="$OUT_DIR/${OBJ_ID}_traj_camera_cotracker.npz"
OBJ_TRAJ_COMPOSED="$OUT_DIR/${OBJ_ID}_traj_composed_cotracker.npz"
OBJ_TRAJ_SMOOTH="$OUT_DIR/${OBJ_ID}_traj_kalman_cotracker.npz"
OBJ_TRAJ="$OUT_DIR/${OBJ_ID}_traj_cotracker.npz"
MANO_OUT="$OUT_DIR/mano_hands_cotracker.npz"
MANO_PARTS="$OUT_DIR/mano_parts_cotracker"
REPLAY="$OUT_DIR/replay_hands_cotracker.mp4"

SCALE_TOLERANCE="${SCALE_TOLERANCE:-0.02}"
HELD_WINDOW="${HELD_WINDOW:-}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-9}"
OBJECT_SMOOTHER="${OBJECT_SMOOTHER:-kalman}"

source "$(conda info --base)/etc/profile.d/conda.sh"
for f in chest.mp4 rokoko_hands.json wilor_hands.json; do
    [ -f "$EPISODE_DIR/$f" ] || { echo "no $f in $EPISODE_DIR" >&2; exit 1; }
done

# ──────────────── Stage 1: frames + SAM3 ────────────────
if [ "${SKIP_EXTRACT_AND_SAM3:-0}" != "1" ]; then
    conda activate "$ENV_SAM3"
    cp "$EPISODE_DIR/chest.mp4" "$VIDEO_PATH"
    echo "=== Extracting all frames ==="
    rm -rf "$FRAMES_DIR" && mkdir -p "$FRAMES_DIR"
    ffmpeg -v error -i "$VIDEO_PATH" -vsync 0 -start_number 0 "$FRAMES_DIR/%06d.png"

    cd "$SCRIPTS_DIR"
    echo "=== SAM3: object '$OBJECT' ==="
    python run_sam3_video.py ${SAM3_OFFLOAD:+--offload-video} \
        --video "$VIDEO_PATH" --text "$OBJECT" --obj_id "$OBJ_ID" --frame_idx "$n"
    echo "=== SAM3: static '$STATIC_PROMPT' (camera motion) ==="
    if [ -n "$TABLE_POINTS" ]; then
        labels="$(tr ';' '\n' <<< "$TABLE_POINTS" | sed 's/.*/1/' | paste -sd';')"
        python run_sam3_video.py ${SAM3_OFFLOAD:+--offload-video} \
            --video "$VIDEO_PATH" --points "$TABLE_POINTS" --point_labels "$labels" \
            --obj_id "$STATIC_ID" --frame_idx "$n"
    else
        python run_sam3_video.py ${SAM3_OFFLOAD:+--offload-video} \
            --video "$VIDEO_PATH" --text "$STATIC_PROMPT" --obj_id "$STATIC_ID" --frame_idx "$n"
    fi
else
    echo "=== SKIP_EXTRACT_AND_SAM3=1 ==="
fi
conda activate "$ENV_GENESIS"
# Both trackers seed their points on the reference frame's mask, so an empty one there
# is fatal -- and SAM3's text prompt does miss tables. Say so rather than fail later.
for mask in "$REF_MASKS/$OBJ_ID.png" "$REF_MASKS/$STATIC_ID.png"; do
    python -c "import cv2,sys; m=cv2.imread(sys.argv[1],0); sys.exit(0 if m is not None and (m>128).sum()>100 else 1)" "$mask" \
        || { echo "empty mask at the reference frame: $mask (for the table, try TABLE_POINTS)" >&2; exit 1; }
done

# ──────────────── Stages 2-3: intrinsics + depth ────────────────
conda activate "$ENV_MOGE"
cd "$SCRIPTS_DIR"
echo "=== Camera intrinsics (MoGe, frame $n) ==="
INTRINSICS_INFO="$(python make_intrinsics.py --from-moge "$FRAME_PATH" --checkpoint "$MOGE_CHECKPOINT" \
    --moge-repo "$MOGE_REPO" --out "$INTRINSICS_YAML")"
echo "$INTRINSICS_INFO"
FOV_X="$(sed -n 's/^FOV_X=//p' <<< "$INTRINSICS_INFO")"
if [ "${SKIP_DEPTH:-0}" != "1" ]; then
    echo "=== MoGe depth, every frame ==="
    # A fresh depth pass invalidates any earlier rescale of it.
    rm -rf "$DEPTH_DIR" "${DEPTH_DIR}_raw"
    python run_moge_depth_batch.py --frames-dir "$FRAMES_DIR" --out-dir "$DEPTH_DIR" \
        --fov-x "$FOV_X" --checkpoint "$MOGE_CHECKPOINT" --moge-repo "$MOGE_REPO"
else
    echo "=== SKIP_DEPTH=1: reusing $DEPTH_DIR ==="
fi

# CoTracker points on one mask, lifted with the current depth, then Kabsch.
track_rigid() {  # MASK_NAME POINTS_OUT POSES_OUT [SMOOTHER]
    conda activate "$ENV_SAM3"
    cd "$SCRIPTS_DIR"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python track_points_cotracker.py \
        --frames-dir "$FRAMES_DIR" --depth-dir "$DEPTH_DIR" --masks-root "$MASKS_ROOT" \
        --object "$1" --ref-frame "$n" --intrinsics "$INTRINSICS_YAML" \
        --out "$2" --overlay "${2%.npz}.mp4"
    conda activate "$ENV_GENESIS"
    python pose_from_points.py --points "$2" --smoother "${4:-window}" --smooth-window "$SMOOTH_WINDOW" --out "$3"
}

# ──────────────── Stage 4: camera motion ────────────────
if [ "${SKIP_CAMERA:-0}" != "1" ]; then
    echo "=== Camera motion from the static '$STATIC_PROMPT' (CoTracker + Kabsch) ==="
    track_rigid "$STATIC_ID" "$TABLE_POINTS_NPZ" "$CAMERA_TRAJ"
else
    echo "=== SKIP_CAMERA=1: reusing $CAMERA_TRAJ ==="
fi

# ──────────────── Stages 5-6: mocap hands in the camera, and the metric ────────────────
place_hands() {  # [--fixed-scale S]
    conda activate "$ENV_GENESIS"
    cd "$SCRIPTS_DIR"
    python touchanything_hands.py --episode-dir "$EPISODE_DIR" --camera-traj "$CAMERA_TRAJ" \
        --intrinsics "$INTRINSICS_YAML" --depth-dir "$DEPTH_DIR" --frames-dir "$FRAMES_DIR" \
        --overlay "$OUT_DIR/hands_overlay_cotracker.mp4" --transform-out "$MOCAP_FIT" \
        --out "$MOCAP_JOINTS" "$@"
}
if [ "${SKIP_HANDS:-0}" != "1" ]; then
    echo "=== Mocap hands (both) into the reference camera ==="
    place_hands
    SCALE="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['scale'])" "$MOCAP_FIT")"
    if python -c "import sys; sys.exit(0 if abs(float(sys.argv[1]) - 1) > float(sys.argv[2]) else 1)" "$SCALE" "$SCALE_TOLERANCE"; then
        echo "=== Depth is off the mocap's metric by s=$SCALE: rescaling depth + camera track ==="
        python scale_depth.py --depth-dir "$DEPTH_DIR" --scale "$SCALE" \
            --points "$TABLE_POINTS_NPZ" --poses "$CAMERA_TRAJ"
        place_hands --fixed-scale 1.0
    fi
else
    echo "=== SKIP_HANDS=1: reusing $MOCAP_JOINTS ==="
fi

# ──────────────── Stage 7: table plane ────────────────
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"
echo "=== Fitting the table plane ==="
python fit_ground_plane.py --depth "$DEPTH_DIR/$(printf "%06d.png" "$n")" --intrinsics "$INTRINSICS_YAML" \
    --exclude "$REF_MASKS/$OBJ_ID.png" --out "$GROUND_PLANE"

# ──────────────── Stage 8: object mesh (RecGen) ────────────────
if [ "${SKIP_MESH:-0}" != "1" ]; then
    conda activate "$ENV_RECGEN"
    cd "$SCRIPTS_DIR"
    echo "=== Object mesh (RecGen): $OBJECT ==="
    # RecGen erodes the mask by 2 px by default, which deletes any part a few pixels
    # wide -- a screwdriver's shaft at 640x480 -- so it is off unless asked for.
    EROSION_ARG=(--no-mask-erosion)
    [ "${RECGEN_EROSION:-0}" = "1" ] && EROSION_ARG=()
    python run_recgen.py "${EROSION_ARG[@]}" --rgb "$FRAME_PATH" --depth "$DEPTH_DIR/$(printf "%06d.png" "$n")" \
        --mask "$REF_MASKS/$OBJ_ID.png" --intrinsics "$INTRINSICS_YAML" --checkpoint "$RECGEN_CHECKPOINT" \
        --out "$OUT_DIR/recgen_out_moge/$OBJ_ID" --name "$OBJ_ID"
else
    echo "=== SKIP_MESH=1: reusing $MESH ==="
fi

# ──────────────── Stage 9: object trajectory ────────────────
if [ "${SKIP_OBJECT:-0}" != "1" ]; then
    echo "=== Object trajectory: CoTracker points lifted with depth + Kabsch ==="
    # Under the Kalman smoother the per-frame solves stay raw here: it runs after the
    # camera's motion is composed out, where constant velocity describes the object.
    OBJ_SOLVE=window
    [ "$OBJECT_SMOOTHER" = "kalman" ] && OBJ_SOLVE=none
    track_rigid "$OBJ_ID" "$OBJ_POINTS" "$OBJ_TRAJ_CAM" "$OBJ_SOLVE"
    echo "=== Carrying it into the reference camera ==="
    python compose_camera_motion.py --object-traj "$OBJ_TRAJ_CAM" --camera-traj "$CAMERA_TRAJ" \
        --intrinsics "$INTRINSICS_YAML" --out "$OBJ_TRAJ_COMPOSED"
else
    echo "=== SKIP_OBJECT=1: reusing $OBJ_TRAJ_COMPOSED ==="
fi
conda activate "$ENV_GENESIS"
cd "$SCRIPTS_DIR"
if [ "$OBJECT_SMOOTHER" = "kalman" ]; then
    echo "=== Kalman filter on the object trajectory (reference camera) ==="
    python kalman_object_traj.py --traj "$OBJ_TRAJ_COMPOSED" --mesh "$MESH" --out "$OBJ_TRAJ_SMOOTH"
else
    cp "$OBJ_TRAJ_COMPOSED" "$OBJ_TRAJ_SMOOTH"
fi
if [ -n "$HELD_WINDOW" ]; then
    echo "=== 9b: held object's depth from the left/right hand, frames $HELD_WINDOW ==="
    python snap_depth_to_hand.py --object-traj "$OBJ_TRAJ_SMOOTH" --camera-traj "$CAMERA_TRAJ" \
        --hand-meshes "$MOCAP_JOINTS" --side "${HELD_SIDE:-left}" --mesh "$MESH" \
        --window "$HELD_WINDOW" --out "$OBJ_TRAJ"
else
    cp "$OBJ_TRAJ_SMOOTH" "$OBJ_TRAJ"
fi

# ──────────────── Stage 10: MANO, both hands ────────────────
if [ "${SKIP_MANO:-0}" != "1" ]; then
    conda activate "$ENV_ECFIT"
    cd "$SCRIPTS_DIR"
    echo "=== Fitting MANO to the mocap joints (left + right) ==="
    python mano_from_mocap.py --joints "$MOCAP_JOINTS" --out "$MANO_OUT" --parts-dir "$MANO_PARTS"
else
    echo "=== SKIP_MANO=1: reusing $MANO_OUT ==="
fi

# ──────────────── Stage 11: Genesis replay ────────────────
if [ "${SKIP_REPLAY:-0}" != "1" ]; then
    conda activate "$ENV_GENESIS"
    cd "$SCRIPTS_DIR"
    echo "=== Replaying both hands + the object in Genesis ==="
    python replay_hands_genesis.py --mano "$MANO_OUT" --parts-dir "$MANO_PARTS" \
        --object-mesh "$MESH" --object-poses "$OBJ_TRAJ" --camera-traj "$CAMERA_TRAJ" \
        --ground-plane "$GROUND_PLANE" --intrinsics "$INTRINSICS_YAML" --frames-dir "$FRAMES_DIR" \
        --hand-render "${HAND_RENDER:-skinned}" --record "$REPLAY"
fi

echo
echo "=== Done ==="
echo "  camera traj:   $CAMERA_TRAJ"
echo "  mocap joints:  $MOCAP_JOINTS   (both hands, reference camera)"
echo "  object mesh:   $MESH"
echo "  object traj:   $OBJ_TRAJ   (CoTracker + Kabsch, reference camera)"
echo "  MANO hands:    $MANO_OUT   (+ parts in $MANO_PARTS)"
[ -f "$REPLAY" ] && echo "  replay:        $REPLAY"
exit 0
