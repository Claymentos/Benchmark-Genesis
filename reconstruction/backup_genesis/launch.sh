#!/usr/bin/env bash
# Re-run the FoundationPose pipeline on one take, skipping the stages set to 1.
# Any flag can be overridden at call time, e.g.  SKIP_TRACK=0 ./launch.sh
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Exported so the pipeline, which runs as a child process, sees them.
export SKIP_EXTRACT_AND_SAM3="${SKIP_EXTRACT_AND_SAM3:-1}"
export SKIP_DEPTH="${SKIP_DEPTH:-1}"
export SKIP_MESH="${SKIP_MESH:-1}"
export SKIP_HAWOR="${SKIP_HAWOR:-1}"
export SKIP_RETARGETING="${SKIP_RETARGETING:-1}"

export SKIP_TRACK="${SKIP_TRACK:-0}"
export SKIP_OVERLAY="${SKIP_OVERLAY:-1}"
export SKIP_WUJI="${SKIP_WUJI:-0}"
export SKIP_FRANKA="${SKIP_FRANKA:-1}"
export SKIP_COTRACKER="${SKIP_COTRACKER:-0}"

export SKIP_POINT_MAP="${SKIP_POINT_MAP:-0}"

"$HERE/run_reconstruction_pipeline_foundationpose.sh" \
    "$HERE/human_videos/grasping_cup/take_006/take006_grasping_cup_rgb.mp4" 40 cup left
