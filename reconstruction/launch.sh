#!/usr/bin/env bash
# Re-run the CoTracker pipeline on one take, skipping the stages set to 1.
# Any flag can be overridden at call time, e.g.  SKIP_HAWOR=0 ./launch.sh
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Exported so the pipeline, which runs as a child process, sees them.
export SKIP_EXTRACT_AND_SAM3="${SKIP_EXTRACT_AND_SAM3:-1}"
export SKIP_DEPTH="${SKIP_DEPTH:-1}"
export SKIP_MESH="${SKIP_MESH:-1}"
export SKIP_HAWOR="${SKIP_HAWOR:-1}"
export SKIP_RETARGETING="${SKIP_RETARGETING:-1}"
export SKIP_TRAJOPT="${SKIP_TRAJOPT:-1}"
export SKIP_OVERLAY="${SKIP_OVERLAY:-1}"
export SKIP_WUJI="${SKIP_WUJI:-0}"
export SKIP_COTRACKER="${SKIP_COTRACKER:-0}"
export SKIP_KEYFRAME_GRASPS="${SKIP_KEYFRAME_GRASPS:-1}"
export SKIP_POINT_MAP="${SKIP_POINT_MAP:-0}"
export SKIP_REFINE="${SKIP_REFINE:-1}"
"$HERE/run_reconstruction_pipeline_cotracker.sh" \
    "$HERE/human_videos/grasping_screwdriver/take_002/take002_grasping_screwdriver_rgb.mp4" 50 screwdriver left
