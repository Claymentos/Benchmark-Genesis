#!/usr/bin/env python3
"""
Cut an EPIC-Contact clip out of its raw EPIC-Kitchens video, ready for the pipeline.

EPIC-Contact ships its labels without pixels (source_rgb/ is an OneDrive-only
download), but every clip name encodes its EPIC-Kitchens video and frame span --
P01_01_42298_42980_left_bottle is frames 42298..42980 of P01_01 -- so the clip can be
cut from the raw video instead.

EPIC frame numbers are 1-based indices of a 60 fps extraction: frame N sits at
(N - 1) / 60 s. The P01 recordings are 59.94 fps containers holding 30 fps content
(every frame appears twice), so the clip is re-timed to 30 fps, which drops only the
duplicates. That was checked against the shipped central frame, which matches the
cut at mean absolute difference 7.8/255, against 12+ one frame either side.

Writes, into --out-dir:
    <clip>.mp4            the clip at --fps, resized to --width
    epic_frames.json      local frame index -> EPIC frame number, plus the central
                          (true-GT) frame's local index, the annotated side and class

Usage:
    python prepare_epic_contact_clip.py --clip P01_01_42298_42980_left_bottle \\
        --epic-root ~/EPIC-KITCHENS --out-dir human_videos/epic_contact/<clip>
"""

import argparse
import glob
import json
import os
import subprocess

import numpy as np


def decode_rgb(args, width, height):
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", *args, "-s", f"{width}x{height}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3).astype(np.float32)


def match_central(video, t0, expected, central_jpg, radius=6):
    """Decoded frame index (from the seek at t0) that best matches the central image."""
    size = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         central_jpg],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    width, height = int(size[0]), int(size[1])
    target = decode_rgb(["-i", central_jpg], width, height)[0]
    lo = max(expected - radius, 0)
    frames = decode_rgb(
        ["-ss", f"{t0:.6f}", "-i", video, "-vf", f"select='between(n\\,{lo}\\,{expected + radius})'",
         "-fps_mode", "passthrough"],
        width, height,
    )
    errors = [(lo + i, float(np.abs(frame - target).mean())) for i, frame in enumerate(frames)]
    return min(errors, key=lambda e: e[1])[0], errors


def parse_args():
    parser = argparse.ArgumentParser(description="Cut an EPIC-Contact clip from raw EPIC-Kitchens")
    parser.add_argument("--clip", required=True, help="EPIC-Contact clip id")
    parser.add_argument("--epic-root", default=os.path.expanduser("~/EPIC-KITCHENS"))
    parser.add_argument(
        "--contact-root",
        default=os.path.join(os.path.dirname(__file__), "..", "epic-contact"),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280, help="Output width; height keeps aspect")
    parser.add_argument(
        "--align-central", action="store_true",
        help="Pick the sampling phase so the shipped central frame is one of the output frames",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parts = args.clip.split("_")
    video_id = "_".join(parts[:2])
    start, end = int(parts[2]), int(parts[3])
    side, obj_class = parts[4], "_".join(parts[5:])

    video = os.path.join(args.epic_root, parts[0], "videos", f"{video_id}.MP4")
    if not os.path.exists(video):
        raise SystemExit(f"no raw video at {video}")

    central = glob.glob(os.path.join(args.contact_root, "central_frames", args.clip, "frame_*.jpg"))
    if not central:
        raise SystemExit(f"no central frame for {args.clip}")
    keyframe = int(os.path.basename(central[0])[len("frame_"):-len(".jpg")])

    step = 60.0 / args.fps
    t0 = (start - 1) / 60.0
    resample = f"fps={args.fps}"
    first_epic = start
    if args.align_central:
        # The fps filter's sampling phase can skip the central frame, and the seek can
        # land a frame off (N - 1) / 60, so the phase is set by matching the decoded
        # frames around the central one against the shipped central image.
        if step != int(step):
            raise SystemExit("--align-central needs --fps dividing 60")
        step = int(step)
        best, errors = match_central(video, t0, keyframe - start, central[0])
        phase = best % step
        first_epic = keyframe - best + phase
        resample = f"select='not(mod(n-{phase}\\,{step}))',setpts=N/{args.fps}/TB"
        print("central-frame match (decoded index: mean abs diff): "
              + ", ".join(f"{n}: {e:.1f}" for n, e in errors) + f" -> {best}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_video = os.path.join(args.out_dir, f"{args.clip}.mp4")
    duration = (end - start + 1) / 60.0
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-ss", f"{t0:.6f}", "-i", video, "-t", f"{duration:.6f}",
            "-vf", f"{resample},scale={args.width}:-2",
            "-r", f"{args.fps}", "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
            out_video,
        ],
        check=True,
    )

    count = int(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", out_video],
        check=True, capture_output=True, text=True,
    ).stdout.strip())

    epic_frames = [int(round(first_epic + i * step)) for i in range(count)]
    keyframe_local = min(range(count), key=lambda i: abs(epic_frames[i] - keyframe))
    meta = {
        "clip": args.clip,
        "video_id": video_id,
        "side": side,
        "object_class": obj_class,
        "fps": args.fps,
        "epic_frames": epic_frames,
        "keyframe_epic": keyframe,
        "keyframe_local": keyframe_local,
    }
    with open(os.path.join(args.out_dir, "epic_frames.json"), "w") as handle:
        json.dump(meta, handle, indent=1)
    print(f"Wrote {out_video}: {count} frames; central frame {keyframe} -> local {keyframe_local}")
    print(f"KEYFRAME_LOCAL={keyframe_local}")


if __name__ == "__main__":
    main()
