#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Run SAM3 video segmentation with a text, point, or interactive click prompt.

Output structure (in <video_dir>/video_segmentation/ by default):
    masks/frame_000000_masks/<obj_id>.png  - binary masks per frame per object
    overlays/               - overlay visualizations as frame_000000.png, ...
    tracked_<prompt>.mp4    - overlay video

Usage examples:
    # Text prompt (output: video_segmentation/tracked_person.mp4)
    python run_sam3_video.py --video /path/to/video.mp4 --text "left hand" --obj_id left_hand_0

    # Text prompt with custom output directory
    python run_sam3_video.py --video video.mp4 --text "dog" --output_dir output/

    # Point prompt (x,y in pixels, label 1=positive 0=negative)
    python run_sam3_video.py --video video.mp4 --points 210,350 --point_labels 1 --obj_id 1

    # Interactive click mode - opens a window to click points on a frame
    python run_sam3_video.py --video /path/to/video.mp4 --click --obj_id scooper_0 --frame_idx 6
"""

import argparse
import os
import subprocess

import cv2
import numpy as np

import sys
sys.path.append(
    os.environ.get(
        "SAM3_PKG_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules", "sam3"),
    )
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 video segmentation")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video (MP4 file or directory of JPEG frames)",
    )
    parser.add_argument("--text", type=str, default=None, help="Text prompt")
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="Point prompts as 'x1,y1;x2,y2;...' in pixel coords",
    )
    parser.add_argument(
        "--point_labels",
        type=str,
        default=None,
        help="Point labels as '1;0;1;...' (1=positive, 0=negative)",
    )
    parser.add_argument(
        "--obj_id",
        type=str,
        required=True,
        help="Object ID used for naming saved masks (e.g. 'flower', 'hand')",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index to add prompt on (default: 0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output masks (default: same as video path)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-download from HF)",
    )
    parser.add_argument(
        "--click",
        action="store_true",
        help="Interactive mode: open a window to click points on a frame. "
        "Left-click = positive, right-click = negative. Press Enter/Space to confirm, Esc to cancel.",
    )
    return parser.parse_args()


def load_video_frames(video_path):
    """Load video frames for visualization."""
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    else:
        IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        frame_paths = [
            os.path.join(video_path, p)
            for p in os.listdir(video_path)
            if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
        ]
        try:
            frame_paths.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            frame_paths.sort()
        return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frame_paths]


def get_highest_score_obj(outputs):
    """From a frame output dict, return the obj_id and mask with the highest score."""
    obj_ids = outputs["out_obj_ids"]
    probs = outputs["out_probs"]
    masks = outputs["out_binary_masks"]

    if len(obj_ids) == 0:
        return None, None, None

    best_idx = np.argmax(probs)
    return obj_ids[best_idx], probs[best_idx], masks[best_idx]


def interactive_click_prompt(frame, obj_id=""):
    """
    Open a window showing `frame` (RGB array). The user can:
      - Left-click  to add a positive point (green dot)
      - Right-click to add a negative point (red dot)
      - Press Enter or Space to confirm
      - Press 'u' to undo the last point
      - Press Esc to abort

    Returns (points, labels) as numpy arrays, or (None, None) if cancelled.
    """
    points = []
    labels = []
    window_name = f"SAM3 [{obj_id}] - Click to add points (Enter=confirm, U=undo, Esc=cancel)"

    def _redraw():
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
        for (x, y), lbl in zip(points, labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 6, color, -1)
            cv2.circle(img, (int(x), int(y)), 6, (255, 255, 255), 1)
        return img

    def _on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y))
            labels.append(0)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame.shape[1], 1280), min(frame.shape[0], 720))
    cv2.setMouseCallback(window_name, _on_mouse)

    print("Left-click = positive point (green), Right-click = negative point (red)")
    print("Press Enter/Space to confirm, 'u' to undo, Esc to cancel")

    while True:
        cv2.imshow(window_name, _redraw())
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):  # Enter or Space
            break
        elif key == 27:  # Esc
            points, labels = None, None
            break
        elif key == ord("u") and points:  # Undo
            points.pop()
            labels.pop()

    cv2.destroyWindow(window_name)

    if points is None or len(points) == 0:
        return None, None

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def overlay_mask_on_frame(frame, mask, color=(30, 144, 255), alpha=0.5):
    """Overlay a binary mask on a frame with the given color and alpha."""
    overlay = frame.copy()
    overlay[mask] = (
        (1 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.uint8)
    ).astype(np.uint8)
    # Draw contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def main():
    args = parse_args()

    assert args.text is not None or args.points is not None or args.click, (
        "Must provide either --text, --points, or --click prompt"
    )

    if args.output_dir is None:
        if args.video.endswith(".mp4"):
            video_dir = os.path.dirname(args.video) or "."
        else:
            video_dir = args.video
        args.output_dir = os.path.join(video_dir, "video_segmentation")

    # Determine prompt name for the output video filename
    if args.text is not None:
        prompt_name = args.text.replace(" ", "_")
    elif args.click:
        prompt_name = "click"
    else:
        prompt_name = "points"

    masks_base_dir = os.path.join(args.output_dir, "masks")
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(masks_base_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Interactive click mode: collect clicks BEFORE loading the model, because
    # CUDA/PyTorch initialization can corrupt the Qt libraries that cv2 uses
    # for its GUI, causing a segfault.
    click_points = None
    click_labels = None
    if args.click:
        video_frames = load_video_frames(args.video)
        frame = video_frames[args.frame_idx]

        print(f"Opening frame {args.frame_idx} for interactive clicking...")
        click_points, click_labels = interactive_click_prompt(frame, obj_id=args.obj_id)
        if click_points is None:
            print("Cancelled. Exiting.")
            return

    # Import torch and SAM3 after any cv2 GUI work, because sam3's imports
    # load libraries that conflict with cv2's Qt backend and cause segfaults.
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    # Build predictor
    print("Loading SAM3 model...")
    predictor = build_sam3_video_predictor()

    # Start session
    print(f"Starting session on: {args.video}")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=args.video,
        )
    )
    session_id = response["session_id"]

    # SAM3 requires an integer obj_id; use 1 internally, keep string for naming
    sam3_obj_id = 1

    # Add prompt
    if args.click:
        h, w = video_frames[0].shape[:2]

        rel_points = [[x / w, y / h] for x, y in click_points.tolist()]
        points_tensor = torch.tensor(rel_points, dtype=torch.float32)
        labels_tensor = torch.tensor(click_labels.tolist(), dtype=torch.int32)

        print(
            f"Adding {len(click_points)} clicked point(s) on frame {args.frame_idx}: "
            f"points={click_points.tolist()}, labels={click_labels.tolist()}"
        )
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=args.frame_idx,
                points=points_tensor,
                point_labels=labels_tensor,
                obj_id=sam3_obj_id,
            )
        )
    elif args.text is not None:
        print(f"Adding text prompt: '{args.text}' on frame {args.frame_idx}")
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=args.frame_idx,
                text=args.text,
                obj_id=sam3_obj_id,
            )
        )
    else:
        # Parse point prompts
        points_list = [
            [float(c) for c in pt.split(",")] for pt in args.points.split(";")
        ]
        labels_list = [int(l) for l in args.point_labels.split(";")]
        assert len(points_list) == len(labels_list), (
            "Number of points must match number of labels"
        )

        # Load a frame to get dimensions for coordinate normalization
        video_frames = load_video_frames(args.video)
        h, w = video_frames[0].shape[:2]
        rel_points = [[x / w, y / h] for x, y in points_list]

        points_tensor = torch.tensor(rel_points, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.int32)

        print(
            f"Adding point prompt on frame {args.frame_idx}: "
            f"points={points_list}, labels={labels_list}"
        )
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=args.frame_idx,
                points=points_tensor,
                point_labels=labels_tensor,
                obj_id=sam3_obj_id,
            )
        )

    # Show prompt frame results
    prompt_out = response["outputs"]
    best_obj_id, best_score, best_mask = get_highest_score_obj(prompt_out)
    if best_obj_id is not None:
        print(
            f"Prompt frame {args.frame_idx}: best obj_id={best_obj_id}, score={best_score:.4f}"
        )
    else:
        print("No objects detected on the prompt frame.")

    # Propagate through video
    print("Propagating through video...")
    outputs_per_frame = {}
    for resp in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[resp["frame_index"]] = resp["outputs"]

    # Load video frames for overlays and video
    if "video_frames" not in locals():
        video_frames = load_video_frames(args.video)

    # Save masks, overlays, and tracked video
    sorted_frames = sorted(outputs_per_frame.keys())
    h, w = video_frames[0].shape[:2]
    out_video_path = os.path.join(args.output_dir, f"tracked_{args.obj_id}_{prompt_name}.mp4")

    print(f"Saving {len(sorted_frames)} frames to {args.output_dir}/...")
    for frame_idx in sorted_frames:
        out = outputs_per_frame[frame_idx]
        _, _, best_mask = get_highest_score_obj(out)

        # Save mask into per-frame directory as <obj_id>.png
        frame_mask_dir = os.path.join(masks_base_dir, f"frame_{frame_idx:06d}_masks")
        os.makedirs(frame_mask_dir, exist_ok=True)
        if best_mask is not None:
            mask_uint8 = (best_mask * 255).astype(np.uint8)
        else:
            mask_uint8 = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(
            os.path.join(frame_mask_dir, f"{args.obj_id}.png"), mask_uint8
        )

        # Save overlay
        if frame_idx < len(video_frames):
            frame = video_frames[frame_idx]
            if best_mask is not None:
                overlay = overlay_mask_on_frame(frame, best_mask)
            else:
                overlay = frame
            cv2.imwrite(
                os.path.join(overlay_dir, f"frame_{frame_idx:06d}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

    # Encode overlay frames to H264 video using ffmpeg
    print(f"Encoding tracked video with H264: {out_video_path}")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", "24",
            "-i", os.path.join(overlay_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            out_video_path,
        ],
        check=True,
    )

    # Cleanup
    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )
    predictor.shutdown()

    print(f"Done! Output saved to: {args.output_dir}/")
    print(f"  masks/frame_XXXXXX_masks/{args.obj_id}.png")
    print(f"  overlays/    - frame_000000.png, ...")
    print(f"  tracked_{args.obj_id}_{prompt_name}.mp4")


if __name__ == "__main__":
    main()
