#!/usr/bin/env python3
"""
Estimate metric depth for every frame of a video with MoGe.

Batch equivalent of recgen's scripts/run_moge_depth.py, which loads the model per
invocation -- prohibitive across a few hundred frames. Writes uint16-millimetre PNGs
in RecGen's format, one per input frame.

Run in the `moge` env:
    python run_moge_depth_batch.py --frames-dir all_frames --out-dir depth --fov-x 100.71
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Batch MoGe metric depth")
    parser.add_argument("--frames-dir", required=True, help="Directory of RGB frames")
    parser.add_argument("--out-dir", required=True, help="Directory for the depth PNGs")
    parser.add_argument(
        "--fov-x",
        type=float,
        default=None,
        help="Known horizontal FOV in degrees; locks MoGe's focal length to the real camera",
    )
    parser.add_argument("--checkpoint", default="Ruicheng/moge-3-vitl")
    parser.add_argument("--version", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moge-repo", default=None, help="Local MoGe checkout for sys.path")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.moge_repo:
        sys.path.insert(0, args.moge_repo)

    from moge.model import import_model_class_by_version

    device = torch.device(args.device)
    model = import_model_class_by_version(args.version).from_pretrained(args.checkpoint)
    model = model.to(device).eval()

    frames = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith((".png", ".jpg")))
    os.makedirs(args.out_dir, exist_ok=True)

    infer_kwargs = {"fov_x": args.fov_x}
    if args.version == "v3":
        infer_kwargs["refine_steps"] = args.refine_steps

    for i, name in enumerate(frames):
        image = cv2.cvtColor(cv2.imread(os.path.join(args.frames_dir, name)), cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            output = model.infer(tensor, **infer_kwargs)

        depth = output["depth"].cpu().numpy()
        finite = np.isfinite(depth) & output["mask"].cpu().numpy()
        depth_mm = np.clip(np.where(finite, depth * 1000.0, 0.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out_dir, os.path.splitext(name)[0] + ".png"), depth_mm)

        if i % 25 == 0 or i == len(frames) - 1:
            print(f"[moge] {i + 1}/{len(frames)} {name}", flush=True)

    print(f"[moge] wrote {len(frames)} depth maps to {args.out_dir}")


if __name__ == "__main__":
    main()
