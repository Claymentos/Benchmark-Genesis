#!/usr/bin/env python3
"""
Run HaWoR hand reconstruction on a video, producing `all_hand_meshes.npz`.

Thin launcher around the vendored HaWoR `demo.py`. It exists because HaWoR's
checkpoint is loaded through Lightning, which calls `torch.load` without
`weights_only=False`; under torch >= 2.6 that default flipped and the load
fails on the checkpoint's embedded omegaconf config.

Usage:
    python run_hawor.py --video-path VIDEO --img-focal FX [--hawor-dir DIR]
"""

import argparse
import os
import runpy
import sys

import torch

DEFAULT_HAWOR_DIR = os.environ.get(
    "HAWOR_DIR",
    os.path.expanduser("~/do-as-i-do/reconstruction/modules/HaWoR"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run HaWoR hand reconstruction")
    parser.add_argument("--video-path", required=True, help="Path to the input video")
    parser.add_argument(
        "--img-focal",
        type=float,
        required=True,
        help="Focal length in pixels (from the camera intrinsics)",
    )
    parser.add_argument(
        "--hawor-dir",
        default=DEFAULT_HAWOR_DIR,
        help="HaWoR checkout containing demo.py and weights/",
    )
    parser.add_argument(
        "--moving-camera",
        action="store_true",
        help="Run SLAM camera tracking instead of assuming a static camera",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = os.path.abspath(args.video_path)
    hawor_dir = os.path.abspath(args.hawor_dir)

    _torch_load = torch.load
    torch.load = lambda *a, **kw: _torch_load(*a, **{**kw, "weights_only": False})

    demo_argv = [
        "demo.py",
        "--video_path", video_path,
        "--vis_mode", "cam",
        "--img_focal", str(args.img_focal),
    ]
    if not args.moving_camera:
        demo_argv.append("--static_camera")

    os.chdir(hawor_dir)
    sys.path.insert(0, hawor_dir)

    if not args.moving_camera:
        # Motion estimation renders MANO meshes only to build the hand masks that
        # DROID-SLAM uses to ignore the hand while tracking the camera. A static
        # camera skips SLAM, so those masks are never read -- stubbing the render
        # avoids requiring a CUDA-enabled pytorch3d build.
        import numpy as np
        from lib.vis.renderer import Renderer

        Renderer.render_multiple = lambda self, *a, **kw: (
            np.zeros((self.height, self.width, 3), dtype=np.uint8),
            np.zeros((self.height, self.width), dtype=bool),
        )

    sys.argv = demo_argv
    runpy.run_path(os.path.join(hawor_dir, "demo.py"), run_name="__main__")


if __name__ == "__main__":
    main()
