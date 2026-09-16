"""
Compute MoGe pointmaps (+ camera intrinsics) for one image or a whole directory.

Single image (e.g. the reference frame) -- saves <base>_pointmap.npy and
<base>_intrinsics.txt (fx, fy, cx, cy, one value per line):
  python get_pointmap_dir.py --image /path/to/0028.png --output /path/to/0028_pointmap.npy

Batch (directory of frames) -- saves <base>_pointmap.npy and <base>_intrinsics.npy (3x3) per image:
  python get_pointmap_dir.py --image_dir /path/to/all_frames
"""

import sys
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate
import argparse
import os
import glob

_RECON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3D_REPO_ROOT = os.environ.get(
    "SAM3D_REPO_ROOT", os.path.join(_RECON_ROOT, "modules", "Fast-SAM3D")
)
if SAM3D_REPO_ROOT not in sys.path:
    sys.path.insert(0, SAM3D_REPO_ROOT)


def load_model():
    """Load the MoGe depth model once."""
    config_path = f"{SAM3D_REPO_ROOT}/checkpoints/hf/pipeline.yaml"
    cfg = OmegaConf.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_model = instantiate(cfg.depth_model, device=device)
    return depth_model


def run_pointmap(depth_model, image_path: str, output_path: str, intrinsics_format: str = "npy"):
    """Compute MoGe pointmap for a single image and save pointmap + intrinsics.

    intrinsics_format:
      "npy" -> denormalized 3x3 matrix at <base>_intrinsics.npy   (batch / all frames)
      "txt" -> fx, fy, cx, cy (one value per line) at <base>_intrinsics.txt
               (reference-frame format consumed by HaWoR --img_focal)
    """
    img = np.array(Image.open(image_path)).astype(np.uint8)

    # Add alpha channel
    alpha = np.full((*img.shape[:2], 1), 255, dtype=img.dtype)
    img = np.concatenate([img, alpha], axis=-1)

    # Compute pointmap
    loaded_image = (img / 255.0).astype(np.float32)
    loaded_image = torch.from_numpy(loaded_image)
    loaded_rgb = loaded_image.permute(2, 0, 1).contiguous()[:3]

    dtype = torch.float16
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = depth_model(loaded_rgb)

    pointmap_np = output["pointmaps"].detach().cpu().numpy()
    np.save(output_path, pointmap_np)
    print(f"Saved pointmap to: {output_path}")

    # Denormalize intrinsics to image size, then save in the requested format
    intrinsics = output["intrinsics"].detach().cpu().numpy().copy()
    H, W = img.shape[:2]
    if intrinsics_format == "txt":
        # Reference-frame format (consumed by HaWoR --img_focal)
        fx = intrinsics[0, 0] * W
        fy = intrinsics[1, 1] * H
        cx = intrinsics[0, 2] * W
        cy = intrinsics[1, 2] * H
        intrinsics_txt_path = output_path.replace('_pointmap.npy', '_intrinsics.txt')
        with open(intrinsics_txt_path, "w") as f:
            f.write(f"{fx}\n{fy}\n{cx}\n{cy}\n")
        print(f"Saved intrinsics (fx, fy, cx, cy; scaled to image size) to: {intrinsics_txt_path}")
    else:
        intrinsics[0, 0] *= W
        intrinsics[1, 1] *= H
        intrinsics[0, 2] *= W
        intrinsics[1, 2] *= H
        intrinsics_path = output_path.replace('_pointmap.npy', '_intrinsics.npy')
        np.save(intrinsics_path, intrinsics)
        print(f"Saved intrinsics (3x3 matrix) to: {intrinsics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Single image (e.g. reference frame); saves <base>_intrinsics.txt")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of frames for batch; saves <base>_intrinsics.npy per frame")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.image_dir is not None:
        # Batch mode: process all PNGs in the directory
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
        if not image_paths:
            print(f"No .png files found in {args.image_dir}")
            sys.exit(1)

        print(f"Found {len(image_paths)} images in {args.image_dir}")
        depth_model = load_model()

        for i, image_path in enumerate(image_paths):
            base, _ = os.path.splitext(image_path)
            output_path = f"{base}_pointmap.npy"
            print(f"\n[{i+1}/{len(image_paths)}] Processing {os.path.basename(image_path)}")
            run_pointmap(depth_model, image_path, output_path, intrinsics_format="npy")

        print(f"\nDone! Processed {len(image_paths)} images.")

    elif args.image is not None:
        # Single image mode
        if args.output is None:
            base, _ = os.path.splitext(args.image)
            args.output = f"{base}_pointmap.npy"

        depth_model = load_model()
        run_pointmap(depth_model, args.image, args.output, intrinsics_format="txt")

    else:
        parser.error("Must specify either --image or --image_dir")
