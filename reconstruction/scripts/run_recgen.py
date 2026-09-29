#!/usr/bin/env python3
"""
Run RecGen's own `scripts/run_inference.py`, with its mask erosion made optional.

RecGen erodes the object mask (5x5, one iteration) before reconstructing, to drop
pixels that bleed onto the background at the object's edge. On a thin part that
removes the part: a TouchAnything screwdriver's shaft is 3-4 px wide at 640x480, the
erosion keeps 44% of its 775-pixel mask and none of the shaft, and RecGen returned an
8 cm handle for a ~15 cm screwdriver. `run_inference.py` exposes no flag for it, so
this launcher patches `recgen_inference.generate` and then runs the script unchanged,
passing every other argument straight through.

Usage (in the `recgen` env, any directory):
    python run_recgen.py --no-mask-erosion --rgb ... --depth ... --mask ... \\
        --intrinsics ... --checkpoint ... --out ... --name ...
"""

import os
import runpy
import sys

RECGEN_DIR = os.environ.get("RECGEN_DIR", os.path.expanduser("~/recgen"))


def main():
    argv = sys.argv[1:]
    erosion = "--no-mask-erosion" not in argv
    argv = [a for a in argv if a != "--no-mask-erosion"]

    sys.path.insert(0, RECGEN_DIR)
    import recgen_inference

    if not erosion:
        original = recgen_inference.generate

        def generate(*args, **kwargs):
            kwargs["mask_erosion_enabled"] = False
            return original(*args, **kwargs)

        # run_inference.py does `from recgen_inference import generate` when it runs,
        # so replacing the package attribute first is enough.
        recgen_inference.generate = generate
        print("[run_recgen] mask erosion disabled")

    os.chdir(RECGEN_DIR)
    sys.argv = ["scripts/run_inference.py", *argv]
    runpy.run_path(os.path.join(RECGEN_DIR, "scripts", "run_inference.py"), run_name="__main__")


if __name__ == "__main__":
    main()
