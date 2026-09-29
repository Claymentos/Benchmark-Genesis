#!/usr/bin/env python3
"""Stage 1 -- Semantic parsing (paper: "Digital Twin Reconstruction from Video").

Sends three key RGB frames from the manipulation video ({I1, I_T/2, I_T}) plus the
reference image (Is) to Gemini, and asks it to infer a structured scene description:
every tabletop object, the manipulated object, the task type, the target object, and
a material label per object (consumed by Stage 4's physical-property estimation).

Run under an env with `google-genai` installed (this repo's `sam3d` conda env has it).
Reads GEMINI_API_KEY from reconstruction/.env (or the environment).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

SCHEMA = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string", "description": "short phrase describing the manipulation task, e.g. 'pouring liquid from a cup'"},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "short id, lowercase, underscores, e.g. 'cup'"},
                    "manipulated_part": {"type": "string", "description" :"short id, lowercase, underscores, e.g. 'handle'"},
                    "description": {"type": "string", "description": "visually grounded description usable as a text prompt for a promptable segmentation model, e.g. 'white ceramic mug with a handle'"},
                    "material": {"type": "string", "description": "one of: ceramic, porcelain, glass, plastic, metal, aluminum, steel, wood, paper, rubber, fabric"},
                    "is_manipulated": {"type": "boolean", "description": "true if this is the object the hand directly grasps/manipulates"},
                    "is_target": {"type": "boolean", "description": "true if this is the destination/target of the manipulation (e.g. a bowl being poured into)"},
                },
                "required": ["name", "description", "material", "is_manipulated", "is_target"],
            },
        },
    },
    "required": ["task_type", "objects"],
}

PROMPT = """You are given four images from a single human manipulation demonstration video:
1) the first frame, 2) the middle frame, 3) the last frame, and 4) a clean reference
frame of the scene. Identify every distinct tabletop/manipulable object visible, the
task being demonstrated, which object is being manipulated by the hand, which part of the object is manipulated by the human hand, and which
object (if any) is the target/destination of the manipulation. For each object give a
short text description suitable as a prompt for a promptable image-segmentation model,
and a single best-guess material label. Respond ONLY with JSON matching the schema."""


def load_image_part(path: str) -> types.Part:
    with open(path, "rb") as f:
        data = f.read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return types.Part.from_bytes(data=data, mime_type=mime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="directory of all_frames/%06d.png")
    ap.add_argument("--reference-frame", required=True, help="path to the reference frame image")
    ap.add_argument("--out", required=True, help="output scene.json path")
    ap.add_argument("--model", default="gemini-flash-lite-latest",
                     help="gemini-flash-latest has been persistently 503-overloaded in testing; "
                          "gemini-flash-lite-latest and gemini-3.6-flash both work reliably")
    args = ap.parse_args()

    frame_files = sorted(f for f in os.listdir(args.frames_dir) if f.endswith(".png"))
    if not frame_files:
        print(f"no frames found in {args.frames_dir}", file=sys.stderr)
        sys.exit(1)
    first = os.path.join(args.frames_dir, frame_files[0])
    middle = os.path.join(args.frames_dir, frame_files[len(frame_files) // 2])
    last = os.path.join(args.frames_dir, frame_files[-1])

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    parts = [
        load_image_part(first),
        load_image_part(middle),
        load_image_part(last),
        load_image_part(args.reference_frame),
        PROMPT,
    ]
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SCHEMA,
    )
    last_err = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(model=args.model, contents=parts, config=config)
            break
        except genai_errors.ServerError as e:
            last_err = e
            print(f"attempt {attempt + 1}/5 failed ({e}); retrying...", file=sys.stderr)
            time.sleep(2 ** attempt)
    else:
        raise last_err
    scene = json.loads(resp.text)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"wrote {args.out}")
    print(json.dumps(scene, indent=2))


if __name__ == "__main__":
    main()
