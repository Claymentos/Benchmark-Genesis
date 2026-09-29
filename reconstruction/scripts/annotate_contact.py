#!/usr/bin/env python3
"""
Annotate hand-object contact on your own video, following EPIC-Contact's protocol.

EPIC-Contact (Bansal et al., ECCV 2026, Sec. 3.1-3.2, Sup. Mat. 8.3-8.4) labels one frame
of a stable-grasp clip in two interfaces, both showing the video clip beside a 3D view:

  1. hand contact -- the annotator paints the contact on the canonical 3106-vertex MANO
     template (not on the image, and not on a posed hand), inferring the parts the object
     hides from the video and the grasp motion. The painted vertices fall into three
     regions (fingers / thumb / palm) by EPIC's part segmentation;
  2. contact transfer -- for each region, the patch is shown on a flat hand whose finger
     spread matches the video, with its contact axis (blue ball = start, red line =
     direction). The annotator clicks twice on the object mesh: where the axis starts and
     which way it runs. The patch is carried over by ContactEdit's log/exp-map transfer
     (contact_transfer.py), giving one object point per painted hand vertex.

The posed meshes then come from EC-fit (run_ecfit.py), EPIC's own fitting code.

Differences from EPIC-Contact, on purpose:
  * the object is the pipeline's RecGen mesh (instance-specific, metric from ZED or MoGe
    depth) instead of a category template scaled by a VLM; it is remeshed to a closed
    manifold (geodesics need one) -- annotation_mesh.obj, which the contact indexes into;
  * the flat hand's finger spread is taken directly from HaWoR's pose at the frame instead
    of the closest of EPIC's 41 flat poses (that pool is not released).

Outputs in <out-dir>:
    annotation.json            the painting and the clicks (reloaded on the next run)
    annotation_mesh.obj        the object mesh the contact faces index into
    hand_fit.npz               HaWoR's hand at the frame as MANO parameters in EC-fit's
                               convention (run_ecfit.py reads it)
    central_frame/contact_vertices_bodyparts.json   dense hand contact, per region
    central_frame/contact_vertices_sparse.json      its 778-vertex image
    central_frame/corresponding_contacts.json       the bijective hand<->object pairs

Run in the `ecfit` env, after run_v2s2r_pipeline.sh (paths default to its layout):
    python annotate_contact.py --video-dir ../human_videos/grasping_cup/take_006v3 \\
        --object cup --side left --frame 145 --contact-frames 114 233
then python run_ecfit.py with the same arguments.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import trimesh
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_transfer import (REGIONS, DenseHand, PatchParam, Surface, fit_mano_to_vertices,  # noqa: E402
                              flat_hand, manifold_remesh, mano_faces, transfer)

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(HERE, "annotate_contact_ui.html")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EPIC-Contact-style contact annotation on a video")
    add_clip_args(parser)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--export-only", action="store_true",
                        help="skip the UI: redo the transfer from the saved annotation.json")
    parser.add_argument("--remesh-pitch", type=float, default=0.0015,
                        help="voxel size (m) of the manifold object remesh")
    return parser.parse_args(argv)


def add_clip_args(parser):
    """Arguments shared with run_ecfit.py."""
    parser.add_argument("--video-dir", required=True, help="pipeline folder of the clip")
    parser.add_argument("--frame", type=int, required=True, help="the annotated (central) frame")
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--object", default="cup", help="object name used by the pipeline")
    parser.add_argument("--contact-frames", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None,
                        help="the stable grasp (inclusive): the clip shown to the annotator, and the "
                             "frames refine_object_contact.py applies the contact to")
    parser.add_argument("--video", default=None, help="RGB video (default: the *_rgb.mp4 / *.mp4 in --video-dir)")
    parser.add_argument("--hand-meshes", default=None, help="HaWoR all_hand_meshes.npz")
    parser.add_argument("--object-mesh", default=None, help="posed_mesh.obj (reference camera frame)")
    parser.add_argument("--object-poses", default=None, help="tracked object trajectory .npz")
    parser.add_argument("--intrinsics", default=None, help="intrinsics.yaml (fu fv pu pv)")
    parser.add_argument("--epic-root", default=os.path.join(HERE, "..", "epic-contact"),
                        help="EPIC-Contact root (for contact_assets/)")
    parser.add_argument("--out-dir", default=None, help="default: <video-dir>/contact_annotation")


# ───────────────────────────── inputs ─────────────────────────────

def first_existing(paths, what):
    for p in paths:
        if p and os.path.exists(p):
            return p
    raise SystemExit(f"no {what} found; tried:\n  " + "\n  ".join(p for p in paths if p) +
                     "\npass it explicitly (did the pipeline run on this clip?)")


def resolve_inputs(args):
    d = os.path.abspath(args.video_dir)
    obj = args.object.replace(" ", "_")
    if args.video is None:
        videos = sorted(glob.glob(os.path.join(d, "*_rgb.mp4"))) or sorted(glob.glob(os.path.join(d, "*.mp4")))
        args.video = first_existing(videos, "RGB video")
    stem = os.path.basename(args.video).split(".")[0]
    args.hand_meshes = args.hand_meshes or first_existing(
        [os.path.join(d, stem, "all_hand_meshes.npz")], "HaWoR hand meshes")
    args.object_mesh = args.object_mesh or first_existing(
        [os.path.join(d, f"recgen_out_{s}", obj, "posed_mesh.obj") for s in ("zed", "moge")], "object mesh")
    args.object_poses = args.object_poses or first_existing(
        [os.path.join(d, f"{obj}_traj_v2s2r{t}.npz") for t in ("", "_moge")] +
        [os.path.join(d, f"{obj}_traj_points{t}.npz") for t in ("", "_moge")], "object trajectory")
    args.intrinsics = args.intrinsics or os.path.join(d, "intrinsics.yaml")
    args.out_dir = args.out_dir or os.path.join(d, "contact_annotation")
    os.makedirs(os.path.join(args.out_dir, "central_frame"), exist_ok=True)
    for name in ("video", "hand_meshes", "object_mesh", "object_poses", "intrinsics"):
        print(f"[inputs] {name:13s} {getattr(args, name)}")


def load_intrinsics(args):
    if os.path.exists(args.intrinsics):
        c = yaml.safe_load(open(args.intrinsics))
        return tuple(float(c[k]) for k in ("fu", "fv", "pu", "pv"))
    metas = glob.glob(os.path.join(args.video_dir, "*_meta.json"))
    if not metas:
        raise SystemExit(f"no {args.intrinsics} and no *_meta.json to read intrinsics from")
    c = json.load(open(metas[0]))["intrinsics_left"]
    return c["fx"], c["fy"], c["cx"], c["cy"]


def read_frame(args, k):
    png = os.path.join(args.video_dir, "all_frames", f"{k:06d}.png")
    if os.path.exists(png):
        return cv2.imread(png)
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, image = cap.read()
    if not ok:
        raise SystemExit(f"could not read frame {k} of {args.video}")
    return image


def load_obj(path):
    mesh = trimesh.load(path, process=False, force="mesh")
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)


def write_obj(path, vertices, faces):
    # Written by hand: face order must survive exactly, the contact indexes into it.
    with open(path, "w") as fh:
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in vertices)
        fh.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)


def preview_video(src, dst, fps, max_width=960):
    """A VP8 WebM copy of the clip: pipeline videos are MPEG-4 Part 2, which browsers do not play."""
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / w)
    size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if not writer.isOpened():
        print("[warn] this OpenCV cannot write VP8; the page will show no video")
        return None
    while True:
        ok, image = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
    writer.release()
    print(f"[video] browser preview {dst}")
    return dst


def annotation_mesh(args, pitch):
    """The object mesh the contact indexes into: RecGen's, remeshed once and kept."""
    path = os.path.join(args.out_dir, "annotation_mesh.obj")
    if os.path.exists(path):
        return load_obj(path)
    v, f = load_obj(args.object_mesh)
    v, f = manifold_remesh(v, f, pitch)
    print(f"[mesh] manifold remesh at {pitch * 1000:.1f} mm: {len(f)} faces")
    write_obj(path, v, f)
    return v, f


class Clip:
    """Everything the annotation needs, prepared once per run."""

    def __init__(self, args, pitch=0.0015):
        self.args = args
        k = self.k = args.frame
        side = args.side
        hands = np.load(args.hand_meshes, allow_pickle=True)
        verts = hands[f"{side}_vertices"]
        self.n_frames = len(verts)
        if not 0 <= k < self.n_frames:
            raise SystemExit(f"--frame {k} outside the clip's {self.n_frames} frames")
        if args.contact_frames and not args.contact_frames[0] <= k <= args.contact_frames[1]:
            raise SystemExit(f"--frame {k} is outside --contact-frames {args.contact_frames}")
        # HaWoR can flag a frame valid and still return NaN vertices.
        valid = hands[f"{side}_valid"].astype(bool) & np.isfinite(verts).all((1, 2))
        if not valid[k]:
            good = np.flatnonzero(valid)
            raise SystemExit(f"HaWoR has no {side} hand at frame {k}; nearest valid: "
                             f"{sorted(good[np.argsort(np.abs(good - k))[:5]])}")
        self.hand = verts[k].astype(np.float64)

        # HaWoR's hand as MANO parameters in EC-fit's convention (cached: run_ecfit reads it).
        fit_path = os.path.join(args.out_dir, "hand_fit.npz")
        cached = np.load(fit_path) if os.path.exists(fit_path) else None
        if cached is not None and int(cached["frame"]) == k and str(cached["side"]) == side:
            self.mano = {key: cached[key] for key in cached.files}
        else:
            init = {"rot": hands[f"{side}_rot"][k], "hand_pose": hands[f"{side}_hand_pose"][k],
                    "betas": hands[f"{side}_betas"][k], "trans": hands[f"{side}_trans"][k]}
            self.mano = fit_mano_to_vertices(side, self.hand, init)
            self.mano.update(frame=k, side=side, hawor_vertices=self.hand)
            np.savez(fit_path, **self.mano)
        print(f"[hand] HaWoR hand -> MANO (EC-fit convention): max error {float(self.mano['max_error']) * 1000:.1f} mm")

        self.sparse_faces = mano_faces(side)
        self.dense = DenseHand(args.epic_root, side, self.sparse_faces)
        flat, _ = flat_hand(side, self.mano["hand_pose"], self.mano["betas"])
        self.flat = self.dense.pose(flat)
        self.flat_surface = Surface(self.flat, self.dense.faces)
        self.wrist = flat[0]
        self.mesh_v, self.mesh_f = annotation_mesh(args, pitch)
        self.object = Surface(self.mesh_v, self.mesh_f)

        poses = np.load(args.object_poses, allow_pickle=True)
        self.rot = poses["rotation"][k].astype(np.float64)
        self.trans = poses["translation"][k].astype(np.float64)
        self.intrinsics = load_intrinsics(args)
        self.image = read_frame(args, k)
        self.fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30.0
        self.preview = preview_video(args.video, os.path.join(args.out_dir, "preview.webm"), self.fps)
        self.annotation_path = os.path.join(args.out_dir, "annotation.json")
        self._params = {}

    def region_patch(self, painted, region):
        painted = np.asarray(sorted(set(int(i) for i in painted)), int)
        if len(painted) == 0:
            return painted
        return painted[self.dense.region[painted] == REGIONS.index(region)]

    def param(self, region, patch):
        key = (region, hashlib.md5(np.asarray(patch, np.int64).tobytes()).hexdigest())
        if key not in self._params:
            self._params[key] = PatchParam(self.flat_surface, patch, self.wrist)
        return self._params[key]

    def payload(self):
        saved = json.load(open(self.annotation_path)) if os.path.exists(self.annotation_path) else None
        if saved and (saved.get("frame") != self.k or saved.get("object_faces") != len(self.mesh_f)):
            print("[warn] annotation.json does not match this frame/mesh; starting empty")
            saved = None
        window = self.args.contact_frames or [0, self.n_frames - 1]
        return {
            "frame": self.k, "side": self.args.side, "object": self.args.object, "regions": REGIONS,
            "fps": self.fps, "window": window, "n_frames": self.n_frames,
            "template": {"vertices": np.round(self.dense.template, 5).ravel().tolist(),
                         "faces": self.dense.faces.ravel().tolist(), "region": self.dense.region.tolist()},
            "flat": {"vertices": np.round(self.flat, 5).ravel().tolist()},
            "object_mesh": {"vertices": np.round(self.mesh_v, 5).ravel().tolist(),
                            "faces": self.mesh_f.ravel().tolist()},
            "annotation": saved,
        }


# ───────────────────────────── export ─────────────────────────────

def export(clip, annotation):
    """Transfer every clicked region and write EPIC-Contact's three contact files."""
    out = os.path.join(clip.args.out_dir, "central_frame")
    painted = annotation["hand_dense"]
    bodyparts, sparse, contacts, summary = {}, {}, [], {}
    for r in REGIONS:
        patch = clip.region_patch(painted, r)
        bodyparts[r] = patch.tolist()
        sparse[r] = sorted(set(clip.dense.dense2sparse[patch].tolist())) if len(patch) else []
        clicks = annotation.get("clicks", {}).get(r)
        if not len(patch):
            continue
        if not clicks:
            print(f"[warn] {r}: painted but not transferred; EC-fit will ignore it")
            summary[r] = {"hand_vertices": len(patch), "transferred": False}
            continue
        # The clicks belong to the patch they were placed for (the page stores it as `key`).
        if clicks.get("key") != ",".join(map(str, patch.tolist())):
            print(f"[warn] {r}: the painting changed since its clicks; not transferred, click again")
            summary[r] = {"hand_vertices": len(patch), "transferred": False}
            continue
        param = clip.param(r, patch)
        _, faces, bary, _ = transfer(param, clip.object, clicks["start"], clicks["direction"])
        # EPIC's layout: objShape<region> and humanShape<region>, the i-th entries paired.
        contacts.append({"name": f"objShape{r}", "contactPoints": [
            f"f {int(f)} {b[0]} {b[1]} {b[2]}" for f, b in zip(faces, bary)]})
        contacts.append({"name": f"humanShape{r}", "contactPoints": [f"v {int(i)}" for i in patch]})
        summary[r] = {"hand_vertices": len(patch), "transferred": True, "axis_mm": param.length * 1000}
    json.dump(bodyparts, open(os.path.join(out, "contact_vertices_bodyparts.json"), "w"))
    json.dump(sparse, open(os.path.join(out, "contact_vertices_sparse.json"), "w"))
    json.dump({"data": contacts}, open(os.path.join(out, "corresponding_contacts.json"), "w"))
    print(f"[export] {out}: " + ", ".join(
        f"{r} {s['hand_vertices']} v{'' if s['transferred'] else ' (not transferred)'}" for r, s in summary.items()))
    return summary


# ───────────────────────────── server ─────────────────────────────

def next_command(args):
    command = (f"conda activate ecfit; cd {HERE}; python run_ecfit.py --video-dir {os.path.abspath(args.video_dir)} "
               f"--object {args.object} --side {args.side} --frame {args.frame}")
    if args.contact_frames:
        command += " --contact-frames {} {}".format(*args.contact_frames)
    return command


def serve(clip, args):
    payload = json.dumps(clip.payload()).encode()
    ok, jpg = cv2.imencode(".jpg", clip.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, body, kind, status=200, headers=()):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def send_video(self):
            # Browsers seek in a <video> with byte-range requests.
            if not clip.preview:
                return self.send_error(404)
            size = os.path.getsize(clip.preview)
            start, end = 0, size - 1
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), size - 1)
            with open(clip.preview, "rb") as fh:
                fh.seek(start)
                body = fh.read(end - start + 1)
            self.reply(body, "video/webm", 206 if match else 200,
                       [("Accept-Ranges", "bytes"), ("Content-Range", f"bytes {start}-{end}/{size}")])

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.reply(open(UI_PATH, "rb").read(), "text/html; charset=utf-8")
            elif self.path == "/data.json":
                self.reply(payload, "application/json")
            elif self.path == "/frame.jpg":
                self.reply(jpg.tobytes(), "image/jpeg")
            elif self.path == "/video.webm":
                self.send_video()
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                with lock:
                    result = self.route(body)
            except Exception as err:  # keep the page alive
                result = {"ok": False, "error": f"{type(err).__name__}: {err}"}
            self.reply(json.dumps(result).encode(), "application/json")

        def route(self, body):
            if self.path == "/axis":
                patch = clip.region_patch(body["hand_dense"], body["region"])
                if not len(patch):
                    return {"ok": False, "error": "nothing painted in this region"}
                param = clip.param(body["region"], patch)
                return {"ok": True, "patch": patch.tolist(), "axis": param.path.round(6).tolist(),
                        "length_mm": param.length * 1000}
            if self.path == "/transfer":
                patch = clip.region_patch(body["hand_dense"], body["region"])
                param = clip.param(body["region"], patch)
                points, _, _, axis = transfer(param, clip.object, body["start"], body["direction"])
                return {"ok": True, "points": points.round(6).tolist(), "axis": axis.round(6).tolist()}
            if self.path == "/save":
                annotation = {"frame": clip.k, "side": args.side, "object": args.object,
                              "object_faces": len(clip.mesh_f),
                              "object_mesh_source": os.path.abspath(args.object_mesh),
                              "contact_frames": args.contact_frames,
                              "hand_dense": sorted(set(int(i) for i in body["hand_dense"])),
                              "clicks": body.get("clicks", {})}
                json.dump(annotation, open(clip.annotation_path, "w"), indent=1)
                return {"ok": True, "summary": export(clip, annotation),
                        "out_dir": os.path.join(args.out_dir, "central_frame"), "next": next_command(args)}
            return {"ok": False, "error": f"unknown endpoint {self.path}"}

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[ui] open {url}  (Ctrl+C to stop; 'Save' writes to {args.out_dir})")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] stopped")


def main():
    # The URL line must show up even when the output is piped to a log.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    resolve_inputs(args)
    clip = Clip(args, args.remesh_pitch)
    print(f"[clip] frame {clip.k}/{clip.n_frames}; {args.side} hand; object {len(clip.mesh_f)} faces, "
          f"extents {np.round(clip.object.mesh.extents, 3)} m")
    if args.export_only:
        if not os.path.exists(clip.annotation_path):
            raise SystemExit(f"no {clip.annotation_path} to export")
        export(clip, json.load(open(clip.annotation_path)))
        return
    serve(clip, args)


if __name__ == "__main__":
    main()
