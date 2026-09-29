"""
EPIC-Contact's hand-to-object contact transfer (Bansal et al., ECCV 2026, Sec. 3.2 and
Sup. Mat. 8.4), which follows ContactEdit (Lakshmipathy et al., SIGGRAPH 2023, Sec. 4.1).

The pieces, as the papers describe them:

  * hand contact is painted on EPIC's canonical 3106-vertex MANO template and split into
    three regions (fingers / thumb / palm) by EPIC's own part segmentation;
  * each region is transferred on its own, from a *flat* hand whose finger spread matches
    the hand in the video. EPIC retrieves the closest of 41 flat poses by geodesic distance
    on the spread dimensions of the WiLoR pose; the pool is not released, so the spread is
    taken directly from the pose (the limit of an unboundedly fine pool): each finger's MCP
    rotation is reduced to its twist about the palm normal, everything else is zero;
  * each region is parametrised by a contact axis A -- ContactEdit's first default, the
    geodesic between the patch's two furthest points -- and every patch point p by the log
    map at its closest axis point a_j: distance r and angle theta relative to A's tangent;
  * the annotator gives two clicks on the object: the start of A' and its direction. A' is
    traced as a geodesic of A's length; each p' is traced from a'_j at the *negated* angle
    (hand and object surfaces face each other, so the patch is mirrored) for distance r.

Every painted hand vertex therefore gets exactly one object point: the bijection
EC-fit's contact loss uses.

Needs potpourri3d (vector heat log maps, edge-flip geodesics, geodesic tracing) and
smplx; both are in the `ecfit` env.
"""

import json
import os

import numpy as np
import potpourri3d as pp
import torch
import trimesh

REGIONS = ("fingers", "thumb", "palm")
MANO_DIR = os.path.expanduser("~/.cache/mano_models")      # MANO_{LEFT,RIGHT}.pkl
# smplx hand_pose rows (joints 1..15): index 0-2, middle 3-5, pinky 6-8, ring 9-11, thumb 12-14.
MCP_ROWS = (0, 3, 6, 9)


# ───────────────────────────── MANO ─────────────────────────────

def mano_model(side, mano_dir=MANO_DIR):
    from smplx import MANO
    return MANO(mano_dir, is_rhand=(side == "right"), flat_hand_mean=True, use_pca=False)


def mano_faces(side, mano_dir=MANO_DIR):
    return np.asarray(mano_model(side, mano_dir).faces, np.int64)


def fit_mano_to_vertices(side, vertices, init, iters=800, mano_dir=MANO_DIR):
    """MANO parameters, in EC-fit's convention, that reproduce a given 778-vertex hand.

    EC-fit rebuilds the hand from MANO parameters (smplx, flat_hand_mean=True, the model of
    the hand's own side). HaWoR's left hand is a mirrored right-hand MANO and the left
    model does not reproduce it exactly (~10 mm), so the parameters are fitted rather than
    converted. The shape is refitted too: the mismatch is largely MANO_LEFT's shape basis.
    init: dict with rot (3,), hand_pose (45,), betas (10,), trans (3,).
    """
    model = mano_model(side, mano_dir)
    f = lambda a: torch.tensor(np.asarray(a, np.float32))[None]
    params = [f(init["rot"]).requires_grad_(), f(init["hand_pose"]).requires_grad_(),
              f(init["trans"]).requires_grad_(), f(init["betas"]).requires_grad_()]
    betas = params[3]
    target = f(vertices)
    opt = torch.optim.Adam(params, lr=1e-2)
    for i in range(iters):
        out = model(betas=betas, global_orient=params[0], hand_pose=params[1], transl=params[2]).vertices
        loss = ((out - target) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == iters // 2:
            for g in opt.param_groups:
                g["lr"] = 2e-3
    with torch.no_grad():
        out = model(betas=betas, global_orient=params[0], hand_pose=params[1], transl=params[2]).vertices[0]
    err = float((out - target[0]).norm(dim=-1).max())
    return {"global_orient": params[0][0].detach().numpy().astype(np.float64),
            "hand_pose": params[1][0].detach().numpy().astype(np.float64),
            "transl": params[2][0].detach().numpy().astype(np.float64),
            "betas": betas[0].detach().numpy().astype(np.float64),
            "vertices": out.numpy().astype(np.float64), "max_error": err}


def flat_hand(side, hand_pose, betas, mano_dir=MANO_DIR):
    """The flat hand EPIC transfers from: fingers straight, spread as in hand_pose."""
    model = mano_model(side, mano_dir)
    with torch.no_grad():
        rest = model(betas=torch.zeros(1, 10))
    joints = rest.joints[0].numpy()
    # Palm normal: the plane through the wrist and the four MCP joints.
    palm = joints[[0, 1, 4, 7, 10]]
    normal = np.linalg.svd(palm - palm.mean(0))[2][2]
    pose = np.zeros(45)
    for row in MCP_ROWS:
        w = np.asarray(hand_pose[row:row + 3], np.float64)
        angle = np.linalg.norm(w)
        if angle < 1e-9:
            continue
        # Swing-twist: the twist of rotation q about `normal` is (q_w, (q_v . n) n).
        qw, qv = np.cos(angle / 2), np.sin(angle / 2) * w / angle
        twist = 2 * np.arctan2(qv @ normal, qw)
        pose[row:row + 3] = twist * normal
    with torch.no_grad():
        out = model(betas=torch.tensor(np.asarray(betas, np.float32))[None],
                    hand_pose=torch.tensor(pose, dtype=torch.float32)[None])
    return out.vertices[0].numpy().astype(np.float64), pose


# ───────────────────────────── EPIC's dense hand ─────────────────────────────

class DenseHand:
    """EPIC's 3106-vertex MANO (contact_assets/), its regions, and its 778-vertex embedding."""

    def __init__(self, epic_root, side, sparse_faces):
        assets = os.path.join(epic_root, "contact_assets")
        dense = trimesh.load(os.path.join(assets, f"{side}_hand_dense.obj"), process=False, force="mesh")
        self.template = np.asarray(dense.vertices, np.float64)
        self.faces = np.asarray(dense.faces, np.int64)
        # The template's first 778 vertices are the canonical MANO; every dense vertex sits
        # on (a median 0.2 mm off) that surface, so it is embedded once, barycentrically,
        # and any posed MANO hand -- HaWoR's, EC-fit's, the flat one -- carries it along.
        canon = trimesh.Trimesh(self.template[:778], sparse_faces, process=False)
        closest, _, tri = trimesh.proximity.closest_point(canon, self.template)
        bary = trimesh.triangles.points_to_barycentric(canon.triangles[tri], closest)
        self.weights = np.zeros((len(self.template), 778))
        np.put_along_axis(self.weights, sparse_faces[tri], bary, axis=1)
        seg = json.load(open(os.path.join(assets, "mano_part_segmentation_dense_three_regions.json")))
        self.region = np.full(len(self.template), -1, int)
        for i, r in enumerate(REGIONS):
            self.region[np.asarray(seg[r], int)] = i
        mapping = json.load(open(os.path.join(assets, "sparse_dense_mapping.json")))
        self.dense2sparse = np.asarray(mapping[f"m{side[0]}_dense2sparse"], int)

    def pose(self, sparse_vertices):
        return self.weights @ sparse_vertices


# ───────────────────────────── surfaces ─────────────────────────────

def outward_sign(vertices, faces):
    """+1 if the faces wind outward (normals point away from the interior), else -1.

    Works on open meshes (the hand is open at the wrist) by comparing face normals with
    the direction from the centroid; for a closed mesh this agrees with the volume sign.
    """
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    d = mesh.triangles_center - vertices.mean(0)
    s = np.sum(mesh.face_normals * d, 1) * mesh.area_faces
    return 1.0 if s.sum() >= 0 else -1.0


def manifold_remesh(vertices, faces, pitch=0.0015):
    """A closed manifold copy of a mesh, for geodesics (geometry-central needs a manifold).

    RecGen meshes, and any quadric decimation of them, can carry duplicate or non-manifold
    edges. Voxelising and running marching cubes gives a watertight manifold at `pitch`
    resolution; snapping its vertices back onto the original surface removes the half-voxel
    offset without touching the connectivity.
    """
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    grid = mesh.voxelized(pitch).fill()
    out = grid.marching_cubes
    out.apply_transform(grid.transform)
    trimesh.smoothing.filter_taubin(out, iterations=10)
    closest, _, _ = trimesh.proximity.closest_point(mesh, out.vertices)
    return np.asarray(closest, np.float64), np.asarray(out.faces, np.int64)


def rotate_about(v, n, angle):
    """Rotate tangent vector v about unit normal n by angle (counter-clockwise about n)."""
    return v * np.cos(angle) + np.cross(n, v) * np.sin(angle)


class Surface:
    def __init__(self, vertices, faces):
        self.v = np.asarray(vertices, np.float64)
        self.f = np.asarray(faces, np.int64)
        self.mesh = trimesh.Trimesh(self.v, self.f, process=False)
        self.sign = outward_sign(self.v, self.f)
        self._heat = self._vector = self._flip = self._tracer = None

    @property
    def heat(self):
        self._heat = self._heat or pp.MeshHeatMethodDistanceSolver(self.v, self.f)
        return self._heat

    @property
    def vector(self):
        self._vector = self._vector or pp.MeshVectorHeatSolver(self.v, self.f)
        return self._vector

    @property
    def flip(self):
        self._flip = self._flip or pp.EdgeFlipGeodesicSolver(self.v, self.f)
        return self._flip

    @property
    def tracer(self):
        self._tracer = self._tracer or pp.GeodesicTracer(self.v, self.f)
        return self._tracer

    def outward_normal(self, face):
        return self.sign * self.mesh.face_normals[face]

    def locate(self, points):
        """Closest surface point, its face and barycentric coordinates."""
        closest, _, face = trimesh.proximity.closest_point(self.mesh, np.atleast_2d(points))
        bary = trimesh.triangles.points_to_barycentric(self.mesh.triangles[face], closest)
        return closest, face, bary

    def trace(self, face, bary, direction):
        """Geodesic from a surface point; the length of `direction` is the distance."""
        n = self.mesh.face_normals[face]
        direction = direction - n * (direction @ n)
        if np.linalg.norm(direction) < 1e-9:
            return np.asarray(self.mesh.triangles[face].T @ bary)[None]
        return np.asarray(self.tracer.trace_geodesic_from_face(int(face), np.asarray(bary, np.float64),
                                                               direction))


# ───────────────────────────── ContactEdit parametrisation ─────────────────────────────

def arc_lengths(path):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])


def point_on_path(path, s):
    cum = arc_lengths(path)
    s = np.clip(s, 0, cum[-1])
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(path) - 2))
    seg = path[i + 1] - path[i]
    length = np.linalg.norm(seg)
    t = 0.0 if length < 1e-12 else (s - cum[i]) / length
    return path[i] + t * seg, seg / max(length, 1e-12)


class PatchParam:
    """A region's patch on the hand, parametrised about its contact axis."""

    def __init__(self, surface, patch, anchor, spacing=0.004):
        patch = np.asarray(patch, int)
        self.patch = patch
        # Axis = geodesic between the two furthest patch points (ContactEdit, default 1).
        d = surface.heat.compute_distance(int(patch[0]))
        a = int(patch[np.argmax(d[patch])])
        d = surface.heat.compute_distance(a)
        b = int(patch[np.argmax(d[patch])])
        if a == b:                                   # a single vertex: any direction
            b = int(np.argsort(np.linalg.norm(surface.v - surface.v[a], axis=1))[1])
        # Start at the end nearer the wrist, so the blue ball is on the proximal end.
        if np.linalg.norm(surface.v[b] - anchor) < np.linalg.norm(surface.v[a] - anchor):
            a, b = b, a
        self.path = np.asarray(surface.flip.find_geodesic_path(a, b))
        self.length = float(arc_lengths(self.path)[-1])

        # Axis sample points, snapped to vertices (log maps are computed at vertices).
        n_samples = max(2, int(np.ceil(self.length / spacing)) + 1)
        samples = [point_on_path(self.path, s)[0] for s in np.linspace(0, self.length, n_samples)]
        verts, s_of = [], []
        for s, p in zip(np.linspace(0, self.length, n_samples), samples):
            v = int(np.argmin(np.linalg.norm(surface.v - p, axis=1)))
            if not verts or v != verts[-1]:
                verts.append(v)
                s_of.append(s)
        if len(verts) < 2:
            verts, s_of = [a, b], [0.0, self.length]
        self.axis_vertices = np.asarray(verts)
        self.axis_s = np.asarray(s_of)

        # Log map at every axis vertex; the tangent angle there is the direction to the
        # next axis vertex (for the last vertex, the reverse of the direction back).
        logs, tangent = [], []
        for j, v in enumerate(self.axis_vertices):
            lm = surface.vector.compute_log_map(int(v))
            logs.append(lm[patch])
            if j + 1 < len(self.axis_vertices):
                nxt = lm[self.axis_vertices[j + 1]]
                tangent.append(np.arctan2(nxt[1], nxt[0]))
            else:
                prv = lm[self.axis_vertices[j - 1]]
                tangent.append(np.arctan2(prv[1], prv[0]) + np.pi)
        logs = np.stack(logs)                                    # (m, n, 2)
        radius = np.linalg.norm(logs, axis=-1)
        self.closest = np.argmin(radius, axis=0)                 # closest axis point per p
        pick = logs[self.closest, np.arange(len(patch))]
        self.r = radius[self.closest, np.arange(len(patch))]
        # Angle relative to the axis tangent, counter-clockwise about the OUTWARD normal
        # (the log map's tangent basis follows the face winding).
        theta = np.arctan2(pick[:, 1], pick[:, 0]) - np.asarray(tangent)[self.closest]
        self.theta = surface.sign * theta

    def to_dict(self, surface):
        start = surface.v[self.axis_vertices[0]]
        return {"path": self.path.round(6).tolist(), "start": start.round(6).tolist(),
                "length": self.length, "patch": self.patch.tolist()}


def transfer(param, obj, start_point, direction_point):
    """Transfer a parametrised hand patch onto object surface `obj` from two clicks.

    Returns (points (n, 3), faces (n,), barycentric (n, 3), axis polyline).
    """
    p1, f1, b1 = obj.locate(start_point)
    p1, f1, b1 = p1[0], int(f1[0]), b1[0]
    n1 = obj.outward_normal(f1)
    d = np.asarray(direction_point, np.float64) - p1
    d = d - n1 * (d @ n1)
    if np.linalg.norm(d) < 1e-9:
        raise ValueError("direction click is at the start point")
    d = d / np.linalg.norm(d)
    axis = obj.trace(f1, b1, d * param.length)
    if len(axis) < 2:
        axis = np.stack([p1, p1 + d * 1e-6])
    # The traced axis may stop short at a mesh boundary; scale the samples onto it.
    traced = arc_lengths(axis)[-1]
    scale = traced / param.length if param.length > 0 else 1.0

    anchors = []
    for s in param.axis_s:
        point, tangent = point_on_path(axis, s * scale)
        q, fq, bq = obj.locate(point)
        n = obj.outward_normal(int(fq[0]))
        tangent = tangent - n * (tangent @ n)
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        anchors.append((q[0], int(fq[0]), bq[0], tangent, n))

    ends = np.empty((len(param.patch), 3))
    for i, (j, r, theta) in enumerate(zip(param.closest, param.r, param.theta)):
        q, fq, bq, tangent, n = anchors[j]
        # Mirror: the object patch faces the hand patch, so the angle is negated.
        direction = rotate_about(tangent, n, -theta) * r
        ends[i] = obj.trace(fq, bq, direction)[-1]
    points, faces, bary = obj.locate(ends)
    return points, faces, bary, axis
