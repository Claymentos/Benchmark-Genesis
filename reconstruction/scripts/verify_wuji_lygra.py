#!/usr/bin/env python3
"""
Re-derive the numbers behind the Wuji Hand's Lightning Grasp config, and check them.

The config itself lives where Lightning Grasp keeps its own hands
(lightning-grasp/lygra/robot/wuji.py). Two of its entries are measurements rather
than choices, and a revised hand description would silently invalidate both:

  * PAD_DIRECTION -- where each fingertip's pad faces, taken as the direction the tip
    travels between the open hand and the flexion joints at their limit.
  * the self-collision whitelist -- link pairs that overlap in the collision meshes
    whatever the pose, as opposed to pairs that interfere only in some poses and must
    stay checked. Without the right set the hand self-collides in its resting pose and
    Lightning Grasp filters out every grasp it synthesises.

This reproduces both from the URDF and compares them with what the config declares.

Run in the `lygra` env (the collision check needs its CUDA kernels):
    python verify_wuji_lygra.py --side left
    python verify_wuji_lygra.py --side left --skip-collision   # geometry only, no GPU
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ElementTree
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_LYGRA_DIR = os.environ.get(
    "LIGHTNING_GRASP_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lightning-grasp"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Check the Wuji Lightning Grasp config")
    parser.add_argument("--side", default="left", choices=("left", "right"))
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--lygra-dir", default=DEFAULT_LYGRA_DIR)
    parser.add_argument("--samples", type=int, default=200,
                        help="Random joint configurations used to separate a permanent "
                             "mesh overlap from real interference")
    parser.add_argument("--skip-collision", action="store_true",
                        help="Only re-derive the pad directions, which needs no GPU")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Allowed disagreement with the declared pad direction. The "
                             "direction only picks a 90 degree cone, so the thumb's 0.036 "
                             "(about 2 degrees) changes nothing; this is set to catch a "
                             "hand whose pads genuinely face elsewhere")
    return parser.parse_args()


def load_joints(urdf_path):
    """Child link -> its joint, as offsets, axes and limits."""
    root = ElementTree.parse(urdf_path).getroot()
    joints = {}
    for joint in root.iter("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints[joint.find("child").get("link")] = dict(
            parent=joint.find("parent").get("link"),
            offset=np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]),
            rotation=Rotation.from_euler(
                "xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]
            ),
            axis=np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else None,
            name=joint.get("name"),
            upper=float(limit.get("upper")) if limit is not None else None,
        )
    return joints


def forward(joints, link, angles):
    """Pose of `link` in the palm frame at the given joint angles."""
    position, orientation = np.zeros(3), Rotation.identity()
    while link in joints:
        node = joints[link]
        if node["axis"] is not None:
            turn = Rotation.from_rotvec(node["axis"] * angles.get(node["name"], 0.0))
            position, orientation = turn.apply(position), turn * orientation
        position = node["offset"] + node["rotation"].apply(position)
        orientation = node["rotation"] * orientation
        link = node["parent"]
    return position, orientation


def check_pad_directions(joints, side, fingers, declared, tolerance):
    """Closing direction per fingertip, against the config's single constant."""
    print("pad directions (the direction a fingertip travels as it closes)")
    worst = 0.0
    for finger in fingers:
        tip = f"{side}_finger{finger}_tip_link"
        closed = {
            f"{side}_finger{finger}_joint{k}": joints[f"{side}_finger{finger}_link{k}"]["upper"]
            for k in (3, 4)
        }
        open_position, _ = forward(joints, tip, {})
        closed_position, closed_orientation = forward(joints, tip, closed)
        travel = closed_position - open_position
        travel /= np.linalg.norm(travel)
        local = closed_orientation.inv().apply(travel)
        gap = float(np.linalg.norm(local - declared))
        worst = max(worst, gap)
        print(f"  finger{finger}: palm frame {np.round(travel, 3)}  "
              f"link frame {np.round(local, 3)}  off by {gap:.4f}")
    verdict = "OK" if worst <= tolerance else "DISAGREES"
    print(f"  declared PAD_DIRECTION {declared}: {verdict} "
          f"(worst {worst:.4f}, tolerance {tolerance})\n")
    return worst <= tolerance


def check_collision_pairs(robot, samples):
    """Which link pairs collide, and how often, over random joint configurations.

    A pair that collides in every configuration is a permanent overlap in the meshes
    and belongs in the whitelist; one that collides in some is real interference.
    """
    import torch

    from lygra.kinematics import build_kinematics_tree
    from lygra.mesh import get_urdf_mesh_decomposed
    import lygra.pipeline.module.collision as collision

    tree = build_kinematics_tree(robot.urdf_path, active_joint_names=robot.get_active_joints())
    meshes = get_urdf_mesh_decomposed(
        urdf_path=robot.urdf_path, tree=tree, mesh_scale=robot.get_mesh_scale()
    )
    unfiltered = tree.get_self_collision_check_link_pairs(
        link_body_id=meshes["link_body_id"], whitelist_link=[], whitelist_pairs=[]
    )
    lower, upper = tree.get_actuated_joint_limit()
    angles = torch.from_numpy(
        np.random.default_rng(0).uniform(lower, upper, (samples, len(lower)))
    ).cuda().float()

    body_to_link, links = meshes["body_id_to_link_id"], tree.links
    hits = Counter()
    for pair in unfiltered:
        single = torch.from_numpy(np.array([pair], dtype=np.int32)).cuda().int()
        free, _, _ = collision.batch_hand_self_collision_check(
            tree, q=angles, self_collision_link_pairs=single, mesh=meshes
        )
        colliding = int((~free.bool()).sum())
        if colliding:
            hits[(links[body_to_link[pair[0]]], links[body_to_link[pair[1]]])] += colliding

    always = {pair for pair, count in hits.items() if count == samples}
    print(f"self-collision over {samples} random configurations")
    for pair, count in hits.most_common(10):
        marker = "always" if count == samples else "      "
        print(f"  {marker}  {pair[0]:26s} {pair[1]:26s} {100 * count / samples:5.1f}%")

    declared = {tuple(pair) for pair in robot.get_white_list_pairs()}
    declared |= {(b, a) for a, b in declared}
    missing = {pair for pair in always if pair not in declared}
    extra = {pair for pair in declared if pair in hits and pair not in always}
    print(f"\n  always-colliding pairs: {len(always)}, declared in the whitelist: "
          f"{len(declared) // 2}")
    if missing:
        print(f"  MISSING from the whitelist (grasps will all be filtered): {sorted(missing)}")
    if extra:
        print(f"  whitelisted but only sometimes colliding (hides real interference): {sorted(extra)}")
    if not missing and not extra:
        print("  whitelist matches the measurement")
    return not missing and not extra


def main():
    args = parse_args()
    sys.path.insert(0, args.lygra_dir)

    from lygra.robot import build_robot
    from lygra.robot.wuji import FINGERS, PAD_DIRECTION

    robot = build_robot(f"wuji_{args.side}", urdf_path=args.urdf)
    print(f"{robot.urdf_path}\n")

    joints = load_joints(robot.urdf_path)
    ok = check_pad_directions(joints, args.side, FINGERS, PAD_DIRECTION, args.tolerance)

    if not args.skip_collision:
        ok &= check_collision_pairs(robot, args.samples)

    print("\nconfig agrees with the hand description" if ok else "\nconfig needs updating")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
