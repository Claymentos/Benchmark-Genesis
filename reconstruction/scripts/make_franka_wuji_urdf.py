#!/usr/bin/env python3
"""
Build a single URDF with the Wuji Hand mounted on a Franka Panda flange.

The two source URDFs live in different checkouts and reference their meshes
relatively ("package://meshes/..." for the Panda, "../meshes/left/..." for the hand),
so a naive concatenation loads with no geometry. Every mesh path is rewritten to an
absolute one, which keeps the result valid from any working directory without
copying mesh files or editing either source repo.

    python make_franka_wuji_urdf.py --side left --out assets/franka_wuji_left.urdf
"""

import argparse
import os
import xml.etree.ElementTree as ET

DEFAULT_PANDA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "..", "miniconda3", "envs", "genesis", "lib", "python3.12",
    "site-packages", "genesis", "assets", "urdf", "panda_bullet", "panda_nohand.urdf",
)
DEFAULT_WUJI_DIR = os.environ.get(
    "WUJI_RETARGETING_DIR",
    os.path.expanduser("~/wuji-ego-mint/eval/simulate/wuji-retargeting"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Mount the Wuji Hand on a Franka Panda")
    parser.add_argument("--panda-urdf", default=None, help="Panda URDF (default: Genesis's panda_nohand)")
    parser.add_argument("--wuji-urdf", default=None, help="Wuji Hand URDF (default: bundled for the side)")
    parser.add_argument("--wuji-dir", default=DEFAULT_WUJI_DIR)
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--parent-link", default=None, help="Panda link to mount on (default: its tip)")
    parser.add_argument(
        "--mount-xyz", default="0,0,0", help="Mount translation from the flange, metres"
    )
    parser.add_argument(
        "--mount-rpy", default="0,0,0", help="Mount rotation from the flange, radians"
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def absolutise_meshes(root, urdf_path):
    """Rewrite every mesh filename in `root` to an absolute path."""
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename is None or os.path.isabs(filename):
            continue
        if filename.startswith("package://"):
            filename = filename[len("package://"):]
        mesh.set("filename", os.path.normpath(os.path.join(urdf_dir, filename)))


def tip_link(root):
    """The link that nothing is attached to, i.e. the end of the chain."""
    children = {joint.find("child").get("link") for joint in root.findall("joint")}
    parents = {joint.find("parent").get("link") for joint in root.findall("joint")}
    tips = [name for name in children if name not in parents]
    if not tips:
        raise SystemExit("could not find a tip link on the arm")
    return tips[-1]


def root_link(root):
    children = {joint.find("child").get("link") for joint in root.findall("joint")}
    roots = [link.get("name") for link in root.findall("link") if link.get("name") not in children]
    if len(roots) != 1:
        raise SystemExit(f"expected exactly one root link, found {roots}")
    return roots[0]


def main():
    args = parse_args()

    panda_path = args.panda_urdf or os.path.normpath(DEFAULT_PANDA)
    wuji_path = args.wuji_urdf or os.path.join(
        args.wuji_dir, "wuji_retargeting", "wuji-description", "hand", "body", "urdf",
        f"{args.side}.urdf",
    )
    for path in (panda_path, wuji_path):
        if not os.path.exists(path):
            raise SystemExit(f"missing URDF: {path}")

    panda = ET.parse(panda_path).getroot()
    wuji = ET.parse(wuji_path).getroot()
    absolutise_meshes(panda, panda_path)
    absolutise_meshes(wuji, wuji_path)

    panda_names = {link.get("name") for link in panda.findall("link")}
    overlapping = panda_names & {link.get("name") for link in wuji.findall("link")}
    if overlapping:
        raise SystemExit(f"link names collide between the two URDFs: {sorted(overlapping)}")

    parent = args.parent_link or tip_link(panda)
    child = root_link(wuji)

    mount = ET.SubElement(panda, "joint", {"name": f"wuji_{args.side}_mount", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": parent})
    ET.SubElement(mount, "child", {"link": child})
    ET.SubElement(
        mount,
        "origin",
        {
            "xyz": " ".join(args.mount_xyz.split(",")),
            "rpy": " ".join(args.mount_rpy.split(",")),
        },
    )

    for element in list(wuji.findall("link")) + list(wuji.findall("joint")):
        panda.append(element)

    panda.set("name", f"franka_wuji_{args.side}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    ET.ElementTree(panda).write(args.out, encoding="utf-8", xml_declaration=True)

    print(f"Wrote {args.out}")
    print(f"  arm: {panda_path}")
    print(f"  hand: {wuji_path}")
    print(f"  mounted {child} on {parent}")


if __name__ == "__main__":
    main()
