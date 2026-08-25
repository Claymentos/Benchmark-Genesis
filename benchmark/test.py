import argparse

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import (
    R_to_quat,
    inv_transform_by_trans_quat,
    trans_quat_to_T,
    transform_by_quat,
    transform_by_trans_quat,
)

# Grasp point on the hammer handle, expressed in the hammer link's local frame. Measured on
# objects/meshes/hammer.obj as the centroid and long axis of its thinnest contiguous cross-section (the
# cylindrical handle, before it flares into the head).
HAMMER_HANDLE_CENTER_LOCAL = (-0.2566, 0.0827, 0.0501)
HAMMER_HANDLE_AXIS_LOCAL = (1.0, 0.0, 0.0)
HAMMER_SPAWN_POS = (0.55, 0.0, 0.084)  # z clears the mesh's lowest local point (-0.0821) above the ground plane.

FINGER_TIP_LINK_NAMES = ("finger1_link4", "finger2_link4", "finger3_link4", "finger4_link4", "finger5_link4")

# Per-joint closing amount, as a fraction of each dof's [lower, upper] range: the thumb (finger1) opposes the
# other four fingers, which curl inward with a moderate spread to wrap around the handle.
THUMB_OPPOSE_FRACTION = 0.7
FINGER_SPREAD_FRACTION = 0.3
FINGER_CURL_FRACTION = 0.75

# Bent-elbow arm configuration to start from and to seed inverse kinematics with, instead of the URDF's all-zero
# default (which is itself outside panda_joint4's range and leaves the arm fully outstretched overhead).
ARM_HOME_QPOS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

GRASP_POSE_AXIS_LENGTH = 0.08
GRASP_POINT_MARKER_RADIUS = 0.015
GRASP_POINT_MARKER_COLOR = (1.0, 0.0, 1.0, 0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.0, 0.8),
            camera_lookat=(0.3, 0.0, 0.1),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        gs.morphs.URDF(
            file="/home/cl-ment-prigent/genesis-world/robots/panda_wuji/urdf/panda_wuji_left.urdf",
            fixed=True,
        ),
    )

    hammer = scene.add_entity(
        gs.morphs.Mesh(
            file="/home/cl-ment-prigent/genesis-world/objects/meshes/hammer.obj",
            pos=HAMMER_SPAWN_POS,
        )
    )

    scene.build()

    # --- DOF layout: 7 arm joints, then 4 joints per Wuji finger (thumb, index, middle, ring, pinky). ---
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, franka.n_dofs)
    wrist_link = franka.get_link("panda_link7")

    franka.set_dofs_kp(4500, motors_dof)
    franka.set_dofs_kp(500, fingers_dof)
    franka.set_dofs_kv(450, motors_dof)
    franka.set_dofs_kv(50, fingers_dof)

    home_qpos = torch.zeros(franka.n_dofs, dtype=gs.tc_float)
    home_qpos[motors_dof] = torch.tensor(ARM_HOME_QPOS, dtype=gs.tc_float)
    franka.set_qpos(home_qpos)

    # Let the hammer settle into its actual resting pose before planning the grasp around it: its spawn
    # orientation is not necessarily a stable one.
    franka.control_dofs_position(home_qpos)
    for _ in range(100):
        scene.step()

    lower, upper = franka.get_dofs_limit(fingers_dof)
    closing_fractions = torch.empty(len(fingers_dof), dtype=gs.tc_float)
    for i in range(5):
        closing_fractions[4 * i] = THUMB_OPPOSE_FRACTION if i == 0 else FINGER_SPREAD_FRACTION
        closing_fractions[4 * i + 1 : 4 * i + 4] = FINGER_CURL_FRACTION
    finger_closing_target = lower + closing_fractions * (upper - lower)

    # Grasp point: centroid of the 5 fingertip links once the hand is closed to `finger_closing_target`,
    # expressed in the wrist link's local frame. This is where the handle should sit for the closed
    # fingertips to converge onto it.
    probe_qpos = home_qpos.clone()
    probe_qpos[fingers_dof] = finger_closing_target
    franka.set_qpos(probe_qpos)
    wrist_pos = wrist_link.get_pos(relative=False)
    wrist_quat = wrist_link.get_quat(relative=False)
    grasp_local_point = torch.stack(
        [
            inv_transform_by_trans_quat(franka.get_link(name).get_pos(relative=False), wrist_pos, wrist_quat)
            for name in FINGER_TIP_LINK_NAMES
        ]
    ).mean(dim=0)
    franka.set_qpos(home_qpos)

    handle_center_local = torch.tensor(HAMMER_HANDLE_CENTER_LOCAL, dtype=gs.tc_float)
    handle_axis_local = torch.tensor(HAMMER_HANDLE_AXIS_LOCAL, dtype=gs.tc_float)
    handle_center_world = transform_by_trans_quat(handle_center_local, hammer.get_pos(), hammer.get_quat())
    handle_axis_world = transform_by_quat(handle_axis_local, hammer.get_quat())

    # Grasp orientation: the wrist's row axis (spanning the 4 opposing fingers) is aligned with the handle,
    # approaching from directly above so the fingers close down around it.
    approach_dir_world = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float)
    row_axis_world = handle_axis_world - (handle_axis_world @ approach_dir_world) * approach_dir_world
    row_axis_world = row_axis_world / row_axis_world.norm()
    mid_axis_world = torch.linalg.cross(approach_dir_world, row_axis_world)
    grasp_quat = R_to_quat(torch.stack([row_axis_world, mid_axis_world, approach_dir_world], dim=-1))

    standoff = 0.12
    pregrasp_pos_world = handle_center_world - approach_dir_world * standoff

    # Display the grasping poses: the handle contact point, the staging pose the hand approaches from, and the
    # target pose its fingers close around. These stay drawn for the rest of the run.
    scene.draw_debug_sphere(handle_center_world, radius=GRASP_POINT_MARKER_RADIUS, color=GRASP_POINT_MARKER_COLOR)
    scene.draw_debug_frame(trans_quat_to_T(pregrasp_pos_world, grasp_quat), axis_length=GRASP_POSE_AXIS_LENGTH)
    scene.draw_debug_frame(trans_quat_to_T(handle_center_world, grasp_quat), axis_length=GRASP_POSE_AXIS_LENGTH)

    qpos_pregrasp = franka.inverse_kinematics(
        link=wrist_link,
        pos=pregrasp_pos_world,
        quat=grasp_quat,
        local_point=grasp_local_point,
    )
    qpos_pregrasp[fingers_dof] = 0.0

    path = franka.plan_path(qpos_goal=qpos_pregrasp, num_waypoints=100)
    path_debug = scene.draw_debug_path(path, franka)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        for _ in range(5):
            scene.step()
    scene.clear_debug_object(path_debug)
    for _ in range(100):
        scene.step()

    qpos_grasp = franka.inverse_kinematics(
        link=wrist_link,
        pos=handle_center_world,
        quat=grasp_quat,
        local_point=grasp_local_point,
    )
    franka.control_dofs_position(qpos_grasp[motors_dof], motors_dof)
    for _ in range(300):
        scene.step()

    franka.control_dofs_position(finger_closing_target, fingers_dof)
    for _ in range(150):
        scene.step()

    # Weld constraint: locks the hammer's current pose relative to the wrist so the grasp survives the lift
    # even where finger contact alone would not hold it.
    rigid = scene.sim.rigid_solver
    link_hammer = hammer.get_link("hammer_obj_baselink").idx
    rigid.add_weld_constraint(link_hammer, wrist_link.idx)

    # The lift keeps whatever orientation the wrist actually settled into after the grasp descent, rather than
    # the nominal `grasp_quat`, so the welded hammer only translates instead of swinging through the residual
    # orientation error.
    achieved_quat = wrist_link.get_quat(relative=False)

    lift_height = 0.2
    lift_pos_world = pregrasp_pos_world + approach_dir_world * (standoff - lift_height)
    qpos_lift = franka.inverse_kinematics(
        link=wrist_link,
        pos=lift_pos_world,
        quat=achieved_quat,
        local_point=grasp_local_point,
    )
    qpos_lift[fingers_dof] = finger_closing_target
    scene.draw_debug_frame(trans_quat_to_T(lift_pos_world, achieved_quat), axis_length=GRASP_POSE_AXIS_LENGTH)
    franka.control_dofs_position(qpos_lift)
    for _ in range(300):
        scene.step()


if __name__ == "__main__":
    main()
