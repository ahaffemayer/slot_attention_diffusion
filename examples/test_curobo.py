#!/usr/bin/env python3
"""
franka_cuboids_example.py

Minimal example:
 - create a small world with cuboid obstacles
 - build MotionGen for franka
 - warm up
 - get a valid start state from MotionGen's retract config
 - create a nearby goal pose (shifted end-effector pose)
 - plan a single collision-aware trajectory
 - print result and interpolated trajectory shape
"""

import torch
import numpy as np
from pathlib import Path
import pinocchio as pin
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.util_file import (
    join_path,
    load_yaml,
    get_robot_configs_path,
    get_world_configs_path,
)
from curobo.rollout.cost.pose_cost import PoseCostMetric

from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda
from conditional_diffusion_motion.utils.panda.params_parser import ParamParser
from conditional_diffusion_motion.utils.panda.visualizer import (
    create_viewer,
    add_sphere_to_viewer,
)

from parser_collision_model import parser_collision_model


def get_device_args():
    # choose CUDA if available, otherwise CPU
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return TensorDeviceType(device=dev)


def main():
    tensor_args = get_device_args()
    print("Using device:", tensor_args.device)

    robot_model, collision_model, visual_model = load_reduced_panda()

    data_dir = Path(__file__).parent.parent / "ressources" / "shelf_example"

    param_path = data_dir / "config" / "scenes.yaml"

    pp = ParamParser(str(param_path), 5)

    cmodel = pp.add_collisions(robot_model, collision_model)
    vis = create_viewer(robot_model, cmodel, visual_model)

    # q0 = pin.randomConfiguration(robot_model)
    q0 = np.array(
        [0.6337731, 0.6594915, 0.16988713, -1.0447954, -2.6341944, 1.9979668, 2.7248776]
    )
    vis.display(q0)
    # --- simple world with cuboids ---
    # dims are x,y,z extents, pose is [x,y,z, qw,qx,qy,qz]
    world_config = parser_collision_model(cmodel)
    goal_pos = torch.tensor(
        # [0.6, 0.5, 0.6], device=tensor_args.device
        [0.5, -0.4, 0.5], device=tensor_args.device
    )  # alternative goal


    add_sphere_to_viewer(
        vis, "goal_sphere", 0.02, goal_pos.cpu().numpy(), color=0xFF0000
    )

    # meshes are optional, omitted here
    reach_vec_weight = tensor_args.to_device([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    reach_vec_weight = tensor_args.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    pose_metric = PoseCostMetric(
        reach_partial_pose=True,
        reach_full_pose=False,  # do not enforce full pose
        reach_vec_weight=reach_vec_weight,
    )

    # --- build MotionGen config for Franka, primitive collision checking ---
    robot_file = "franka.yml"  # assumes this file is in your curobo robot configs
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_config,
        tensor_args,
        interpolation_dt=0.02,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        num_trajopt_seeds=12,
        num_ik_seeds=200,
    )

    # --- create MotionGen and warm it up ---
    motion_gen = MotionGen(motion_gen_cfg)
    motion_gen.warmup()  # warms GPU kernels, IK solvers, etc

    # --- get a safe starting joint state from the retract config ---
    retract_cfg = motion_gen.get_retract_config()
    retract_cfg = torch.tensor(q0, device=tensor_args.device, dtype=torch.float32)
    # retract_cfg = torch.randn_like(retract_cfg) * 0.05 + retract_cfg
    # retract_cfg is a tensor of joint positions, shape (n_seeds, nq) typically
    start_state = JointState.from_position(retract_cfg.view(1, -1))

    # --- compute current end-effector pose and create a shifted goal ---
    kin_state = motion_gen.rollout_fn.compute_kinematics(start_state)
    ee_pos = kin_state.ee_pos_seq.squeeze().detach()  # (3,) or (1,3)
    ee_quat = kin_state.ee_quat_seq.squeeze().detach()  # (4,)

    # create goal pose by shifting the EE a bit in x, keep same orientation
    # place on same device as tensor_args
    # goal_pos = ee_pos + torch.tensor([0.15, 0.0, 0.0], device=tensor_args.device)
    goal_quat = ee_quat
    goal_pose = Pose(position=goal_pos.unsqueeze(0), quaternion=goal_quat.unsqueeze(0))

    # --- plan a single collision-aware trajectory ---
    plan_cfg = MotionGenPlanConfig(
        max_attempts=3, timeout=6.0, pose_cost_metric=pose_metric
    )
    result = motion_gen.plan_single(start_state, goal_pose, plan_cfg)

    # --- report results ---
    print("Plan success:", bool(result.success))
    print("Status:", result.status)
    print("Attempts:", int(result.attempts) if hasattr(result, "attempts") else "n/a")
    print(
        "Solve time (s):",
        float(result.solve_time) if hasattr(result, "solve_time") else "n/a",
    )

    if bool(result.success):
        interp = result.get_interpolated_plan()  # trajectory object
        print("Interpolated plan dt:", result.interpolation_dt)
        # show some shapes and the first few joint positions
        pos = interp.position.cpu().numpy()  # shape (T, 7)
        print("Trajectory timesteps, dims:", pos.shape)
        print("First timestep joints:", pos[0])
        print("Last timestep joints:", pos[-1])
    else:
        # optional debug info if available
        if hasattr(result, "debug_info"):
            print("Debug info keys:", list(result.debug_info.keys()))
        print(
            "Planning failed, consider more seeds, different goal, or inspect world configuration."
        )

    add_sphere_to_viewer(
        vis, "start_sphere", 0.02, ee_pos.cpu().numpy(), color=0x00FF00
    )

    for q in interp:
        vis.display(q.position.cpu().numpy())
        print(q.position.cpu().numpy())
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()
