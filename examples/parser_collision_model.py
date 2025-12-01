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
import hppfcl

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


def parser_collision_model(collision_model) -> dict:
    """Returns a dict of the like:
        world_config = {
        "cuboid": {
            "table": {
                "dims": [2.0, 2.0, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0],
            },
            "obs_1": {
                "dims": [0.2, 0.2, 0.4],
                "pose": [0.4, 0.0, 0.2, 1, 0, 0, 0],
            },
            "obs_2": {
                "dims": [0.15, 0.15, 0.3],
                "pose": [0.2, -0.15, 0.15, 1, 0, 0, 0],
            },
        },

    }

    Args:
        collision_model (_type_): _description_
    """
    world_config = {
        "cuboid": {},
    }

    for obj in collision_model.geometryObjects:
        if isinstance(obj.geometry, hppfcl.Box):
            name = obj.name
            pose_se3 = obj.placement
            xyzquat = pin.SE3ToXYZQUAT(pose_se3)
            translation = [float(v) for v in xyzquat[:3]]
            rotation = [float(v) for v in xyzquat[3:]]
            qx, qy, qz, qw = rotation # Pinocchio uses (qx, qy, qz, qw) order
            world_config["cuboid"][name] = {
                "dims": [float(v) for v in (obj.geometry.halfSide * 2)],
                "pose": [
                    translation[0],
                    translation[1],
                    translation[2],
                    float(qw),
                    float(qx),
                    float(qy),
                    float(qz),
                ],
            }

    return world_config


if __name__ == "__main__":

    data_dir = Path(__file__).parent.parent / "ressources" / "shelf_example"

    param_path = data_dir / "config" / "scenes.yaml"

    pp = ParamParser(str(param_path), 5)

    # Load your robot model
    robot_model, collision_model, visual_model = load_reduced_panda()

    # Initialize the shelf environment
    # shelf_env = ShelfEnv()
    cmodel = pp.add_collisions(robot_model, collision_model)
    world_config = parser_collision_model(cmodel)
    print(world_config)
