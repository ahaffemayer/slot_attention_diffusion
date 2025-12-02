from typing import Tuple, List, Dict
import json
from pathlib import Path
import random

import torch
import numpy as np
import pinocchio as pin
import hppfcl
from conditional_diffusion_motion.utils.panda.visualizer import (
    create_viewer,
    add_sphere_to_viewer,
)
from conditional_diffusion_motion.utils.panda.params_parser import ParamParser
from conditional_diffusion_motion.utils.panda.panda_wrapper import (
    load_reduced_panda,
    robot_links,
)
from conditional_diffusion_motion.utils.panda.create_boxes_wall_env import BoxEnv

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


from conditional_diffusion_motion.utils.panda.parser_collision_model import (
    parser_collision_model,
)


from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# ---------------------------
# Progress Bar Configuration
# ---------------------------
progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("\u2022"),
    TimeElapsedColumn(),
    TextColumn("\u2022"),
    TimeRemainingColumn(),
)


def get_device_args():
    # choose CUDA if available, otherwise CPU
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return TensorDeviceType(device=dev)


def display_traj(
    vis,
    xs: List[np.ndarray],
    target_translation: np.ndarray,
    scene_name: str = "scene",
):
    """Display the trajectory in the viewer."""

    q = pin.neutral(rmodel)
    vis.display(q)
    add_sphere_to_viewer(
        vis,
        sphere_name="target",
        position=target_translation,
        radius=0.05,
        color=(1, 0, 0),
    )  # Display target as a red sphere
    for x in xs:
        q = x[: rmodel.nq]
        vis.display(q)
        input("Press Enter to continue...")
    print(f"Trajectory displayed for scene: {scene_name}")


def store_results(xs: List[np.ndarray], target: np.ndarray, rmodel: pin.Model) -> Dict:
    """Converts trajectory data and metadata into a dictionary format for saving."""
    q0 = xs[0][: rmodel.nq]
    qfinal = xs[-1][: rmodel.nq]

    return {
        "target": target.tolist(),
        "q0": q0.tolist(),
        "qfinal": qfinal.tolist(),
        "trajectory": [X[:7].tolist() for X in xs],
    }


def create_robot_models(param_path: Path, n_scene: int, n_obstacles: int) -> Tuple[
    pin.Model,
    pin.GeometryModel,
    pin.GeometryModel,
    pin.GeometryModel,
    pin.GeometryModel,
    pin.Model,
    BoxEnv,
]:
    """Create and return robot and environment models with obstacles."""

    rmodel, cmodel, vmodel = load_reduced_panda()

    pp = ParamParser(str(param_path), n_scene)
    env = BoxEnv()
    cmodel_box, vmodel_box, rmodel_box, cmodel = env.create_model_with_obstacles(
        cmodel, num_obstacles=n_obstacles
    )
    env.add_collision_pairs_with_boxes(cmodel, robot_links)
    return rmodel, cmodel, vmodel, cmodel_box, vmodel_box, rmodel_box, env


def take_picture_of_scene(
    rmodel,
    cmodel_box,
    vmodel_box,
    box_env: BoxEnv,
    scene_name: str,
    output_dir: Path,
):
    """Take a picture of the current scene in the viewer."""
    image_path = output_dir / f"{scene_name}.png"
    vis = create_viewer(
        rmodel,
        cmodel_box,
        vmodel_box,
    )
    box_env.setup_cam(vis)

    vis.display(pin.randomConfiguration(rmodel))

    # Save screenshot
    image = vis.viewer.get_image()
    image.save(image_path)


def generate_collision_free_configuration(
    rmodel: pin.Model,
    cmodel: pin.GeometryModel,
    max_tries: int = 1000,
) -> np.ndarray:
    """Generate a random collision-free configuration within given bounds."""
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    for _ in range(max_tries):
        q = pin.randomConfiguration(rmodel)
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        if not pin.computeCollisions(cmodel, cdata, stop_at_first_collision=True):
            return q
    raise RuntimeError("Failed to find a collision-free configuration.")


def generate_reachable_target(
    rmodel: pin.Model,
    cmodel: pin.GeometryModel,
    max_tries: int = 100,
) -> List[pin.SE3]:
    """Generate reachable target poses on the table surface."""
    rdata = rmodel.createData()
    q = generate_collision_free_configuration(rmodel, cmodel, max_tries)
    pin.framesForwardKinematics(rmodel, rdata, q)
    ee_frame_id = rmodel.getFrameId("panda_hand_tcp")
    return rdata.oMf[ee_frame_id]


def create_motion_gen_curobo(
    cmodel: pin.GeometryModel,
) -> MotionGen:
    """Create and return a curobo MotionGen instance for the Franka robot."""
    world_config = parser_collision_model(cmodel)

    # --- build MotionGen config for Franka, primitive collision checking ---
    robot_file = "franka.yml"  # assumes this file is in your curobo robot configs
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_config,
        tensor_args,
        interpolation_dt=0.02,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        num_trajopt_seeds=4,
        num_ik_seeds=50,
    )
    # --- create MotionGen and warm it up ---
    motion_gen = MotionGen(motion_gen_cfg)
    # motion_gen.warmup()  # warms GPU kernels, IK solvers, etc

    return motion_gen


def create_motion_gen_plan_config():

    reach_vec_weight = tensor_args.to_device([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    reach_vec_weight = tensor_args.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    pose_metric = PoseCostMetric(
        reach_partial_pose=True,
        reach_full_pose=False,  # do not enforce full pose
        reach_vec_weight=reach_vec_weight,
    )
    plan_cfg = MotionGenPlanConfig(
        max_attempts=3, timeout=6.0, pose_cost_metric=pose_metric
    )
    return plan_cfg


def plan_with_curobo(
    motion_gen: MotionGen,
    q_start: np.ndarray,
    target_pose: pin.SE3,
    plan_cfg: MotionGenPlanConfig,
) -> List[np.ndarray]:
    """Plan a trajectory using curobo's motion generation. Normally here we have a SE3 but only the position matters.
    1. Run create_motion_gen_curobo to create the motion_gen
    2. Run create_motion_gen_plan_config to create the plan_cfg
    3. Call this function with the motion_gen, start configuration, target pose, and plan_cfg
    Returns the interpolated trajectory as a list of numpy arrays."""

    q_start_torch = torch.tensor(
        q_start, device=tensor_args.device, dtype=torch.float32
    )
    start_state = JointState.from_position(q_start_torch.view(1, -1))

    goal = pin.SE3ToXYZQUAT(target_pose)
    goal_quat = goal[-4:]
    w, x, y, z = goal_quat[3], goal_quat[0], goal_quat[1], goal_quat[2]
    goal_pose_curobo = Pose(
        position=torch.tensor(
            target_pose.translation, device=tensor_args.device
        , dtype=torch.float32).unsqueeze(0),
        quaternion=torch.tensor([w, x, y, z], device=tensor_args.device, dtype=torch.float32).unsqueeze(
            0
        ),  # Different convention for quaternions
    )

    result = motion_gen.plan_single(start_state, goal_pose_curobo, plan_cfg)
    if bool(result.success):
        interp = result.get_interpolated_plan()  # trajectory object
        # show some shapes and the first few joint positions
        return interp
    else:
        raise RuntimeError("Curobo planning failed.")


def resample_trajectory(q_traj: np.ndarray, T: int) -> np.ndarray:
    """
    q_traj: ndarray of shape (N, dof)
    T: desired number of timesteps
    Returns an ndarray of shape (T, dof)
    """
    N = q_traj.shape[0]
    dof = q_traj.shape[1]

    if N == T:
        return q_traj.copy()

    # Time indices of original and new trajectories
    original_idx = np.linspace(0, 1, N)
    target_idx = np.linspace(0, 1, T)

    q_resampled = np.zeros((T, dof), dtype=q_traj.dtype)

    for j in range(dof):
        q_resampled[:, j] = np.interp(target_idx, original_idx, q_traj[:, j])

    # Ensure exact matching of start and end
    q_resampled[0] = q_traj[0]
    q_resampled[-1] = q_traj[-1]

    assert q_resampled.shape == (T, dof)
    return q_resampled


if __name__ == "__main__":
    tensor_args = get_device_args()
    print("Using device:", tensor_args.device)

    output_dir = Path("generated_scenes")
    output_dir.mkdir(exist_ok=True)

    num_scenes: int = 300
    num_trajectories_per_scene: int = 50

    n_obstacles_range = (1, 3)  # Min and max number of obstacles per scene

    n_scene = 6

    display = False  # Whether to display the viewer

    # Load configuration and models
    yaml_path = "/home/arthur/Desktop/Code/slot_attention_diffusion/ressources/shelf_example/config/scenes.yaml"
    pp: ParamParser = ParamParser(str(yaml_path), n_scene)

    results: List[Dict] = []
    scene_data: Dict = {}

    with progress:
        obstacle_task = progress.add_task("Generating scenes", total=num_scenes)
        traj_task = progress.add_task(
            "  • Trajectories for current scene", total=num_trajectories_per_scene
        )

        for scene_idx in range(num_scenes):
            scene_name = f"scene_{scene_idx:04d}"

            progress.reset(traj_task)  # Reset trajectory progress for each obstacle

            print(f"Generating scene {scene_idx + 1}/{num_scenes}")
            # Fresh models for each scene

            rmodel, cmodel, vmodel, cmodel_box, vmodel_box, rmodel_box, env = (
                create_robot_models(
                    yaml_path,
                    n_scene,
                    n_obstacles=random.randint(*n_obstacles_range),
                )
            )

            # Take a picture of the scene for the slot attention algorithm
            take_picture_of_scene(
                rmodel,
                cmodel_box,
                vmodel_box,
                env,
                scene_name,
                output_dir,
            )

            # Storing the data about the obstacles in the scene
            obstacle_data = []
            for obj in cmodel_box.geometryObjects:
                # Assume `obj` is a dict with keys: type, transform (position), and shape
                if isinstance(obj.geometry, hppfcl.Box):
                    dim = {
                        "width": obj.geometry.halfSide[0] * 2,
                        "depth": obj.geometry.halfSide[1] * 2,
                        "height": obj.geometry.halfSide[2] * 2,
                    }
                obstacle_data.append(
                    {
                        "position": obj.placement.translation.tolist(),
                        "dimensions": dim,
                    }
                )

            scene_data[scene_name] = {
                "image_path": f"generated_scenes/{scene_name}.png",
                "obstacles": obstacle_data,
            }

            # Curobo motion generator
            motion_gen = create_motion_gen_curobo(cmodel)
            plan_cfg = create_motion_gen_plan_config()

            if display:
                vis = create_viewer(rmodel, cmodel, vmodel)

            n_failed = 0
            for i in range(num_trajectories_per_scene):
                try:
                    q_start = generate_collision_free_configuration(
                        rmodel,
                        cmodel,
                    )
                    target = generate_reachable_target(
                        rmodel,
                        cmodel,
                    )
                    traj = plan_with_curobo(motion_gen, q_start, target, plan_cfg)

                    resampled_traj = resample_trajectory(
                        traj.position.cpu().numpy(), T=pp.T
                    )
                    xs = [resampled_traj[t] for t in range(resampled_traj.shape[0])]
                    if display:
                        display_traj(vis, xs, target.translation, scene_name)
                        input()
                    # print(
                    #     f"Trajectory {i + 1}/{num_trajectories_per_scene} computed successfully."
                    # )
                    result = store_results(xs, target.translation, rmodel)
                    result["scene"] = scene_name
                    results.append(result)
                except Exception as e:
                    print(f"[Warning] Failed trajectory {i + 1}: {e}")
                    n_failed += 1
                progress.advance(traj_task)

            del motion_gen
            torch.cuda.empty_cache()
            print(
                f"Scene {scene_idx + 1} done with {n_failed} failed trajectories."
            )
            progress.advance(obstacle_task)  # Update obstacle progress
            if display:
                del vis
            del rmodel, cmodel, vmodel

    with open(output_dir / "trajectories_data_shelf.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "scenes_data.json", "w") as f:
        json.dump(scene_data, f, indent=2)

    print(
        f"Saved {len(results)} trajectories to {output_dir / 'trajectories_data_shelf.json'}"
    )
