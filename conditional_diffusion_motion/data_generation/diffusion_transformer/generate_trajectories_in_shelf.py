from typing import Tuple, List, Dict
import json
from pathlib import Path
import random

import numpy as np
import pinocchio as pin
import hppfcl
from conditional_diffusion_motion.utils.panda.visualizer import create_viewer, add_sphere_to_viewer
from conditional_diffusion_motion.utils.panda.params_parser import ParamParser
from conditional_diffusion_motion.utils.panda.ocp import OCP
from conditional_diffusion_motion.utils.panda.plan_and_optimize import PlanAndOptimize
from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda
from conditional_diffusion_motion.utils.panda.create_shelf import ShelfEnv
from conditional_diffusion_motion.utils.panda.panda_wrapper import robot_links


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
        vis, sphere_name="target", position=target_translation, radius=0.05, color=(1, 0, 0)
    )  # Display target as a red sphere
    for x in xs:
        q = x[: rmodel.nq]
        vis.display(q)
        input("Press Enter to continue...")
    print(f"Trajectory displayed for scene: {scene_name}")


def shrink_bounds_toward_mid(upper: np.ndarray, lower: np.ndarray, scale: float):
    assert 0 < scale <= 1
    mid = 0.5 * (upper + lower)
    up = mid + scale * (upper - mid)
    lo = mid + scale * (lower - mid)
    up = np.minimum(up, upper)
    lo = np.maximum(lo, lower)
    return up, lo


def generate_reachable_targets(
    rmodel: pin.Model,
    cmodel: pin.GeometryModel,
    vmodel: pin.GeometryModel,
    shelf_allocation: Dict[str, int],
    num_obstacles: int = 2,
    max_rrt_tries: int = 100,
    q_upper_bounds=None,
    q_lower_bounds=None,
):

    results = {}

    # Initialize the shelf environment
    shelf_env = ShelfEnv()

    # Add shelf to both robot and a new dummy model
    cmodel_shelf, vmodel_shelf, rmodel_shelf, cmodel, vmodel = shelf_env.create_model_with_shelf(cmodel, vmodel)

    # Collect all current objects (shelves)
    existing_objects = shelf_env.create_scene_objects()

    # Add n non-colliding small obstacles
    cmodel, cmodel_shelf = shelf_env.add_random_obstacles(cmodel, cmodel_shelf, num_obstacles)
    # Add collision pairs for shelf and obstacles
    shelf_env.add_collision_pairs_with_shelf(cmodel, robot_links)
    shelf_env.add_collision_pairs_with_obstacles(cmodel, robot_links)

    vis = create_viewer(rmodel_shelf, cmodel_shelf, vmodel_shelf)
    q = pin.neutral(rmodel_shelf)
    shelf_env.setup_cam(vis)
    vis.display(q)
    img = vis.viewer.get_image()
    img.save(f"generated_scenes/{scene_name}.png")

    rrt = PlanAndOptimize(rmodel, cmodel, "panda_hand_tcp", 5)

    q = np.array([-0.1882631, -1.25520716, -0.8290775, -0.56, -2.0212958, 0.38120324, 0.01])
    for key, count in shelf_allocation.items():
        mode_parts = key.split("_")
        if mode_parts[0] == "around":
            mode = "around_robot"
            shelf_level = None
        else:
            mode = "in_shelf"
            shelf_level = mode_parts[-1]  # 'bot', 'mid', or 'top'

        collected_targets = []
        collected_qs = []
        tries = 0
        print(f"\nGenerating {count} samples for '{key}'...")
        while len(collected_targets) < count and tries < max_rrt_tries:
            tries += 1
            try:
                t = shelf_env.generate_target_pose_not_in_obstacles(mode, shelf_level)
                qs_rrt = rrt.compute_traj_rrt(q, t, q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds)
                collected_targets.append(t)
                collected_qs.append(qs_rrt[-1])
                print(f"[{key}] {len(collected_targets)}/{count} success")
            except Exception as e:
                print(f"Error generating target for {key}: {e}")
                continue

        results[key] = {
            "targets": collected_targets,
            "final_q": collected_qs,
        }
    return results, cmodel, cmodel_shelf


def sample_trajectories_from_repartition(
    data: Dict[str, Dict[str, List[np.ndarray]]], traj_repartition: Dict[str, int]
) -> List[Tuple[np.ndarray, pin.SE3, str]]:
    result_pairs = []

    def get_qs_targets(space):
        return list(zip(data[space]["final_q"], data[space]["targets"]))

    def get_qs(space):
        return data[space]["final_q"]

    def get_targets(space):
        return data[space]["targets"]

    levels = ["bot", "mid", "top"]
    around_key = "around_robot"

    # SAME LEVEL TRAJS
    for lvl in levels:
        key = f"in_shelf_{lvl}"
        samples = get_qs_targets(key)
        available = len(samples) // 2
        requested = traj_repartition["same_level"]
        count = min(available, requested)

        if count < requested:
            print(f"[Warning] Not enough for same_level:{lvl}, requested {requested}, using {count}")

        for _ in range(count):
            q_entry, t_entry = random.sample(samples, 2)
            result_pairs.append((q_entry[0], t_entry[1], f"same_level_{lvl}"))

    # DIFFERENT LEVEL TRAJS
    for lvl_from in levels:
        for lvl_to in levels:
            if lvl_from == lvl_to:
                continue
            from_key = f"in_shelf_{lvl_from}"
            to_key = f"in_shelf_{lvl_to}"

            qs_from = get_qs(from_key)
            ts_to = get_targets(to_key)
            available = min(len(qs_from), len(ts_to))
            requested = traj_repartition["different_level"]
            count = min(available, requested)

            if count < requested:
                print(
                    f"[Warning] Not enough for different_level:{lvl_from}->{lvl_to}, requested {requested}, using {count}"
                )

            q_start_samples = random.sample(qs_from, count)
            t_goal_samples = random.sample(ts_to, count)
            for q, t in zip(q_start_samples, t_goal_samples):
                result_pairs.append((q, t, f"{lvl_from}_to_{lvl_to}"))

    # GOING OUT
    for lvl in levels:
        from_key = f"in_shelf_{lvl}"
        to_key = around_key

        qs = get_qs(from_key)
        ts = get_targets(to_key)
        count = min(len(qs), len(ts), traj_repartition["going_out"])

        if count < traj_repartition["going_out"]:
            print(f"[Warning] Not enough for going_out:{lvl}->around, using {count}")

        q_start_samples = random.sample(qs, count)
        t_goal_samples = random.sample(ts, count)
        for q, t in zip(q_start_samples, t_goal_samples):
            result_pairs.append((q, t, f"{lvl}_to_around"))

    # GOING IN
    for lvl in levels:
        from_key = around_key
        to_key = f"in_shelf_{lvl}"

        qs = get_qs(from_key)
        ts = get_targets(to_key)
        count = min(len(qs), len(ts), traj_repartition["going_in"])

        if count < traj_repartition["going_in"]:
            print(f"[Warning] Not enough for going_in:around->{lvl}, using {count}")

        q_start_samples = random.sample(qs, count)
        t_goal_samples = random.sample(ts, count)
        for q, t in zip(q_start_samples, t_goal_samples):
            result_pairs.append((q, t, f"around_to_{lvl}"))

    return result_pairs


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


if __name__ == "__main__":

    output_dir = Path("generated_scenes")
    output_dir.mkdir(exist_ok=True)

    shelf_allocation = {"in_shelf_bot": 30, "in_shelf_mid": 30, "in_shelf_top": 30, "around_robot": 10}
    traj_repartition = {
        "same_level": 10,
        "different_level": 10,
        "going_out": 5,
        "going_in": 5,
    }

    num_scenes: int = 300

    # Load configuration and models
    yaml_path = Path(__file__).parent / "config" / "scenes.yaml"
    pp: ParamParser = ParamParser(str(yaml_path), 2)

    results: List[Dict] = []
    scene_data: Dict = {}

    with progress:
        obstacle_task = progress.add_task("Generating scenes", total=num_scenes)
        traj_task = progress.add_task("  • Trajectories for current scene", total=120)

        for scene_idx in range(num_scenes):
            scene_name = f"scene_{scene_idx:04d}"

            progress.reset(traj_task)  # Reset trajectory progress for each obstacle

            print(f"Generating scene {scene_idx + 1}/{num_scenes}")
            # Fresh models for each scene
            rmodel, cmodel, vmodel = load_reduced_panda()
            rdata: pin.Data = rmodel.createData()

            shrink_factor = 0.8
            q_upper_bounds, q_lower_bounds = shrink_bounds_toward_mid(
                rmodel.upperPositionLimit, rmodel.lowerPositionLimit, shrink_factor
            )
            print(f"Using bounds: {q_lower_bounds} to {q_upper_bounds}")
            
            data, cmodel, cmodel_shelf = generate_reachable_targets(
                rmodel, cmodel, vmodel, shelf_allocation, num_obstacles=np.random.randint(1, 6), max_rrt_tries=100,
                q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds
            )
            
            # Store data
            obstacle_data = []
            for obj in cmodel_shelf.geometryObjects:
                # Assume `obj` is a dict with keys: type, transform (position), and shape
                if isinstance(obj.geometry, hppfcl.Box):
                    dim = {
                        "width": obj.geometry.halfSide[0] * 2,
                        "depth": obj.geometry.halfSide[1] * 2,
                        "height": obj.geometry.halfSide[2] * 2,
                    }
                obstacle_data.append({
                    "position": obj.placement.translation.tolist(),
                    "dimensions": dim,
                })

            scene_data[scene_name] = {
                "image_path": f"generated_scenes/{scene_name}.png",
                "obstacles": obstacle_data,
            }
            
            rrt = PlanAndOptimize(rmodel, cmodel, "panda_hand_tcp", pp.T)
            vis = create_viewer(rmodel, cmodel, vmodel)
            # Save the data to a JSON file
            trajectory_pairs = sample_trajectories_from_repartition(data, traj_repartition)

            for i, (q_start, target, mode) in enumerate(trajectory_pairs):
                try:
                    print(q_start)
                    ocp_creator = OCP(
                        rmodel, cmodel, TARGET_POSE=target, x0=np.concatenate((q_start, np.zeros(rmodel.nv))), pp=pp
                    )
                    ocp = ocp_creator.create_OCP()
                    print(f"Computing trajectory {i + 1}/{len(trajectory_pairs)}: {mode}")
                    xs, _ = rrt.compute_traj(q_start, target, ocp,  q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds)
                    # display_traj(vis, xs, target.translation, scene_name)
                    print(f"Trajectory {i + 1}/{len(trajectory_pairs)} computed successfully: {mode}")
                    result = store_results(xs, target.translation, rmodel)
                    result["scene"] = scene_name
                    result["mode"] = mode  # optional, but helpful
                    results.append(result)
                except Exception as e:
                    print(f"[Warning] Failed trajectory {i} ({mode}): {e}")

                progress.advance(traj_task)

            progress.advance(obstacle_task)  # Update obstacle progress
            del vis
            del rmodel, cmodel, vmodel

    with open(output_dir / "trajectories_data_shelf.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "scenes_data.json", "w") as f:
        json.dump(scene_data, f, indent=2)

    print(f"Saved {len(results)} trajectories to {output_dir / 'trajectories_data_shelf.json'}")
