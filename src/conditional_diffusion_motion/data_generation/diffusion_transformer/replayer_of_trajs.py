import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import time 
from conditional_diffusion_motion.utils.panda.cspace_helpers import (
    load_scene_from_json,
    build_geometry_model_from_fcl_objects,
    check_collision,
    add_collision_pairs_with_obstacles,
)

if __name__ == "__main__":
    from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda, robot_links
    from conditional_diffusion_motion.occupancy_grid.occupancy_grid import OccupancyGrid3D, populate_grid_from_scene
    from conditional_diffusion_motion.utils.panda.visualizer import create_viewer, add_sphere_to_viewer
    from conditional_diffusion_motion.cspace.cspace import CSpace
    from rich.progress import Progress

    rmodel, cmodel, vmodel = load_reduced_panda()
    scene_path = Path("/home/arthur/Desktop/Code/slot_attention_diffusion/generated_scenes/scenes_data.json")
    configurations_path = Path("/home/arthur/Desktop/Code/slot_attention_diffusion/generated_scenes/trajectories_data_shelf.json")
    with open(configurations_path, "r") as f:
        configurations = json.load(f)
    occupancy_tensor_path =  f""
    cspace_path = f""

    tensors_grid_results = []
    cspace_results = []
    with open(scene_path, "r") as f:
        all_scenes = json.load(f)

    # Build mapping: scene_name -> list of configs
    configs_by_scene = {}
    for cfg in configurations:
        sname = cfg["scene"]  # already "scene_0000"
        if sname not in configs_by_scene:
            configs_by_scene[sname] = []
        configs_by_scene[sname].append(cfg)

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing scenes...", total=len(all_scenes))

        for scene_name in sorted(all_scenes.keys()):
            progress.update(task, advance=1, description=f"[cyan]Scene: {scene_name}")
            print(f"Processing scene: {scene_name}")

            rmodel, cmodel, vmodel = load_reduced_panda()

            fcl_obs, list_of_obstacles = load_scene_from_json(scene_path, scene_name)
            obs_model = build_geometry_model_from_fcl_objects(fcl_obs, list_of_obstacles)
            for geom in obs_model.geometryObjects:
                cmodel.addGeometryObject(geom)
            add_collision_pairs_with_obstacles(cmodel, robot_links)

            vis = create_viewer(rmodel, cmodel, vmodel)

            # Retrieve all configs belonging to this scene
            scene_configs = configs_by_scene.get(scene_name, [])

            for config in scene_configs:
                add_sphere_to_viewer(vis,"goal" , 0.02,config["target"], color=0xFF0000)

                # input("press enter to see the initial configuration...")
                q0 = np.array(config["q0"])
                # vis.display(q0)

                # input("press enter to see the final configuration...")
                qfinal = np.array(config["qfinal"])
                # vis.display(qfinal)

                input("press enter to see the trajectory...")
                traj = np.array(config["trajectory"])
                for q in traj:
                    vis.display(q)
                    # input("Press Enter to continue...")
                    time.sleep(0.03)
            input("Press Enter to continue... (new scene)")
