import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from conditional_diffusion_motion.utils.panda.cspace_helpers import (
    load_scene_from_json,
    build_geometry_model_from_fcl_objects,
    check_collision,
    add_collision_pairs_with_obstacles,
)

if __name__ == "__main__":
    from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda, robot_links
    from conditional_diffusion_motion.occupancy_grid.occupancy_grid import OccupancyGrid3D, populate_grid_from_scene
    from conditional_diffusion_motion.cspace.cspace import CSpace
    from rich.progress import Progress

    rmodel, cmodel, vmodel = load_reduced_panda()
    scene_path = Path("/home/arthur/Desktop/Code/slot_attention_diffusion/ressources/table_example/generated_scenes/box_for_diffusion/scenes_data.json")
    configurations_path = Path("/home/arthur/Desktop/Code/slot_attention_diffusion/ressources/table_example/cspace/c_space_configuration_vectors/cspace_configurations.npy")
    configurations = np.load(configurations_path, allow_pickle=True)

    # occupancy_tensor_path =  f""
    cspace_path = f"/home/arthur/Desktop/Code/slot_attention_diffusion/ressources/table_example/processed_datasets/indirect_motion_with_obstacles_cache_with_shelf_cspace.npy"

    tensors_grid_results = []
    cspace_results = []
    with open(scene_path, "r") as f:
        all_scenes = json.load(f)

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing scenes...", total=len(all_scenes))

        for scene_name in sorted(all_scenes.keys()):
            progress.update(task, advance=1, description=f"[cyan]Scene: {scene_name}")
            print(f"Processing scene: {scene_name}")
            
            # Reload base model for clean state
            rmodel, cmodel, vmodel = load_reduced_panda()

            fcl_obs, list_of_obstacles = load_scene_from_json(scene_path, scene_name)
            obs_model = build_geometry_model_from_fcl_objects(fcl_obs, list_of_obstacles)
            for geom in obs_model.geometryObjects:
                cmodel.addGeometryObject(geom)
            add_collision_pairs_with_obstacles(cmodel, robot_links)
            q = pin.neutral(rmodel)
            # grid = OccupancyGrid3D(
            #     x_min=0.4, x_max=1.2,
            #     y_min=-0.8, y_max=0.8,
            #     z_min=-0.2, z_max=1.0,
            #     resolution=0.05
            # )

            # tensors_grid_results.append(grid.generate_binary_occupancy_tensor(collision_model=cmodel))

            cspace = CSpace(rmodel=rmodel, cmodel=cmodel)
            cspace_result = []
            for i in range(len(configurations)):
                q = configurations[i]
                in_collision = check_collision(rmodel, cmodel, q)
                # if collision = 1 else collision = 0
                if not in_collision:
                    cspace_result.append(0)
                else:
                    cspace_result.append(1)
            cspace_results.append(cspace_result)
            print(f"Scene {scene_name} processed. {len(cspace_result) - sum(cspace_result)} / {len(cspace_result)} valid configurations.")


    # np.save(occupancy_tensor_path,tensors_grid_results, allow_pickle=True) 
    np.save(cspace_path, cspace_results, allow_pickle=True)
    
    # print(f"Occupancy grid tensors saved to {occupancy_tensor_path}")
    print(f"C-space validity vectors saved to {cspace_path}")