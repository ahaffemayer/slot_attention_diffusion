from pathlib import Path
import numpy as np
import random

import pinocchio as pin
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from conditional_diffusion_motion.utils.panda.create_boxes_wall_env import BoxEnv
from conditional_diffusion_motion.utils.panda.visualizer import create_viewer
from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda, robot_links
from conditional_diffusion_motion.utils.panda.params_parser import ParamParser

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = Path("/home/arthur/Desktop/Code/slot_attention_diffusion/ressources/table_example/generated_scenes/table_scenes")
OUTPUT_DIR.mkdir(exist_ok=True)

N_SCENES = 1000  # Number of scenes to generate
OBSTACLE_RANGE = (1, 4)  # Range for number of random obstacles

# -----------------------------
# Generate a single scene
# -----------------------------
def generate_shelf_only_scene(scene_idx: int):

    scene = 6

    yaml_path = "/home/arthur/Desktop/Code/slot_attention_diffusion/ressources/shelf_example/config/scenes.yaml"

    rmodel, cmodel, vmodel = load_reduced_panda()


    pp = ParamParser(str(yaml_path), scene)
    box_env = BoxEnv()
    cmodel_shelf, vmodel_shelf, rmodel_shelf, cmodel = box_env.create_model_with_obstacles(cmodel, num_obstacles=np.random.randint(*OBSTACLE_RANGE), for_slot=True)

    vis = create_viewer(rmodel, cmodel_shelf, vmodel_shelf)
    box_env.setup_cam(vis)

    vis.display(pin.randomConfiguration(rmodel))

    # Save screenshot
    image = vis.viewer.get_image()
    image.save(OUTPUT_DIR / f"table_scene_{scene_idx:04d}.png")

    del vis  # Explicitly free viewer

# -----------------------------
# Main Execution with Rich Progress
# -----------------------------
if __name__ == "__main__":
    progress = Progress(
        TextColumn("[bold green]Generating scenes"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task("Table scenes", total=N_SCENES)
        for idx in range(N_SCENES):
            generate_shelf_only_scene(idx)
            progress.advance(task)

    print(f"\n✅ Saved {N_SCENES} table scenes to: {OUTPUT_DIR.resolve()}")
