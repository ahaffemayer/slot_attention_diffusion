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

from conditional_diffusion_motion.utils.panda.create_shelf import ShelfEnv
from conditional_diffusion_motion.utils.panda.visualizer import create_viewer

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = Path("generated_scenes")
OUTPUT_DIR.mkdir(exist_ok=True)

N_SCENES = 1000  # Number of scenes to generate
OBSTACLE_RANGE = (1, 6)  # Range for number of random obstacles

# -----------------------------
# Generate a single scene
# -----------------------------
def generate_shelf_only_scene(scene_idx: int):
    shelf_env = ShelfEnv()

    dummy_model = pin.Model()
    dummy_geom = pin.GeometryModel()
    dummy_visu = pin.GeometryModel()

    # Create scene with shelf
    cmodel_scene, vmodel_scene, rmodel_dummy, _, _ = shelf_env.create_model_with_shelf(dummy_geom, dummy_visu)

    # Add random number of obstacles
    num_obstacles = random.randint(*OBSTACLE_RANGE)
    cmodel_scene, _ = shelf_env.add_random_obstacles(cmodel_scene, cmodel_scene, num_obstacles)

    # Viewer
    vis = create_viewer(rmodel_dummy, cmodel_scene, vmodel_scene)
    shelf_env.setup_cam(vis)
    vis.display(np.zeros(rmodel_dummy.nq))

    # Save screenshot
    image = vis.viewer.get_image()
    image.save(OUTPUT_DIR / f"shelf_scene_{scene_idx:04d}.png")

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
        task = progress.add_task("Shelf scenes", total=N_SCENES)
        for idx in range(N_SCENES):
            generate_shelf_only_scene(idx)
            progress.advance(task)

    print(f"\n✅ Saved {N_SCENES} shelf scenes to: {OUTPUT_DIR.resolve()}")
