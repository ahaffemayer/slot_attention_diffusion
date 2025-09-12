import pinocchio
import hppfcl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


class OccupancyGrid3D:
    """
    A 3D occupancy grid (voxel grid) with two methods for checking occupancy:
    1. AABB-based (fast, coarse)
    2. Precise collision checking (slower, accurate)
    """

    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, resolution):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.resolution = resolution
        self.nx = math.ceil((x_max - x_min) / resolution)
        self.ny = math.ceil((y_max - y_min) / resolution)
        self.nz = math.ceil((z_max - z_min) / resolution)
        print(f"Grid initialized: {self.nx} x {self.ny} x {self.nz} voxels.")

    def world_to_grid(self, wx, wy, wz):
        """Converts world coordinates to grid cell indices (ix, iy, iz)."""
        wx_clamped = np.clip(wx, self.x_min, self.x_max - 1e-9)
        wy_clamped = np.clip(wy, self.y_min, self.y_max - 1e-9)
        wz_clamped = np.clip(wz, self.z_min, self.z_max - 1e-9)
        
        ix = int((wx_clamped - self.x_min) / self.resolution)
        iy = int((wy_clamped - self.y_min) / self.resolution)
        iz = int((wz_clamped - self.z_min) / self.resolution)
        return ix, iy, iz

    def grid_to_world_center(self, ix, iy, iz):
        """Gets the world coordinates of a voxel's center."""
        wx = self.x_min + ix * self.resolution + self.resolution / 2.0
        wy = self.y_min + iy * self.resolution + self.resolution / 2.0
        wz = self.z_min + iz * self.resolution + self.resolution / 2.0
        return wx, wy, wz

    def check_and_get_occupied_cells_aabb(self, geom_obj):
        """
        Fast but coarse check based on the object's 3D AABB.
        This is our 'broad phase'.
        """
        collision_obj = hppfcl.CollisionObject(geom_obj.geometry, geom_obj.placement)
        collision_obj.computeAABB()
        aabb = collision_obj.getAABB()
        min_corner = aabb.min_
        max_corner = aabb.max_

        if (
            max_corner[0] < self.x_min or min_corner[0] > self.x_max or
            max_corner[1] < self.y_min or min_corner[1] > self.y_max or
            max_corner[2] < self.z_min or min_corner[2] > self.z_max
        ):
            return False, []

        min_ix, min_iy, min_iz = self.world_to_grid(min_corner[0], min_corner[1], min_corner[2])
        max_ix, max_iy, max_iz = self.world_to_grid(max_corner[0], max_corner[1], max_corner[2])

        occupied_cells = []
        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                for iz in range(min_iz, max_iz + 1):
                    occupied_cells.append((ix, iy, iz))

        return True, occupied_cells

    def check_and_get_occupied_cells_accurate(self, geom_obj):
        """
        Slow but precise check using hppfcl.collide.
        This is our 'narrow phase'.
        """
        # 1. BROAD PHASE: Get candidate cells from the fast AABB check
        is_candidate, candidate_cells = self.check_and_get_occupied_cells_aabb(geom_obj)
        if not is_candidate:
            return False, []

        # 2. NARROW PHASE: Perform precise collision check for each candidate voxel
        accurate_occupied_cells = []

        # Create the shape for a grid cell (voxel) once
        cell_shape = hppfcl.Box(self.resolution, self.resolution, self.resolution)

        # HPP-FCL collision request and result objects
        req = hppfcl.CollisionRequest()
        res = hppfcl.CollisionResult()

        for ix, iy, iz in candidate_cells:
            # Get the center of the current voxel
            wx, wy, wz = self.grid_to_world_center(ix, iy, iz)
            cell_placement = pinocchio.SE3(np.eye(3), np.array([wx, wy, wz]))

            # Perform the precise collision check
            res.clear()
            is_colliding = hppfcl.collide(
                geom_obj.geometry, geom_obj.placement,
                cell_shape, cell_placement,
                req, res
            )

            if is_colliding:
                accurate_occupied_cells.append((ix, iy, iz))

        return len(accurate_occupied_cells) > 0, accurate_occupied_cells
    

    def generate_binary_occupancy_tensor(self, collision_model):
        """
        Returns a 3D binary tensor of shape (nx, ny, nz) where:
        - 1 indicates an occupied voxel
        - 0 indicates a free voxel

        Args:
            collision_model (pinocchio.CollisionModel): The scene to voxelize.

        Returns:
            np.ndarray: Binary occupancy grid (shape: nx x ny x nz)
        """
        # Initialize the tensor with zeros
        occupancy_tensor = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)

        print(f"Generating binary occupancy tensor from collision model with {len(collision_model.geometryObjects)} objects...")
        
        for i, geom_obj in enumerate(collision_model.geometryObjects):
            print(f"  Processing object {i+1}/{len(collision_model.geometryObjects)}: {geom_obj.name}...", end="")

            is_occupied, occupied_cells = self.check_and_get_occupied_cells_accurate(geom_obj)

            if is_occupied:
                print(f" marking {len(occupied_cells)} voxels.")
                for ix, iy, iz in occupied_cells:
                    occupancy_tensor[ix, iy, iz] = 1
            else:
                print(" no occupied voxels.")
        
        print("Binary occupancy tensor generation complete.")
        return occupancy_tensor

    
    def __len__(self):
        """Returns the total number of voxels in the grid."""
        return self.nx * self.ny * self.nz

    def shape(self):
        """Returns the dimensions of the grid as a tuple (nx, ny, nz)."""
        return (self.nx, self.ny, self.nz)

def visualize_grid_3d(grid, geom_obj, occupied_cells):
    """Visualizes the 3D grid, the object's AABB, and occupied voxels."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # A more robust way to plot voxels is to create a full color map
    # Initialize a transparent color map for the entire grid
    colors = np.zeros((grid.nx, grid.ny, grid.nz, 4)) # RGBA format

    # Initialize a boolean array for the voxel structure
    voxel_data = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)

    # Set the color and structure for occupied cells
    if occupied_cells:
        green_rgba = [0, 1, 0, 0.7]  # Green with 70% opacity
        for ix, iy, iz in occupied_cells:
            voxel_data[ix, iy, iz] = True
            colors[ix, iy, iz] = green_rgba

        # Plot using the boolean structure and the color map.
        # CRUCIALLY, we do not specify an edgecolor.
        ax.voxels(voxel_data, facecolors=colors)

    # Draw the object's AABB (this part remains the same)
    collision_obj = hppfcl.CollisionObject(geom_obj.geometry, geom_obj.placement)
    collision_obj.computeAABB()
    aabb = collision_obj.getAABB()
    min_c, max_c = aabb.min_, aabb.max_

    pts = np.array([
        [min_c[0], min_c[1], min_c[2]], [max_c[0], min_c[1], min_c[2]],
        [max_c[0], max_c[1], min_c[2]], [min_c[0], max_c[1], min_c[2]],
        [min_c[0], min_c[1], max_c[2]], [max_c[0], min_c[1], max_c[2]],
        [max_c[0], max_c[1], max_c[2]], [min_c[0], max_c[1], max_c[2]]
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
        (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        ax.plot3D(pts[[i, j], 0], pts[[i, j], 1], pts[[i, j], 2], color="r", linewidth=2)

    ax.set_title("3D Occupancy Grid Visualization (Accurate Check)")
    ax.set_xlabel("X-coordinate (m)"), ax.set_ylabel("Y-coordinate (m)"), ax.set_zlabel("Z-coordinate (m)")
    ax.set_xlim(grid.x_min, grid.x_max), ax.set_ylim(grid.y_min, grid.y_max), ax.set_zlim(grid.z_min, grid.z_max)
    
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))


    red_line = plt.Line2D([0], [0], color="r", lw=2, label="Object's 3D AABB")
    green_patch = patches.Patch(color='green', alpha=0.7, label='Occupied Voxels')
    ax.legend(handles=[red_line, green_patch])

    plt.show()

def populate_grid_from_scene(grid, collision_model):
    """
    Populates the grid by voxelizing every object in a Pinocchio CollisionModel.

    Args:
        grid (OccupancyGrid3D): The grid instance to populate.
        collision_model (pinocchio.CollisionModel): The scene containing all geometric objects.

    Returns:
        list: A list of unique (ix, iy, iz) tuples for all occupied voxels.
    """
    all_occupied_cells = set()
    num_geoms = len(collision_model.geometryObjects)
    print(f"Starting to populate grid from scene with {num_geoms} geometry objects...")

    for i, geom_obj in enumerate(collision_model.geometryObjects):
        # We can ignore the robot links if they are in this model, 
        # as we are only interested in the static environment.
        # This check depends on how the scene was constructed.
        # Assuming all objects in cmodel_shelf are obstacles.
        print(f"  Processing object {i+1}/{num_geoms}: {geom_obj.name}...", end="")
        
        is_occupied, occupied_cells = grid.check_and_get_occupied_cells_accurate(geom_obj)
        
        if is_occupied:
            print(f" found {len(occupied_cells)} voxels.")
            all_occupied_cells.update(occupied_cells)
        else:
            print(" outside grid or no collision.")

    print(f"\nFinished populating grid. Total unique occupied voxels: {len(all_occupied_cells)}")
    return list(all_occupied_cells)

def visualize_voxelized_scene(grid, occupied_cells):
    """
    Visualizes the populated grid using ax.scatter, which is more robust than ax.voxels.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    print("Preparing visualization using robust scatter plot method...")

    if occupied_cells:
        world_points = np.array([grid.grid_to_world_center(ix, iy, iz) for ix, iy, iz in occupied_cells])
        
        ax.scatter(
            world_points[:, 0],  # All X coordinates
            world_points[:, 1],  # All Y coordinates
            world_points[:, 2],  # All Z coordinates
            c='blue',
            alpha=0.6,
            marker='s',  # 's' for square marker, looks like a cube
            s=25         # Marker size, may need tuning depending on voxel resolution
        )

    ax.set_title("Voxelized Scene from Collision Model (Scatter Method)")
    ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)"), ax.set_zlabel("Z (m)")
    ax.set_xlim(grid.x_min, grid.x_max)
    ax.set_ylim(grid.y_min, grid.y_max)
    ax.set_zlim(grid.z_min, grid.z_max)
    
    # Ensure the aspect ratio is equal to prevent distortion
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
    
    blue_patch = patches.Patch(color='blue', alpha=0.6, label='Occupied Scene Voxels')
    ax.legend(handles=[blue_patch])
    
    # Add a grid for better spatial perception
    ax.grid(True)

    plt.show()


# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    
    import pinocchio
    import random
    
    from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda, robot_links
    from conditional_diffusion_motion.utils.panda.create_shelf import ShelfEnv
    
    # Load the reduced Panda model
    rmodel, cmodel, vmodel = load_reduced_panda()
    # Create the shelf environment
    shelf_env = ShelfEnv()
    
    cmodel_shelf, vmodel_shelf, rmodel_shelf, cmodel, vmodel = shelf_env.create_model_with_shelf(cmodel, vmodel)
    existing_objects = shelf_env.create_scene_objects()
    cmodel, cmodel_shelf = shelf_env.add_random_obstacles(cmodel, cmodel_shelf, random.randint(1, 5))
    shelf_env.add_collision_pairs_with_shelf(cmodel, robot_links)
    shelf_env.add_collision_pairs_with_obstacles(cmodel, robot_links)
    
    # 1. Define the 3D Occupancy Grid
    grid = OccupancyGrid3D(
        x_min=0.4, x_max=1.2,
        y_min=-0.8, y_max=0.8,
        z_min=-0.2, z_max=1.0,
        resolution=0.05  # Using a slightly larger resolution for 3D visualization clarity
    )

    print(f"Length of grid: {len(grid)} voxels.")
    print(f"Grid shape: {grid.shape()} (nx, ny, nz)")
    
    # 3. Populate the grid using our new function
    occupied_voxels = populate_grid_from_scene(grid, cmodel_shelf)
    
    print(grid.generate_binary_occupancy_tensor(cmodel_shelf).shape)
    
    # 4. Visualize the entire voxelized scene
    if occupied_voxels:
        visualize_voxelized_scene(grid, occupied_voxels)
    else:
        print("No occupied voxels were found in the scene.")

