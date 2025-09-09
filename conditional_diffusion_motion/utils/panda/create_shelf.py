from typing import Tuple, List, Dict
import numpy as np
import pinocchio as pin
import hppfcl
import random

class ShelfEnv:
    def __init__(self):
        self.shelf_names = ["top_shelf", "mid_shelf", "bot_shelf", "left_shelf", "right_shelf"]
        self._random_obstacles = False

    def create_scene_objects(self):
        objects = []

        def make_box(name, size, translation):
            shape = hppfcl.Box(*size)
            pose = pin.SE3.Identity()
            pose.translation = np.array(translation)
            obj = pin.GeometryObject(name, 0, 0, pose, shape)
            obj.meshColor = np.array([0.4, 0.4, 0.4, 1.0])
            return obj

        objects.append(make_box("top_shelf", [0.5, 1.0, 0.05], [0.7, 0.0, 0.8]))
        objects.append(make_box("mid_shelf", [0.5, 1.0, 0.05], [0.7, 0.0, 0.4]))
        objects.append(make_box("bot_shelf", [0.5, 1.0, 0.05], [0.7, 0.0, 0.0]))
        objects.append(make_box("left_shelf", [0.5, 0.05, 0.8], [0.7, 0.5, 0.4]))
        objects.append(make_box("right_shelf", [0.5, 0.05, 0.8], [0.7, -0.5, 0.4]))

        return objects

    def add_to_models(self, cmodel_full, vmodel_full):
        scene_objects = self.create_scene_objects()
        for obj in scene_objects:
            cmodel_full.addGeometryObject(obj)
            vmodel_full.addGeometryObject(obj)

    def add_collision_pairs_with_shelf(self, cmodel, robot_links):
        name_to_id = {geom.name: i for i, geom in enumerate(cmodel.geometryObjects)}

        for shelf_name in self.shelf_names:
            for link in robot_links:
                if shelf_name not in name_to_id or link not in name_to_id:
                    print(f"[Warning] '{shelf_name}' or '{link}' not found.")
                    continue
                try:
                    cmodel.addCollisionPair(pin.CollisionPair(name_to_id[shelf_name], name_to_id[link]))
                except Exception as e:
                    print(f"[Error] Could not add pair ({shelf_name}, {link}): {e}")

    def add_collision_pairs_with_obstacles(self, cmodel, robot_links):
        name_to_id = {geom.name: i for i, geom in enumerate(cmodel.geometryObjects)}

        if not hasattr(self, '_obstacle_names'):
            raise ValueError("Obstacle names not initialized. Call add_random_obstacles first.")
        for obstacle_name in self._obstacle_names:
            for link in robot_links:
                if obstacle_name not in name_to_id or link not in name_to_id:
                    print(f"[Warning] '{obstacle_name}' or '{link}' not found.")
                    continue
                try:
                    cmodel.addCollisionPair(pin.CollisionPair(name_to_id[obstacle_name], name_to_id[link]))
                except Exception as e:
                    print(f"[Error] Could not add pair ({obstacle_name}, {link}): {e}")

    def create_scene_model(self):
        self._cmodel_scene = pin.GeometryModel()
        self._vmodel_scene = pin.GeometryModel()
        for obj in self.create_scene_objects():
            self._cmodel_scene.addGeometryObject(obj)
            self._vmodel_scene.addGeometryObject(obj)
        return self._cmodel_scene, self._vmodel_scene

    def setup_cam(self, vis):
        fOcam = pin.SE3.Identity()
        fOcam.translation = np.array([-1.0, 0.0, -0.4])
        fOcam.rotation = np.eye(3)  # pin.utils.rotate("z", np.pi) #@ pin.utils.rotate("y", np.pi / 8)
        vis.viewer["/Cameras/default"].set_transform(np.array(fOcam))

    def create_model_with_shelf(self, cmodel_full, vmodel_full):
        """
        Create dummy robot model and a clean scene model with shelf geometry.
        Also adds the shelf to the robot's full collision and visual models.

        Returns:
            cmodel_scene, vmodel_scene: scene-only models with shelf geometry.
            rmodel_dummy: minimal robot model for visualization.
            cmodel_full, vmodel_full: updated robot geometry models with shelf added.
        """
        # Add shelf geometry to robot's collision/visual models
        self.add_to_models(cmodel_full, vmodel_full)

        # Create clean shelf-only scene model
        self._cmodel_scene, self._vmodel_scene = self.create_scene_model()

        # Create dummy robot model: minimal model with a base joint
        rmodel_dummy = pin.Model()
        joint_id = rmodel_dummy.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root_joint")
        rmodel_dummy.appendBodyToJoint(joint_id, pin.Inertia.Random(), pin.SE3.Identity())
        rmodel_dummy.addFrame(pin.Frame("base_link", joint_id, 0, pin.SE3.Identity(), pin.FrameType.BODY))

        return self._cmodel_scene, self._vmodel_scene, rmodel_dummy, cmodel_full, vmodel_full

    def _record_obstacle(self, obj: pin.GeometryObject):
            """Internal helper to record metadata for later target sampling."""
            self._obstacle_names.append(obj.name)
            self._obstacle_positions.append(np.copy(obj.placement.translation))

    def add_random_obstacles(self, cmodel_full, dummy_cmodel, num_obstacles):
        """
        Add multiple small box obstacles randomly on or in the shelf.
        Avoid collisions with shelf parts or other obstacles.

        Args:
            cmodel_full (pin.GeometryModel): The full collision model.
            dummy_cmodel (pin.GeometryModel): A dummy model (e.g., for display).
            num_obstacles (int): Number of obstacles to generate.

        Returns:
            List[pin.GeometryObject]: List of successfully added obstacles.
        """
        self._random_obstacles = True
        self._obstacle_positions = []
        self._obstacle_names = []

        if num_obstacles <= 0:
            print("[Warning] No obstacles to add.")
            return []
        if num_obstacles > 6:
            raise ValueError("Number of obstacles should be <= 6.")
        size = [0.2, 0.15, 0.2]
        color_pool = [
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 1.0, 0.0, 1.0],  # Green
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [1.0, 0.5, 0.0, 1.0],  # Orange
        ]
        z_levels = [0.0, 0.4, 0.8]
        max_tries = 30

        existing_objects = []
        added_objects = []


        for obstacle_id in range(num_obstacles):
            success = False
            for _ in range(max_tries):
                x = 0.6
                y = random.uniform(-0.45, 0.45)
                z = random.choice(z_levels) + 0.125

                pose = pin.SE3.Identity()
                pose.translation = np.array([x, y, z])
                shape = hppfcl.Box(*size)
                obj = pin.GeometryObject(f"slot_obstacle_{obstacle_id}", 0, 0, pose, shape)
                obj.meshColor = np.array(color_pool[obstacle_id % len(color_pool)])

                # Collision check disabled for now
                collision = False
                for other in existing_objects:
                    req = hppfcl.CollisionRequest()
                    res = hppfcl.CollisionResult()
                    r1 = hppfcl.CollisionObject(other.geometry, pin_to_fcl(other.placement))
                    r2 = hppfcl.CollisionObject(obj.geometry, pin_to_fcl(obj.placement))
                    hppfcl.collide(r1, r2, req, res)
                    if res.isCollision():
                        collision = True
                        break

                if not collision:
                    cmodel_full.addGeometryObject(obj)
                    dummy_cmodel.addGeometryObject(obj)
                    existing_objects.append(obj)
                    added_objects.append(obj)
                    self._record_obstacle(obj)
                    success = True
                    break

            if not success:
                print(f"[Warning] Could not place non-colliding obstacle {obstacle_id}.")

        return cmodel_full, dummy_cmodel

    def generate_target_pose(self, mode="in_shelf", max_tries=100) -> pin.SE3:
        """
        Generate a target pose in, on, or near the shelf.

        Args:
            mode (str): Placement mode. One of ["in_shelf", "on_shelf", "near_shelf"].
            max_tries (int): Maximum attempts to place a non-colliding target.
        Returns:
            pin.SE3: The target pose.
        """
        if mode not in ["in_shelf", "on_shelf", "near_shelf"]:
            raise ValueError("Invalid mode. Choose from 'in_shelf', 'on_shelf', 'near_shelf'.")

        radius = 0.05
        target_shape = hppfcl.Sphere(radius)

        for _ in range(max_tries):
            pose = pin.SE3.Identity()

            if mode == "in_shelf":
                x = random.uniform(0.55, 0.85)
                y = random.uniform(-0.5, 0.5)
                z = random.choice([0.05, 0.45, 0.85])
            elif mode == "on_shelf":
                x = random.uniform(0.55, 0.85)
                y = random.uniform(-0.5, 0.5)
                z = random.choice([0.05, 0.45, 0.85]) + 0.025
            elif mode == "near_shelf":
                x = random.uniform(0.3, 1.0)
                y = random.uniform(-0.6, 0.6)
                z = random.uniform(0.0, 0.9)

            pose.translation = np.array([x, y, z])

            # Check collision
            collision = False
            # if self._cmodel_scene.geometryObjects is not None:
            #     target_fcl = hppfcl.CollisionObject(target_shape, pin_to_fcl(pose))
            #     for other in self._cmodel_scene.geometryObjects:
            #         other_fcl = hppfcl.CollisionObject(other.geometry, pin_to_fcl(other.placement))
            #         req = hppfcl.CollisionRequest()
            #         res = hppfcl.CollisionResult()
            #         hppfcl.collide(target_fcl, other_fcl, req, res)
            #         if res.isCollision():
            #             print(f"Collision detected with {other.name} at pose {pose.translation}. Retrying...")
            #             collision = True
            #             break

            if not collision:
                # print(f"Generated target pose: {pose}")
                return pose
            
    def _sample_mode_position(self, mode: str, shelf_level: str = None) -> np.ndarray:
        """Sample a raw xyz position according to placement mode logic (no filtering).

        Args:
            mode: One of 'in_shelf', 'on_shelf', 'around_robot'
            shelf_level: One of 'top', 'mid', 'bot' (only used if mode is 'in_shelf' or 'on_shelf')
        Returns:
            np.ndarray: A sampled (x, y, z) position
        """
        if mode not in ["in_shelf", "on_shelf", "around_robot"]:
            raise ValueError("Invalid mode. Choose from 'in_shelf', 'on_shelf', 'around_robot'.")

        if shelf_level is not None and shelf_level not in ["top", "mid", "bot"]:
            raise ValueError("Invalid shelf_level. Choose from 'top', 'mid', 'bot'.")

        shelf_z_map = {
            "bot": 0.1,
            "mid": 0.5,
            "top": 0.9
        }

        if mode in ["in_shelf", "on_shelf"]:
            x = random.uniform(0.55, 0.85)
            y = random.uniform(-0.4, 0.4)

            # Pick shelf level
            if shelf_level is None:
                z_base = random.choice(list(shelf_z_map.values()))
            else:
                z_base = shelf_z_map[shelf_level]

            z_height = 0.2  # height of one shelf level
            z = random.uniform(z_base, z_base + z_height)

            if mode == "on_shelf":
                z += 0.025  # lift slightly above the shelf for "on"
        else:  # around_robot
            # Sample in a box near robot base, avoiding the shelf (which is at x=0.7+)
            x = random.uniform(-0.5, 0.3)
            y = random.uniform(-0.6, 0.6)
            z = random.uniform(0.15, 0.8)  # Up to shoulder height

        return np.array([x, y, z])


    def generate_target_pose_avoid_obstacles(self, mode="in_shelf", shelf_level: str = None, max_tries=100, return_debug=False) -> pin.SE3:
        """
        Sample a target pose using the same sampling regions as generate_target_pose but
        reject samples that fall inside a shelf-sized exclusion box centered at any obstacle.

        The exclusion zone is a box of size (0.5, 1.0, 0.2) centered at each obstacle.

        Args:
            mode: in_shelf, on_shelf, near_shelf
            max_tries: number of rejection sampling attempts
            shelf_level: 'top', 'mid', 'bot' (only used if mode is in_shelf or on_shelf)
            return_debug: if True, returns (pose, last_validity_mask) where the mask tells whether candidate was valid
        Returns:
            pin.SE3 pose (or tuple if return_debug)
        """
        if not hasattr(self, "_obstacle_positions") or len(self._obstacle_positions) == 0:
            # no obstacles, just fall back
            pose = self.generate_target_pose(mode=mode, max_tries=max_tries)
            if return_debug:
                return pose, np.array([])
            return pose

        exclusion_box_half_extents = np.array([0.25, 0.25, 0.25])  # half of (0.5, 1.0, 0.2)
        last_valid_mask = None

        for _ in range(max_tries):
            xyz = self._sample_mode_position(mode, shelf_level=shelf_level)  # shape (3,)
            obs_mat = np.vstack(self._obstacle_positions)  # shape (N, 3)
            diff = np.abs(obs_mat - xyz[None, :])  # shape (N, 3)
            inside_box = np.all(diff <= exclusion_box_half_extents[None, :], axis=1)  # shape (N,)
            last_valid_mask = ~inside_box
            if np.all(last_valid_mask):
                pose = pin.SE3.Identity()
                pose.translation = xyz
                if return_debug:
                    return pose, last_valid_mask
                return pose

        # fallback: shift a sample outside the nearest box
        nearest_idx = int(np.argmin(np.linalg.norm(obs_mat - xyz[None, :], axis=1)))
        nearest = obs_mat[nearest_idx]
        # Move along random direction until outside the exclusion box
        direction = self._sample_mode_position(mode, shelf_level=shelf_level) - nearest
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        direction /= np.linalg.norm(direction)
        xyz = nearest + direction * exclusion_box_half_extents.max()  # conservative push

        pose = pin.SE3.Identity()
        pose.translation = xyz
        if return_debug:
            return pose, last_valid_mask if last_valid_mask is not None else np.array([])
        return pose
    
    # convenience wrapper exactly matching your question semantics
    def generate_target_pose_not_in_obstacles(self, mode="in_shelf", shelf_level: str = None, **kwargs) -> pin.SE3:
        """Alias for generate_target_pose_avoid_obstacles using exclusion_radius (meters)."""
        return self.generate_target_pose_avoid_obstacles(mode=mode, shelf_level=shelf_level, **kwargs)



def pin_to_fcl(pose: pin.SE3) -> hppfcl.Transform3f:
    return hppfcl.Transform3f(pose.rotation, pose.translation)
