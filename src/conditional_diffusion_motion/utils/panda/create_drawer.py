from typing import Tuple, List, Dict
import numpy as np
import pinocchio as pin
import hppfcl
import random


class DrawerEnv:
    def __init__(self):
        self._obstacle_names = []
        self._obstacle_positions = []

    # ===============================================================
    # COLLISION PAIRS
    # ===============================================================

    def add_collision_pairs_with_boxes(self, cmodel, robot_links):
        if not self._obstacle_names:
            raise ValueError("Obstacle names not initialized, call add_random_obstacles first.")

        name_to_id = {geom.name: i for i, geom in enumerate(cmodel.geometryObjects)}

        for obs in self._obstacle_names:
            for link in robot_links:
                if obs not in name_to_id or link not in name_to_id:
                    print(f"[Warning] '{obs}' or '{link}' not in collision model.")
                    continue
                try:
                    cmodel.addCollisionPair(pin.CollisionPair(name_to_id[obs], name_to_id[link]))
                except Exception as e:
                    print(f"[Error] Could not add collision pair ({obs}, {link}): {e}")

    # ===============================================================
    # SCENE CONSTRUCTION
    # ===============================================================

    def create_scene_model(self):
        self._cmodel_scene = pin.GeometryModel()
        self._vmodel_scene = pin.GeometryModel()

        for obj in self.create_scene_objects():
            self._cmodel_scene.addGeometryObject(obj)
            self._vmodel_scene.addGeometryObject(obj)

        return self._cmodel_scene, self._vmodel_scene

    def setup_cam(self, vis):        
        
        roll, pitch, yaw = 0, -25, 135

        # Rotation matrix from roll, pitch, yaw
        R_roll  = pin.utils.rotate('x', roll)
        R_pitch = pin.utils.rotate('y', pitch)
        R_yaw   = pin.utils.rotate('z', yaw)
        
        # Combine rotations: R = Rz * Ry * Rx
    
        cam_pose = pin.SE3.Identity()
        cam_pose.translation = np.array([ 2.2,-0.5,-0.5])
        # cam_pose.rotation = np.eye(3)
        cam_pose.rotation = R_yaw @ R_pitch @ R_roll
        
        vis.viewer["/Cameras/default"].set_transform(np.array(cam_pose))

    def create_model_with_obstacles(self, cmodel_full, num_obstacles=3, for_slot = False) -> Tuple[pin.GeometryModel, pin.GeometryModel, pin.Model, pin.GeometryModel]:
        # Add shelf objects to robot’s models
        self.dummy_cmodel = pin.GeometryModel()
        self.dummy_vmodel = pin.GeometryModel()

        # Dummy robot for display
        rmodel_dummy = self._create_dummy_robot()
        
        # Create the random obstacles
        cmodel_full, self.dummy_cmodel = self.add_random_obstacles_for_slots(cmodel_full, self.dummy_cmodel, num_obstacles=num_obstacles)
        cmodel_full, self.dummy_cmodel = self.add_walls_and_table(cmodel_full, self.dummy_cmodel)

        return self.dummy_cmodel, self.dummy_vmodel, rmodel_dummy, cmodel_full

    @staticmethod
    def _create_dummy_robot():
        model = pin.Model()
        jid = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root_joint")
        model.appendBodyToJoint(jid, pin.Inertia.Random(), pin.SE3.Identity())
        model.addFrame(pin.Frame("base_link", jid, 0, pin.SE3.Identity(), pin.FrameType.BODY))
        return model

    # ===============================================================
    # OBSTACLE GENERATION
    # ===============================================================

    def add_walls_and_table(self, cmodel_full, dummy_cmodel):
        obstacles = {
            "obstacle1": {
                "dim": [1.2, 0.7, 0.3],
                "translation": [0.3, 0.0, -0.1],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "color": [0.5, 0.5, 0.5, 1.0]
            },
            "obstacle2": {
                "dim": [0.5, 0.6, 0.03],
                "translation": [0.65, 0.0, 0.8],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "color": [0.5, 0.5, 0.5, 1.0]
            },
            "obstacle3": {
                "dim": [0.03, 0.6, 0.75],
                "translation": [0.9, 0.0, 0.42],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "color": [0.5, 0.5, 0.5, 1.0]
            },
            "obstacle4": {
                "dim": [0.5, 0.03, 0.75],
                "translation": [0.65, 0.3, 0.42],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "color": [0.5, 0.5, 0.5, 1.0]
            },
            "obstacle5": {
                "dim": [0.5, 0.03, 0.75],
                "translation": [0.65, -0.3, 0.42],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "color": [0.5, 0.5, 0.5, 1.0]
            }
        }

        for name, params in obstacles.items():
            size_x, size_y, size_z = params["dim"]
            shape = hppfcl.Box(size_x, size_y, size_z)

            quat = params["orientation"]
            R = pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()

            pose = pin.SE3(
                R,
                np.array(params["translation"])
            )

            obj = pin.GeometryObject(name, 0, 0, pose, shape)
            obj.meshColor = np.array(params["color"])

            cmodel_full.addGeometryObject(obj)
            dummy_cmodel.addGeometryObject(obj)
            self._record_obstacle(obj)

        return cmodel_full, dummy_cmodel


    def _record_obstacle(self, geom_obj: pin.GeometryObject):
        self._obstacle_names.append(geom_obj.name)
        self._obstacle_positions.append(np.copy(geom_obj.placement.translation))


    def add_random_obstacles_for_slots(self, cmodel_full, dummy_cmodel, num_obstacles):
        """
        Generate random obstacles within given boundaries and ensure they do NOT
        collide with the robot base links. Retries until a valid sample is found.
        """
        robot_base_links = [
            "panda_link0_capsule_0",
            "panda_link1_capsule_0",
            "panda_link2_capsule_0",
            "panda_link3_capsule_0",
        ]
        accepted_obstacles = []
        if num_obstacles <= 0:
            print("[Warning] No obstacles to add.")
            return cmodel_full, dummy_cmodel
        if num_obstacles > 3:
            raise ValueError("Too many obstacles, keep it <= 3.")

        box_dim = [0.03, 0.6,0.07]
        x_fixed, y_fixed = 0.4, 0 
        z_min, z_max = 0.1, 0.7
        max_tries = 50  # Allow enough retries
        colors = [
            [1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1],
            [1,0,1,1], [0,1,1,1],
        ]
        colors_available = colors.copy()

        # Pre-fetch IDs for speed and safety
        link_ids = {}
        for link in robot_base_links:
            if not cmodel_full.existGeometryName(link):
                print(f"[Warning] Link '{link}' not found in geometry model.")
                continue
            link_ids[link] = cmodel_full.getGeometryId(link)

        for i in range(num_obstacles):

            # Selecting the color
            color = random.choice(colors_available)
            colors_available.remove(color)

            obstacle_obj = None

            for _ in range(max_tries):

                # Position
                z = random.uniform(z_min, z_max)
                pos = np.array([x_fixed, y_fixed, z])
                pose = pin.SE3(np.eye(3), pos)

                # Shape
                shape = hppfcl.Box(*box_dim)

                name = f"slot_obstacle_{i}"
                obj = pin.GeometryObject(name, 0, 0, pose, shape)
                obj.meshColor = np.array(color)

                # Check collision against base links
                collision = False
                for link, gid in link_ids.items():
                    print(f"Checking collision between obstacle {name} and link {link}")
                    collision_geom = cmodel_full.geometryObjects[gid]

                    req = hppfcl.CollisionRequest()
                    res = hppfcl.CollisionResult()

                    r1 = hppfcl.CollisionObject(
                        collision_geom.geometry, 
                        pin_to_fcl(collision_geom.placement)
                    )
                    r2 = hppfcl.CollisionObject(
                        obj.geometry,
                        pin_to_fcl(obj.placement)
                    )

                    hppfcl.collide(r1, r2, req, res)
                    if res.isCollision():
                        collision = True
                        break

                if not collision:
                    obstacle_obj = obj
                    for prev_obj in accepted_obstacles:
                        req2 = hppfcl.CollisionRequest()
                        res2 = hppfcl.CollisionResult()

                        r_prev = hppfcl.CollisionObject(prev_obj.geometry,
                                                        pin_to_fcl(prev_obj.placement))
                        r_new  = hppfcl.CollisionObject(obj.geometry,
                                                        pin_to_fcl(obj.placement))

                        hppfcl.collide(r_prev, r_new, req2, res2)

                        if res2.isCollision():
                            collision = True
                            break

                    print(f"[Info] Placed obstacle {i} at {pos} after {_+1} tries.")
                    break  # got a valid sample

            if obstacle_obj is None:
                print(f"[Warning] Could not place obstacle {i} after {max_tries} tries.")
                continue

            # Add obstacle to both collision and visual models
            cmodel_full.addGeometryObject(obstacle_obj)
            dummy_cmodel.addGeometryObject(obstacle_obj)
            accepted_obstacles.append(obstacle_obj.copy())
            # Save metadata
            self._record_obstacle(obstacle_obj)

        return cmodel_full, dummy_cmodel


def pin_to_fcl(pose: pin.SE3) -> hppfcl.Transform3f:
    return hppfcl.Transform3f(pose.rotation, pose.translation)


if __name__ == "__main__":
    # Simple test of the environment creation
    from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda, robot_links
    from conditional_diffusion_motion.utils.panda.visualizer import create_viewer
    from conditional_diffusion_motion.utils.panda.params_parser import ParamParser
    from pathlib import Path
    import pinocchio as pin

    scene = 7

    yaml_path = "/home/arthur/Desktop/Code/slot_attention_diffusion/ressources/shelf_example/config/scenes.yaml"

    rmodel, cmodel, vmodel = load_reduced_panda()


    pp = ParamParser(str(yaml_path), scene)

    # cmodel = pp.add_collisions(rmodel, cmodel)
    print("Number of collision pairs:", len(cmodel.collisionPairs))   
    drawer_env = DrawerEnv()
    cmodel_shelf, vmodel_shelf, rmodel_shelf, cmodel = drawer_env.create_model_with_obstacles(cmodel, num_obstacles=3, for_slot=True)
    drawer_env.add_collision_pairs_with_boxes(cmodel, robot_links)

    # vis = create_viewer(rmodel, cmodel, vmodel)
    vis = create_viewer(rmodel_shelf, cmodel_shelf, vmodel_shelf)
    drawer_env.setup_cam(vis)

    # robot_data = rmodel.createData()
    # collision_data = cmodel.createData()

    # pin.framesForwardKinematics(rmodel, robot_data, pin.randomConfiguration(rmodel))
    # for cp in cmodel.collisionPairs:
    #     print(cp)

    vis.display(pin.randomConfiguration(rmodel))
