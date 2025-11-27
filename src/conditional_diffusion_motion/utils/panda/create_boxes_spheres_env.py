from typing import Tuple, List, Dict
import numpy as np
import pinocchio as pin
import hppfcl
import random


class SphereEnv:
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

        
        
        roll, pitch, yaw = 0, 90, 0

        # Rotation matrix from roll, pitch, yaw
        R_roll  = pin.utils.rotate('x', roll)
        R_pitch = pin.utils.rotate('y', pitch)
        R_yaw   = pin.utils.rotate('z', yaw)
        
        # Combine rotations: R = Rz * Ry * Rx
    
        cam_pose = pin.SE3.Identity()
        cam_pose.translation = np.array([ 0,0,2])
        # cam_pose.rotation = np.eye(3)
        cam_pose.rotation = R_yaw @ R_pitch @ R_roll
        
        vis.viewer["/Cameras/default"].set_transform(np.array(cam_pose))

    def create_model_with_obstacles(self, cmodel_full, num_obstacles=4) -> Tuple[pin.GeometryModel, pin.GeometryModel, pin.Model, pin.GeometryModel]:
        # Add shelf objects to robot’s models
        self.dummy_cmodel = pin.GeometryModel()
        self.dummy_vmodel = pin.GeometryModel()

        # Dummy robot for display
        rmodel_dummy = self._create_dummy_robot()
        
        # Create the random obstacles
        cmodel_full, self.dummy_cmodel = self.add_random_obstacles(cmodel_full, self.dummy_cmodel, num_obstacles=num_obstacles)

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

    def _record_obstacle(self, geom_obj: pin.GeometryObject):
        self._obstacle_names.append(geom_obj.name)
        self._obstacle_positions.append(np.copy(geom_obj.placement.translation))

    def add_random_obstacles(self, cmodel_full, dummy_cmodel, num_obstacles):
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
        if num_obstacles > 4:
            raise ValueError("Too many obstacles, keep it <= 4.")

        self._obstacle_names = []
        self._obstacle_positions = []

        box_dim = [0.05, 0.7, 1.2]
        z_fixed = 0.67
        xy_min, xy_max = -0.6, 0.6
        max_tries = 50  # Allow enough retries

        colors = [
            [1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1],
            [1,0,1,1], [0,1,1,1],
        ]

        # Pre-fetch IDs for speed and safety
        link_ids = {}
        for link in robot_base_links:
            if not cmodel_full.existGeometryName(link):
                print(f"[Warning] Link '{link}' not found in geometry model.")
                continue
            link_ids[link] = cmodel_full.getGeometryId(link)

        for i in range(num_obstacles):

            obstacle_obj = None

            for _ in range(max_tries):

                # Position
                x = random.uniform(xy_min, xy_max)
                y = random.uniform(xy_min, xy_max)
                pos = np.array([x, y, z_fixed])

                # Rotation around X axis
                angle = random.uniform(0, 2*np.pi)
                s, c = np.sin(angle / 2), np.cos(angle / 2)
                quat = np.array([0.0, 0.0, s, c])  # xyzw

                rot = pin.Quaternion(quat).toRotationMatrix()
                pose = pin.SE3(rot, pos)

                # Shape
                shape = hppfcl.Box(*box_dim)

                name = f"slot_obstacle_{i}"
                obj = pin.GeometryObject(name, 0, 0, pose, shape)
                obj.meshColor = np.array(colors[i % len(colors)])

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
    from conditional_diffusion_motion.utils.panda.create_boxes_spheres_env import SphereEnv
    import pinocchio as pin

    rmodel, cmodel, vmodel = load_reduced_panda()
    sphere_env = SphereEnv()
    cmodel_shelf, vmodel_shelf, rmodel_shelf, cmodel = sphere_env.create_model_with_obstacles(cmodel, num_obstacles=4)
    sphere_env.add_collision_pairs_with_boxes(cmodel, robot_links)

    vis = create_viewer(rmodel, cmodel, vmodel)
    sphere_env.setup_cam(vis)

    robot_data = rmodel.createData()
    collision_data = cmodel.createData()

    pin.framesForwardKinematics(rmodel, robot_data, pin.randomConfiguration(rmodel))

    vis.display(pin.randomConfiguration(rmodel))
    input("Press Enter to continue...")
    camera = vis.viewer["/Cameras/default"]

    # Get the current transform (4x4 matrix)
    transform = camera.get_transform()  # returns a 4x4 numpy array

    # Extract translation
    position = transform[:3, 3]