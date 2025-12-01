import hppfcl
import numpy as np
import pinocchio as pin
import yaml
class ParamParser:
    def __init__(self, path: str, scene: int):
        self.path = path
        self.params = None
        self.scene = scene

        with open(self.path) as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.data = self.params["scene" + str(self.scene)]

    @staticmethod
    def _parse_obstacle_shape(shape: str, size: list):
        if shape == "box":
            return hppfcl.Box(*size)
        elif shape == "sphere":
            return hppfcl.Sphere(size[0])
        elif shape == "cylinder":
            return hppfcl.Cylinder(size[0], size[1])
        elif shape == "ellipsoid":
            return hppfcl.Ellipsoid(*size)
        elif shape == "capsule":
            return hppfcl.Capsule(*size)
        else:
            raise ValueError(f"Unknown shape {shape}")

    def _add_ellipsoid_on_robot(self, rmodel: pin.Model, cmodel: pin.GeometryModel):
        """Add ellipsoid on the robot model"""
        if "ROBOT_ELLIPSOIDS" in self.data:
            for ellipsoid in self.data["ROBOT_ELLIPSOIDS"]:
                rob_hppfcl = hppfcl.Ellipsoid(
                    *self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["dim"]
                )
                idf_rob = rmodel.getFrameId(
                    self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["parentFrame"]
                )
                print(idf_rob)
                idj_rob = rmodel.frames[idf_rob].parentJoint
                if (
                    "translation" in self.data["ROBOT_ELLIPSOIDS"][ellipsoid]
                    and "orientation" in self.data["ROBOT_ELLIPSOIDS"][ellipsoid]
                ):
                    rot_mat = (
                        pin.Quaternion(
                            *tuple(
                                self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["orientation"]
                            )
                        )
                        .normalized()
                        .toRotationMatrix()
                    )
                    Mrob = pin.SE3(
                        rot_mat,
                        np.array(
                            self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["translation"]
                        ),
                    )
                else:
                    Mrob = rmodel.frames[idf_rob].placement
                rob_geom = pin.GeometryObject(
                    ellipsoid, idj_rob, idf_rob, Mrob, rob_hppfcl
                )
                rob_geom.meshColor = np.r_[1, 1, 0, 0.7]
                cmodel.addGeometryObject(rob_geom)
        return cmodel

    def add_collisions(self, rmodel: pin.Model, cmodel: pin.GeometryModel):
        """Add collisions to the robot model"""
        cmodel = self._add_ellipsoid_on_robot(rmodel, cmodel)
        for obs in self.data["OBSTACLES"]:
            obs_hppfcl = self._parse_obstacle_shape(
                self.data["OBSTACLES"][obs]["type"], self.data["OBSTACLES"][obs]["dim"]
            )
            Mobs = pin.SE3(
                pin.Quaternion(*tuple(self.data["OBSTACLES"][obs]["orientation"]))
                .normalized()
                .toRotationMatrix(),
                np.array(self.data["OBSTACLES"][obs]["translation"]),
            )
            obs_id_frame = rmodel.addFrame(pin.Frame(obs, 0, 0, Mobs, pin.OP_FRAME))
            obs_geom = pin.GeometryObject(
                obs, 0, obs_id_frame, rmodel.frames[obs_id_frame].placement, obs_hppfcl
            )
            if "color" in self.data["OBSTACLES"][obs]:
                obs_geom.meshColor = np.array(self.data["OBSTACLES"][obs]["color"])
            else:
                obs_geom.meshColor = np.concatenate(
                    (np.random.rand(3), np.array([0.7]))
                )
            _ = cmodel.addGeometryObject(obs_geom)

        for col in self.data["collision_pairs"]:
            if cmodel.existGeometryName(col[0]) and cmodel.existGeometryName(col[1]):
                cmodel.addCollisionPair(
                    pin.CollisionPair(
                        cmodel.getGeometryId(col[0]),
                        cmodel.getGeometryId(col[1]),
                    )
                )
            else:
                raise ValueError(
                    f"Collision pair {col} does not exist in the collision model"
                )
        return cmodel

    @property
    def target_pose(self):
        return pin.SE3(
            pin.Quaternion(
                *tuple(self.data["TARGET_POSE"]["orientation"])
            ).toRotationMatrix(),
            np.array(self.data["TARGET_POSE"]["translation"]),
        )

    @target_pose.setter
    def target_pose(self, value):
        self.data["TARGET_POSE"]["orientation"] = value.rotation.tolist()
        self.data["TARGET_POSE"]["translation"] = value.translation.tolist()

    @property
    def initial_config(self):
        return np.array(self.data["INITIAL_CONFIG"])

    @initial_config.setter
    def initial_config(self, value):
        self.data["INITIAL_CONFIG"] = value.tolist()

    @property
    def X0(self):
        return np.concatenate(
            (self.initial_config, np.array(self.data["INITIAL_VELOCITY"]))
        )

    @X0.setter
    def X0(self, value):
        self.initial_config = value[:len(self.initial_config)]
        self.data["INITIAL_VELOCITY"] = value[len(self.initial_config):].tolist()

    @property
    def safety_threshold(self):
        return self.data["SAFETY_THRESHOLD"]

    @safety_threshold.setter
    def safety_threshold(self, value):
        self.data["SAFETY_THRESHOLD"] = value

    @property
    def T(self):
        return self.data["T"]

    @T.setter
    def T(self, value):
        self.data["T"] = value

    @property
    def dt(self):
        return self.data["dt"]

    @dt.setter
    def dt(self, value):
        self.data["dt"] = value

    @property
    def di(self):
        return self.data["di"]

    @di.setter
    def di(self, value):
        self.data["di"] = value

    @property
    def ds(self):
        return self.data["ds"]

    @ds.setter
    def ds(self, value):
        self.data["ds"] = value

    @property
    def ksi(self):
        return self.data["ksi"]

    @ksi.setter
    def ksi(self, value):
        self.data["ksi"] = value

    @property
    def W_xREG(self):
        return self.data["WEIGHT_xREG"]

    @W_xREG.setter
    def W_xREG(self, value):
        self.data["WEIGHT_xREG"] = value

    @property
    def W_uREG(self):
        return self.data["WEIGHT_uREG"]

    @W_uREG.setter
    def W_uREG(self, value):
        self.data["WEIGHT_uREG"] = value

    @property
    def W_gripper_pose(self):
        return self.data["WEIGHT_GRIPPER_POSE"]

    @W_gripper_pose.setter
    def W_gripper_pose(self, value):
        self.data["WEIGHT_GRIPPER_POSE"] = value

    @property
    def W_gripper_pose_term(self):
        return self.data["WEIGHT_GRIPPER_POSE_TERM"]

    @W_gripper_pose_term.setter
    def W_gripper_pose_term(self, value):
        self.data["WEIGHT_GRIPPER_POSE_TERM"] = value

    @property
    def W_limit(self):
        return self.data["WEIGHT_LIMIT"]

    @W_limit.setter
    def W_limit(self, value):
        self.data["WEIGHT_LIMIT"] = value