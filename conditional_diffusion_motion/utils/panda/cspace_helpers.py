import json
from typing import List, Tuple

import hppfcl
import meshcat
import numpy as np
import pinocchio as pin

from pinocchio.visualize import MeshcatVisualizer

def fcl_transform_to_pin_se3(fcl_transform):
    rotation = np.array(fcl_transform.getRotation())
    translation = np.array(fcl_transform.getTranslation()).squeeze()
    return pin.SE3(rotation, translation)

def load_scene_from_json(json_path: str, scene_name: str) -> Tuple[List[Tuple[hppfcl.CollisionObject, str]], List[str]]:
    with open(json_path, "r") as f:
        scene_data = json.load(f)

    if scene_name not in scene_data:
        raise ValueError(f"Scene '{scene_name}' not found in scene data.")

    obstacles = scene_data[scene_name]["obstacles"]
    fcl_objects = []

    for i, obs in enumerate(obstacles):
        dims = obs["dimensions"]
        pos = obs["position"]
        shape = hppfcl.Box(dims["width"], dims["depth"], dims["height"])
        tf = pin.SE3(np.eye(3), np.array(pos))
        obj = hppfcl.CollisionObject(shape, tf)
        name = f"obs_{i:03d}"
        list_of_obstacles = []
        dims_list = [float(dims["width"]), float(dims["depth"]), float(dims["height"])]
        if dims_list == [0.2, 0.15, 0.2]:
            list_of_obstacles.append(name)
        fcl_objects.append((obj, name))

    return fcl_objects, list_of_obstacles

def load_scene_from_benchmark(scene:dict) -> Tuple[List[Tuple[hppfcl.CollisionObject, str]], List[str]]:

    obstacles = scene["obstacles"]
    fcl_objects = []
    list_of_obstacles = []

    for i, obs in enumerate(obstacles):
        dims = obs["dimensions"]
        pos = obs["position"]
        shape = hppfcl.Box(dims["width"], dims["depth"], dims["height"])
        tf = pin.SE3(np.eye(3), np.array(pos))
        obj = hppfcl.CollisionObject(shape, tf)
        name = f"obs_{i:03d}"
        dims_list = [float(dims["width"]), float(dims["depth"]), float(dims["height"])]
        if dims_list == [0.2, 0.15, 0.2]:
            list_of_obstacles.append(name)
        fcl_objects.append((obj, name))

    return fcl_objects, list_of_obstacles

def build_geometry_model_from_fcl_objects(fcl_objects: List[Tuple[hppfcl.CollisionObject, str]],list_of_obstacles: List[str] ) -> pin.GeometryModel:
    geom_model = pin.GeometryModel()
    color_pool = [
    [1.0, 0.0, 0.0, 1.0],  # Red
    [0.0, 1.0, 0.0, 1.0],  # Green
    [0.0, 0.0, 1.0, 1.0],  # Blue
    [1.0, 1.0, 0.0, 1.0],  # Yellow
    [1.0, 0.0, 1.0, 1.0],  # Magenta
    [1.0, 0.5, 0.0, 1.0],  # Orange
]
    i = 0
    for obj, name in fcl_objects:
        
        geom = pin.GeometryObject(name, 0,0, fcl_transform_to_pin_se3(obj.getTransform()), obj.collisionGeometry())
        if name in list_of_obstacles:
            geom.meshColor = np.array(color_pool[i])
            i += 1
        else:
            geom.meshColor = np.array([0.4, 0.4, 0.4, 1.0])
        geom_model.addGeometryObject(geom)
    return geom_model

def create_visualizer(rmodel: pin.Model, cmodel: pin.GeometryModel, vmodel: pin.GeometryModel) -> MeshcatVisualizer:
    viz = MeshcatVisualizer(model=rmodel, collision_model=cmodel, visual_model=vmodel)
    viz.initViewer(viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000"))
    viz.clean()
    viz.loadViewerModel("pinocchio")
    viz.displayCollisions(True)
    viz.viewer['/Grid'].set_property("visible", False)
    return viz

def pin_to_fcl(pose: pin.SE3) -> hppfcl.Transform3f:
    return hppfcl.Transform3f(pose.rotation, pose.translation)

def check_collision(rmodel, cmodel, q):
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
    for cp in cmodel.collisionPairs:
        cr = hppfcl.CollisionRequest()
        result = hppfcl.CollisionResult()
        o1 = hppfcl.CollisionObject(cmodel.geometryObjects[cp.first].geometry, pin_to_fcl(cdata.oMg[cp.first]))
        o2 = hppfcl.CollisionObject(cmodel.geometryObjects[cp.second].geometry, pin_to_fcl(cdata.oMg[cp.second]))
        hppfcl.collide(o1, o2, cr, result)
        if result.isCollision():
            return True
    return False

def add_collision_pairs_with_obstacles( cmodel, robot_links):
    name_to_id = {geom.name: i for i, geom in enumerate(cmodel.geometryObjects)}

    obstacle_names = [name for name in name_to_id if name.startswith("obs_")]
    
    for obstacle_name in obstacle_names:
        for link in robot_links:
            if obstacle_name not in name_to_id or link not in name_to_id:
                print(f"[Warning] '{obstacle_name}' or '{link}' not found.")
                continue
            try:
                cmodel.addCollisionPair(pin.CollisionPair(name_to_id[obstacle_name], name_to_id[link]))
            except Exception as e:
                print(f"[Error] Could not add pair ({obstacle_name}, {link}): {e}")