import os
from pathlib import Path
import json
import hppfcl
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pinocchio as pin
import time
import torch
import random

from conditional_diffusion_motion.slot_attention.slot_attention_wrapper import (
    SlotEncoderWrapper,
)
from conditional_diffusion_motion.diffusion_transformer.datasets.indirect_motion_with_obstacles_with_shelf_slot_attention_data_module import (
    IndirectDataModuleWithObstacleWithShelfSlotAttention,
)
from conditional_diffusion_motion.diffusion_transformer.models.wrapper_models.model_slot_attention import (
    ModelSlotAttention,
)
from conditional_diffusion_motion.utils.panda.params_parser import ParamParser
from conditional_diffusion_motion.utils.panda.visualizer import (
    create_viewer,
    add_sphere_to_viewer,
)
from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda, robot_links
from conditional_diffusion_motion.diffusion_transformer.inference.trajectory_predictor_with_slot_attention import (
    TrajectoryPredictorWithSlotAttention,
)

from conditional_diffusion_motion.utils.panda.create_shelf import ShelfEnv
from conditional_diffusion_motion.utils.panda.ocp import OCP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Helpers ###


def is_collision_free(q, rmodel, rdata, cmodel, cdata):
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    col = pin.computeCollisions(rmodel, rdata, cmodel, cdata, q, stop_at_first_collision=True)
    return col


def sample_valid_configuration():
    max_trials = 100
    for _ in range(max_trials):
        q0 = pin.randomConfiguration(rmodel)
        if not is_collision_free(q0, rmodel, rdata, cmodel, cdata):
            return q0
        else:
            print("Initial config in collision, resampling...")
    raise RuntimeError("Failed to sample a collision-free initial configuration.")


def sample_valid_goal():
    max_trials = 100
    for _ in range(max_trials):
        goal = np.array(
            [
                np.random.uniform(0.3, 0.6),  # X
                np.random.uniform(-0.2, 0.2),  # Y
                np.random.uniform(0.1, 0.4),  # Z
            ]
        )
        # Use a small sphere to test if the goal is inside an obstacle
        sphere = hppfcl.Sphere(0.04)
        sphere_tf = pin.SE3(np.eye(3), goal)
        for geom in cmodel.geometryObjects:
            req = hppfcl.DistanceRequest()
            res = hppfcl.DistanceResult()
            dist = hppfcl.distance(geom.geometry, geom.placement, sphere, sphere_tf, req, res)
            if dist < 0:
                break  # goal in collision
        else:
            return goal
    raise RuntimeError("Failed to sample a collision-free goal.")


def sample_valid_initial_and_goal():
    q0 = sample_valid_configuration()
    goal = sample_valid_goal()
    return q0, goal


### Paths ###

data_dir = Path(__file__).parent.parent.parent / "ressources" / "shelf_example"
file = data_dir / "trajectories" / "trajectories_data_shelf.json"

# Load configuration and models
yaml_path = data_dir / "config" / "scenes.yaml"
diffusion_checkpoint_dir = data_dir / "slot_attention" / "diffusion" / "lightning_logs" / "version_19045852" / "checkpoints"

slot_attention_checkpoint_dir = data_dir / "slot_attention" / "model_slot_attention" / "slot_attention_shelf.ckpt"

### Load the data ###
trajs = json.load(open(file, "r"))
pp = ParamParser(str(yaml_path), 2)
print(len(trajs[0]["trajectory"]))  
# get latest checkpoint
checkpoint_path = natsorted(diffusion_checkpoint_dir.glob("*.ckpt"))[-1]  # get the latest .ckpt
print(f"Loading model from {checkpoint_path}")

### Create the slot encoder ###
slot_encoder = SlotEncoderWrapper(model_dir=str(slot_attention_checkpoint_dir), num_slots=6)
data = IndirectDataModuleWithObstacleWithShelfSlotAttention(
    data_dir=data_dir, slot_attention_model_dir=slot_attention_checkpoint_dir
)

### Load the diffusion transformer model ###
predictor = TrajectoryPredictorWithSlotAttention(
    diffusion_ckpt_path=checkpoint_path,
    slot_attention_ckpt_path=slot_attention_checkpoint_dir,
    device=device,
)

### CREATE ROBOT ###
rmodel, cmodel, vmodel = load_reduced_panda()
### Generating the problem ###
# Initialize the shelf environment
shelf_env = ShelfEnv()

# Add shelf to both robot and a new dummy model
cmodel_shelf, vmodel_shelf, rmodel_shelf, cmodel, vmodel = shelf_env.create_model_with_shelf(cmodel, vmodel)

# Collect all current objects (shelves)
existing_objects = shelf_env.create_scene_objects()
# Obstacle generation
num_obstacles = random.randint(1, 2)
# Add n non-colliding small obstacles
cmodel, cmodel_shelf = shelf_env.add_random_obstacles(cmodel, cmodel_shelf, num_obstacles)
# Add collision pairs for shelf and obstacles
shelf_env.add_collision_pairs_with_shelf(cmodel, robot_links)
shelf_env.add_collision_pairs_with_obstacles(cmodel, robot_links)


cdata = cmodel.createData()
rdata = rmodel.createData()

# initial_pose = torch.tensor(pp.initial_config, dtype=torch.float32).to(device)

# goal = data.dataset.goal[np.random.choice(len(samples), size=1, replace=False)]
# goal = pp.target_pose.translation
# goal[0] = 0.3
# goal = torch.tensor(goal, dtype=torch.float32).to(device)

# q0_np, goal_np = sample_valid_initial_and_goal()


rand1 = random.randint(0, len(trajs) - 1)
rand2 = random.randint(0, len(trajs) - 1)

q0 = trajs[rand1]["q0"]
goal = trajs[rand2]["target"]
q0_np = np.array(q0)
goal_np = np.array(goal)
# q0_np = np.array([ 2.51000746,-0.305086   ,-2.84942981 ,-1.39436553 ,-1.68961783 , 2.08410124,
#  0.02483012])
# goal_np = np.array([0.7, 0.2, 1.0])  # Example goal position
print(f"Sampled initial pose: {q0_np}, goal: {goal_np}")
initial_pose = torch.tensor(q0_np, dtype=torch.float32).to(device)
goal = torch.tensor(goal_np, dtype=torch.float32).to(device)

samples = data.dataset.samples
print(samples.shape)  # (bs, seq_length, configuration_size)
seq_length = samples.shape[1]
configuration_size = samples.shape[2]
bs = 1  # Batch size

# Display and save
q = pin.neutral(rmodel)
pin.framesForwardKinematics(rmodel, rmodel.createData(), q)  # Proper kinematics init

### Setup viewer ###
vis1 = create_viewer(rmodel_shelf, cmodel_shelf, vmodel_shelf)
shelf_env.setup_cam(vis1)
vis1.display(q)
img = vis1.viewer.get_image()
img = img.convert("RGB")

### Inference ###
# Measure inference time
start_time = time.time()
sample = predictor.predict(
    image=img,
    initial_pose=initial_pose,
    goal=goal,
    seq_length=seq_length,
    configuration_size=configuration_size,
    bs=bs,
)
print(sample[0].shape) 
end_time = time.time()
print(f"Inference completed in {end_time - start_time:.2f} seconds.")
# Convert to numpy for visualization
goal_np = goal.cpu().numpy()
initial_pose_np = initial_pose.cpu().numpy()

# End-effector trajectories
list_of_trajs_ee = []
q0 = np.array(initial_pose_np)
pin.framesForwardKinematics(rmodel, rdata, q0)
p0 = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")].translation.copy()

for traj in sample:
    traj_ee = []
    for q in traj:
        pin.framesForwardKinematics(rmodel, rdata, q)
        ee_pos = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")].translation
        traj_ee.append(ee_pos.copy())
    list_of_trajs_ee.append(traj_ee)

list_of_trajs_ee = np.array(list_of_trajs_ee)  # (bs, T, 3)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
axes_titles = ["End-Effector X", "End-Effector Y", "End-Effector Z"]

for i, ax in enumerate(axes):
    for traj in list_of_trajs_ee:
        ax.plot(traj[:, i])
    ax.plot([p0[i]] * seq_length, color="red", linestyle="--", label="p0")
    ax.plot([goal_np[i]] * seq_length, color="blue", linestyle="--", label="pT")
    ax.set_ylabel(axes_titles[i])
    ax.grid(True)

axes[-1].set_xlabel("Timestep")
plt.suptitle("End-Effector Position Over Time (X, Y, Z)")
plt.tight_layout()
plt.legend()
plt.show()
vis = create_viewer(rmodel, cmodel, vmodel)
# Viewer visualization
add_sphere_to_viewer(vis, "goal", 0.05, goal_np, color=0x006400)
add_sphere_to_viewer(vis, "p0", 0.05, p0, color=0xFF0000)

# OCP refinement
ocp_creator = OCP(rmodel, cmodel, TARGET_POSE=pin.SE3(np.eye(3), goal_np), x0=np.concatenate((q0_np, np.zeros(rmodel.nv))), pp=pp, with_callbacks=True)
ocp = ocp_creator.create_OCP()
X_init = [np.concatenate((q0_np, np.zeros(rmodel.nv)))]


for q in sample[0]:
    X_init.append(np.concatenate((q, np.zeros(rmodel.nv))))

U_init = ocp.problem.quasiStatic(X_init[:-1])
ocp.solve(X_init, U_init, 100)

# Initialize the output tensor with the right shape
xt_new = []
Qs = [xs[:rmodel.nq] for xs in ocp.xs]  # Extract only the configuration part

#     xt_new.append(torch.tensor(qs, dtype=torch.float32))


for i, traj in enumerate(sample):
    for q in traj:
        vis.display(np.array(q))
        input("Step displayed. Press Enter for next...")
    print(f"Trajectory {i} displayed. Press Enter to continue...")

for q in Qs:
    vis.display(np.array(q))
    input("Step displayed. Press Enter for next...")