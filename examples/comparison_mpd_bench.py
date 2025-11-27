import time
import numpy as np
import pinocchio as pin
from pathlib import Path
from rich.progress import Progress

from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda
from conditional_diffusion_motion.utils.panda.params_parser import ParamParser
from conditional_diffusion_motion.utils.panda.ocp import OCP
from conditional_diffusion_motion.utils.panda.visualizer import create_viewer, add_sphere_to_viewer
from metrics import compute_success

## Helpers

def sample_box(low, high, n):
    low = np.array(low)
    high = np.array(high)
    return low + np.random.rand(n, 3) * (high - low)


n_pairs = 10
scene = 5
data_dir = Path(__file__).parent.parent / "ressources" / "shelf_example"

yaml_path = data_dir / "config" / "scenes.yaml"

rmodel, cmodel, vmodel = load_reduced_panda()


pp = ParamParser(str(yaml_path), scene)

cmodel = pp.add_collisions(rmodel, cmodel)
print("Number of collision pairs:", len(cmodel.collisionPairs))

vis = create_viewer(rmodel, cmodel, vmodel)

q0 = pin.randomConfiguration(rmodel)
vis.display(q0)
rdata = rmodel.createData()
cdata = cmodel.createData()
input("Press Enter to continue...")

p_ee_list = []
q_list = []
with Progress() as progress:
    task = progress.add_task("Generating the pairs...", total=n_pairs)
    while len(p_ee_list) < n_pairs:
        q_rand = pin.randomConfiguration(rmodel)
        pin.framesForwardKinematics(rmodel, rdata, q_rand)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q_rand)
        col = pin.computeCollisions(rmodel, rdata, cmodel, cdata, q_rand, True)
        if col:
            continue  # skip this sample entirely

        vis.display(q_rand)
        p_ee = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")].translation.copy()
        add_sphere_to_viewer(vis, f"sample_{len(p_ee_list)}", 0.03, p_ee, color=0x0000FF)
        p_ee_list.append(p_ee)
        q_list.append(q_rand.copy())
        progress.update(task, advance=1)    


print(f"Collected {len(p_ee_list)} valid end-effector positions.")
if scene == 3:
    n = 20
    nA = n // 2
    nB = n - nA

    boxA = sample_box([-0.5, 0.5, 0.5], [-0.3, 0.7, 0.9], nA)
    boxB = sample_box([0.2, 0.5, 0.6], [0.6, 0.7, 0.9], nB)

    p_ee_list = [p for p in boxA] + [p for p in boxB]
    # for i, pos in enumerate(p_ee_list):
    #     add_sphere_to_viewer(vis, f"p_ee_{i}", 0.03, pos, color=0x006400)


# OCP refinement
n_success = 0
total_pairs = len(p_ee_list) * (len(q_list) - 1)

total_pairs = len(q_list) * len(p_ee_list)
n_success = 0

with Progress() as progress:
    task = progress.add_task("Refining...", total=total_pairs)

    for j in range(len(q_list)):               # for each start config
        q0 = q_list[j]                         # this is now the start
        pin.framesForwardKinematics(rmodel, rdata, q0)
        p_start = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")].translation
        add_sphere_to_viewer(vis, f"starting_{j}", 0.03, p_start, color=0x000000) # Black sphere at start
        for i in range(len(p_ee_list)):        # try each goal
            p_goal = p_ee_list[i]

            print(f"Refining path #{j} -> goal #{i}")

            target_pose = pin.SE3(np.eye(3), p_goal)
            x0 = np.concatenate((q0, np.zeros(rmodel.nv)))

            ocp_creator = OCP(
                rmodel, cmodel,
                TARGET_POSE=target_pose,
                x0=x0,
                pp=pp,
                with_callbacks=True
            )
            ocp = ocp_creator.create_OCP()
            ocp.solve()

            Qs = [xs[:rmodel.nq] for xs in ocp.xs]
            success = compute_success(rmodel, rdata, cmodel, cdata, Qs, p_goal)
            add_sphere_to_viewer(vis, f"goal_{i}", 0.03, p_goal, color=0xFF0000) # Red sphere at goal

            for x in ocp.xs:
                vis.display(x[:rmodel.nq])
                time.sleep(0.05)
            if success:
                n_success += 1

            print(f"Success: {success}")
            progress.update(task, advance=1)

print(f"Total successful refinements: {n_success} out of {total_pairs}")
