import numpy as np
from pathlib import Path
import pinocchio as pin
import random
from tqdm import tqdm

from scipy.optimize import minimize 
from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda
from conditional_diffusion_motion.utils.panda.create_shelf import ShelfEnv
from conditional_diffusion_motion.utils.panda.visualizer import create_viewer, add_sphere_to_viewer


def ik_scipy(rmodel, rdata, target_pose, q0=None, joint_bounds=False):
    """
    Inverse kinematics using scipy.optimize.minimize with position cost only.
    
    Args:
        rmodel: Pinocchio robot model
        rdata: Pinocchio data
        target_pose: pin.SE3 target end-effector pose
        q0: Optional initial guess
        joint_bounds: Whether to apply joint limits from model
        
    Returns:
        q_opt: Optimal joint configuration
        success: Whether solver succeeded
    """
    if q0 is None:
        q0 = pin.neutral(rmodel)

    ee_frame = rmodel.getFrameId("panda_hand_tcp")  # change as needed

    def cost(q):
        q = np.asarray(q)
        pin.forwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        oMf = rdata.oMf[ee_frame]
        error_SE3 = oMf.inverse() * target_pose
        error = pin.log6(error_SE3).vector
        return np.dot(error, error)  # squared norm

    bounds = None
    if joint_bounds:
        qmin = rmodel.lowerPositionLimit
        qmax = rmodel.upperPositionLimit
        bounds = [(l, u) for l, u in zip(qmin, qmax)]

    result = minimize(
        cost,
        q0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-6}
    )

    q_opt = result.x
    return q_opt, result.success


def generate_configurations(N=100, save_path="configs_shelf.npy", visualize=False):
    # Load robot
    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()

    # Initialize shelf environment
    shelf_env = ShelfEnv()
    _, _, _, cmodel, vmodel = shelf_env.create_model_with_shelf(cmodel, vmodel)

    # Visualizer
    if visualize:
        vis = create_viewer(rmodel, cmodel, vmodel)
    else:
        vis = None

    configs = []
    attempts = 0

    pbar = tqdm(total=N, desc="Generating IK configurations")

    while len(configs) < N and attempts < N * 5:
        mode = random.choice(["in_shelf", "on_shelf", "near_shelf"])
        target_pose = shelf_env.generate_target_pose(mode=mode)
        q_sol, success = ik_scipy(rmodel, rmodel.createData(), target_pose)

        if success:
            configs.append(q_sol)
            pbar.update(1)

            if vis:
                idx = len(configs)
                vis.display(q_sol)
                add_sphere_to_viewer(
                    vis,
                    f"target_sphere_{idx}",
                    radius=0.05,
                    position=target_pose.translation,
                    color=0x00FF00  # green
                )
        attempts += 1

    pbar.close()

    configs = np.array(configs)
    np.save(save_path, configs, allow_pickle=True)
    print(f"Saved {len(configs)} configurations to {save_path}.")


if __name__ == "__main__":
    path = Path(__file__).parent / "results"
    path.mkdir(exist_ok=True)
    generate_configurations(N=10, visualize=True, save_path=path / "cspace_configurations.npy")
