import numpy as np
import pinocchio as pin
from pathlib import Path
import time


from conditional_diffusion_motion.utils.panda.panda_wrapper import load_reduced_panda, robot_links
from conditional_diffusion_motion.utils.panda.rrt_connect import RRTConnect
from conditional_diffusion_motion.utils.panda.visualizer import create_viewer, add_sphere_to_viewer
from conditional_diffusion_motion.utils.panda.create_shelf import ShelfEnv
from conditional_diffusion_motion.utils.panda.ocp import OCP
from conditional_diffusion_motion.utils.panda.params_parser import ParamParser

def IK_solver(robot_model, cmodel, target, q_start):
            # OCP refinement
    ocp_creator = OCP(robot_model, cmodel, TARGET_POSE=pin.SE3(np.eye(3), target), x0=np.concatenate((q_start, np.zeros(robot_model.nv))), pp=pp, with_callbacks=True)
    ocp = ocp_creator.create_OCP()
    X_init = [np.concatenate((q_start, np.zeros(robot_model.nv)))]*pp.data["T"]
    
    U_init = ocp.problem.quasiStatic(X_init[:-1])
    ocp.solve(X_init, U_init, 100)

    ocp.solve()
    
    return ocp.xs[-1][:robot_model.nq]
    


def plan_and_refine(
    robot_model,
    collision_model,
    q_start,
    target_translation,
    viewer=None,
    step_size=0.1,
    max_iter=1500,
    ik_T=4,
    ocp_T=15,
    with_visualization=True
):
    # --------------------------
    # 1. IK Refinement to get q_goal
    # --------------------------
    print("Solving IK...")
    pp_ik = ParamParser(param_path, 4)   # create small horizon OCP settings
    pp_ik.data["T"] = ik_T
    ocp_creator = OCP(
        robot_model,
        collision_model,
        TARGET_POSE=pin.SE3(np.eye(3), target_translation),
        x0=np.concatenate((q_start, np.zeros(robot_model.nv))),
        pp=pp_ik,
        with_callbacks=False
    )
    ocp = ocp_creator.create_OCP()

    X_init = [
        np.concatenate((q_start, np.zeros(robot_model.nv)))
        for _ in range(pp_ik.data["T"])
    ]
    U_init = ocp.problem.quasiStatic(X_init[:-1])
    ocp.solve(X_init, U_init, 100)
    ocp.solve()

    q_goal = ocp.xs[-1][:robot_model.nq]

    print("IK found q_goal:", q_goal)

    # --------------------------
    # 2. RRT* Connect Planning
    # --------------------------
    print("Running RRT*...")
    robot_data = robot_model.createData()
    collision_data = collision_model.createData()

    planner = RRTConnect(
        robot_model,
        robot_data,
        collision_model,
        collision_data,
        step_size=step_size,
        max_iter=max_iter
    )

    path = planner.plan(q_start, q_goal)

    if path is None:
        raise RuntimeError("RRT* failed to find a path")

    print(f"RRT* returned path of {len(path)} configurations")

    # optional visualization of raw RRT path
    if viewer is not None and with_visualization:
        for q in path:
            viewer.display(q)
            input("Press Enter to continue...")

    # --------------------------
    # 3. Resample path for OCP warm-start
    # --------------------------
    print("Resampling for OCP warm-start...")
    N = len(path)

    pp_ocp = ParamParser(param_path, 4)
    pp_ocp.data["T"] = ocp_T


    indices = np.linspace(0, N - 1, ocp_T).astype(int)

    X_init = [
        np.concatenate((path[i], np.zeros(robot_model.nv)))
        for i in indices
    ]

    # ensure final node matches IK endpoint exactly
    X_init[-1][:robot_model.nq] = q_goal

    # --------------------------
    # 4. Final OCP refinement
    # --------------------------
    print("Solving final OCP...")

    ocp_creator = OCP(
        robot_model,
        collision_model,
        TARGET_POSE=pin.SE3(np.eye(3), target_translation),
        x0=np.concatenate((q_start, np.zeros(robot_model.nv))),
        pp=pp_ocp,
        with_callbacks=False
    )
    ocp = ocp_creator.create_OCP()

    U_init = ocp.problem.quasiStatic(X_init[:-1])
    ocp.solve(X_init, U_init, 100)
    ocp.solve()

    print("Refinement complete")

    # optional visualization of refined trajectory
    if viewer is not None and with_visualization:
        for x in ocp.xs:
            viewer.display(x[:robot_model.nq])
            input("Enter for next refined step...")

    return ocp.xs, ocp.us, path


if __name__ == "__main__":
    
    data_dir = Path(__file__).parent.parent / "ressources" / "shelf_example"
    
    param_path = data_dir / "config" / "scenes.yaml"

    pp = ParamParser(str(param_path), 4)

    # Load your robot model
    robot_model, collision_model, visual_model = load_reduced_panda()
    
    # Initialize the shelf environment
    shelf_env = ShelfEnv()
    
    # Add shelf to both robot and a new dummy model
    cmodel_shelf, vmodel_shelf, rmodel_shelf, cmodel, vmodel = shelf_env.create_model_with_shelf(collision_model, visual_model)

    # Collect all current objects (shelves)
    existing_objects = shelf_env.create_scene_objects()

    # Add n non-colliding small obstacles
    cmodel, cmodel_shelf = shelf_env.add_random_obstacles(cmodel, cmodel_shelf, 2)
    # Add collision pairs for shelf and obstacles
    shelf_env.add_collision_pairs_with_shelf(cmodel, robot_links)
    shelf_env.add_collision_pairs_with_obstacles(cmodel, robot_links)
    
    q_start = pin.randomConfiguration(robot_model)
    q_goal = pin.randomConfiguration(robot_model)
    vis = create_viewer(robot_model, cmodel, vmodel)
    
    robot_data = robot_model.createData()
    collision_data = cmodel.createData()
    # Position the robot at start and goal configurations
    pin.framesForwardKinematics(robot_model, robot_data, q_start)
    p_start = robot_data.oMf[robot_model.getFrameId('panda_hand_tcp')].translation
    add_sphere_to_viewer(vis, "start_sphere", 0.02, p_start, color=0x00FF00)
    
    # Position the robot at goal configuration
    # pin.framesForwardKinematics(robot_model, robot_data, q_goal)
    # p_goal = robot_data.oMf[robot_model.getFrameId('panda_hand_tcp')].translation
    p_goal = np.array([0.7, 0.0, 0.6])
    add_sphere_to_viewer(vis, "goal_sphere", 0.02, p_goal, color=0xFF0000)
    ##
    
    qs, us, rrt_path = plan_and_refine(
        robot_model,
        cmodel,
        q_start,
        target_translation=p_goal,
        viewer=vis,
        with_visualization=True
    )