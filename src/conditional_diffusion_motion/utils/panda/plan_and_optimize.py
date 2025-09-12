from typing import List
import warnings
import numpy as np
import pydynorrt as pyrrt
import pinocchio as pin


class PlanAndOptimize:
    """This class takes a robot model and a collision model, an initial configuration and a SE3 pose and a name of the frame
    that is wanted to reach the SE3 in inputs, and returns a trajectory and its dynamics, linking the initial configuration to the SE3.
    """

    def __init__(self, rmodel: pin.Model, cmodel: pin.GeometryModel, ee_name: str, T: int) -> None:
        """Args:
        rmodel (pin.Model): pinocchio model of the robot.
        cmodel (pin.GeometryModel): collision model of the robot.
        ee_name (str): end effector frame name.
        T (int): number of nodes of the trajectory.
        """
        # Models of the robot
        self._rmodel = rmodel
        self._cmodel = cmodel

        # Planning problem setup
        self._ee_name = ee_name

        # Number of nodes in the OCP at the end
        self._T = T

        # Booleans describing the state of the planner
        self._set_IK = False
        self._solved_IK = False
        self._set_collision_planner = False
        self._set_lim = False
        self._set_planner = False
        self._shortcut_done = False

    def _set_limits(self, q_upper_bounds=None, q_lower_bounds=None) -> None:
        """Setting the bounds for the joints.

        Args:
            q_upper_bounds (np.ndarray, optional): Upper bounds of the joints. Defaults to None, uses URDF.
            q_lower_bounds (np.ndarray, optional): Lower bounds of the joints. Defaults to None, uses URDF.
        """
        if q_upper_bounds is None:
            self._q_upper_bounds = self._rmodel.upperPositionLimit
            warnings.warn("No upper bounds specified, will use the default ones specified in the URDF.")
        else:
            assert np.all(q_upper_bounds <= self._rmodel.upperPositionLimit), "An upper bound exceeds the URDF limit."
            self._q_upper_bounds = q_upper_bounds

        if q_lower_bounds is None:
            self._q_lower_bounds = self._rmodel.lowerPositionLimit
            warnings.warn("No lower bounds specified, will use the default ones specified in the URDF.")
        else:
            assert np.all(q_lower_bounds >= self._rmodel.lowerPositionLimit), "A lower bound is below the URDF limit."
            self._q_lower_bounds = q_lower_bounds

        # Sanity check ordering
        assert np.all(
            self._q_lower_bounds < self._q_upper_bounds
        ), "Lower bounds must be strictly less than upper bounds per joint."

        self._set_lim = True

    def set_ik_solver(
        self,
        q_upper_bounds=None,
        q_lower_bounds=None,
        oMgoal=pin.SE3.Identity(),
        max_num_attempts=1000,
        max_time_ms=300,
        max_solutions=20,
        max_it=1000,
        use_gradient_descent=False,
        use_finite_diff=False,
    ) -> "pyrrt.Pin_ik_solver":
        """Set the IK solver.

        Returns:
            pyrrt.Pin_ik_solver: IK solver.
        """
        self._set_IK = True
        self._solver = pyrrt.Pin_ik_solver()
        pyrrt.set_pin_model_ik(self._solver, self._rmodel, self._cmodel)

        # Setting the bounds used for both IK and RRT
        self._set_limits(q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds)

        self._solver.set_frame_positions([oMgoal.translation])
        self._solver.set_bounds(self._q_lower_bounds, self._q_upper_bounds)
        self._solver.set_max_num_attempts(max_num_attempts)
        self._solver.set_frame_names([self._ee_name])
        self._solver.set_max_time_ms(max_time_ms)
        self._solver.set_max_solutions(max_solutions)
        self._solver.set_max_it(max_it)
        self._solver.set_use_gradient_descent(use_gradient_descent)
        self._solver.set_use_finite_diff(use_finite_diff)

        return self._solver

    def solve_IK(self) -> List[np.ndarray]:
        """Solves the constrained IK problem.
        This IK problem is finding configuration(s) that satisfy collision constraints and reach the target with the end effector.

        Returns:
            list: list of configurations that are solutions to the IK problem
        """
        if not self._set_IK:
            self.set_ik_solver()
            warnings.warn(
                "The IK problem has not been set. Using default parameters. To change them, call set_ik_solver first."
            )
        if not self._set_collision_planner:
            self.set_collision_planner()

        if not self._set_lim:
            self._set_limits()

        _ = self._solver.solve_ik()
        ik_solutions = self._solver.get_ik_solutions()

        self._solved_IK = True

        # Filter any numerically marginal solutions
        self._ik_solutions = [
            s
            for s in ik_solutions
            if self.cm.is_collision_free(s) and np.all(s >= self._q_lower_bounds) and np.all(s <= self._q_upper_bounds)
        ]
        return self._ik_solutions

    def set_collision_planner(self):
        """Set the collision planner."""
        self.cm = pyrrt.Collision_manager_pinocchio()
        pyrrt.set_pin_model(self.cm, self._rmodel, self._cmodel)
        self.cm.reset_counters()
        self._set_collision_planner = True

    def _generate_random_collision_free_configuration(self) -> np.ndarray:
        """Generate random feasible configurations within the current bounds."""
        if not self._set_collision_planner:
            self.set_collision_planner()
        if not self._set_lim:
            self._set_limits()

        valid_start = False
        while not valid_start:
            s = np.random.uniform(self._q_lower_bounds, self._q_upper_bounds)
            if self.cm.is_collision_free(s):
                valid_start = True
        return s

    def get_random_reachable_target(self) -> pin.SE3:
        """Generate a random reachable target as an SE3."""
        random_config = self._generate_random_collision_free_configuration()
        rdata = self._rmodel.createData()
        pin.forwardKinematics(self._rmodel, rdata, random_config)
        pin.updateFramePlacements(self._rmodel, rdata)
        oMee = rdata.oMf[self._rmodel.getFrameId(self._ee_name)]
        return oMee

    def init_planner(
        self,
        start=None,
        ik_solutions: List[np.ndarray] = [],
        q_upper_bounds=None,
        q_lower_bounds=None,
    ):
        """Initialize the RRT with current or provided bounds, same bounds as IK by default."""
        # Configuration of the RRT
        self._rrt = pyrrt.PlannerRRT_Rn()
        config_str = """
        [RRT_options]
        max_it = 20000
        max_num_configs = 20000
        max_step = 1.0
        goal_tolerance = 0.001
        collision_resolution = 0.05
        goal_bias = 0.1
        store_all = false
        """

        # Setting up start and goal
        if start is None:
            start = self._generate_random_collision_free_configuration()
        self._rrt.set_start(start)
        self._rrt.set_goal_list(ik_solutions)
        self._rrt.init(self._rmodel.nq)

        # Ensure bounds exist, prefer provided ones, else reuse already set ones
        if not self._set_lim or (q_upper_bounds is not None or q_lower_bounds is not None):
            self._set_limits(q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds)

        self._rrt.set_bounds_to_state(self._q_lower_bounds, self._q_upper_bounds)
        self._rrt.set_is_collision_free_fun_from_manager(self.cm)
        self._rrt.read_cfg_string(config_str)

        self._set_planner = True
        return self._rrt

    def plan(self) -> List[np.ndarray]:
        """Run RRT, then shortcut and resample."""
        assert self._set_planner is True, "Set up the planner first"
        out = self._rrt.plan()
        assert out == pyrrt.TerminationCondition.GOAL_REACHED, "RRT did not reach the goal"
        self._fine_path = self._rrt.get_fine_path(0.5)
        self._shortcut()
        self._sol = self._ressample_path()
        return self._sol

    def _shortcut(self) -> List[np.ndarray]:
        """Shortcut the path with collision checking."""
        self._shortcut_done = True
        path_shortcut = pyrrt.PathShortCut_RX()
        path_shortcut.init(self._rmodel.nq)
        path_shortcut.set_bounds_to_state(self._q_lower_bounds, self._q_upper_bounds)
        self.cm.reset_counters()
        path_shortcut.set_is_collision_free_fun_from_manager(self.cm)
        path_shortcut.set_initial_path(self._fine_path)
        path_shortcut.shortcut()
        self._new_path_fine = path_shortcut.get_fine_path(0.01)
        return self._new_path_fine

    def _ressample_path(self) -> List[np.ndarray]:
        """Uniformly pick T nodes along the shortcut path, including endpoints."""
        if not self._shortcut_done:
            raise ValueError("Call the plan method before resampling.")
        start = self._new_path_fine[0]
        end = self._new_path_fine[-1]

        T = self._T  # Number of nodes in the optimized trajectory
        T_remaining = T - 2  # exclude start and end

        T_new_path = len(self._new_path_fine) - 2
        step = max(1, T_new_path // max(1, T_remaining))

        ressample_path = [start]
        for t in range(T_remaining):
            idx = min(1 + step * t, len(self._new_path_fine) - 2)
            ressample_path.append(self._new_path_fine[idx])
        ressample_path.append(end)

        return ressample_path

    def optimize(self, OCP):
        """Warm start the OCP with the RRT path and solve."""
        X_init = []
        for q in self._sol:
            X_init.append(np.concatenate((q, np.zeros(self._rmodel.nv))))
        U_init = OCP.problem.quasiStatic(X_init[:-1])
        OCP.solve(X_init, U_init)
        return OCP.xs, OCP.us

    def compute_traj(self, q0: np.ndarray, oMgoal: pin.SE3, OCP, q_upper_bounds=None, q_lower_bounds=None):
        """End to end, same bounds for IK and RRT if provided."""
        self.set_ik_solver(oMgoal=oMgoal, q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds)
        sol = self.solve_IK()
        self.init_planner(start=q0, ik_solutions=sol)  # bounds already set and shared
        _ = self.plan()
        xs, us = self.optimize(OCP)
        return xs, us

    def compute_traj_rrt(self, q0: np.ndarray, oMgoal: pin.SE3, q_upper_bounds=None, q_lower_bounds=None):
        """Compute a trajectory using RRT only, same bounds for IK and RRT if provided."""
        self.set_ik_solver(oMgoal=oMgoal, q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds)
        sol = self.solve_IK()
        self.init_planner(start=q0, ik_solutions=sol)
        fine_path = self.plan()
        return fine_path



def shrink_bounds_toward_mid(upper: np.ndarray, lower: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Shrink joint bounds by 'scale' toward the midpoint of each joint's interval.
    scale in (0,1], 1 means unchanged, 0 would collapse to the midpoint.
    The result never exceeds the URDF limits and preserves ordering.
    """
    assert 0 < scale <= 1, "scale must be in (0, 1]"
    mid = 0.5 * (upper + lower)

    up = mid + scale * (upper - mid)
    lo = mid + scale * (lower - mid)

    # Clip to URDF and enforce ordering
    up = np.minimum(up, upper)
    lo = np.maximum(lo, lower)

    # In pathological cases where a joint had zero range, keep the original
    eps = 1e-12
    mask = lo >= up - eps
    up[mask] = upper[mask]
    lo[mask] = lower[mask]

    return up, lo



def random_pose_within_bounds(lower: np.ndarray,
                              upper: np.ndarray,
                              margin: float = 1e-6) -> np.ndarray:
    """
    Generate a random configuration strictly inside the given bounds.
    
    Args:
        lower (np.ndarray): Lower joint bounds.
        upper (np.ndarray): Upper joint bounds.
        margin (float): Safety margin to stay away from the exact bounds.
        
    Returns:
        np.ndarray: Random configuration inside bounds.
    """
    assert lower.shape == upper.shape, "Lower and upper bounds must have same shape."
    assert np.all(lower < upper), "Lower bounds must be strictly less than upper bounds."
    
    lo = lower + margin
    up = upper - margin
    return np.random.uniform(lo, up)


if __name__ == "__main__":

    import os
    import example_robot_data as robex

    from conditional_diffusion_motion.utils.panda.visualizer import create_viewer, add_sphere_to_viewer
    from conditional_diffusion_motion.utils.panda.params_parser import ParamParser
    from conditional_diffusion_motion.utils.panda.ocp import OCP

    # Creating the robot
    panda = robex.load("panda_collision")
    rmodel, cmodel, vmodel = panda.model, panda.collision_model, panda.visual_model

    # Set the initial configuration
    initial_config = np.array([ 2.3, 1.4 , 2.3 ,-0.3  ,   2.3 , 3.3 , 0.03, 0 , 0])
    yaml_path = ""
    pp = ParamParser(yaml_path, 1)

    geom_models = [vmodel, cmodel]
    rmodel, geometric_models_reduced = pin.buildReducedModel(
        rmodel,
        list_of_geom_models=geom_models,
        list_of_joints_to_lock=[7, 8],
        reference_configuration=initial_config,
    )
    # geometric_models_reduced is a list, ordered as passed in geom_models
    vmodel, cmodel = geometric_models_reduced[0], geometric_models_reduced[1]
    # cmodel = pp.add_collisions(rmodel, cmodel)

    X0 = np.append( random_pose_within_bounds(
        rmodel.lowerPositionLimit.copy(),
        rmodel.upperPositionLimit.copy(),
        margin=1e-3,
    ) , np.zeros(rmodel.nv) )

    cdata = cmodel.createData()
    rdata = rmodel.createData()
    ocp_creation = OCP(rmodel, cmodel, pp.target_pose, X0, pp)
    OCP_ = ocp_creation.create_OCP()

    # Generate the meshcat visualizer
    vis = create_viewer(rmodel, cmodel, vmodel)
    add_sphere_to_viewer(vis, "goal", 5e-2, pp.target_pose.translation, color=0x006400)

    shrink = 0.8
    q_upper_bounds, q_lower_bounds = shrink_bounds_toward_mid(
        rmodel.upperPositionLimit.copy(),
        rmodel.lowerPositionLimit.copy(),
        shrink,
    )
    print(f"q_upper_bounds: {q_upper_bounds}"
          f"\nq_lower_bounds: {q_lower_bounds}")
    print(f"Initial upper bounds: {rmodel.upperPositionLimit}"
          f"\nInitial lower bounds: {rmodel.lowerPositionLimit}")

    print("X0:", X0)

    PaO = PlanAndOptimize(rmodel, cmodel, "panda_hand_tcp", pp.T)
    xs, us = PaO.compute_traj(
        X0[:7], pp.target_pose, OCP_,
        q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds
    )


    print("ready to visualize")
    while True:
        vis.display(initial_config[:7])
        input()
        for x in xs:
            vis.display(np.array(x[: rmodel.nq]))
            input()
        print("replay")
