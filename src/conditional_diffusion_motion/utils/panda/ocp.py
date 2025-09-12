import sys
from typing import Any
import numpy as np
import colmpc as col
import crocoddyl
import pinocchio as pin
import mim_solvers

from .params_parser import ParamParser


class OCP:
    """This class is creating a optimal control problem of a panda robot reaching for a target while taking a collision between a given previously given shape of the robot and an obstacle into consideration"""

    def __init__(
        self,
        rmodel: pin.Model,
        cmodel: pin.GeometryModel,
        TARGET_POSE: pin.SE3,
        x0: np.ndarray,
        pp: ParamParser,
        joint_limits: bool = False,
        penalisation: bool = True,
        constraint: bool = False,
        with_callbacks: bool = False,
    ) -> None:
        """Creating the class for optimal control problem of a panda robot reaching for a target while taking a collision between a given previously given shape of the robot and an obstacle into consideration.

        Args:
            rmodel (pin.Model): pinocchio Model of the robot
            cmodel (pin.GeometryModel): Collision model of the robot
            TARGET_POSE (pin.SE3): Pose of the target in WOLRD ref
            x0 (np.ndarray): Initial state of the problem
            pp (ParamParser): Param parser containing the parameters of the problem, such as time horizon, time step, weights, etc.
            joint_limits (bool, optional): Describe wether the joint limits will be taken into account or not. Defaults to False.
            penalisation (bool, optional): If True, the joint limits will be penalised in the cost function, otherwise they will be treated as constraints. Defaults to True.
            constraint (bool, optional): If True, the joint limits will be treated as constraints, otherwise they will be penalised in the cost function. Defaults to False.
            with_callbacks (bool, optional): If True, the OCP will be created with callbacks. Defaults to False.
        """
        # Models of the robot
        self._rmodel = rmodel
        self._cmodel = cmodel

        self.pp = pp
        # Poses & dimensions of the target & obstacle
        self._TARGET_POSE = TARGET_POSE
        self._SAFETY_THRESHOLD = self.pp.safety_threshold

        # Params of the problem
        self._T = self.pp.T
        self._dt = self.pp.dt
        self._x0 = x0

        self._joints_limits = joint_limits
        self._penalisation = penalisation
        self._constraint = constraint

        # Weights
        self._WEIGHT_xREG = self.pp.W_xREG
        self._WEIGHT_uREG = self.pp.W_uREG
        self._WEIGHT_GRIPPER_POSE = self.pp.W_gripper_pose
        self._WEIGHT_GRIPPER_POSE_TERM = self.pp.W_gripper_pose_term
        self._WEIGHT_LIMIT = self.pp.W_limit

        # Data models
        self._rdata = rmodel.createData()
        self._cdata = cmodel.createData()
        
        # Debugging
        self._with_callbacks = with_callbacks

        # Frames
        self._endeff_frame = self._rmodel.getFrameId("panda_hand_tcp")

        # Making sure that the frame exists
        assert self._endeff_frame <= len(self._rmodel.frames)

    def create_OCP(self) -> "OCP":
        "Setting up croccodyl OCP"

        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._rmodel)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Running & terminal cost models
        self._runningCostModel = crocoddyl.CostModelSum(self._state)
        self._terminalCostModel = crocoddyl.CostModelSum(self._state)

        ### Creation of cost terms

        # State Regularization cost
        xResidual = crocoddyl.ResidualModelState(self._state, self._x0)
        xRegCost = crocoddyl.CostModelResidual(self._state, xResidual)

        # Control Regularization cost
        uResidual = crocoddyl.ResidualModelControl(self._state)
        uRegCost = crocoddyl.CostModelResidual(self._state, uResidual)

        # # End effector frame cost
        framePlacementResidual = crocoddyl.ResidualModelFrameTranslation(
            self._state,
            self._endeff_frame,
            self._TARGET_POSE.translation,
        )

        goalTrackingCost = crocoddyl.CostModelResidual(
            self._state, framePlacementResidual
        )

        # Obstacle cost with hard constraint
        self._runningConstraintModelManager = crocoddyl.ConstraintModelManager(
            self._state, self._actuation.nu
        )
        self._terminalConstraintModelManager = crocoddyl.ConstraintModelManager(
            self._state, self._actuation.nu
        )
        # Creating the residual
        if len(self._cmodel.collisionPairs) != 0:
            for col_idx in range(len(self._cmodel.collisionPairs)):
                obstacleDistanceResidual = col.ResidualDistanceCollision(
                    self._state, self._rmodel.nq, self._cmodel, col_idx
                )
                # Creating the inequality constraint
                constraint = crocoddyl.ConstraintModelResidual(
                    self._state,
                    obstacleDistanceResidual,
                    np.array([self._SAFETY_THRESHOLD]),
                    np.array([np.inf]),
                )

                # Adding the constraint to the constraint manager
                self._runningConstraintModelManager.addConstraint(
                    "col_" + str(col_idx), constraint
                )
                self._terminalConstraintModelManager.addConstraint(
                    "col_term_" + str(col_idx), constraint
                )

        # Add joint limits
        
        if self._joints_limits:
            maxfloat = sys.float_info.max
            xlb = np.concatenate(
                [
                    self._rmodel.lowerPositionLimit,
                    -maxfloat * np.ones(self._state.nv),
                ]
            )
            xub = np.concatenate(
                [
                    self._rmodel.upperPositionLimit,
                    maxfloat * np.ones(self._state.nv),
                ]
            )
            bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
            xLimitResidual = crocoddyl.ResidualModelState(self._state, self._x0, self._actuation.nu)
            if self._penalisation:
                xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
                limitCost = crocoddyl.CostModelResidual(self._state, xLimitActivation, xLimitResidual)
                self._runningCostModel.addCost("limitCostRM", limitCost, self._WEIGHT_LIMIT)
                self._terminalCostModel.addCost("limitCost", limitCost, self._WEIGHT_LIMIT)
            elif self._constraint:
                constraint = crocoddyl.ConstraintModelResidual(
                    self._state,
                    xLimitResidual,
                    xlb,
                    xub,
                )
                # Adding the constraint to the constraint manager
                self._runningConstraintModelManager.addConstraint(
                    "lim" , constraint
                )
                self._terminalConstraintModelManager.addConstraint(
                    "lim_term", constraint
                )

        # Adding costs to the models
        self._runningCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
        self._runningCostModel.addCost("ctrlRegGrav", uRegCost, self._WEIGHT_uREG)
        self._runningCostModel.addCost(
            "gripperPoseRM", goalTrackingCost, self._WEIGHT_GRIPPER_POSE
        )
        self._terminalCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
        self._terminalCostModel.addCost(
            "gripperPose", goalTrackingCost, self._WEIGHT_GRIPPER_POSE_TERM
        )

        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        self._running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state,
            self._actuation,
            self._runningCostModel,
            self._runningConstraintModelManager,
        )
        self._terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state,
            self._actuation,
            self._terminalCostModel,
            self._terminalConstraintModelManager,
        )

        # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
        self._runningModel = crocoddyl.IntegratedActionModelEuler(
            self._running_DAM, self._dt
        )
        self._terminalModel = crocoddyl.IntegratedActionModelEuler(
            self._terminal_DAM, 0.0
        )

        self._runningModel.differential.armature = np.array(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
        )
        self._terminalModel.differential.armature = np.array(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.0]
        )

        problem = crocoddyl.ShootingProblem(
            self._x0, [self._runningModel] * (self._T-1), self._terminalModel
        )
        # Create solver + callbacks
        # Define mim solver with inequalities constraints
        ocp = mim_solvers.SolverCSQP(problem)

        # Merit function
        ocp.use_filter_line_search = False

        # Parameters of the solver
        ocp.termination_tolerance = 1e-3
        ocp.max_qp_iters = 25
        ocp.eps_abs = 1e-6
        ocp.eps_rel = 0

        ocp.with_callbacks = True if self._with_callbacks else False

        return ocp