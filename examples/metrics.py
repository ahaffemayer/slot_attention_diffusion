import time
import numpy as np
import torch
import pinocchio as pin

def compute_distance_to_target(rmodel, rdata, q, target):
    """Compute the distance from the end-effector to the target position.

    Args:
        rmodel (_type_): _description_
        rdata (_type_): _description_
        q (_type_): _description_
        target (_type_): _description_

    Returns:
        _type_: _description_
    """

    pin.framesForwardKinematics(rmodel, rdata, q)
    target_position = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")].translation
    distance = np.linalg.norm(target_position - target)

    return distance


def compute_success(rmodel, rdata, cmodel, cdata, traj, target):
    """
    Compute the success of the trajectory.
    The trajectory is successful if it ends in the target configuration and it is collision-free.
    """

    for q in traj:
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        if pin.computeCollisions(cmodel, cdata, stop_at_first_collision=True):
            return False

    # Check if the last configuration is close to the target
    distance_to_target = compute_distance_to_target(rmodel, rdata, traj[-1], target)

    if distance_to_target < 0.1:  # Threshold for success
        return True
