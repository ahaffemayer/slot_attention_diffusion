import numpy as np
import pinocchio as pin


class Node:
    def __init__(self, q, parent=None, cost=0.0):
        self.q = q
        self.parent = parent
        self.cost = cost

class RRTConnect:
    def __init__(self, rmodel, rdata, cmodel, cdata, step_size=0.1, max_iter=1000, neighbor_radius=0.2):
        self.rmodel = rmodel
        self.rdata = rdata
        self.cmodel = cmodel
        self.cdata = cdata
        self.step_size = step_size
        self.max_iter = max_iter
        self.neighbor_radius = neighbor_radius

    # -------------------
    # Core functions
    # -------------------
    def distance(self, q1, q2):
        return np.linalg.norm(q1 - q2)

    def nearest_node(self, tree, q_rand):
        return min(tree, key=lambda node: self.distance(node.q, q_rand))

    def find_nearby_nodes(self, tree, q_new):
        return [node for node in tree if self.distance(node.q, q_new) <= self.neighbor_radius]

    def interpolate(self, q1, q2):
        dist = self.distance(q1, q2)
        n_steps = max(int(dist / self.step_size), 1)
        return [q1 + (q2 - q1) * i / n_steps for i in range(1, n_steps + 1)]

    def is_collision_free(self, q):
        pin.framesForwardKinematics(self.rmodel, self.rdata, q)
        pin.updateGeometryPlacements(self.rmodel, self.rdata, self.cmodel, self.cdata, q)
        return not pin.computeCollisions(self.cmodel, self.cdata, True)

    def is_collision_free_path(self, q1, q2):
        for q in self.interpolate(q1, q2):
            if not self.is_collision_free(q):
                return False
        return True

    # -------------------
    # RRT* operations
    # -------------------
    def rewire(self, tree, q_new_node):
        nearby_nodes = self.find_nearby_nodes(tree, q_new_node.q)
        for node in nearby_nodes:
            potential_cost = q_new_node.cost + self.distance(q_new_node.q, node.q)
            if potential_cost < node.cost and self.is_collision_free_path(q_new_node.q, node.q):
                node.parent = q_new_node
                node.cost = potential_cost

    def extend(self, tree, q_rand):
        nearest = self.nearest_node(tree, q_rand)
        for q_new in self.interpolate(nearest.q, q_rand):
            if not self.is_collision_free(q_new):
                return None
            nearest = Node(q_new, parent=nearest, cost=nearest.cost + self.distance(nearest.q, q_new))
        tree.append(nearest)
        self.rewire(tree, nearest)
        return nearest

    def connect(self, tree, q_target):
        while True:
            node_new = self.extend(tree, q_target)
            if node_new is None:
                return False
            if self.distance(node_new.q, q_target) < 1e-3:
                return True

    def ik_translation_multi_seed(self, target_translation, n_seeds=200, max_iter=1000, max_dq=0.05):
        for _ in range(n_seeds):
            q0 = pin.randomConfiguration(self.rmodel)
            q = q0.copy()
            for _ in range(max_iter):
                pin.forwardKinematics(self.rmodel, self.rdata, q)
                ee_translation = self.rdata.oMf[-1].translation
                error = target_translation - ee_translation
                if np.linalg.norm(error) < 1e-4:
                    return q  # success

                J = pin.computeJointJacobian(self.rmodel, self.rdata, q, self.rmodel.nq - 1)
                J_translation = J[:3, :]
                dq = np.linalg.pinv(J_translation).dot(error)
                if np.linalg.norm(dq) > max_dq:
                    dq = dq / np.linalg.norm(dq) * max_dq
                q = pin.integrate(self.rmodel, q, dq)
                q = np.clip(q, self.rmodel.lowerPositionLimit, self.rmodel.upperPositionLimit)

        return None  # failed after all seeds


    # -------------------
    # Planning
    # -------------------
    def plan(self, q_start, q_goal_or_translation):
        # Generate goal configuration if given a translation
        if isinstance(q_goal_or_translation, np.ndarray) and q_goal_or_translation.shape[0] == 3:
            q_goal = self.ik_translation_multi_seed(q_goal_or_translation)
            if q_goal is None:
                raise RuntimeError("Failed to find IK solution for target translation")
        else:
            q_goal = q_goal_or_translation

        start_tree = [Node(q_start)]
        goal_tree = [Node(q_goal)]

        for _ in range(self.max_iter):
            q_rand = np.random.uniform(low=-np.pi, high=np.pi, size=q_start.shape)

            node_new = self.extend(start_tree, q_rand)
            if node_new:
                node_goal_connect = self.extend(goal_tree, node_new.q)
                if node_goal_connect and self.distance(node_goal_connect.q, node_new.q) < 1e-3:
                    path_start, path_goal = [], []
                    node = node_new
                    while node:
                        path_start.append(node.q)
                        node = node.parent
                    node = node_goal_connect
                    while node:
                        path_goal.append(node.q)
                        node = node.parent
                    return path_start[::-1] + path_goal

            start_tree, goal_tree = goal_tree, start_tree

        return None


if __name__ == "__main__":
    # Load your robot model
    import time
    
    from panda_wrapper import load_reduced_panda, robot_links
    from conditional_diffusion_motion.utils.panda.visualizer import create_viewer, add_sphere_to_viewer
    from conditional_diffusion_motion.utils.panda.create_shelf import ShelfEnv

    
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
    input()
    
    # Position the robot at goal configuration
    # pin.framesForwardKinematics(robot_model, robot_data, q_goal)
    # p_goal = robot_data.oMf[robot_model.getFrameId('panda_hand_tcp')].translation
    p_goal = np.array([0.5, 0.0, 0.6])
    add_sphere_to_viewer(vis, "goal_sphere", 0.02, p_goal, color=0xFF0000)
    
    
    planner = RRTConnect(robot_model, robot_data, cmodel, collision_data, step_size=0.1, max_iter=1000)
    t = time.time()

    path = planner.plan(q_start, p_goal)
    print(f"RRT-Connect planning time: {(time.time() - t)*1000:.2f} ms")
    if path:
        print(f"Found path with {len(path)} configurations")
    else:
        print("No path found")

    for q in path:
        vis.display(q)
        input("Press Enter to continue to next configuration...")