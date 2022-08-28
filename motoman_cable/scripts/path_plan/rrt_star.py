import math
import os
import sys
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise

show_animation = True


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, a, P = None, r = None):
            # super().__init__(a, P, r)
            RRT.Node.__init__(self, a, P, r)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 sae=None,
                 obs=None,
                 mu=np.array([-0.06990777, -0.21053933, -0.12817508, -3.40543442, -0.07793685, 2.78281192]),
                 std = np.array([0.02894388, 0.40500549, 0.32527397, 0.64293891, 1.02876501, 1.15996231]),
                 collision_radius=0.05,
                 expand_dis=1.0,
                 path_resolution=0.2,
                 goal_sample_rate=20,
                 max_iter=300,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        # super().__init__(start, goal, expand_dis,
        #                  path_resolution, goal_sample_rate, max_iter)
        RRT.__init__(self, start, goal, expand_dis,
                         path_resolution, goal_sample_rate, max_iter)
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal)
        self.search_until_max_iter = search_until_max_iter
        self.obs = obs
        self.sae = sae
        self.mu = mu
        self.std = std
        self.collision_r = collision_radius
        self.distance_cost = True
        self.opt_func = min if self.distance_cost else min
        self.K = 3

    def planning(self, population=None):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """
        print()
        print('Planning from:')
        print('Start: ', self.start.a)
        print('to goal: ', self.goal.a)
        print('Running...')

        best_cost = -1
        self.node_list = [self.start]
        self.History = []
        for i in range(self.max_iter):
            if best_cost > 0:
                self.History.append(best_cost)
            print("Iter:", i, ", number of nodes:", len(self.node_list), " Best cost: %.2f"%best_cost)
            rnd = self.get_random_node(population=population)
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.r = self.get_distance(np.array(new_node.a))
            if self.distance_cost:
                new_node.cost = near_node.cost + np.linalg.norm(np.array(new_node.a) - near_node.a)
            else:
                d = np.linalg.norm(np.array(new_node.a) - near_node.a)
                new_node.cost = near_node.cost + (self.K * d) / new_node.r

            # if self.check_stability(new_node):
            #     P = self.recent_P.copy()
            #     near_inds = self.find_near_nodes(new_node)
            #     node_with_updated_parent = self.choose_parent(new_node, near_inds)
            #     if node_with_updated_parent:
            #         self.rewire(node_with_updated_parent, near_inds)
            #         node_with_updated_parent.r = self.get_distance(node_with_updated_parent.a)
            #         self.node_list.append(node_with_updated_parent)
            #     else:
            #         new_node.P = P
            #         new_node.r = self.get_distance(new_node.a)
            #         self.node_list.append(new_node)

            if self.check_collision(new_node):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    node_with_updated_parent.r = self.get_distance(node_with_updated_parent.a)
                    self.node_list.append(node_with_updated_parent)
                else:
                    new_node.r = self.get_distance(new_node.a)
                    self.node_list.append(new_node)

            converge = True if (len(self.History) > 201 and np.all(self.History[200:] == self.History[-1])) else False
                
            if ((not self.search_until_max_iter) and new_node) or (near_node and converge):  # if reaches goal
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    print('Found solution with cost: %.2f'%self.node_list[last_index].cost)
                    return self.generate_final_course(last_index, smooth=False)
            elif new_node:
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    best_cost = self.node_list[last_index].cost

        print("Reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index, smooth=False)

        return None

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node

            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_stability(t_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = self.opt_func(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.a) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_stability(t_node):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = self.opt_func([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [np.linalg.norm(node.a - np.array(new_node.a))**2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree

                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_stability(edge_node)

            if self.distance_cost:
                improved_cost = near_node.cost > edge_node.cost
            else:
                improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.a = edge_node.a
                near_node.cost = edge_node.cost
                near_node.path_a = edge_node.path_a
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_uvector(from_node, to_node)
        # d = d if d > 0. else 0.001

        # if d == 0:
        #     print(from_node.a, to_node.a)

        if not self.distance_cost:
            if to_node.r is None:
                to_node.r = self.get_distance(to_node.a)
        
            return from_node.cost + (self.K * d) / to_node.r
        else:
            return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def check_collision(self, node):
        if self.obs is None or self.sae is None:
            return True
        else:
            a = node.a.copy() - self.mu
            a /= self.std
            p = self.sae.decode(torch.tensor(a).float()).squeeze().detach().numpy()
            r = R.from_euler('x', -45, degrees=True)
            r = r.as_dcm()
            obs_tranformed = np.dot(r, self.obs)
            # obs_ang = np.arctan2(self.obs[2], self.obs[1]) * 180 / np.pi
            for row in p:
                if np.linalg.norm(row - self.obs) <= self.collision_r:
                    return False
                # curr_point_ang = np.arctan2(row[2], [1]) * 180 / np.pi
                if (obs_tranformed[2] - 1.5) <= np.dot(r, row)[2] <= (obs_tranformed[2] + 0.15) and \
                        (obs_tranformed[1] - 0.05 <= np.dot(r, row)[1] <= obs_tranformed[1] + 0.05) and\
                        (abs(row[0] - self.obs[0]) <= 0.05):
                    return False
            return True


def main():
    print("Start " + __file__)

    # with open('../stability_region/dataset_b.pkl', 'rb') as h:
    #     data_estimation = pickle.load(h)
    # data_estimation = [d[0] for d in data_estimation if d[3]]
    # start = data_estimation[98384]
    # goal = data_estimation[98496]
    start = np.array([ 0.0123,  0.32,   -0.043, 3.27,  -1.61, -4.44] )
    goal = np.array([  0.65,   1.81,  0.002,   5.08,  -2.02, -6.68])

    # Set Initial parameters
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        expand_dis=0.5,
        path_resolution=0.1,
        max_iter = 100,
        search_until_max_iter = True)
    path = rrt_star.planning()

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        print(len(path))
        rrt_star.save_path(postfix = '_rrtstar')

        plt.plot(rrt_star.History)
        plt.show()

        # Draw final path
        if show_animation:
            rrt_star.animate()
            plt.show()


if __name__ == '__main__':
    main()