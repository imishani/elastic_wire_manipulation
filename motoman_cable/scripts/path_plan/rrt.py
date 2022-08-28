import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle

import sys
sys.path.insert(1, '../')
from rodeval_avishai import rod
# sys.path.insert(1, '../stability_region/gp_radii')
# from gp_radii import gp_radii

show_animation = True


class RRT(rod): #, gp_radii
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, a, P = None, r = None):
            self.a = a
            self.path_a = []
            self.parent = None
            self.P = P
            self.r = r

    def __init__(self,
                 start,
                 goal,
                 expand_dis=1.0,
                 path_resolution=0.2,
                 goal_sample_rate=5,
                 max_iter=1000):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        randArea:Random Sampling Area [min,max]

        """
        # super().__init__(L = 0.88, c=[0.77, 0.15, 0.15], short=True)
        rod.__init__(self, L = 0.85, c=[0.4, 0.4, 0.4], short=True)

        # gp_radii.__init__(self)

        self.dim = len(start)
        try:
            Q, stable_start, _, Jl= self.solve(start)
        except:
            Q, stable_start, _ = self.solve(start)

        P = Q[::10, :3, 3]
        self.start = self.Node(start, P = P, r = self.get_distance(start))
        try:
            Q, stable_goal, _, Jl = self.solve(goal)
        except:
            Q, stable_goal, _ = self.solve(goal)

        P = Q[::10, :3, 3]
        self.goal = self.Node(goal, P = P, r = self.get_distance(goal))

        if not stable_start or not stable_goal:
            print('Error: start and/or goal not stable!')
            exit(1)

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

    def planning(self, population=None):
        """
        rrt path planning

        """
        print()
        print('Planning from:')
        print('Start: ', self.start.a)
        print('to goal: ', self.goal.a)
        print('Running...')

        self.node_list = [self.start]
        # return self.generate_final_course(len(self.node_list) - 1) # Return straight line

        for _ in range(self.max_iter):
            rnd_node = self.get_random_node(population)
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_stability(new_node):
                new_node.path_a = None
                new_node.P = self.recent_P
                new_node.r = self.get_distance(new_node.a)
                self.node_list.append(new_node)

            if self.calc_dist_to_goal(self.node_list[-1].a) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                if self.check_stability(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.a.copy())
        d, v = self.calc_distance_and_uvector(new_node, to_node)

        new_node.path_a = [new_node.a.copy()]

        if d > extend_length:
            d = extend_length

        n_expand = math.floor(d / self.path_resolution)

        for _ in range(int(n_expand)):
            new_node.a += self.path_resolution * v
            new_node.path_a.append(new_node.a.copy())

        d, v = self.calc_distance_and_uvector(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_a.append(to_node.a.copy())
            new_node.a = to_node.a

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind, smooth = False):
        path = [self.goal]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(node)

        if smooth:
            path = self.smooth_path(path)
        self.path = path

        return path

    def calc_dist_to_goal(self, a):
        return np.linalg.norm(a - np.array(self.goal.a))

    def get_random_node(self, population=None):
        if population is not None:
            if random.randint(0, 100) > self.goal_sample_rate:
                a = population[np.random.choice(range(len(population)))]
                rnd = self.Node(a)
            else:  # goal point sampling
                rnd = self.Node(self.goal.a)
            return rnd

        if random.randint(0, 100) > self.goal_sample_rate:
            a = self.gen_random_rod()
            rnd = self.Node(a)
        else:  # goal point sampling
            rnd = self.Node(self.goal.a)
        return rnd

    def smooth_path(self, path):

        print('Initial path size: %d.'%len(path))
        print('Initiating smoothing...')
        for _ in range(50):
            if len(path) == 2:
                break

            pickPoints = [int(random.uniform(0, len(path))), int(random.uniform(0, len(path)))]
            pickPoints.sort()

            node1, node2 = path[pickPoints[0]], path[pickPoints[1]]

            if pickPoints[1]-pickPoints[0] <= 1 or not self.line_stability_check(self.Node(node1.a.copy()), self.Node(node2.a.copy())):
                continue
            
            del path[pickPoints[0]+1:pickPoints[1]]

        print('Final path size: %d.'%len(path))

        return self.break_path(path)

    def break_path(self, path):
        da = self.path_resolution

        new_path = [path[0]]
        for i in range(1, len(path)):
            # print(path[i-1].a, path[i].a)
            d, v = self.calc_distance_and_uvector(path[i-1], path[i])
            # print(i, d, da)
            if d < da:
                new_path.append(path[i])
                continue

            n_expand = math.floor(d / da)
            node = self.Node(path[i-1].a.copy())
            for _ in range(n_expand):
                node.a += da * v
                Q, _, _ = self.solve(node.a, check_collision = False)
                node.P = Q[::10, :3, 3]
                temp = self.Node(node.a.copy(), node.P.copy())
                temp.r = self.get_distance(temp.a.copy())
                new_path.append(temp)

            d, v = self.calc_distance_and_uvector(node, path[i])
            if d > 0:
                new_path.append(path[i])

        return new_path


    def line_stability_check(self, first, second):
        # Line Equation

        d, v = self.calc_distance_and_uvector(first, second)
        n_expand = math.floor(d / self.path_resolution)

        node = self.Node(first.a.copy())

        for _ in range(n_expand):
            node.a += self.path_resolution * v
            if not self.check_stability(node):
                return False

        return True  # OK

    def check_stability(self, node):

        if node is None:
            return False
        try:
            Q, stable, _, Jl = self.solve(node.a, check_collision = False)
        except:
            Q, stable, _ = self.solve(node.a, check_collision=False)

        P = Q[::10, :3, 3]
        self.recent_P = P

        return stable

    def save_path(self, postfix = ''):
        path = np.array([p.a for p in self.path])
        R = np.array([p.r for p in self.path])
        with open('path' + postfix + '.pkl', 'wb') as h: # + '_' + str(int(time.time()))
            pickle.dump([path, R], h) 
        print('Path saved.')


    def get_distance(self, to_node):
        # u = np.array(to_node.a) - np.array(self.a)
        # d = np.linalg.norm(u)
        return None

    def animate(self):

        if 1:
            for node in self.path:
                plt.clf()
                ax = plt.axes(projection='3d')
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                ax.plot3D(self.start.P[:,0], self.start.P[:,1], self.start.P[:,2], ':r')
                ax.plot3D(self.goal.P[:,0], self.goal.P[:,1], self.goal.P[:,2], ':g')

                if node.P is None:
                    Q, _, _ = self.solve(node.a, check_collision = False)
                    node.P = Q[::10, :3, 3]                    
                ax.plot3D(node.P[:,0], node.P[:,1], node.P[:,2], '-k')

                plt.pause(0.05)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [ np.linalg.norm(node.a - np.array(rnd_node.a)) for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def calc_distance_and_uvector(from_node, to_node):
        u = np.array(to_node.a) - np.array(from_node.a)
        d = np.linalg.norm(u)
        if d == 0:
            v = np.zeros((len(u),))
        else:
            v = u / d
        return d, v


def main():
    print("start " + __file__)

    # with open('../stability_region/dataset_b.pkl', 'rb') as h:
    #     data_estimation = pickle.load(h)
    # data_estimation = [d[0] for d in data_estimation if d[3]]

    # i = np.random.randint(0, len(data_estimation), size = (2,))
    # print(i)
    # start = data_estimation[i[0]]
    # goal = data_estimation[i[1]]
    # start = data_estimation[98384]
    # goal = data_estimation[98496]

    start = np.array([ -3.33322856,  -5.15552991,   7.26877387, -15.75311516,  38.47484264, 14.22537231] )
    goal = np.array([  4.94857222,   0.87371828,  -1.65167861,   2.30105848,  -8.35991513, -39.16818416])

    # Set Initial parameters
    rrt = RRT(start=start, goal=goal)
    path = rrt.planning()

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        print(len(path))
        rrt.save_path()

        # Draw final path
        if show_animation:
            rrt.animate()
            plt.show()


if __name__ == '__main__':
    main()