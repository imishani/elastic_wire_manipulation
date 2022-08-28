#!/usr/bin/env python

"""
Author: Itamar Mishani
Mail: imishani@gmail.com
"""

# General imports:
import os
import tf
import numpy as np
from scipy import optimize
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colorama import Fore
from PSO_itamar import pso
import time
import torch
from torch.autograd import Variable
from scipy.spatial.transform import Rotation as R
# model import:
from rodeval_new import rod
import sys
sys.path.insert(0, '/home/roblab2/catkin_ws/src/motoman/mish_motoman/motoman_cable/scripts/Optimization')
from Optimish import Optimizer

# ROS imports:
from interactive_markers.interactive_marker_server import *
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, WrenchStamped, PoseStamped
from natnet_msgs.msg import MarkerList
from std_msgs.msg import Header
import message_filters

path = os.getcwd()



class c_approximator(object):
    def __init__(self):
        super(c_approximator, self).__init__()
        self.count = 0
        self.robot_right_array = None
        self.forces_array = None
        self.counter = 1
        self.history = []
        self.iteration, self.iteration_num = [], 0
        self.p_best, self.p_worst = [], []
        self.tl = tf.TransformListener()
        rospy.sleep(1.2)
        _, _, self.tobase_mat = self.tf_trans('base_link', 'left_cable_contact_pos')

    def func_min(self, x, forces_array, right_array):
        t = time.time()
        sum = 0
        for i in range(len(forces_array)):
            a, right_i = forces_array[i,:], right_array[i,:]
            L = x[1]
            Rod = rod(c=[0.77*x[0], x[0], x[0]], L=L, short=True)
            Q, _, _, _ = Rod.solve(np.array(a), check_collision=False)
            pred_pos = Q[-1, :3, 3]
            pred_quat = R.from_dcm(np.matmul(Q[-1, :3,:3], np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])))
            pred_quat = pred_quat.as_quat()
            pred = np.hstack((pred_pos, pred_quat))
            sum += np.sum((pred - right_i) ** 2)
            # sum += np.sum((pred_pos - right_i[:3]) ** 2)
        return sum

    def func_min_J(self, x, forces_array, right_array):
        sum = 0
        for i in range(1, len(forces_array)):
            L = x[1]
            Rod = rod(c=[0.77*x[0], x[0], x[0]], L=L, short=False)
            Q, _, _, Jl = Rod.solve(np.array(forces_array[i - 1, :]), check_collision=False)
            delta_b = np.dot(Jl, forces_array[i, :] - forces_array[i-1, :])
            b_xyz = np.dot(self.tobase_mat, np.dot(Q[-1], np.hstack((delta_b[3:], 1))))[:3]
            rotvec = R.from_rotvec(delta_b[:3])
            RR = rotvec.as_dcm()
            RR = np.matmul(RR, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
            rot = np.dot(self.tobase_mat[:3, :3], np.dot(Q[-1, :3, :3], RR))
            quat = R.from_dcm(rot)
            quat = -quat.as_quat()
            sum += np.linalg.norm(right_array[i, :] - np.hstack((b_xyz, quat)))
        return sum

    def tf_trans(self, target_frame, source_frame):
        try:
            # listen to transform
            print(self.tl.allFramesAsString())
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            mat = self.tl.fromTranslationRotation(trans, rot)
            return trans, rot, mat

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None, None


def callback(a_, right, c_app):
    st = time.time()

    forces = np.array([[a_.wrench.torque.z, a_.wrench.torque.x - 0.11*a_.wrench.force.y, a_.wrench.torque.y - 0.11*a_.wrench.force.x,
                        a_.wrench.force.z, a_.wrench.force.x, a_.wrench.force.y]])
    try:
        c_app.forces_array = np.vstack((c_app.forces_array, forces))
    except:
        c_app.forces_array = forces

    robot_right_quat = R.from_quat([right.pose.orientation.x, right.pose.orientation.y, right.pose.orientation.z, right.pose.orientation.w])
    robot_right_quat = robot_right_quat.as_quat()
    robot_right = np.hstack((np.array([right.pose.position.x, right.pose.position.y, right.pose.position.z]), robot_right_quat))
    try:
        c_app.robot_right_array = np.vstack((c_app.robot_right_array, robot_right))
    except:
        c_app.robot_right_array = robot_right

    if c_app.count == 2:
        # options = {'w': 0.7, 'c1': 1.5, 'c2': 1.5} # 'c1': 0.5, 'c2': 0.3, 'w':0.9
        print(Fore.LIGHTYELLOW_EX + '----- Calculating Rod shape and length -----')
        try:
            lower_bound = np.array([max(c_app.c-0.2, 0.01), max(c_app.l-0.03, 0.5)])
            upper_bound = np.array([min(c_app.c+0.2, 1.), min(c_app.l+0.03, 0.9)])
        except:
            lower_bound = np.array([0.01, 0.5])
            upper_bound = np.array([1., 1.])

        counter = 1
        history = []
        xopt, fopt, history_g, history_fg = pso(c_app.func_min, lower_bound, upper_bound,
                                args=(c_app.forces_array, c_app.robot_right_array),
                                swarmsize=20, maxiter=10, debug=True, omega=0.5, phip=0.5, phig=0.5,
                                particle_output=True)  # f_ieqcons=con

        if raw_input('Save iterations convergence to pickle? y/n') == 'y':
            try:
                with open('PSOProcess.pkl', 'rb') as h:
                    lst = pickle.load(h)
                    if lst is None:
                        lst = []
                    lst.append([history_g, history_fg])
                    print(len(lst))
            except:
                lst = [[history_g, history_fg]]
            with open('PSOProcess.pkl', 'wb') as h:
                pickle.dump(lst, h)

        try:
            if fopt < c_app.fopt:
                c_app.c = xopt[0]
                c_app.l = xopt[1]
                c_app.fopt = fopt
        except:
            c_app.c = xopt[0]
            c_app.l = xopt[1]
            c_app.fopt = fopt

        # cost, pos = optimizer.optimize(func_min, iters=10)
        print(Fore.LIGHTGREEN_EX +'---------------------------')
        print('c opt: {:.3f}, l opt: {:.3f}, f_opt: {:.3f}'.format(c_app.c, c_app.l, c_app.fopt))
        # print(str(time.time() - st) + ' sec')
        print(Fore.LIGHTGREEN_EX + '---------------------------')
        print(Fore.WHITE + 'Iteration time: {:.2f} sec'.format((time.time() - st)))
        print(Fore.WHITE + 'Total time from start: {:.2f} min'.format((time.time() - total_time) / 60))

        c_app.count = 0
        c_app.forces_array = None
        c_app.robot_right_array = None
    else:
        c_app.count += 1

if __name__ == "__main__":
    rospy.init_node("approx_c_live", anonymous=True)

    a = message_filters.Subscriber('/cable_forces', WrenchStamped)
    right_pose = message_filters.Subscriber('/right_pose', PoseStamped)
    c_app = c_approximator()
    ts = message_filters.ApproximateTimeSynchronizer([a, right_pose], 1, 0.1)
    ts.registerCallback(callback, c_app)
    rospy.spin()
