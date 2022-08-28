#!/usr/bin/env python

"""
Author: Itamar Mishani
Mail: imishani@gmail.com
"""

import rospy
import tf
import numpy as np
import pandas as pd
import rodeval
from std_msgs.msg import Float64MultiArray, Header
from visualization_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, PoseStamped, WrenchStamped, Wrench, Vector3
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import torch, torch.utils.data
from torch.autograd import Variable
from scipy.spatial.transform import Rotation as R
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../learn_rod_path/itamar/"))
from sklearn.preprocessing import StandardScaler
from shape_predictor_live import nn_shape_estimator


class cable_location_nn(object):
    def __init__(self, Q, ref_frame="LeftHand"): #left_cable_contact_pos
        super(cable_location_nn, self).__init__()
        self.ref_frame = ref_frame
        self.pub_line_min_dist = rospy.Publisher('~cable_line', Marker, queue_size=1)
        self.marker = Marker()
        self.marker.header.frame_id = ref_frame
        self.marker.type = self.marker.LINE_STRIP
        self.marker.action = self.marker.ADD

        # self.marker scale
        self.marker.scale.x = 0.02
        self.marker.scale.y = 0.02
        self.marker.scale.z = 0.02

        # self.marker color

        self.marker.scale.x = 0.01
        self.marker.color.r = 0.5
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.
        # self.marker orientaiton
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0

        # self.marker position
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0

        self.tl = tf.TransformListener()
        self.df = pd.DataFrame({'x': Q[:, 0, 3], 'y': Q[:, 1, 3], 'z': Q[:, 2, 3]})
        rospy.sleep(1.2)
        self.A = self.tf_trans('base_link', 'left_cable_contact_pos') #arm_left_end_effector_link
        self.A_mocap = self.tf_trans('LeftHand', 'mocap')
        self.A_relative = self.tf_trans('left_cable_contact_pos', 'right_cable_contact_pos')

        self.df['poses'] = self.df.apply(lambda row: self.position(row), axis=1)

    def add_cable_to_rviz(self): # , dele=False
        self.marker.points = []
        for index, row in self.df[::-1].iterrows():
            pos = Point()
            pos.x, pos.y, pos.z = row.x, row.y, row.z
            self.marker.points.append(pos)
        self.pub_line_min_dist.publish(self.marker)

    def update_cable_pos(self, positions):
        self.df['x'] = positions[:, 0, 3]
        self.df['y'] = positions[:, 1, 3]
        self.df['z'] = positions[:, 2, 3]
        self.df['poses'] = self.df.apply(lambda row: self.position(row), axis=1)

        self.add_cable_to_rviz() # True
        # rospy.loginfo('change approved!')

    def processFeedback(self, feedback):
        p = feedback.pose.position
        print(feedback.marker_name + " is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z))

    def to_base_link(self, row):
        v1 = np.array([[row.x], [row.y], [row.z], [1]])
        v0 = np.dot(self.A[-1], v1)
        return Pose(Point(x=v0[0], y=v0[1], z=v0[2]),
                    Quaternion(x=0, y=0,
                               z=0, w=0))

    def position(self, row):
        transf = self.tf_trans('left_cable_contact_pos', 'right_cable_contact_pos')
        return Pose(Point(x=row.x, y=row.y, z=row.z),
                    Quaternion(x=transf[1][0], y=transf[1][1],
                               z=transf[1][2], w=transf[1][3]))

    def tf_trans(self, target_frame, source_frame):
        try:
            # listen to transform, from source to target. if source is 0 and tagret is 1 than A_0^1
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            mat = self.tl.fromTranslationRotation(trans, rot)
            return trans, rot, mat
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None, None

first_time = True
forces_register = None
def callback(data, net):

    forces = np.array([data.wrench.torque.z,
                        data.wrench.torque.x - 0.11*data.wrench.force.y,
                        data.wrench.torque.y - 0.11*data.wrench.force.x,
                        data.wrench.force.z,
                        data.wrench.force.x,
                        data.wrench.force.y])


    with torch.no_grad():
        forces = net(Variable(torch.tensor(forces).float()))
    forces = forces.cpu().detach().numpy()

    global first_time, forces_register
    if first_time:
        forces_register = forces
        Q = estimator.predictor_nn(forces)
        global cable_nn
        cable_nn = cable_location_nn(Q=Q)
        cable_nn.add_cable_to_rviz()
        first_time = False
    else:
        forces = 0.7*forces + 0.3*forces_register
        forces_register = forces
        Q2 = estimator.predictor_nn(forces)
        cable_nn.update_cable_pos(Q2)


if __name__ == "__main__":
    rospy.init_node("nn_marker")
    r = rospy.Rate(30)

    net = torch.nn.Sequential(
        torch.nn.Linear(6, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 6),
    )
    net.load_state_dict(torch.load('net_save_3mm_all.pt')) # net_save_3mm_all , net_save_2mm, net_save_cylinder
    estimator = nn_shape_estimator(step=2)
    a0 = rospy.Subscriber('/ft_compensated', WrenchStamped, callback, (net), queue_size=1)
    r.sleep()