#!/usr/bin/env python

"""
Author: Itamar Mishani
Mail: imishani@gmail.com
"""

# General imports:
import os
import rospy
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
# ROS imports:
from interactive_markers.interactive_marker_server import *
from geometry_msgs.msg import WrenchStamped, PoseStamped
from natnet_msgs.msg import MarkerList
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import message_filters

cable_diam = 2
ind = 0
test = True
sampling_rate = 4


def callback(forces, right, marker, A, angel):
    global ind
    a_ = np.array([forces.wrench.torque.z, forces.wrench.torque.x - 0.11 * forces.wrench.force.y,
                   forces.wrench.torque.y - 0.11 * forces.wrench.force.x, forces.wrench.force.z,
                   forces.wrench.force.x, forces.wrench.force.y])
    robot_right = np.array([right.pose.position.x, right.pose.position.y, right.pose.position.z,
                            right.pose.orientation.x, right.pose.orientation.y, right.pose.orientation.z,
                            right.pose.orientation.w])

    x, y, z = [], [], []
    for vec in marker.positions:
        x.append(vec.x)
        y.append(vec.y)
        z.append(vec.z)
    markers_arr = np.array([x, y, z])
    transform = np.array(A.data).reshape(4, 4)
    config = np.array(angel.position)
    if test:
        path = './b2a_data/test/diam_{}_p_{}.pkl'.format(cable_diam, ind)
    else:
        path = './b2a_data/diam_{}_p_{}.pkl'.format(cable_diam, ind)
    with open(path, 'wb') as h:
        pickle.dump([a_, robot_right, markers_arr, transform, config], h)
        print('Saved data_estimation point ' + str(ind))
        ind += 1
    rate.sleep()


# if __name__ == "__main":
rospy.init_node("recorder", anonymous=True)
rate = rospy.Rate(sampling_rate)
a = message_filters.Subscriber('ft_compensated', WrenchStamped)
right_pose = message_filters.Subscriber('right_pose', PoseStamped)
markers = message_filters.Subscriber('client/markers/leftovers', MarkerList)
a_mocap = message_filters.Subscriber('A_mocap', Float64MultiArray)
angles = message_filters.Subscriber('sda10f/sda10f_r2_controller/joint_states', JointState)
ts = message_filters.ApproximateTimeSynchronizer([a, right_pose, markers, a_mocap, angles], 10, 0.1,
                                                 allow_headerless=True)
ts.registerCallback(callback)
rospy.spin()