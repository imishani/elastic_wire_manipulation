#!/usr/bin/env python

"""
Author: Itamar Mishani
Mail: imishani@gmail.com
"""
import rospy
from visualization_msgs.msg import *
from geometry_msgs.msg import Point
import pandas as pd


class CableGoalRviz(object):
    def __init__(self, Q, ref_frame="LeftHand"):  # left_cable_contact_pos
        super(CableGoalRviz, self).__init__()
        self.ref_frame = ref_frame
        self.pub_line_min_dist = rospy.Publisher('goal_cable_line', Marker, queue_size=1)
        self.marker = Marker()
        self.marker.header.frame_id = ref_frame
        self.marker.type = self.marker.LINE_STRIP
        self.marker.action = self.marker.ADD

        # self.marker scale
        self.marker.scale.x = 0.02
        self.marker.scale.y = 0.02
        self.marker.scale.z = 0.02

        # self.marker color
        self.marker.color.r = 0.0
        self.marker.color.g = 0.5
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0
        # self.marker orientaiton
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0

        # self.marker position
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0

        self.df = pd.DataFrame({'x': Q[:, 0, 3], 'y': Q[:, 1, 3], 'z': Q[:, 2, 3]})

    def add_cable_to_rviz(self, dele=False):
        self.marker.points = []
        for index, row in self.df[::-1].iterrows():
            pos = Point()
            pos.x, pos.y, pos.z = row.x, row.y, row.z
            self.marker.points.append(pos)
        # self.marker.lifetime = rospy.Duration(10)
        st = rospy.Time.now()

        while rospy.Time.now().secs - st.secs < 5:
            self.pub_line_min_dist.publish(self.marker)
            rospy.rostime.wallsleep(1.0)
        # rospy.sleep(0.01)

class CablePathRviz(object):
    def __init__(self, ref_frame="LeftHand"):  # left_cable_contact_pos
        super(CablePathRviz, self).__init__()
        self.ref_frame = ref_frame
        self.pub_path = rospy.Publisher('path_cable_line', MarkerArray, queue_size=1)
        self.marker_array = MarkerArray()

    def add_marker_to_array(self, Q):
        self.marker = Marker()
        self.marker.header.frame_id = self.ref_frame
        self.marker.type = self.marker.LINE_STRIP
        self.marker.action = self.marker.ADD
        # self.marker scale
        self.marker.scale.x = 0.005
        self.marker.scale.y = 0.005
        self.marker.scale.z = 0.005
        # self.marker color
        self.marker.color.r = 0.8
        self.marker.color.g = 0.8
        self.marker.color.b = 0.4
        self.marker.color.a = 0.5
        # self.marker orientaiton
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
        # self.marker position
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0
        self.df = pd.DataFrame({'x': Q[:, 0, 3], 'y': Q[:, 1, 3], 'z': Q[:, 2, 3]})
        self.marker.points = []
        for index, row in self.df[::-1].iterrows():
            pos = Point()
            pos.x, pos.y, pos.z = row.x, row.y, row.z
            self.marker.points.append(pos)
        self.marker_array.markers.append(self.marker)

    def PublishRviz(self):
        # st = rospy.Time.now()
        # while rospy.Time.now().secs - st.secs < 5:
        ind = 0
        for marker in self.marker_array.markers:
            marker.id = ind
            ind += 1
        self.pub_path.publish(self.marker_array)
        rospy.sleep(0.1)
        # rospy.rostime.wallsleep(1.0)


class CableGoalRviz_nn(object):
    def __init__(self, p, ref_frame="LeftHand"):  # left_cable_contact_pos
        super(CableGoalRviz_nn, self).__init__()
        self.ref_frame = ref_frame
        self.pub_line_min_dist = rospy.Publisher('goal_cable_line', Marker, queue_size=1)
        self.marker = Marker()
        self.marker.header.frame_id = ref_frame
        self.marker.type = self.marker.LINE_STRIP
        self.marker.action = self.marker.ADD

        # self.marker scale
        self.marker.scale.x = 0.02
        self.marker.scale.y = 0.02
        self.marker.scale.z = 0.02

        # self.marker color
        self.marker.color.r = 0.0
        self.marker.color.g = 0.5
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0
        # self.marker orientaiton
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0

        # self.marker position
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0

        self.df = pd.DataFrame({'x': p[:, 0], 'y': p[:, 1], 'z': p[:, 2]})

    def add_cable_to_rviz(self, dele=False):
        self.marker.points = []
        for index, row in self.df[::-1].iterrows():
            pos = Point()
            pos.x, pos.y, pos.z = row.x, row.y, row.z
            self.marker.points.append(pos)
        # self.marker.lifetime = rospy.Duration(10)
        st = rospy.Time.now()

        while rospy.Time.now().secs - st.secs < 5:
            self.pub_line_min_dist.publish(self.marker)
            rospy.rostime.wallsleep(1.0)
        # rospy.sleep(0.01)


class CablePathRviz_nn(object):
    def __init__(self, ref_frame="LeftHand"):  # left_cable_contact_pos
        super(CablePathRviz_nn, self).__init__()
        self.ref_frame = ref_frame
        self.pub_path = rospy.Publisher('path_cable_line', MarkerArray, queue_size=1)
        self.marker_array = MarkerArray()

    def add_marker_to_array(self, p):
        self.marker = Marker()
        self.marker.header.frame_id = self.ref_frame
        self.marker.type = self.marker.LINE_STRIP
        self.marker.action = self.marker.ADD
        # self.marker scale
        self.marker.scale.x = 0.005
        self.marker.scale.y = 0.005
        self.marker.scale.z = 0.005
        # self.marker color
        self.marker.color.r = 0.8
        self.marker.color.g = 0.8
        self.marker.color.b = 0.4
        self.marker.color.a = 0.5
        # self.marker orientaiton
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
        # self.marker position
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0
        self.df = pd.DataFrame({'x': p[:, 0], 'y': p[:, 1], 'z': p[:, 2]})
        self.marker.points = []
        for index, row in self.df[::-1].iterrows():
            pos = Point()
            pos.x, pos.y, pos.z = row.x, row.y, row.z
            self.marker.points.append(pos)
        self.marker_array.markers.append(self.marker)

    def PublishRviz(self):
        # st = rospy.Time.now()
        # while rospy.Time.now().secs - st.secs < 5:
        ind = 0
        for marker in self.marker_array.markers:
            marker.id = ind
            ind += 1
        self.pub_path.publish(self.marker_array)
        rospy.sleep(0.1)
        # rospy.rostime.wallsleep(1.0)


