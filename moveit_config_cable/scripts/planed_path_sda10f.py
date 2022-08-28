#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Empty, EmptyResponse
from motoman_msgs.msg import DynamicJointTrajectory, DynamicJointPoint, DynamicJointsGroup
import sys
sys.path.append('/home/roblab2/catkin_ws/src/bota_driver/bota_demo/scripts')
import set_desired_force


PATH_RIGHT = [[1.46697998046875, 1.2045669555664062, -1.4096978902816772, -1.3364769220352173, -0.9337234497070312, -1.5575220584869385, 0.], \
              [1.46697998046875, 1.2045669555664062, -1.4096978902816772, np.deg2rad(-100), -0.9337234497070312, -1.5575220584869385, 0.], \
              [1.46697998046875, np.deg2rad(90), -1.4096978902816772, np.deg2rad(-100), -0.9337234497070312, -1.5575220584869385, 0.], \
              [1.46697998046875, np.deg2rad(90), -1.4096978902816772, np.deg2rad(-100), np.deg2rad(-90), -1.5575220584869385, 0.], \
              [1.46697998046875, np.deg2rad(90), -1.4096978902816772, np.deg2rad(-100), -0.9337234497070312, -1.5575220584869385, 0.], \
              [1.46697998046875, 1.2045669555664062, -1.4096978902816772, np.deg2rad(-100), -0.9337234497070312, -1.5575220584869385, 0.], \
              [1.46697998046875, 1.2045669555664062, -1.4096978902816772, -1.3364769220352173, -0.9337234497070312, -1.5575220584869385, 0.]]


class move_routine(object):
    def __init__(self, path, controll_group_number=1):

        rospy.loginfo("Enter move_routine")
        self.robot_current_state = rospy.wait_for_message('/sda10f/sda10f_r2_controller/joint_states', JointState) #[1.46697998046875, 1.2045669555664062, -1.4096978902816772, -1.3364769220352173, -0.9337234497070312, -1.5575220584869385, -0.18837887048721313]
        self.robot_current_pos = list(self.robot_current_state.position)
        self.robot_joint_names = self.robot_current_state.name
        self.robot_joint_velocity = list(self.robot_current_state.velocity)
        self.duration = 15
        self.controll_group_number = controll_group_number
        self.path = path
        self.path.insert(0, self.robot_current_pos)
        self.traj = self.prepare_traj(self.path)


    def CalculateJointVelocety(self, time):
        velocity_vector = []
        if self.controll_group_number == 1:
            if time == 1:
                pos = self.robot_current_pos
                for joint in range(0, len(self.path[time - 1])):
                    joint_velocity = self.path[time - 1][joint] - pos[joint]
                    joint_velocity = joint_velocity / self.duration
                    velocity_vector.append(joint_velocity)
            else:
                for joint in range(0, len(self.path[time - 1])):
                    joint_velocity = self.path[time - 1][joint] - self.path[time - 2][joint]
                    joint_velocity = joint_velocity / self.duration
                    velocity_vector.append(joint_velocity)
        else:
            if time == 1:
                pos = self.robot_current_pos
                next_pose = self.path[time]
                for joint in range(0, len(next_pose)):
                    joint_velocity = next_pose[joint] - pos[joint]
                    joint_velocity = joint_velocity / self.duration
                    velocity_vector.append(joint_velocity)
            else:
                pos = self.path[time - 2]
                next_pose = self.path[time - 1]
                for joint in range(0, len(next_pose)):
                    joint_velocity = next_pose[joint] - pos[joint]
                    joint_velocity = joint_velocity / self.duration
                    velocity_vector.append(joint_velocity)
        return velocity_vector

    def DynamicJointsGroup_right_from_path(self, point_time):
        group_1_next_pt = DynamicJointsGroup()
        group_1_next_pt.group_number = 1
        group_1_next_pt.num_joints = 7
        if point_time == 0:
            pos = self.robot_current_pos
            group_1_next_pt.positions = pos
            group_1_next_pt.velocities = [0.0] * len(pos)
            group_1_next_pt.time_from_start = rospy.Duration(0)
        else:
            pos = self.path[point_time - 1]
            new_pos = np.zeros(7)
            for i in range(group_1_next_pt.num_joints):
                new_pos[i] = pos[i]
            group_1_next_pt.positions = new_pos
            group_1_next_pt.velocities = [0.0] * len(pos)  #self.CalculateJointVelocety(point_time)
            group_1_next_pt.time_from_start = rospy.Duration(point_time*6 + 0.5)
        return group_1_next_pt

    def DynamicJointPoint_assembling_from_path(self, time):
        djp = DynamicJointPoint()
        djp.num_groups = self.controll_group_number
        djp.groups = [self.DynamicJointsGroup_right_from_path(time)]
        return djp

    def prepare_traj(self, path):
        traj = DynamicJointTrajectory()
        traj.points = []
        traj.joint_names = self.robot_joint_names
        for time in range(0, len(self.path) + 1):
            next_p = self.DynamicJointPoint_assembling_from_path(time)
            traj.points.append(next_p)
        haeder = Header()
        haeder.frame_id = ''
        haeder.seq = 0
        traj.header = haeder
        return traj


class PubTrajService(object):
    def __init__(self, traj):
        self.pub = rospy.Publisher("/sda10f/sda10f_r2_controller/joint_path_command", DynamicJointTrajectory, queue_size=1)
        self.traj = traj

    def Pub(self, req):
        rospy.loginfo("Moving right arm to follow trajectory")
        rate = rospy.Rate(10)
        sensor_reset = set_desired_force.desired_set_sensor()
        sensor_reset.set()
        self.pub.publish(self.traj)
        rate.sleep()
        return EmptyResponse()


def main_traj(traj):
    a = PubTrajService(traj)
    rospy.Service("Pub_right_arm_traj", Empty, a.Pub)
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('move_right_arm', anonymous=True)
    manipulator = move_routine(PATH_RIGHT)
    main_traj(manipulator.traj)