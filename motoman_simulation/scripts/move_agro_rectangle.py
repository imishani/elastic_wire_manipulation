#!/usr/bin/env python


import sys
import os
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import pickle
import pandas as pd
import numpy as np
from moveit_commander.conversions import pose_to_list
import tf
from scipy.spatial.transform import Rotation as R
from rodeval_avishai import rod
sys.path.append('/../../bota_driver/bota_demo/scripts')
import set_desired_force
sys.path.insert(1, "/../../motoman_cable/scripts/path_plan/")


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class TrajectoryRRTstar(rod, object):
    """TrajectoryRRTstar"""

    def __init__(self):
        super(TrajectoryRRTstar, self).__init__()
        rod.__init__(self, L=0.88, c=[0.77, 0.4, 0.4], short=True)
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_sda10f',
                        anonymous=True)
        self.tl = tf.TransformListener()
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        group_arm_left = moveit_commander.MoveGroupCommander('arm_left')
        group_arm_right = moveit_commander.MoveGroupCommander('arm_right')
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        planning_frame = group_arm_left.get_planning_frame()
        eef_link = group_arm_left.get_end_effector_link()

        group_names = robot.get_group_names()
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.group_arm_left = group_arm_left
        # self.group_arm_left_torso = group_arm_left_torso
        self.group_arm_right = group_arm_right
        # self.group_arm_right_torso = group_arm_right_torso
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names


    def go_to_joint_state(self):

        joint_goal = [0.336743, 0.2129, 0.230598, -1.610194, 0.012469, -1.230465,
                      -1.355588]    # regular spot!!

        self.group_arm_right.set_max_velocity_scaling_factor(0.1)
        self.group_arm_right.set_max_acceleration_scaling_factor(0.1)
        self.group_arm_right.go(joint_goal, wait=True)
        self.group_arm_right.stop()

        current_joints = self.group_arm_left.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def plan_cartesian_path(self, group, scale=1):

        waypoints = []

        wpose = group.get_current_pose().pose
        x = np.random.uniform(low=0.43, high=0.68, size=(50,))
        y = np.random.uniform(low=-0.1, high=0.05, size=(50,))
        z = np.random.uniform(low=0.8, high=1.4, size=(50,))
        x_o = np.random.uniform(low=-0.75, high=-0.45, size=(50,))
        y_o = np.random.uniform(low=-0.75, high=-0.45, size=(50,))
        z_o = np.random.uniform(low=0.05, high=0.4, size=(50,))
        w = np.random.uniform(low=0.05, high=0.4, size=(50,))

        for ind in range(50):
            wpose.position.x = x[ind]
            wpose.position.y = y[ind]
            wpose.position.z = z[ind]
            quat = np.array([x_o[ind], y_o[ind], z_o[ind], w[ind]])
            quat = quat / np.linalg.norm(quat)
            wpose.orientation.x, wpose.orientation.y, wpose.orientation.z, wpose.orientation.w = quat[0], quat[1], \
                                                                                                 quat[2], quat[3]
            waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0 disabling:
        (plan, fraction) = group.compute_cartesian_path(
            waypoints,  # waypoints to follow
            0.00015,  # eef_step
            0.0)  # jump_threshold

        return plan, fraction

    @ staticmethod
    def execute_plan(self, plan, group):
        group.execute(plan, wait=True)

    def tf_trans(self, target_frame, source_frame):
        try:
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            mat = self.tl.fromTranslationRotation(trans, rot)
            return trans, rot, mat

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None, None


def main():
    reset = True
    try:
        print("============ Press `Enter` to begin (press ctrl-d to exit) ...")
        raw_input()
        traj_movement = TrajectoryRRTstar()

        traj_movement.go_to_joint_state()
        if reset:
            sensor_reset = set_desired_force.desired_set_sensor()
            sensor_reset.set()
            rospy.sleep(1)
            # sensor_reset.set()
        print("============ Press `Enter` to plan and display a Cartesian path ...")
        raw_input()
        cartesian_plan, fraction = traj_movement.plan_cartesian_path(traj_movement.group_arm_right)

        print("============ Press `Enter` to execute a saved path ...")
        raw_input()
        traj_movement.execute_plan(cartesian_plan, traj_movement.group_arm_right)
        print("============ Press `Enter` to go back to zero position ...")
        raw_input()
        traj_movement.go_to_joint_state()

        print("============ Python traj_movement complete!")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
