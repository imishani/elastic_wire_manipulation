#!/usr/bin/env python
import random
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
from Markers_goal import CableGoalRviz, CablePathRviz

import torch
from torch.autograd import Variable

sys.path.insert(0, "/../../motoman_cable/scripts/path_plan")
sys.path.append('/../../bota_driver/bota_demo/scripts')
import set_desired_force
from rrt_star import RRTStar
from rrt import RRT


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


class MovementRRTStar(rod, object):
    """MovementRRTStar"""

    def __init__(self):
        super(MovementRRTStar, self).__init__()
        rod.__init__(self, L=0.63, c=[0.77*0.055, 0.055, 0.055], short=True)
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('RRTStar_agro',
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
        rospy.sleep(1.2)
        _, _, self.mat = self.tf_trans('right_cable_contact_pos', 'left_cable_contact_pos')
        _, _, self.tobase_mat = self.tf_trans(planning_frame, 'left_cable_contact_pos')

        eef_link = group_arm_left.get_end_effector_link()

        group_names = robot.get_group_names()
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.group_arm_left = group_arm_left
        self.group_arm_right = group_arm_right
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_joint_state(self):
        joint_goal = [np.deg2rad(-7), np.deg2rad(15), np.deg2rad(-9), np.deg2rad(-106), np.deg2rad(-15), np.deg2rad(-39), np.deg2rad(-23)] # for obstacle video
        self.group_arm_right.set_max_velocity_scaling_factor(0.1)
        self.group_arm_right.set_max_acceleration_scaling_factor(0.1)
        self.group_arm_right.go(joint_goal, wait=True)

        self.group_arm_right.stop()

        current_joints = self.group_arm_left.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def plan_cartesian_path(self, group, path_a, scale=1):

        waypoints = []
        self.df = pd.DataFrame(path_a, columns=['Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz'])
        self.df['poses'] = self.df.apply(lambda row:
                                         self.solve(
                                             np.array([row['Mx'], row['My'], row['Mz'],
                                                       row['Fx'], row['Fy'], row['Fz']]))[0],
                                         axis=1)

        goal_cable = CableGoalRviz(self.df['poses'].iloc[-1])
        goal_cable.add_cable_to_rviz()
        path_cable = CablePathRviz()
        for Q in self.df['poses'][::len(self.df['poses'])/10]: # int(round(0.9*len(self.df['poses'])))
            path_cable.add_marker_to_array(Q)
        path_cable.PublishRviz()

        self.df['xyz_end'] = self.df['poses'].apply(lambda row: np.matmul(self.tobase_mat, row[-1, :, 3])[:3])
        self.df['quat_end'] = self.df['poses'].apply(lambda row:
                                                     np.matmul(row[-1, :3, :3], np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])))
        self.df['quat_end'] = self.df['quat_end'].apply(lambda row:
                                                        R.from_dcm(np.matmul(self.tobase_mat[:3,:3], row)).as_quat())

        wpose = curr = group.get_current_pose().pose
        for index, row in self.df.iterrows():
            wpose.position.x = row['xyz_end'][0]
            wpose.position.y = row['xyz_end'][1] # - 0.02
            wpose.position.z = row['xyz_end'][2]
            wpose.orientation.x = row['quat_end'][0]
            wpose.orientation.y = row['quat_end'][1]
            wpose.orientation.z = row['quat_end'][2]
            wpose.orientation.w = row['quat_end'][3]
            waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
            waypoints,  # waypoints to follow
            0.0001,  # eef_step
            0.0)  # jump_threshold

        return plan, fraction


    def display_trajectory(self, plan):

        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        display_trajectory_publisher.publish(display_trajectory)

    def execute_plan(self, plan, group):

        group.execute(plan, wait=True)


    def tf_trans(self, target_frame, source_frame):
        try:
            # listen to transform
            print(self.tl.allFramesAsString())
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            mat = self.tl.fromTranslationRotation(trans, rot)
            return trans, rot, mat

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None, None

    def a_space(self):
        for i in range(0, 87167): # 54815   45952
            with open('aSpaceData/solutions3/solution_' + str(i) + '.pkl', 'rb') as h: # solutions_horizontal
                try:
                    a.append(pickle.load(h)[0])
                except:
                    a = [pickle.load(h)[0]]
        return a

    def plan_RRTStar(self, start, goal, population=None):
        rrt_star = RRTStar(
            start=start,
            goal=goal,
            expand_dis=0.4,
            path_resolution=0.2,
            max_iter=300,
            search_until_max_iter=False)
        path = rrt_star.planning(population=population)

        if path is None:
            raise ValueError("Couldn't find path")

        else:
            print("found path!!")
            print(len(path))
            return np.array([p.a for p in path])[::-1, :]

    def plan_RRT(self, start, goal, population=None):
        rrt = RRT(
            start=start,
            goal=goal,
            expand_dis=1.,
            path_resolution=0.5,
            max_iter=300,
            search_until_max_iter=False)
        path = rrt.planning(population=population)

        if path is None:
            raise ValueError("Couldn't find path")

        else:
            print("found path!!")
            print(len(path))
            return np.array([p.a for p in path])[::-1, :]


def main():
    try:

        print "============ Press `Enter` to set moveit_commander up (press ctrl-d to exit) ..."
        raw_input()
        traj_movement = MovementRRTStar()

        print "============ Press `Enter` to execute a zero calibration by movement using a joint state goal ..."
        raw_input()
        traj_movement.go_to_joint_state()
        print(traj_movement.group_arm_right.get_current_pose())
        sensor_reset = set_desired_force.desired_set_sensor()
        sensor_reset.set()

        print "============ Press `Enter` to plan RRTStar"
        raw_input()

        a_start = rospy.wait_for_message('/cable_forces', geometry_msgs.msg.WrenchStamped)
        a_start = np.array([[a_start.wrench.torque.x,
                            a_start.wrench.torque.y,
                            a_start.wrench.torque.z,
                            a_start.wrench.force.x,
                            a_start.wrench.force.y,
                            a_start.wrench.force.z]])

        a_start = a_start.T.reshape((6,))

        # Getting forces space from recorded data_estimation and defining a goal:
        a_space = traj_movement.a_space()
        a_goal = a_space[random.randint(0, len(a_space))]
        path = traj_movement.plan_RRTStar(a_start, a_goal, population=a_space)

        print "============ Press `Enter` to plan and display a Cartesian path ..."
        raw_input()
        cartesian_plan, fraction = traj_movement.plan_cartesian_path(traj_movement.group_arm_right, path_a=path)

        print "============ Press `Enter` to display a saved trajectory (this will replay the Cartesian path)  ..."
        raw_input()
        traj_movement.display_trajectory(cartesian_plan)

        print "============ Press `Enter` to execute a saved path ..."
        raw_input()
        traj_movement.execute_plan(cartesian_plan, traj_movement.group_arm_right)

        print "============ Press `Enter` to plan and display a Cartesian path ..."
        raw_input()

        print "============ Python traj_movement complete!"
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
