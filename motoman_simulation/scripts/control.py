#!/usr/bin/env python
import random
import sys
import copy

import natnet_msgs.msg
from std_msgs.msg import Float64MultiArray
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
from rodeval import rod
from Markers_goal import CableGoalRviz, CablePathRviz
import PID


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


class PathControl(rod, object):
    """PathControl"""

    def __init__(self):
        super(PathControl, self).__init__()
        rod.__init__(self, L=0.63, c=[0.77*0.055, 0.055, 0.055], short=True)
        # rod.__init__(self, L=0.82, c=[0.77*0.2785, 0.2785, 0.2785], short=True)

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('path_control',
                        anonymous=True)
        self.tl = tf.TransformListener()

        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_arm_left = moveit_commander.MoveGroupCommander('arm_left')
        group_arm_right = moveit_commander.MoveGroupCommander('arm_right')

        planning_frame = group_arm_right.get_planning_frame()
        # print "============ Reference frame: %s" % planning_frame
        rospy.sleep(1.2)
        _, _, self.mat = self.tf_trans('right_cable_contact_pos', 'left_cable_contact_pos')
        _, _, self.tobase_mat = self.tf_trans(planning_frame, 'left_cable_contact_pos')

        eef_link = group_arm_left.get_end_effector_link()

        group_names = robot.get_group_names()

        # Misc variables
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.group_arm_left = group_arm_left
        self.group_arm_right = group_arm_right
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        self.error_shape, self.b_curr = [], []
        self.error_a_m, self.error_a_f, self.a_list = [], [], []

    def go_to_joint_state(self):
        joint_goal = [np.deg2rad(-7), np.deg2rad(15), np.deg2rad(-9), np.deg2rad(-106), np.deg2rad(-15), np.deg2rad(-39), np.deg2rad(-23)]
        self.group_arm_right.set_max_velocity_scaling_factor(0.1)
        self.group_arm_right.set_max_acceleration_scaling_factor(0.1)
        self.group_arm_right.go(joint_goal, wait=True)
        self.group_arm_right.stop()
        current_joints = self.group_arm_left.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def plan_cartesian_path(self, group, path_a, scale=1, ind=None):

        def a_current():
            counter = 0
            while counter < 10:
                a_ = rospy.wait_for_message('/cable_forces', geometry_msgs.msg.WrenchStamped)
                try:
                    a_current_ = np.vstack((a_current_, a_))
                except:
                    a_current_ = np.array([[a_.wrench.torque.x,
                                            a_.wrench.torque.y,
                                            a_.wrench.torque.z,
                                            a_.wrench.force.x,
                                            a_.wrench.force.y,
                                            a_.wrench.force.z]])
                counter += 1
            a_current_ = np.mean(a_current_, axis=0)
            a_current_ = a_current_.T.reshape((6,))
            return a_current_

        self.df = pd.DataFrame(path_a, columns=['Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz'])
        self.df['poses'] = self.df.apply(lambda row:
                                         self.solve(
                                             np.array([row['Mx'], row['My'], row['Mz'],
                                                       row['Fx'], row['Fy'], row['Fz']]))[0],
                                         axis=1)

        goal_cable = CableGoalRviz(self.df['poses'].iloc[-1])
        goal_cable.add_cable_to_rviz()
        path_cable = CablePathRviz()
        for Q in self.df['poses'][::]: #len(self.df['poses'])   /10
            path_cable.add_marker_to_array(Q)
        path_cable.PublishRviz()

        self.df['xyz_end'] = self.df['poses'].apply(lambda row: np.matmul(self.tobase_mat, row[-1, :, 3])[:3])
        self.df['quat_end'] = self.df['poses'].apply(lambda row:
                                                     np.matmul(row[-1, :3, :3], np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])))
        self.df['quat_end'] = self.df['quat_end'].apply(lambda row:
                                                        R.from_dcm(np.matmul(self.tobase_mat[:3,:3], row)).as_quat())

        self.error_shape, self.a_list, self.b_curr = [], [], []
        for index, row in self.df.iterrows():
            waypoints = []
            wpose = group.get_current_pose().pose
            # Adding noise to the system:
            wpose.position.x = row['xyz_end'][0] + np.random.normal(loc=0., scale=0.02)
            wpose.position.y = row['xyz_end'][1] + np.random.normal(loc=0., scale=0.02)
            wpose.position.z = row['xyz_end'][2] + np.random.normal(loc=0., scale=0.08)
            wpose.orientation.x = row['quat_end'][0] + np.random.normal(loc=0., scale=0.05)
            wpose.orientation.y = row['quat_end'][1] + np.random.normal(loc=0., scale=0.05)
            wpose.orientation.z = row['quat_end'][2] + np.random.normal(loc=0., scale=0.05)
            wpose.orientation.w = row['quat_end'][3] + np.random.normal(loc=0., scale=0.05)

            waypoints.append(copy.deepcopy(wpose))
            (plan, fraction) = group.compute_cartesian_path(
                waypoints,  # waypoints to follow
                0.001,  # eef_step
                0.0)  # jump_threshold
            group.execute(plan, wait=True)
            rospy.sleep(1)
            a_curr = a_current()
            a_wanted = np.array([row['Mx'], row['My'], row['Mz'], row['Fx'], row['Fy'], row['Fz']])
            shape = rospy.wait_for_message('/client/markers/leftovers', natnet_msgs.msg.MarkerList)
            mocap = rospy.wait_for_message('/A_mocap', Float64MultiArray)
            a_mocap = np.array(mocap.data).reshape(4, 4)
            array = np.array([[0, 0, 0]])
            for marker in shape.positions:
                mark = np.dot(a_mocap, np.array([[marker.x], [marker.y], [marker.z], [1]]))[:-1]
                array = np.vstack((array, mark.T))
                points = array
            err = self.Error(a_wanted, points)
            self.error_shape.append(err)
            self.error_a_m.append(np.linalg.norm(a_wanted[:3] - a_curr[:3]))
            self.error_a_f.append(np.linalg.norm(a_wanted[3:] - a_curr[3:]))
            self.a_list.append(a_curr)
            current = group.get_current_pose().pose
            self.b_curr.append(np.array([current.position.x, current.position.y, current.position.z, current.orientation.x,
                                current.orientation.y, current.orientation.z, current.orientation.w]))

        if ind is None:
            with open('OpenLoopErrors.pkl', 'wb') as h:
                pickle.dump([self.error_shape, self.error_a_m, self.error_a_f], h)
        else:
            with open('a_path_rrt/a_pathr{}.pkl'.format(ind), 'wb') as h:
                pickle.dump([self.a_list, self.error_shape, self.b_curr, path_a], h)

    def control_a_b(self, group, path_a, scale=1, ind=None):

        self.df = pd.DataFrame(path_a, columns=['Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz'])
        self.df['poses'] = self.df.apply(lambda row:
                                         self.solve(
                                             np.array([row['Mx'], row['My'], row['Mz'],
                                                       row['Fx'], row['Fy'], row['Fz']]))[0],
                                         axis=1)
        self.df['xyz_end'] = self.df['poses'].apply(lambda row: np.matmul(self.tobase_mat, row[-1, :, 3])[:3])
        self.df['quat_end'] = self.df['poses'].apply(lambda row:
                                                     np.matmul(row[-1, :3, :3], np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])))
        self.df['quat_end'] = self.df['quat_end'].apply(lambda row:
                                                        R.from_dcm(np.matmul(self.tobase_mat[:3,:3], row)).as_quat())
        goal_cable = CableGoalRviz(self.df['poses'].iloc[-1])
        goal_cable.add_cable_to_rviz()
        path_cable = CablePathRviz()
        self.error_shape, self.a_list, self.b_curr = [], [], []
        for index, row in self.df.iterrows():
            next_goal = CableGoalRviz(row['poses'])
            next_goal.add_cable_to_rviz()

            a_wanted = np.array([row['Mx'], row['My'], row['Mz'], row['Fx'], row['Fy'], row['Fz']])

            def a_current():
                counter = 0
                while counter < 20:
                    a_ = rospy.wait_for_message('/cable_forces', geometry_msgs.msg.WrenchStamped)
                    try:
                        a_current_ = np.vstack((a_current_, a_))
                    except:
                        a_current_ = np.array([[a_.wrench.torque.x,
                                            a_.wrench.torque.y,
                                            a_.wrench.torque.z,
                                            a_.wrench.force.x,
                                            a_.wrench.force.y,
                                            a_.wrench.force.z]])
                    counter += 1
                a_current_ = np.mean(a_current_, axis=0)
                a_current_ = a_current_.T.reshape((6,))
                return a_current_

            a_curr = a_current()
            pid = PID.PID(0.8, 0., 0.)
            pid.SetPoint = a_wanted
            while (np.linalg.norm(a_wanted[:3] - a_curr[:3]) > 0.08) or (np.linalg.norm(a_wanted[3:] - a_curr[3:]) > 0.5): # 2 * 1e-1
                print(np.linalg.norm(a_wanted[:3] - a_curr[:3]), np.linalg.norm(a_wanted[3:] - a_curr[3:]))
                Rod = rod(c=[0.77*0.055, 0.055, 0.055], L=0.63, short=False)

                Q, _, _, Jl = Rod.solve(a_curr, check_collision=False)
                pid.update(a_curr)
                delta_b = np.dot(Jl, pid.output)
                b_xyz = np.dot(self.tobase_mat, np.dot(Q[-1], np.hstack((delta_b[3:], 1))))[:3]

                xyz_start = np.matmul(self.tobase_mat, Q[-1, :, 3])[:3]
                rot_start = np.matmul(Q[-1, :3, :3], np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
                rot_start = rot_start_mat = np.matmul(self.tobase_mat[:3, :3], rot_start)
                rot_start = -R.from_dcm(rot_start).as_quat()

                rotvec = R.from_rotvec(delta_b[:3])
                RR = rotvec.as_dcm()
                RR = np.matmul(RR, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
                rot = np.dot(self.tobase_mat[:3,:3], np.dot(Q[-1, :3, :3], RR))
                quat = R.from_dcm(rot)
                print(quat.as_rotvec())
                quat = -quat.as_quat()
                waypoints = []
                wpose = current = group.get_current_pose().pose
                current = np.array([current.position.x, current.position.y, current.position.z, current.orientation.x,
                                    current.orientation.y, current.orientation.z, current.orientation.w])
                r = R.from_quat(current[3:])
                new_b_orien = R.from_rotvec(r.as_rotvec() + 0.1*delta_b[:3])
                new_b_orien = new_b_orien.as_quat()

                xyz_diff = b_xyz - xyz_start # Without Noise
                xyz_diff = np.random.normal(loc=xyz_diff, scale=np.abs(xyz_diff/100)) # With Noise
                wpose.position.x += xyz_diff[0]
                wpose.position.y += xyz_diff[1]
                wpose.position.z += xyz_diff[2]

                diff = np.matmul(rot, np.linalg.inv(rot_start_mat))
                curr_quat = R.from_quat(np.array([wpose.orientation.x, wpose.orientation.y, wpose.orientation.z, wpose.orientation.w]))
                curr_quat = np.matmul(diff, curr_quat.as_dcm())
                curr_quat = R.from_dcm(curr_quat).as_quat()
                wpose.orientation.x = curr_quat[0]
                wpose.orientation.y = curr_quat[1]
                wpose.orientation.z = curr_quat[2]
                wpose.orientation.w = curr_quat[3]

                waypoints.append(copy.deepcopy(wpose))
                print('delta b: {},{}'.format(b_xyz, new_b_orien - current[3:])) # delta_b[3:]
                (plan, fraction) = group.compute_cartesian_path(waypoints, 0.001, 0.0)
                group.execute(plan, wait=True)
                rospy.sleep(1)
                a_curr = a_current()

            shape = rospy.wait_for_message('/client/markers/leftovers', natnet_msgs.msg.MarkerList)
            mocap = rospy.wait_for_message('/A_mocap', Float64MultiArray)
            a_mocap = np.array(mocap.data).reshape(4, 4)
            array = np.array([[0, 0, 0]])
            for marker in shape.positions:
                mark = np.dot(a_mocap, np.array([[marker.x], [marker.y], [marker.z], [1]]))[:-1]
                array = np.vstack((array, mark.T))
                points = array

            err = self.Error(a_wanted, points) # a_wanted
            self.error_shape.append(err)
            self.error_a_m.append(np.linalg.norm(a_wanted[:3] - a_curr[:3]))
            self.error_a_f.append(np.linalg.norm(a_wanted[3:] - a_curr[3:]))
            self.a_list.append(a_curr)
            current = group.get_current_pose().pose
            self.b_curr.append(np.array([current.position.x, current.position.y, current.position.z, current.orientation.x,
                                current.orientation.y, current.orientation.z, current.orientation.w]))

        if ind is None:
            with open('ControlErrors.pkl', 'wb') as h:
                pickle.dump([self.error_shape, self.error_a_m, self.error_a_f], h)
        else:
            with open('a_path_rrt/a_path_obs{}.pkl'.format(ind+9), 'wb') as h:
                pickle.dump([self.a_list, self.error_shape, self.b_curr, path_a], h)

    def Error(self, a, P):

        def closest(p, P_):
            d = np.sum((P_ - p) ** 2, axis=1)
            return np.min(d)

        c2 = c3 = 0.055
        c1 = 0.77 * c2
        L = 0.63
        Rod = rod(c=[c1, c2, c3], L=L, short=True)
        Q, _, _, _ = Rod.solve(a, check_collision=False)
        Pr = Q[:, :3, 3]
        F = np.sqrt(np.sum([closest(p, Pr) for p in P]))
        F = (F / P.shape[0]) * 1000
        return F

    def tf_trans(self, target_frame, source_frame):
        """
        Transformation between two frames
        From source to target
        """
        try:
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            mat = self.tl.fromTranslationRotation(trans, rot)
            return trans, rot, mat

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None, None

    def a_space(self):
        for i in range(0, 87167):
            try:
                with open('aSpaceData/solutions3/solution_' + str(i) + '.pkl', 'rb') as h:
                    try:
                        a.append(pickle.load(h)[0])
                    except:
                        a = [pickle.load(h)[0]]
            except:
                print('could not load data_estimation!')
        return a


    def plan_RRTStar(self, start, goal, population=None):
        rrt_star = RRTStar(
            start=start,
            goal=goal,
            expand_dis=0.3,
            path_resolution=0.3,
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

    def planLineA(self, start, goal, resolution=20.):
        step = (goal - start)/resolution
        for s in range(int(resolution)+1):
            try:
                path = np.vstack((path, path[-1,:]+step))
            except:
                path = np.array([start])
        return path


    def GoToGoal(self, a_space, a_start):
        a_goals = [a_space[random.randint(0, len(a_space))] for i in range(3)]

        for j in range(0, 1): #len(a_goals)
            self.go_to_joint_state()

            with open('a_path_rrt/a_path_obs{}.pkl'.format(4), 'rb') as h:
                path = pickle.load(h)[3]

            self.plan_cartesian_path(self.group_arm_right, path_a=path, ind=j)


def main():
    try:

        print "============ Press `Enter` to set moveit_commander up (press ctrl-d to exit) ..."
        raw_input()
        traj_movement = PathControl()

        print "============ Press `Enter` to execute a zero calibration by movement using a joint state goal ..."
        raw_input()
        traj_movement.go_to_joint_state()
        print(traj_movement.group_arm_right.get_current_pose())
        sensor_reset = set_desired_force.desired_set_sensor()
        sensor_reset.set()
        rospy.sleep(1)

        print "============ Press `Enter` to plan RRTStar / Line"
        raw_input()

        a_start = rospy.wait_for_message('/cable_forces', geometry_msgs.msg.WrenchStamped)
        a_start = np.array([[a_start.wrench.torque.x,
                            a_start.wrench.torque.y,
                            a_start.wrench.torque.z,
                            a_start.wrench.force.x,
                            a_start.wrench.force.y,
                            a_start.wrench.force.z]])
        a_start = a_start.T.reshape((6,))
        #   --------------------------------- #

        # Getting forces space from recorded data_estimation and defining a goal:
        a_space = traj_movement.a_space()

        # Go to point:
        traj_movement.GoToGoal(a_space=a_space, a_start=a_start)

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
