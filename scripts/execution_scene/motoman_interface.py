"""
NOTE:
motoman robot has to specify the full joint names to track the trajectory.
"""

import numpy as np
import toppra as ta
import toppra.algorithm as ta_algo
import toppra.constraint as ta_constraint
import rospy
import rospkg
import actionlib
import message_filters
import copy
import sys
import os
import time
import random

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryGoal
from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperAction, CommandRobotiqGripperFeedback, CommandRobotiqGripperGoal

from pracsys_vision_tamp_manipulation.srv import ExecuteTrajectory, ExecuteTrajectoryRequest, ExecuteTrajectoryResponse, \
                            EEControl, EEControlResponse

rp = rospkg.RosPack()
package_path = rp.get_path('pracsys_vision_tamp_manipulation')
sys.path.insert(0, os.path.join(package_path, 'scripts'))

from execution_scene.execution_interface import ExecutionInterface
import utils.conversions as conversions


class MotomanInterface(ExecutionInterface):

    def __init__(self):
        super(MotomanInterface, self).__init__()
        self.joint_names = [
            "torso_joint_b1",
            "torso_joint_b2",  # this is a dummy joint. We only use for tracking
            "arm_left_joint_1_s",
            "arm_left_joint_2_l",
            "arm_left_joint_3_e",
            "arm_left_joint_4_u",
            "arm_left_joint_5_r",
            "arm_left_joint_6_b",
            "arm_left_joint_7_t",
            "arm_right_joint_1_s",
            "arm_right_joint_2_l",
            "arm_right_joint_3_e",
            "arm_right_joint_4_u",
            "arm_right_joint_5_r",
            "arm_right_joint_6_b",
            "arm_right_joint_7_t",
        ]

        # TODO chane values:
        rospy.set_param('/execution_interface/rgb_topic', 'rgb_image')
        rospy.set_param('/execution_interface/depth_topic', 'depth_image')

        # subscribe to split joint state topics
        self.left_state_sub = message_filters.Subscriber(
            '/sda10f/sda10f_r1_controller/joint_states', JointState
        )
        self.right_state_sub = message_filters.Subscriber(
            '/sda10f/sda10f_r2_controller/joint_states', JointState
        )
        self.torso_b1_sub = message_filters.Subscriber(
            '/sda10f/sda10f_b1_controller/joint_states', JointState
        )
        self.joint_state_sub = message_filters.ApproximateTimeSynchronizer(
            [self.left_state_sub, self.right_state_sub, self.torso_b1_sub],
            10,
            0.02,
        )
        self.joint_state_sub.registerCallback(self.joint_state_callback)
        self.joint_state_pub = rospy.Publisher(
            '/joint_state', JointState, queue_size=5
        )

        # action clients
        self.follow_trajectory_client = actionlib.SimpleActionClient(
            '/joint_trajectory_action', FollowJointTrajectoryAction
        )
        self.robotiq_client = actionlib.SimpleActionClient(
            '/command_robotiq_action', CommandRobotiqGripperAction
        )

    def joint_state_callback(
        self, left_msg: JointState, right_msg: JointState,
        torso_b1_msg: JointState
    ):
        joint_names = \
                list(left_msg.name) + \
                list(right_msg.name) + \
                list(torso_b1_msg.name)
        joint_pos = \
                list(left_msg.position) + \
                list(right_msg.position) + \
                list(torso_b1_msg.position)
        joint_vels = \
                list(left_msg.velocity) + \
                list(right_msg.velocity) + \
                list(torso_b1_msg.velocity)
        total_joint_msg = JointState()
        total_joint_msg.name = joint_names
        total_joint_msg.position = joint_pos
        total_joint_msg.velocity = joint_vels
        sec = left_msg.header.stamp.secs + right_msg.header.stamp.secs + torso_b1_msg.header.stamp.secs
        nsec = left_msg.header.stamp.nsecs + right_msg.header.stamp.nsecs + torso_b1_msg.header.stamp.nsecs
        sec = sec / 3
        nsec = nsec / 3
        total_joint_msg.header.stamp = rospy.Time(int(sec), int(nsec))
        self.joint_state_pub.publish(total_joint_msg)

    def retime_trajectroy(self, points, vel_ang_lim=15, acc_ang_lim=20):
        vel_lim = vel_ang_lim * np.pi / 180
        acc_lim = acc_ang_lim * np.pi / 180
        vel_limit = [-vel_lim, vel_lim]
        acc_limit = [-acc_lim, acc_lim]

        raw_plan = [p.positions for p in points]
        np.random.seed(0)
        random.seed(0)
        ss = np.linspace(0, 1, len(raw_plan))
        path = ta.SplineInterpolator(ss, raw_plan)
        vlims = [vel_limit] * len(raw_plan[0])
        alims = [acc_limit] * len(raw_plan[0])
        pc_vel = ta_constraint.JointVelocityConstraint(vlims)
        pc_acc = ta_constraint.JointAccelerationConstraint(alims)
        instance = ta_algo.TOPPRA([pc_vel, pc_acc], path)
        jnt_traj = instance.compute_trajectory()
        timestep = rospy.get_param('/execution_node/trajectory_timestep', 0.2)
        print('duration: ', jnt_traj.duration)
        times = np.linspace(
            0, jnt_traj.duration,
            np.ceil((jnt_traj.duration / timestep)).astype(int)
        )
        positions = jnt_traj(times)
        velocities = jnt_traj(times, 1)
        return positions, velocities, times, jnt_traj

    def execute_trajectory(self, req):
        """
        the real motoman has to specify the full joint names to track.
        It also has one dummy joint name torso_joint_b2.
        To do that we just add the dummy joint name and use the same
        value as torso_joint_b1 for it.
        """
        # * for unspecified joints, set the joint value using the current one
        # * append the current joint to the start of the trajectory so that
        # * we can execute that on the robot.
        joint_names = req.trajectory.joint_names
        # obtain the joints that are not specified
        fixed_joint_names = set(self.joint_names) - set(joint_names)
        fixed_joint_names = list(fixed_joint_names)
        cur_pos_dict = {}
        cur_joint_state = rospy.wait_for_message('joint_state', JointState)
        cur_pos_dict, _ = conversions.joint_state_to_dict(cur_joint_state)
        # default value:
        cur_pos_dict['torso_joint_b2'] = cur_pos_dict['torso_joint_b1']

        points = copy.deepcopy(req.trajectory.points)
        # add one to the start of the trajectory
        point = JointTrajectoryPoint()
        point.positions = [cur_pos_dict[name] for name in joint_names]
        point.time_from_start = 0.0
        points.insert(0, point)
        positions, velocities, times, interped_traj = self.retime_trajectroy(
            points
        )
        interpolated_joint_names = copy.deepcopy(joint_names)

        # * append the fixed joint values to the retimed trajectory
        cur_pos_list = [cur_pos_dict[name] for name in fixed_joint_names]
        cur_pos_list = np.array(cur_pos_list).reshape((1, -1)).repeat(
            len(positions), axis=0
        )
        cur_vel_list = [0 for name in fixed_joint_names]
        cur_vel_list = np.array(cur_vel_list).reshape((1, -1)).repeat(
            len(velocities), axis=0
        )
        positions = np.append(positions, cur_pos_list, axis=1)
        velocities = np.append(velocities, cur_vel_list, axis=1)
        joint_names = list(joint_names)
        joint_names += fixed_joint_names

        # * set torso_joint_b2 as torso_joint_b1 (it may be specified)
        b2_idx = joint_names.index('torso_joint_b2')
        b1_idx = joint_names.index('torso_joint_b1')
        positions[:, b2_idx] = positions[:, b1_idx]
        velocities[:, b2_idx] = velocities[:, b1_idx]

        # * formatting: format the position and velocities in the order of self.joint_names
        ordered_indices = [
            joint_names.index(self.joint_names[i])
            for i in range(len(self.joint_names))
        ]
        ordered_indices = np.array(ordered_indices)
        positions = positions[:, ordered_indices]
        velocities = velocities[:, ordered_indices]
        joint_names = copy.deepcopy((self.joint_names))  # change the order

        traj = JointTrajectory()
        traj.joint_names = joint_names
        points = []
        for i in range(len(times)):
            point = JointTrajectoryPoint()
            point.positions = positions[i]
            point.velocities = velocities[i]
            point.time_from_start = rospy.Duration(times[i])
            points.append(point)
        # last point we want the vel to be zero
        point = JointTrajectoryPoint()
        point.positions = positions[-1]
        point.velocities = np.zeros(len(positions[-1]))
        point.time_from_start = rospy.Duration(
            times[-1] + 2.0
        )  # allow a large enoguh time for stablizing
        points.append(point)
        traj.points = points

        client = self.follow_trajectory_client
        client.wait_for_server()
        action_goal = FollowJointTrajectoryGoal()
        action_goal.trajectory = traj

        self.tas_done = False

        def feedback_cb(feedback: FollowJointTrajectoryFeedback):
            # NOTE: the motoman robot is not able to send feedback
            # rospy.loginfo('Receiving Trajectory Feedback...')
            pass

        def done_cb(state, result):
            rospy.loginfo(
                f"""Trajectory Action Server is Done.
                State: {state}, Result: {result.error_code}"""
            )
            self.tas_done = True

        client.send_goal(action_goal, feedback_cb=feedback_cb, done_cb=done_cb)
        start_ros_time = rospy.get_time()
        # joint_state = rospy.wait_for_message('joint_state', JointState)
        # start_ros_time = joint_state.header.stamp.to_sec()
        print('Trajectory Goal is Sent!')

        # * write the tracked traj
        rate = rospy.Rate(30)
        tracked_time = []
        desired_time = []
        tracked_pos = []
        desired_pos = []
        joint_names = []
        while not self.tas_done:
            joint_state = rospy.wait_for_message('joint_state', JointState)
            cur_time = joint_state.header.stamp.to_sec()-start_ros_time
            tracked_time.append(joint_state.header.stamp.to_sec()-start_ros_time)
            joint_dict, _ = conversions.joint_state_to_dict(joint_state)
            cur_joint_name = list(joint_dict.keys())
            cur_pos = list(joint_dict.values())
            tracked_pos.append(cur_pos)
            inter_time = min(cur_time, times[-1])
            desired_time.append(inter_time)
            desired_pos_i = interped_traj(inter_time)
            desired_pos_i = {
                interpolated_joint_names[i]: desired_pos_i[i]
                for i in range(len(desired_pos_i))
            }
            for i in range(len(fixed_joint_names)):
                desired_pos_i[fixed_joint_names[i]
                              ] = cur_pos_dict[fixed_joint_names[i]]
            desired_pos_i = [desired_pos_i[name] for name in cur_joint_name]
            desired_pos.append(desired_pos_i)
            joint_names.append(cur_joint_name)
            rate.sleep()

        result = client.get_result()
        print('Done!', result.error_code)
        feedback_data = {}
        feedback_data['joint_names'] = joint_names
        feedback_data['desired_time'] = desired_time
        feedback_data['tracked_time'] = tracked_time
        feedback_data['tracked_pos'] = tracked_pos
        feedback_data['desired_pos'] = desired_pos
        import pickle
        f = open('tracked_data.pkl', 'wb')
        pickle.dump(feedback_data, f)
        f.close()

        return ExecuteTrajectoryResponse(result.error_code == 0)

    def robotiq_gripper_control(self, req):
        control = req.control

        client = self.robotiq_client
        client.wait_for_server()
        action_goal = CommandRobotiqGripperGoal()
        action_goal.position = control

        self.gas_done = False

        def feedback_cb(feedback):
            # rospy.loginfo('Receiving Gripper Feedback...')
            pass

        def done_cb(state, result):
            rospy.loginfo(
                f"""Gripper Action Server is Done.
                State: {state}, Result: {result.fault_status}"""
            )
            self.gas_done = True

        client.send_goal(action_goal, feedback_cb=feedback_cb, done_cb=done_cb)
        print('Gripper Goal is Sent!')

        rate = rospy.Rate(30)
        while not self.gas_done:
            rate.sleep()

        result = client.get_result()
        print('Done!', result.fault_status)
        return EEControlResponse(result.fault_status == 0)

    def ee_control(self, req):
        ee_name = req.name
        if ee_name == 'robotiq':
            return self.robotiq_gripper_control(req)
        else:
            print('No such gripper:', ee_name, '!', file=sys.stderr)
            return EEControlResponse(False)

    def reset(self, init_joint_dict: dict = None):
        """
        reset the robot to the given init_joint_dict
        """
        if init_joint_dict is None:
            init_joint_dict = \
            {
                "torso_joint_b1": 0,
                "arm_left_joint_1_s": 1.75,
                "arm_left_joint_2_l": 0.8,
                "arm_left_joint_3_e": 0,
                "arm_left_joint_4_u": -0.66,
                "arm_left_joint_5_r": 0,
                "arm_left_joint_6_b": 0,
                "arm_left_joint_7_t": 0,
                "arm_right_joint_1_s": 1.75,
                "arm_right_joint_2_l": 0.8,
                "arm_right_joint_3_e": 0.0,
                "arm_right_joint_4_u": -0.66,
                "arm_right_joint_5_r": 0,
                "arm_right_joint_6_b": 0,
                "arm_right_joint_7_t": 0
            }
        # get current state
        cur_joint_state = rospy.wait_for_message('joint_state', JointState)
        cur_joint_dict, _ = conversions.joint_state_to_dict(cur_joint_state)
        joint_names = list(init_joint_dict.keys())
        positions = list(init_joint_dict.values())
        cur_joint_pos = [cur_joint_dict[name] for name in joint_names]
        # if the current position is close enough to the init, don't need to track
        diff = np.array(positions) - np.array(cur_joint_pos)
        if np.linalg.norm(diff, ord=np.inf) <= 1e-2:
            return
        # linearly interpolate between current and target
        interpolated_pos = np.linspace(
            np.array(cur_joint_pos), np.array(positions), 20
        )
        msg = JointTrajectory()
        msg.joint_names = joint_names
        pts = []
        for i in range(len(interpolated_pos)):
            pt = JointTrajectoryPoint()
            pt.positions = interpolated_pos[i]
            pts.append(pt)
        msg.points = pts
        rospy.wait_for_service('execute_trajectory')
        execute_traj = rospy.ServiceProxy(
            'execute_trajectory', ExecuteTrajectory
        )
        req = ExecuteTrajectoryRequest()
        req.trajectory = msg
        execute_traj(req)


if __name__ == "__main__":
    rospy.init_node("execution_interface")
    # rospy.on_shutdown(lambda: os.system('pkill -9 -f execution_interface'))
    # rospy.sleep(1.0)
    execution_interface = MotomanInterface()
    # reset the trajectory
    rospy.sleep(1.0)
    execution_interface.reset()
    execution_interface.run()
