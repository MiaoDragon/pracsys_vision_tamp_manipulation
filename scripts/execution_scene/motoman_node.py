"""
motoman execution node inheriting general execution_node
NOTE:
on the real robot, torso_joint_b2 mimics torso_joint_b1.
This only affects when we execute the trajectory.
We can add b2 if b1 is in the execute_trajectory
"""
import sys
import time
import copy
import numpy as np
from scipy.interpolate import CubicHermiteSpline

import rospy
import actionlib
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult
from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperFeedback, CommandRobotiqGripperResult, CommandRobotiqGripperAction, CommandRobotiqGripperGoal

from execution_node import ExecutionNode


class MotomanNode(ExecutionNode):

    def __init__(self, xml_file, gui=True):
        """
        define motoman-specific fields and interfaces
        """
        self.joint_names = [
            "torso_joint_b1",
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
        self.init_joints = {
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
        self.gripper_name = 'left_driver_joint'
        # self.ee_names = [self.gripper_name]

        super(MotomanNode, self).__init__(xml_file, gui)

        self.max_stroke = rospy.get_param('~stroke', 0.085)

        # * init the motoman-specific interfaces
        self.left_pub = rospy.Publisher(
            '/sda10f/sda10f_r1_controller/joint_states',
            JointState,
            queue_size=5
        )
        self.right_pub = rospy.Publisher(
            '/sda10f/sda10f_r2_controller/joint_states',
            JointState,
            queue_size=5
        )
        self.torso_b1_pub = rospy.Publisher(
            '/sda10f/sda10f_b1_controller/joint_states',
            JointState,
            queue_size=5
        )

        # ** action servers
        # robot
        self.follow_trajectory_as = actionlib.SimpleActionServer(
            "/joint_trajectory_action",
            FollowJointTrajectoryAction,
            execute_cb=self.follow_trajectory_cb,
            auto_start=False
        )
        self.follow_trajectory_as.start()

        # gripper
        self.robotiq_gripper_as = actionlib.SimpleActionServer(
            "command_robotiq_action",
            CommandRobotiqGripperAction,
            execute_cb=self.robotiq_gripper_cb,
            auto_start=False
        )
        self.robotiq_gripper_as.start()

    def follow_trajectory_cb(self, goal_command):
        points = goal_command.trajectory.points
        joint_names = goal_command.trajectory.joint_names
        # change torso_joint_b2 to torso_joint_b1
        rospy.logdebug(f"New goal received! Trajectory Length:{len(points)}")
        result = FollowJointTrajectoryResult()

        if self.arm_trajectory:
            rospy.logdebug('Already executing a trajectory!')
            result.error_code = -6
            self.follow_trajectory_as.set_aborted(result)
            return

        times = np.zeros(len(points))
        positions = np.zeros((len(points), len(points[0].positions)))
        velocities = np.zeros((len(points), len(points[0].velocities)))
        for i, p in enumerate(points):
            times[i] = p.time_from_start.to_sec()
            positions[i] = p.positions
            velocities[i] = p.velocities

        # interpolate by spline interpolation
        path = CubicHermiteSpline(times, positions, velocities)
        new_times = np.arange(0, times[-1], self.model.opt.timestep)
        new_pos = path(new_times)
        new_vel = path(new_times, 1)
        new_vel[-1] = velocities[-1]
        new_acc = path(new_times, 2)


        times = new_times
        positions = new_pos
        velocities = new_vel

        # if 'torso_joint_b2' is in the joint_name, change to b1
        mjk_joint_names = copy.deepcopy(joint_names)
        if 'torso_joint_b2' in joint_names:
            idx = joint_names.index('torso_joint_b2')
            mjk_joint_names[idx] = 'torso_joint_b1'

        self.arm_trajectory = list(
            zip([mjk_joint_names] * len(times), positions, velocities, times)
        )

        # While moving provide feedback and check for result
        rate = rospy.Rate(30)
        start_time = time.time()
        while not rospy.is_shutdown() and len(self.arm_trajectory) > 0:
            positions, velocities = self.get_joint_state(mjk_joint_names)
            actual = JointTrajectoryPoint()
            actual.positions = positions
            actual.velocities = velocities
            cur_time = time.time() - start_time
            actual.time_from_start = rospy.Time(cur_time)
            desired = JointTrajectoryPoint()
            desired.positions = path([cur_time])[0]
            desired.velocities = path([cur_time], 1)[0]

            feedback = FollowJointTrajectoryFeedback()
            feedback.joint_names = joint_names
            feedback.actual = actual
            feedback.desired = desired
            self.follow_trajectory_as.publish_feedback(feedback)
            rate.sleep()

        result.error_code = 0
        self.follow_trajectory_as.set_succeeded(result)

    def robotiq_gripper_cb(self, goal_command):
        """
        ref: https://github.com/Danfoa/robotiq_2finger_grippers/blob/master/robotiq_2f_gripper_control/scripts/robotiq_2f_action_server.py
        goal_command:
            position: 0 ~ max_stroke. 0 means closing.
            (in Mujoco we control in range [0,255], where 0 means open and 255 means closing)
        NOTE:
        updated code: gripper xml, control. Need testing.
        """
        result = CommandRobotiqGripperResult()
        if self.ee_trajectory:
            rospy.logdebug('Already controlling a gripper!')
            result.fault_status = 1
            self.follow_trajectory_as.set_aborted(result)
            return
        gctrl = self.get_ctrl_indices([self.gripper_name])
        # * set the trajectory
        # get the start position in [0,255]
        start_position = self.get_joint_state([self.gripper_name])[0][0]
        start_position = np.interp(start_position, [0.004, 0.79], [0, 255])
        requested_pos = np.interp(
            goal_command.position,
            [0, self.max_stroke],
            [255, 0],
        )
        requested_vel = goal_command.speed if goal_command.speed > 0 else 1
        requested_vel *= 255 / self.max_stroke
        requested_time = np.linalg.norm(
            start_position - requested_pos
        ) / requested_vel
        positions = np.linspace(
            start_position,
            requested_pos,
            int(requested_time / self.model.opt.timestep),
        )

        self.ee_trajectory = list(
            zip([[self.gripper_name]] * len(positions), positions)
        )

        # Wait until goal is achieved and provide feedback
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            position = self.get_joint_state([self.gripper_name])[0][0]
            position = np.interp(position, [0.004, 0.79], [0, 255])
            position = np.interp(position, [0, 255], [self.max_stroke, 0])
            feedback = CommandRobotiqGripperFeedback()
            feedback.requested_position = goal_command.position
            feedback.position = position
            feedback.is_moving = True
            self.robotiq_gripper_as.publish_feedback(feedback)
            rate.sleep()
            if len(self.ee_trajectory) == 0:
                n_position = self.get_joint_state([self.gripper_name])[0][0]
                n_position = np.interp(n_position, [0.004, 0.79], [0, 255])
                n_position = np.interp(
                    n_position, [0, 255], [self.max_stroke, 0]
                )
                if np.allclose(position, n_position):
                    result.requested_position = goal_command.position
                    result.position = n_position
                    result.is_moving = False
                    break

        self.robotiq_gripper_as.set_succeeded(result)

    def publish_joint_state(self):
        """
        obtain joint state from Mujoco and publish
        """
        torso_b1 = ["torso_joint_b1"]
        left = [
            "arm_left_joint_1_s",
            "arm_left_joint_2_l",
            "arm_left_joint_3_e",
            "arm_left_joint_4_u",
            "arm_left_joint_5_r",
            "arm_left_joint_6_b",
            "arm_left_joint_7_t",
        ]
        right = [
            "arm_right_joint_1_s",
            "arm_right_joint_2_l",
            "arm_right_joint_3_e",
            "arm_right_joint_4_u",
            "arm_right_joint_5_r",
            "arm_right_joint_6_b",
            "arm_right_joint_7_t",
        ]
        pub_jnts = [
            (self.torso_b1_pub, torso_b1),
            (self.left_pub, left),
            (self.right_pub, right),
        ]
        for pub, joint_names in pub_jnts:
            position, velocitiy = self.get_joint_state(joint_names)
            msg = JointState()
            msg.name = joint_names
            msg.position = position
            msg.velocity = velocitiy
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("execution_node")
    # rospy.on_shutdown(lambda: os.system('pkill -9 -f execution_node'))
    # rospy.sleep(1.0)
    scene_xml = sys.argv[1].strip() if len(sys.argv) > 1 else None
    gui = sys.argv[2][0] in ('t', 'y') if len(sys.argv) > 2 else False

    execution_node = MotomanNode(scene_xml, gui)
    execution_node.run()
