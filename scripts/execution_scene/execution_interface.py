import numpy as np

import rospy
import actionlib
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryGoal

from pracsys_vision_tamp_manipulation.srv import ExecuteTrajectory, ExecuteTrajectoryResponse, EEControl, EEControlResponse


class ExecutionInterface():

    def __init__(self):
        rospy.Service(
            "execute_trajectory",
            ExecuteTrajectory,
            self.execute_trajectory,
        )
        rospy.Service("ee_control", EEControl, self.ee_control)

    def execute_trajectory(self, req):
        pass

    def ee_control(self, req):
        pass

    def run(self):
        while not rospy.is_shutdown():
            pass


if __name__ == "__main__":
    rospy.init_node("execution_interface")
    # rospy.on_shutdown(lambda: os.system('pkill -9 -f execution_interface'))
    # rospy.sleep(1.0)
    execution_interface = ExecutionInterface()
    execution_interface.run()
