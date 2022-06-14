"""
The interface to the task planner to set up the required components.
- task-related description
- human instruction interface to communicate with the task planner
This is only for demonstration and software purpose. For ease of experiments,
try to use command line as input to construct task_planner
--------------------------------------------------------------------------
namespace: task_interface/
provided services:
- task_interface/solve: solve the specified task by the task planner
- task_interface/reset: reset the task planner using the current rosparam
"""
import rospy
import json
from task_planner.task_planner import TaskPlanner

from pracsys_vision_tamp_manipulation.srv import TaskSolve

class TaskInterface():
    def __init__(self):
        self.get_rosparam()  # this reads from rosparam which defines the problem file
        # the rosparam is specified through roslaunch file    
        self.task_planner = TaskPlanner(self.scene_name, self.prob_id)
        # initialize ROS services
        rospy.Service("task_interface/solve", TaskSolve, self.solve_fb)
        # rospy.Service("task_interface/reset", TaskReset, self.reset_fb)


    def get_rosparam(self):
        while not rospy.has_param('/task_interface/scene_name'):
            rospy.sleep(0.2)
        self.scene_name = scene_name = rospy.get_param('/task_interface/scene_name')
        while not rospy.has_param('/task_interface/prob_id'):
            rospy.sleep(0.2)
        self.prob_id = scene_name = rospy.get_param('/task_interface/prob_id')
