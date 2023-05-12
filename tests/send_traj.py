import os
import sys
import pickle

import rospy
from trajectory_msgs.msg import JointTrajectory

from pracsys_vbnpm.srv import ExecuteTrajectory, ExecuteTrajectoryResponse, EEControl, EEControlResponse

param = sys.argv[1] if len(sys.argv) > 1 else 'a'

if param[0] in ('1', 'a'):
    with open('traj.pkl', 'rb') as f:
        trajectory = pickle.load(f)
    # print(trajectory)
    print('execute_trajectory:')
    rospy.wait_for_service('execute_trajectory', timeout=60)
    try:
        execute_trajectory = rospy.ServiceProxy('execute_trajectory', ExecuteTrajectory)
        resp = execute_trajectory(trajectory)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if param[0] in ('2', 'a', 'g'):
    print('ee_control: close')
    rospy.wait_for_service('ee_control', timeout=60)
    try:
        ee_control = rospy.ServiceProxy('ee_control', EEControl)
        resp = ee_control('robotiq', 0.028)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if param[0] in ('3', 'a', 'g'):
    print('ee_control: open')
    rospy.wait_for_service('ee_control', timeout=60)
    try:
        ee_control = rospy.ServiceProxy('ee_control', EEControl)
        resp = ee_control('robotiq', 0.085)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
