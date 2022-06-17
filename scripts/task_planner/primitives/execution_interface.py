"""
interface from planning system to execution system
"""
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pracsys_vision_tamp_manipulation.srv import ExecuteTrajectory, AttachObject

import time
import rospy
import numpy as np

class ExecutionInterface():
    def __init__(self, scene, perception):
        self.bridge = CvBridge()
        self.attached_obj = None
        self.ros_time = 0
        self.execution_calls = 0
        self.num_executed_actions = 0
        self.num_collision = 0
        self.scene = scene
        self.perception = perception

    def execute_traj(self, joint_dict_list, ignored_obj_id=-1, duration=0.001):
        """
        call execution_system to execute the trajectory
        if an object has been attached, update the object model transform at the end
        """
        if len(joint_dict_list) == 0 or len(joint_dict_list) == 1:
            return
        
        start_time = time.time()

        # convert joint_dict_list to JointTrajectory
        traj = JointTrajectory()
        traj.joint_names = list(joint_dict_list[0].keys())
        points = []
        for i in range(len(joint_dict_list)):
            point = JointTrajectoryPoint()
            positions = []
            for name in traj.joint_names:
                if name in joint_dict_list[i]:
                    positions.append(joint_dict_list[i][name])
                else:
                    positions.append(joint_dict_list[i-1][name])
                    joint_dict_list[i][name] = joint_dict_list[i-1][name]
            point.positions = positions
            # point.time_from_start = i * 
            points.append(point)
        traj.points = points

        rospy.wait_for_service('execute_trajectory', timeout=10)
        try:
            execute_trajectory = rospy.ServiceProxy('execute_trajectory', ExecuteTrajectory)
            resp1 = execute_trajectory(traj, ignored_obj_id)
            self.num_collision += resp1.num_collision
            # print('number of collision: ', self.num_collision)
            # update object pose using the last joint angle if an object is attached
            if self.attached_obj is not None:
                start_pose = self.scene.robot.get_tip_link_pose(joint_dict_list[0])
                end_pose = self.scene.robot.get_tip_link_pose(joint_dict_list[-1])
                rel_transform = end_pose.dot(np.linalg.inv(start_pose))
                self.perception.objects[self.attached_obj].update_transform_from_relative(rel_transform)
            # update the planning scene
            for i in range(len(joint_dict_list)):
                self.scene.robot.set_joint_from_dict(joint_dict_list[i])
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        self.ros_time += time.time() - start_time
        self.execution_calls += 1
    def attach_obj(self, obj_id):
        """
        call execution_system to attach the object
        """
        start_time = time.time()
        rospy.wait_for_service('attach_object', timeout=10)
        try:
            attach_object = rospy.ServiceProxy('attach_object', AttachObject)
            resp1 = attach_object(True, self.perception.data_assoc.obj_ids_reverse[obj_id])
            self.attached_obj = obj_id
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        self.ros_time += time.time() - start_time
        self.execution_calls += 1
    def detach_obj(self):
        """
        call execution_system to detach the object
        UPDATE April 14, 2022:
        each action is finished with a detach action. So we count
        how many detach is called, this will indicate how many actions
        are performed
        """
        start_time = time.time()

        rospy.wait_for_service('attach_object', timeout=10)
        try:
            attach_object = rospy.ServiceProxy('attach_object', AttachObject)
            resp1 = attach_object(False, -1)
            self.attached_obj = None
            self.num_executed_actions += 1

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        self.ros_time += time.time() - start_time
        self.execution_calls += 1
    def get_image(self):
        print('waiting for message...')
        start_time = time.time()
        # rospy.sleep(0.2)

        color_img = rospy.wait_for_message('rgb_image', Image, timeout=10)
        depth_img = rospy.wait_for_message('depth_image', Image, timeout=10)
        seg_img = rospy.wait_for_message('seg_image', Image, timeout=10)


        color_img = self.bridge.imgmsg_to_cv2(color_img, 'passthrough')
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, 'passthrough')
        seg_img = self.bridge.imgmsg_to_cv2(seg_img, 'passthrough')

        self.ros_time += time.time() - start_time
        self.execution_calls += 1

        # visualize the images
        # cv2.imshow('img', color_img)
        # print('Press space...')
        # cv2.waitKey()

        return color_img, depth_img, seg_img
