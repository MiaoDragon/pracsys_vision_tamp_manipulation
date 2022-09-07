#!/usr/bin/env python
import numpy as np
import os
import shutil
import time
import sys
import copy
import IPython

import rospy
import rospkg

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point

import cv2
import open3d as o3d

from cv_bridge import CvBridge

import matplotlib.pyplot as plt
import skg
import pickle
from pracsys_vision_tamp_manipulation.srv import SegmentationSrv
from scene.sim_scene import SimScene
import transformations as tf
import trimesh
### This file defines the real camera for the purpose of getting the images of the physical camera

class CylinderSegmentation():

    def __init__(self, scene: SimScene):
        self.scene = scene
        # self.camera = camera  # camera model
        self.bridge = CvBridge()
        self.getDepthIntrinsicInfo()
    def getDepthIntrinsicInfo(self):
        intrinsics = self.scene.camera.info['intrinsics']
        self.depth_camera_info = dict()
        self.depth_camera_info['intrinsics'] = dict()
        self.depth_camera_info['intrinsics']['fx'] = intrinsics[0,0]
        self.depth_camera_info['intrinsics']['fy'] = intrinsics[1,1]
        self.depth_camera_info['intrinsics']['ppx'] = intrinsics[0,2]
        self.depth_camera_info['intrinsics']['ppy'] = intrinsics[1,2]
        self.depth_camera_info['height'] = self.scene.camera.info['img_shape'][0]
        self.depth_camera_info['width'] = self.scene.camera.info['img_shape'][1]

        extrinsics = self.scene.camera.info['extrinsics']
        self.depth_camera_info['extrinsics'] = np.array(extrinsics)

        # depth_camera_info_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        # self.depth_camera_info = dict()
        # self.depth_camera_info['intrinsics'] = dict()
        # self.depth_camera_info['intrinsics']['fx'] = depth_camera_info_msg.K[0]
        # self.depth_camera_info['intrinsics']['fy'] = depth_camera_info_msg.K[4]
        # self.depth_camera_info['intrinsics']['ppx'] = depth_camera_info_msg.K[2]
        # self.depth_camera_info['intrinsics']['ppy'] = depth_camera_info_msg.K[5]
        # self.depth_camera_info['height'] = depth_camera_info_msg.height        
        # self.depth_camera_info['width'] = depth_camera_info_msg.width
        # print(self.depth_camera_info)

    def convert_pcd_to_indices(self, pcd):
        height = self.depth_camera_info['height']
        width = self.depth_camera_info['width']
        pcd = pcd / pcd[:,2].reshape((-1,1))
        x = pcd[:,0] * self.depth_camera_info['intrinsics']['fx'] + self.depth_camera_info['intrinsics']['ppx']
        y = pcd[:,1] * self.depth_camera_info['intrinsics']['fy'] + self.depth_camera_info['intrinsics']['ppy']
        indices = np.array([y,x]).T
        indices = np.floor(indices).astype(int)
        return indices


    def remove_brightness(self, rgb):
        """
        This function removes the light impact on the object by
        increasing the saturation and value of the color cluster

        Args:
            color_cluster: a cluster with points (np.array, #points * 3 channels),
            each of which has 3 color channels (RGB)

        Return:
            the color_cluster without light impact
        """
        print('original color: ', rgb)
        rgb = np.array(rgb).reshape((1,1,3)).astype(float)  # input: 0-255
        print('before color: ', rgb)
        rgb_float32 = np.float32(rgb) ### change to float type
        rgb_hsv = cv2.cvtColor(rgb_float32, cv2.COLOR_RGB2HSV_FULL) ### convert rgb to hsv
        rgb_hsv[0,0,1] = np.float32(1.0) ## maximize saturation
        rgb_hsv[0,0,2] = np.float32(1.0) ## maximize value
        rgb = cv2.cvtColor(rgb_hsv, cv2.COLOR_HSV2RGB_FULL).astype(np.float32).reshape((3))
        # self.visualize_point_cloud(pcd_cluster, color_cluster, show_normal=False)
        return rgb*255

    def segmentation(self, num_objs):
        rospy.wait_for_service("segmentation")
        try:
            segmentation_srv = rospy.ServiceProxy('segmentation', SegmentationSrv)
            resp = segmentation_srv(num_objs)
            # handle the return
            cylinder_models = []
            for i in range(len(resp.cylinders)):
                cylinder_i = {}
                mid_center = np.array([resp.cylinders[i].mid_center.x,resp.cylinders[i].mid_center.y,resp.cylinders[i].mid_center.z])
                radius = resp.cylinders[i].radius
                height = resp.cylinders[i].height
                axis = np.array([resp.cylinders[i].axis.x,resp.cylinders[i].axis.y,resp.cylinders[i].axis.z])
                tran = np.array([resp.cylinders[i].transform.translation.x,resp.cylinders[i].transform.translation.y,
                                 resp.cylinders[i].transform.translation.z])
                qw = resp.cylinders[i].transform.rotation.w
                qx = resp.cylinders[i].transform.rotation.x
                qy = resp.cylinders[i].transform.rotation.y
                qz = resp.cylinders[i].transform.rotation.z
                transform = tf.quaternion_matrix([qw,qx,qy,qz])
                transform[:3,3] = tran
                # NOTE: update to consider only full tall objects in the scene. Use the table shape to change the shape of the object
                z_min = self.scene.workspace.workspace_low[2]
                # get the transform of the object in the world frame
                extrinsics = self.depth_camera_info['extrinsics']
                pose_in_world = extrinsics.dot(transform)
                z_len = pose_in_world[2,3] + height/2 - z_min
                pose_in_world[2,3] = z_min + z_len/2
                transform = np.linalg.inv(extrinsics).dot(pose_in_world)

                cylinder_i['mid_center'] = mid_center
                cylinder_i['radius'] = radius
                cylinder_i['height'] = z_len#height
                cylinder_i['axis'] = axis
                cylinder_i['transform'] = transform
                cylinder_i['shape'] = 'cylinder'

                # decide the color
                color = [resp.cylinders[i].color[0], resp.cylinders[i].color[1], resp.cylinders[i].color[2]]

                if color[0] > 180 and color[1] > 180 and color[2] > 180:
                    # considered uninteresting objects
                    pass
                else:
                    color = self.remove_brightness(color)  # preprocess the color

                cylinder_i['color'] = color


                pts = resp.cylinders[i].pcd.points
                pcd = []
                for j in range(len(pts)):
                    pcd.append([pts[j].x,pts[j].y,pts[j].z])
                pcd = np.array(pcd)

                cylinder_i['pcd'] = pcd
                cylinder_models.append(cylinder_i)

            poses = [cylinder_models[i]['transform'] for i in range(len(cylinder_models))]
            # cv2.imshow("segmentation", seg_img)
            # cv2.waitKey(0)

            seg_img = np.zeros(self.scene.camera.info['img_shape']).astype(int)-1

            for i in range(len(cylinder_models)):
                cylinder = cylinder_models[i]
                pcd_i = np.array(cylinder['pcd'])
                ind_i = self.convert_pcd_to_indices(pcd_i)
                print('pcd.max: ', ind_i.max(axis=0))

                seg_img[ind_i[:,0], ind_i[:,1]] = i
            
            # transform the object to world frame
            for i in range(len(cylinder_models)):
                extrinsics = self.depth_camera_info['extrinsics']
                cylinder_models[i]['transform'] = extrinsics.dot(cylinder_models[i]['transform'])
                pcd = cylinder_models[i]['pcd']
                pcd = extrinsics[:3,:3].dot(pcd.T).T + extrinsics[:3,3]
                cylinder_models[i]['pcd'] = pcd
                # construct a cylinder mesh
                mesh = trimesh.primitives.Cylinder(radius=cylinder_models[i]['radius'], height=cylinder_models[i]['height'])
                cylinder_models[i]['mesh'] = mesh
            return seg_img, cylinder_models

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

def main(args):
    # self.depth_image_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_image_callback, queue_size = 1)
    rospy.init_node("real_camera", anonymous=True)
    segmentation = CylinderSegmentation(None)
    rate = rospy.Rate(10) ### 10hz
    rospy.sleep(1.0)
    color_image_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
    depth_image_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
    # depth_numpy_image = ros_numpy.image.image_to_numpy(depth_image_msg)
    color_numpy_image = segmentation.bridge.imgmsg_to_cv2(color_image_msg, 'passthrough') / 255
    depth_numpy_image = segmentation.bridge.imgmsg_to_cv2(depth_image_msg, 'passthrough')

    segmentation.segmentation(color_numpy_image, depth_numpy_image, filter_plane=False)
    count = 0

    # time.sleep(1)
    # real_camera.getPointCloudFromDepth()

    # while not rospy.is_shutdown():
    #     ### get the time stamp
    #     time_stamp = rospy.get_time()
    #     count += 1
    #     rate.sleep()

if __name__ == '__main__':
    main(sys.argv)
