import os
import rospy
import pickle
import pyransac3d
import numpy as np
import open3d as o3d
import transformations as tf


def calib_ws(pcd, pcd_color, ws_fname, extrinsics, plane_threshold=0.02):
    """
    calibrate the workspace: pose and size
    the x axis of the workspace is in the reverse direction of y axis of the robot
    the y axis of the workspace is in the x axis of the robot
    the z axis of the workspace is the same as the z of the robot

    The axes might not be exact (in real world we have some errors)
    we will just use the fitted cuboid plane model
    """
    # scene_xml = rospy.get_param('/perception/mjcf', 'test.xml')
    # print('Exists:', scene_xml, '?:', os.path.exists(scene_xml))

    pose = np.array(
        rospy.get_param(
            '/workspace/pose', [
                [0, 1, 0, 0.62],
                [-1, 0, 0, 0.4],
                [0, 0, 1, 0.9],
                [0, 0, 0, 1],
            ]
        )
    )
    size = np.array(rospy.get_param('/workspace/size', [0.8, 0.5, 0.3]))

    print('pose: ', pose)
    print('size: ', size)
    plane_models = None

    return plane_models, pose, size
