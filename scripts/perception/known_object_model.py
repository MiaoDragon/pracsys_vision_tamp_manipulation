"""
define the object model
"""
import os
import sys
import pickle
import xml.etree.ElementTree as ET

import rospy
import rospkg
import cv_bridge
import tf as tf_ros
from sensor_msgs.msg import Image, CameraInfo

import trimesh
import mesh2sdf
import numpy as np
import open3d as o3d
from dm_control import mjcf
import transformations as tf
from trimesh.primitives import Box, Cylinder, Capsule, Sphere

import putils
import utils.visual_utils as visual_utils
"""
NOTE: OpenVDB seems to be a fast and easy way to access voxels.
It allows expanding the voxels dynamically
https://www.openvdb.org/documentation/doxygen/codeExamples.html
"""

from perception.object_model import ObjectModel as ObjectModelInterface


class ObjectModel(ObjectModelInterface):

    def __init__(self, obj_id, obj_name, obj_pose, resols, tsdf_threshold=0.03):
        """
        obj_id: object id
        obj_pose: real pose of object in workspace
        resols: resolution for the voxel
        """

        xml_path = rospy.get_param('ws_xml_path', None)
        if xml_path is None:
            print(
                'No workspace xml path specified in ros params. Is the execution node running?',
                file=sys.stderr
            )
            return
        root = ET.parse(xml_path).getroot()
        compiler = root.find('compiler')
        meshdir = compiler.get('meshdir', compiler.get('assetdir', './'))
        assets = {None: None}
        for asset in root.find('asset'):
            if 'file' in asset.keys():
                name = asset.get('name')
                path = os.path.abspath(meshdir + asset.get('file'))
                assets[name] = path

        obj_arr = root.findall(f"./worldbody/body[@name='{obj_name}']")
        obj_node = obj_arr[0] if len(obj_arr) > 0 else None

        geom = None
        for geom in obj_node.findall('./geom'):
            if geom.get('contype', 1) != '0':
                break
        if geom is None:
            print(
                'No geometry specified for object',
                obj_name,
                'in file',
                xml_path,
                '!',
                file=sys.stderr
            )
            return

        gtype = geom.get('type', 'sphere')
        gsize = [float(x) for x in geom.get('size', '0').split()]
        print(gsize)
        meshfile = assets[geom.get('mesh', None)]
        if gtype == 'mesh':
            print(meshfile)
            obj = trimesh.load(meshfile, force='mesh')
        elif gtype == 'box':
            # hx, hy, hz = gsize
            obj = Box(np.multiply(2, gsize))
        elif gtype == 'cylinder':
            r, hh = gsize
            obj = Cylinder(r, 2 * hh)
        elif gtype == 'capsule':
            r, hh = gsize
            obj = Capsule(r, 2 * hh)
        elif gtype == 'sphere':
            r = gsize[0]
            obj = Sphere(r)
        else:
            print('Unsupported object type:', gtype, '!', file=sys.stderr)
            return

        size = obj.bounds[1] - obj.bounds[0]
        shape = np.ceil(size / resols).astype(int)
        obj_T_voxel = np.eye(4)
        obj_T_voxel[:3, 3] = obj.bounds[0]
        sdf = mesh2sdf.compute(obj.vertices, obj.faces, shape)
        print(obj)
        print(obj.bounds)
        print(shape)
        # print(sdf)
        self.tsdf = sdf
        self.tsdf_count = np.zeros(shape).astype(int)  # observed times
        self.color_tsdf = np.zeros(shape.tolist() + [3])
        self.voxel_x, self.voxel_y, self.voxel_z = \
                np.indices(self.tsdf.shape).astype(float)

        self.size = size  # real-valued size
        self.shape = shape  # integer shape of the voxel
        self.resols = resols
        self.world_T_obj = obj_pose  # object pose in the world
        self.obj_T_world = np.linalg.inv(self.world_T_obj)
        self.obj_T_voxel = obj_T_voxel  # voxel pose in the object frame
        self.voxel_T_obj = np.linalg.inv(self.obj_T_voxel)
        self.world_T_voxel = self.world_T_obj.dot(self.obj_T_voxel)
        self.voxel_T_world = np.linalg.inv(self.world_T_voxel)
        self.revealed = True  # if the object is fully revealed
        self.obj_id = obj_id

        self.tsdf_max = tsdf_threshold
        self.tsdf_min = -tsdf_threshold * 1.1
