"""
the object representation to be used.
This implements the model-free version, which is through TSDF reconstruction of the object.
An extension can use model-based version of it. belief_scene.py also needs to be changed
accordingly.
"""
import sys

sys.path.insert(0,'/home/yinglong/Documents/research/task_motion_planning/infrastructure/motoman_ws/src/pracsys_vision_tamp_manipulation/scripts')

import numpy as np
from utils.visual_utils import *
import open3d as o3d
import copy
import trimesh
DEBUG = False

class ObjectBelief():
    def __init__(self, obj_id, resol, obj_model):
        """
        initialize the object model to be the bounding box containing the conservative volume of the object
        obj_model: a dictionary recording the shape, and details of the model
        """
        obj_mesh = copy.deepcopy(obj_model['mesh'])
        bounds = obj_mesh.bounds
        xmin = bounds[0][0]
        xmax = bounds[1][0]
        ymin = bounds[0][1]
        ymax = bounds[1][1]
        zmin = bounds[0][2]
        zmax = bounds[1][2]

        self.obj_mesh = obj_mesh
        self.obj_shape = obj_model['shape']
        self.obj_model = obj_model
        pcd = trimesh.sample.volume_mesh(obj_mesh, 2500)
        self.pcd = np.array(pcd)

        self.origin_x = xmin
        self.origin_y = ymin
        self.origin_z = zmin
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin

        self.resol = resol
        size_x = int(np.ceil((xmax - xmin) / resol[0]))
        size_y = int(np.ceil((ymax - ymin) / resol[1]))
        size_z = int(np.ceil((zmax - zmin) / resol[2]))
        
        self.xmax = xmin + size_x * resol[0]
        self.ymax = ymin + size_y * resol[1]
        self.zmax = zmin + size_z * resol[2]

        self.axis_x = np.array([1.,0,0])
        self.axis_y = np.array([0.,1,0])
        self.axis_z = np.array([0.,0,1])

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        
        self.voxel_default = 0#-np.inf
        self.voxel = np.zeros((size_x,size_y,size_z)) + self.voxel_default  # default value: -1

        self.voxel_x, self.voxel_y, self.voxel_z = np.indices(self.voxel.shape).astype(float)

        self.active = 0
        self.threshold = 1
        # 0 means this object might be hidden by others. 1 means we can safely move the object now
        # we will only expand the object tsdf volume when it's hidden. Once it's active we will keep
        # the tsdf volume fixed
        self.sensed = 0
        # sensed means if the object has been completely reconstructed

        self.voxel_transform = np.zeros((4,4))
        self.voxel_transform[:3,0] = self.axis_x
        self.voxel_transform[:3,1] = self.axis_y
        self.voxel_transform[:3,2] = self.axis_z
        self.voxel_transform[:3,3] = np.array([self.origin_x, self.origin_y, self.origin_z])
        self.voxel_transform[3,3] = 1.

        # self.transform = transform  # the transform of the voxel grid cooridnate system in the world as {world}T{voxel}
        self.world_in_voxel = np.linalg.inv(self.voxel_transform)
        self.world_in_voxel_rot = self.world_in_voxel[:3,:3]
        self.world_in_voxel_tran = self.world_in_voxel[:3,3]

        self.transform = np.array(obj_model['transform'])  # this is the transform of the mesh, and the pcd, NOT the voxel

        self.obj_id = obj_id 
        self.pybullet_id = obj_id
        self.depth_img = None
        


    def get_optimistic_model(self):
        return (self.voxel > 0)
    
    def get_conservative_model(self):
        # unseen parts below to the conservative model
        return (self.voxel > 0)

    
    def update_transform(self, transform):
        del_transform = transform.dot(np.linalg.inv(self.transform))
        self.transform = transform
        self.voxel_transform = del_transform.dot(self.voxel_transform)
        self.world_in_voxel = np.linalg.inv(self.voxel_transform)
        self.world_in_voxel_rot = self.world_in_voxel[:3,:3]
        self.world_in_voxel_tran = self.world_in_voxel[:3,3]

    def update_transform_from_relative(self, rel_transform):
        # when the object is moved, update the transform
        # previous: W T O1
        # relative transform: delta T
        self.transform = rel_transform.dot(self.transform)
        self.world_in_voxel = np.linalg.inv(self.transform)
        self.world_in_voxel_rot = self.world_in_voxel[:3,:3]
        self.world_in_voxel_tran = self.world_in_voxel[:3,3]
        # self.transform = self.transform.dot(rel_transform)

    def sample_pcd(self, mask, n_sample=10):
        # sample voxels in te mask
        # obtain sample in one voxel cell
        return np.array(self.pcd)

    def sample_conservative_pcd(self, n_sample=10):
        # obtain the pcd of the conservative volume
        return self.sample_pcd(self.get_conservative_model(), n_sample)
    def sample_optimistic_pcd(self, n_sample=10):
        # obtain the pcd of the conservative volume
        return self.sample_pcd(self.get_optimistic_model(), n_sample)

    def get_center_frame(self, pcd):
        """
        get the rotation  C R O  (object frame relative to center frame)
        translation C T O
        (center uses the same z value as the object)
        """
        # find the center of the pcd
        midpoint = pcd.mean(axis=0)
        transform = np.zeros((4,4))
        transform[3,3] = 1.0
        transform[:3,:3] = np.eye(3)
        transform[:2,3] = -midpoint[:2]
        return transform
    
    def get_net_transform_from_center_frame(self, pcd, transform):
        """
        given transform from center to world, get the net transform
        from obstacle frame to world
        """
        obj_in_center = self.get_center_frame(pcd)
        net_transform = np.array(transform)
        net_transform[:3,:3] = transform[:3,:3].dot(obj_in_center[:3,:3])
        net_transform[:3,3] = transform[:3,:3].dot(obj_in_center[:3,3]) + transform[:3,3]
        return net_transform
