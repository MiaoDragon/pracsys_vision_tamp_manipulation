"""
integrated in the task planner for faster response
- occlusion
- objects
"""
from re import S
from .scene_belief import SceneBelief
from .object_belief import ObjectBelief
from .real_data_association import CylinderDataAssociation
from .real_segmentation import CylinderSegmentation
from scene.sim_scene import SimScene
import numpy as np
import gc

from utils.visual_utils import *

import cv2

LOG = 0
class PerceptionSystem():
    def __init__(self, occlusion_params, object_params, target_params, scene: SimScene):
        occlusion = SceneBelief(**occlusion_params)
        self.object_params = object_params  # resol and scale
        self.target_params = target_params  # for recognizing target object
        self.occlusion = occlusion
        self.objects = {}  # this stores the recognized objects. assuming object ids start from 0
        self.obj_initial_poses = {}  # useful to backpropagate
        self.sensed_imgs = []
        self.sensed_poses = []
        self.filtered_occluded = None
        self.filtered_occlusion_label = None

        self.filtered_occluded_dict = None

        self.scene = scene
        self.data_assoc = CylinderDataAssociation(object_params)
        self.segmentation = CylinderSegmentation(scene.camera)
        
    def perceive(self, depth_img, color_img, seg_img, seen_obj_models: dict):
        """
        for new object models, append to the object list
        use the images to construct occlusion space
        label the occlusion space using the recognized objects
        filter the occlusion space

        obj_models: associated id -> obj_model
        """
        scene = self.scene
        camera_extrinsics = scene.camera.info['extrinsics']
        camera_intrinsics = scene.camera.info['intrinsics']

        # * for new object models, add them to the scene
        obj_ids = []
        obj_models = []
        for obj_id, obj_model in seen_obj_models.items():
            if obj_id in self.objects:
                continue
            obj_ids.append(obj_id)
            obj_models.append(obj_model)
        # sort the object ids so we can add the new objects to the scene
        print('remaining obj_ids: ')
        print(obj_ids)
        sort_indices = np.argsort(obj_ids)
        for i in range(len(sort_indices)):
            new_obj_id = obj_ids[sort_indices[i]]
            obj_model = obj_models[sort_indices[i]]
            obj = ObjectBelief(new_obj_id, self.object_params['resol'], obj_model)
            self.objects[new_obj_id] = obj
            self.obj_initial_poses[new_obj_id] = np.array(obj_model['transform'])

        # * use the images to construct occlusion space
        # get raw occlusion
        occluded = self.occlusion.scene_occlusion(depth_img, color_img, camera_extrinsics, camera_intrinsics)

        # generate point cloud for each of the object using conservative volumes
        obj_pcds = {}
        obj_opt_pcds = {}
        obj_poses = {}
        for i, obj in self.objects.items():
            # for checking occupied space, use conservative volume
            pcd = obj.sample_conservative_pcd(n_sample=10)
            opt_pcd = obj.sample_optimistic_pcd(n_sample=10)

            obj_poses[i] = np.array(obj.transform)
            obj_pcds[i] = pcd
            obj_opt_pcds[i] = opt_pcd

        # label the occlusion
        occlusion_label, occupied_label, occluded_dict, occupied_dict = \
            self.occlusion.label_scene_occlusion(occluded, camera_extrinsics, camera_intrinsics, depth_img.shape,
                                                obj_poses, obj_pcds, obj_opt_pcds)


        voxels = []
        for i, obj in self.objects.items():
            if occluded_dict[i].astype(int).sum() == 0:
                continue
            color = from_color_map(i, 16)
            vis_voxel = visualize_voxel(self.occlusion.voxel_x, self.occlusion.voxel_y, self.occlusion.voxel_z, occluded_dict[i], color)
            voxels.append(vis_voxel)
        o3d.visualization.draw_geometries(voxels)

        if len(self.sensed_imgs) == 0:
            self.sensed_imgs.append(depth_img)
            self.sensed_poses.append(obj_poses)
        else:
            self.sensed_imgs[0] = depth_img
            self.sensed_poses[0] = obj_poses

        self.occluded_t = occluded
        self.occlusion_label_t = occlusion_label
        self.occupied_label_t = occupied_label
        self.occupied_dict_t = occupied_dict
        self.occluded_dict_t = occluded_dict
        print("*** labels written ***")

        filtered_occluded, filtered_occlusion_label, filtered_occluded_dict = \
            self.filtering(camera_extrinsics, camera_intrinsics)        
        self.filtered_occluded = filtered_occluded
        self.filtered_occlusion_label = filtered_occlusion_label
        # self.filtered_occupied_label = filtered_occupied_label
        self.filtered_occluded_dict = filtered_occluded_dict


    def filtering(self, camera_extrinsics, camera_intrinsics):
        """
        since we remove each object and sense at each time, recording the list of past sensed depth images
        is not necessary. We just need to keep track of the intersection of occlusion to represent it

        A complete approach is to keep track of the list of past sensed depth images and get occlusion for each
        of them and then obtain the intersection
        """
        obj_poses = {}
        obj_opt_pcds = {}
        obj_conserv_pcds = {}
        for i, obj in self.objects.items():
            obj_poses[i] = self.obj_initial_poses[i]
            pcd = self.objects[i].sample_optimistic_pcd(n_sample=10)
            obj_opt_pcds[i] = pcd
            obj_conserv_pcds[i] = self.objects[i].sample_conservative_pcd(n_sample=10)

        net_occluded = self.filtered_occluded

        for i in range(len(self.sensed_imgs)):
            depth_img = self.sensed_imgs[i]
            occluded = self.occlusion.scene_occlusion(depth_img, None, camera_extrinsics, camera_intrinsics)

            for obj_id in range(len(self.sensed_poses[i])):
                obj_poses[obj_id] = self.sensed_poses[i][obj_id]
            occlusion_label, occupied_label, occluded_dict, _ = \
                    self.occlusion.label_scene_occlusion(occluded, camera_extrinsics, camera_intrinsics, depth_img.shape,
                                                        obj_poses, obj_conserv_pcds, obj_opt_pcds)


            opt_occupied_dict = self.occlusion.obtain_object_occupancy(camera_extrinsics, camera_intrinsics, obj_poses, obj_opt_pcds)

            opt_occupied_label = np.zeros(occupied_label.shape).astype(int)
            for obj_id, opt_occupied in opt_occupied_dict.items():
                opt_occupied_label[opt_occupied] = obj_id+1


            for obj_id, obj in self.objects.items():
                pcd = np.array([obj.voxel_x, obj.voxel_y, obj.voxel_z]).transpose([1,2,3,0])
                pcd = np.array([pcd,pcd,pcd,pcd,pcd,pcd,pcd,pcd,pcd,pcd]).transpose(1,2,3,0,4)
                ori_pcd = np.array(pcd).astype(int).reshape(-1,3)
                rand_pcd = np.random.uniform(low=0.0,high=1.0,size=(1,1,1,10,3))
                pcd = pcd + rand_pcd
                # pcd = pcd + 0.5
                pcd = pcd * obj.resol
                pcd = pcd.reshape(-1,3)
                pcd = obj_poses[obj_id][:3,:3].dot(pcd.T).T + obj_poses[obj_id][:3,3]
                pcd = self.occlusion.world_in_voxel_rot.dot(pcd.T).T + self.occlusion.world_in_voxel_tran
                pcd = pcd / self.occlusion.resol
                pcd = np.floor(pcd).astype(int)
                valid_mask = (pcd[:,0] >= 0) & (pcd[:,0] < self.occlusion.voxel_x.shape[0]) & \
                             (pcd[:,1] >= 0) & (pcd[:,1] < self.occlusion.voxel_x.shape[1]) & \
                             (pcd[:,2] >= 0) & (pcd[:,2] < self.occlusion.voxel_x.shape[2])
                occupied_mask = (opt_occupied_label[pcd[valid_mask][:,0],pcd[valid_mask][:,1],pcd[valid_mask][:,2]]>0) & \
                                ((opt_occupied_label[pcd[valid_mask][:,0],pcd[valid_mask][:,1],pcd[valid_mask][:,2]]!=obj_id+1)).reshape(-1)

            if net_occluded is None:
                net_occluded = occlusion_label != 0 #occlusion_label > 0
            else:
                net_occluded = net_occluded & (occlusion_label!=0)#(occlusion_label>0)

        # obtain occlusion for each of the object
        new_occlusion_label = np.zeros(occlusion_label.shape).astype(int)
        for obj_id, obj_occlusion in occluded_dict.items():
            occluded_dict[obj_id] = occluded_dict[obj_id] & net_occluded
            new_occlusion_label[(occlusion_label==obj_id+1) & net_occluded] = obj_id+1

        del obj_conserv_pcds
        del obj_opt_pcds
        del obj_poses
        del pcd
        del occluded
        del ori_pcd
        del occupied_mask
        del occlusion_label
        gc.collect()

        return net_occluded, new_occlusion_label, occluded_dict

    def pipeline_sim(self, color_img, depth_img, seg_img):
        """
        given the camera input, segment the image, and data association
        """
        scene = self.scene
        camera = scene.camera
        workspace_ids = scene.workspace.component_ids
        robot_ids = [scene.robot.robot_id]
        # visualzie segmentation image

        seg_img, sensed_obj_models = self.segmentation.segmentation(num_objs=8)
        # sensed_obj_model: seg_id -> obj_model

        # self.target_recognition.set_ground_truth_seg_img(seg_img)
        # target_seg_img = self.target_recognition.recognize(color_img, depth_img)

        self.depth_img = depth_img
        self.color_img = color_img
        # self.target_seg_img = target_seg_img

        seg_img, obj_models = self.data_assoc.data_association(seg_img, sensed_obj_models, self.objects)
        # obj_models: obj_id -> obj_model

        # sensed_obj_ids: currently seen objects in the scene
        """
        in reality the association can change in time, but the object
        label shouldn't change. So we should only remember the result
        after applying data association
        """
        self.seg_img = seg_img

        # objects that have been revealed will stay revealed
        self.perceive(depth_img, color_img, seg_img, obj_models)


    def sample_pcd(self, mask, n_sample=10):
        # sample voxels in te mask
        # obtain sample in one voxel cell
        grid_sample = np.random.uniform(low=[0,0,0], high=[1,1,1], size=(n_sample, 3))
        voxel_x = self.voxel_x[mask]
        voxel_y = self.voxel_y[mask]
        voxel_z = self.voxel_z[mask]

        total_sample = np.zeros((len(voxel_x), n_sample, 3))
        total_sample = total_sample + grid_sample
        total_sample = total_sample + np.array([voxel_x, voxel_y, voxel_z]).T.reshape(len(voxel_x),1,3)

        total_sample = total_sample.reshape(-1, 3) * np.array(self.resol)

        del voxel_x
        del voxel_y
        del voxel_z

        return total_sample

