"""
integrated in the task planner for faster response
- occlusion
- objects
"""
from .scene_belief import SceneBelief
from .object_belief import ObjectBelief
from .data_association import GroundTruthDataAssociation
from .segmentation import GroundTruthSegmentation

import numpy as np
import gc

from utils.visual_utils import *

import cv2

LOG = 0

class PerceptionSystem():
    def __init__(self, occlusion_params, object_params, target_params, tsdf_color_flag=False):
        occlusion = SceneBelief(**occlusion_params)
        self.object_params = object_params  # resol and scale
        self.target_params = target_params  # for recognizing target object
        self.occlusion = occlusion
        self.objects = []  # this stores the recognized objects.
        self.obj_initial_poses = {}  # useful to backpropagate
        self.sensed_imgs = []
        self.sensed_poses = []
        self.filtered_occluded = None
        self.filtered_occlusion_label = None

        # self.table_z = self.occlusion.z_base
        # self.filtered_occupied_label = filtered_occupied_label
        self.filtered_occluded_dict = None

        self.data_assoc = GroundTruthDataAssociation()
        self.segmentation = GroundTruthSegmentation()
        self.tsdf_color_flag = tsdf_color_flag

    def perceive(self, depth_img, color_img, seg_img, obj_models, scene: SimScene, visualize=False):
        """
        Given raw images, and recognized object models (in mesh), do a perception step to update the state
        """
        camera_extrinsics = scene.camera.info['extrinsics']
        camera_intrinsics = scene.camera.info['intrinsics']
        camera_far = scene.camera.info['far']
        camera_near = scene.camera.info['near']
        workspace_ids = scene.workspace.workspace.component_ids
        robot_ids = [scene.robot.robot_id]

        depth_img = np.array(depth_img)
        for cid in workspace_ids:
            depth_img[seg_img==cid] = 0  # workspace
        depth_img[seg_img==-1] = 0  # background

        for i in range(len(obj_models)):
            obj_id = obj_models[i]['obj_id']  # idx in the list
            transform = obj_models[i]['transform']
            if obj_id > len(self.objects):
                # create new object
                # TODO: add the object mesh into the planning scene
                new_object = ObjectBelief(obj_id, self.data_assoc.obj_ids_reverse[obj_id], obj_mesh, self.object_params['resol'], transform)
                self.objects.append(new_object)
                self.obj_initial_poses[obj_id] = new_object.transform

        # * Occlusion
        # get raw occlusion
        occluded = self.occlusion.scene_occlusion(depth_img, color_img, camera_extrinsics, camera_intrinsics)

        # generate point cloud for each of the object using conservative volumes
        obj_pcds = []
        obj_opt_pcds = []
        obj_poses = []
        for i in range(len(self.objects)):
            obj = self.objects[i]
            # for checking occupied space, use conservative volume
            pcd = obj.sample_conservative_pcd(n_sample=10)
            opt_pcd = obj.sample_optimistic_pcd(n_sample=10)
            obj_poses.append(obj.transform)
            obj_pcds.append(pcd)
            obj_opt_pcds.append(opt_pcd)
        # label the occlusion
        occlusion_label, occupied_label, occluded_dict, occupied_dict = \
            self.occlusion.label_scene_occlusion(occluded, camera_extrinsics, camera_intrinsics, depth_img.shape,
                                                obj_poses, obj_pcds, obj_opt_pcds)

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


        del occluded
        del occlusion_label
        del occupied_label
        del occupied_dict
        del occluded_dict
        del filtered_occluded
        del filtered_occlusion_label
        del filtered_occluded_dict
        del obj_pcds
        del obj_opt_pcds

        if len(sensed_obj_ids) > 0:
            del seg_depth_img

    def filtering(self, camera_extrinsics, camera_intrinsics):
        """
        since we remove each object and sense at each time, recording the list of past sensed depth images
        is not necessary. We just need to keep track of the intersection of occlusion to represent it

        A complete approach is to keep track of the list of past sensed depth images and get occlusion for each
        of them and then obtain the intersection
        """
        obj_poses = []
        obj_opt_pcds = []
        obj_conserv_pcds = []
        for i in range(len(self.objects)):
            obj = self.objects[i]
            obj_poses.append(self.obj_initial_poses[obj_id])
            pcd = self.objects[i].sample_optimistic_pcd(n_sample=10)
            obj_opt_pcds.append(pcd)
            obj_conserv_pcds.append(self.objects[i].sample_conservative_pcd(n_sample=10))

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

    def pipeline_sim(self, color_img, depth_img, seg_img, camera, robot_ids, workspace_ids):
        """
        given the camera input, segment the image, and data association
        """
        # color_img, depth_img, seg_img = camera.sense()

        # visualzie segmentation image

        self.segmentation.set_ground_truth_seg_img(seg_img)
        seg_img = self.segmentation.segment_img(color_img, depth_img)

        # self.target_recognition.set_ground_truth_seg_img(seg_img)
        # target_seg_img = self.target_recognition.recognize(color_img, depth_img)

        self.depth_img = depth_img
        self.color_img = color_img
        # self.target_seg_img = target_seg_img

        assoc, seg_img, sensed_obj_ids = self.data_assoc.data_association(seg_img, robot_ids, workspace_ids)
        # sensed_obj_ids: currently seen objects in the scene
        """
        in reality the association can change in time, but the object
        label shouldn't change. So we should only remember the result
        after applying data association
        """
        self.last_assoc = assoc
        self.seg_img = seg_img

        # objects that have been revealed will stay revealed

        self.current_hide_set = object_hide_set
        self.perceive(depth_img, color_img, seg_img,
                    sensed_obj_ids, object_hide_set,
                    camera.info['extrinsics'], camera.info['intrinsics'], camera.info['far'],
                    robot_ids, workspace_ids)

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
