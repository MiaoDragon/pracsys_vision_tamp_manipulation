"""
a process handling perception of images
"""
import os
import sys
import cv2
import pickle
import cv_bridge
import numpy as np
import tf as tf_ros
import open3d as o3d
import transformations as tf

import rospy
import rospkg
from sensor_msgs.msg import Image, CameraInfo

import putils
import utils.visual_utils as visual_utils
from perception.odg import OcclusionDependencyGraph
from perception.scene_occlusion import SceneOcclusion
from perception.ground_truth_modules import GroundTruthSegmentation, GroundTruthTracker

known_models = True
auto_ws_fitting = False

if known_models:
    from perception.known_object_model import ObjectModel
else:
    from perception.object_model import ObjectModel

if auto_ws_fitting:
    import perception.shelf_fitting as ws_fitting
else:
    import perception.workspace_fitting as ws_fitting


class PerceptionSystem():

    def __init__(self, occ_resols, obj_resols, tsdf_color_flag=False):
        """
        occlusion_params: (transform and size are estimated by workspace recognition module)
        - transform
        - resols
        - size
        object_params:
        - resols

        the transform of the scene occlusion can be found by plane fitting
        """
        self.bridge = cv_bridge.CvBridge()
        tf_listener = tf_ros.TransformListener()

        # * get information for the camera
        # Instrinsics
        camera_info = rospy.wait_for_message('camera_info', CameraInfo)
        intrinsics = np.array(camera_info.P).reshape((3, 4))

        # Extrinsics
        rospy.sleep(1.0)
        # transformation of camera in base, i.e.  R T C
        trans, rot = tf_listener.lookupTransform(
            'base',
            'camera_link',
            rospy.Time(0),
        )
        qx, qy, qz, qw = rot
        rot_mat = tf.quaternion_matrix([qw, qx, qy, qz])
        rot_mat[:3, 3] = trans
        extrinsics = rot_mat

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.img_shape = (camera_info.width, camera_info.height)

        # * workspace fitting (here we use cuboid fitting for this task)
        ws_models, ws_pose, ws_size = self.fit_workspace()
        scene_occlusion = SceneOcclusion(
            pose=ws_pose,
            size=ws_size,
            resols=occ_resols,
        )
        self.ws_pose = ws_pose
        self.ws_size = ws_size
        self.ws_models = ws_models
        self.scene_occlusion = scene_occlusion

        self.objects = {}  # objects are stored as dict: id/name -> obj_model

        # * modules
        self.odg = OcclusionDependencyGraph()
        self.segmentation = GroundTruthSegmentation()
        self.tracker = GroundTruthTracker()

        # * other data
        self.total_occluded = None  # this is the net observation for the occluded space

        # * parameter
        self.occ_resols = occ_resols
        self.obj_resols = obj_resols

    def perceive(self, rgb_img, depth_img, visualize=False):
        # rgb_img, depth_img = self.get_img()  # for unit testing
        rgb_img = np.array(rgb_img)
        depth_img = np.array(depth_img)
        seg_img = self.segmentation.segment_img(
            rgb_img,
            depth_img,
        )  # this gives the obj_ids
        # track the object current pose
        self.tracker.track_poses(
            rgb_img,
            depth_img,
            self.objects,
        )  # before TSDF, get the poses
        self.tracker.update_poses(
            self.objects
        )  # update the object belief poses
        # mask the invalid parts to have 0 depth value
        depth_img[seg_img == -1] = 0
        cam_far = 1.2
        # background and workspace marks as far
        depth_img[seg_img == -2] = cam_far
        # * update object models
        # for unrevealed ones, expand the model and update the TSDF
        # for revealed ones, update the TSDF

        # update object models for newly sensed ones
        sensed_obj_ids = list(set(seg_img.flatten().tolist()))
        print('sensed_obj_ids: ')
        print(sensed_obj_ids)
        obj_hiding_dict = self.odg.partial_odg_from_img(depth_img, seg_img)
        # obj_id -> objs that are hiding it
        # TODO we could also segment into names and types of the objects for the sorting task
        for obj_id in sensed_obj_ids:
            if obj_id in [-1, -2]:
                continue

            # NOTE: we need to consider parts that are hiding the target objects and that are
            # hidden by the target object to correctly update the TSDF model.
            # assumptions:
            # 1. the workspace and background (id=-2) are assumed to be hidden by the object always,
            # and thus we set them to have cam_far value
            # 2. the robot is assumed to hide the object, so we set the depth to be 0
            # 3. we assume that if one object is hiding the target object, we will set its depth to
            # be 0. Otherwise we set the depth to be infty

            seg_depth_img = np.zeros(depth_img.shape) + cam_far
            seg_depth_img[seg_img == obj_id] = depth_img[seg_img == obj_id]
            seg_depth_img[seg_img == -2] = cam_far
            seg_depth_img[seg_img == -1] = 0  # invalid
            # mark the objs hiding this obj to have 0 value too
            for hiding_obj_id in list(obj_hiding_dict[obj_id]):
                seg_depth_img[seg_img == hiding_obj_id] = 0
            seg_color_img = np.zeros(rgb_img.shape)
            seg_color_img[seg_img == obj_id, :] = rgb_img[seg_img == obj_id, :]
            if visualize:
                cv2.imshow('seg_color_img', seg_color_img / 255)
                cv2.imshow('seg_depth_img', seg_depth_img)
                print('DepthImg', seg_depth_img)
                cv2.waitKey()

            occluded = self.scene_occlusion.get_occlusion(
                seg_depth_img,
                seg_color_img,
                self.extrinsics,
                self.intrinsics,
            )
            # print(occluded.any())
            sampled_pcd = self.scene_occlusion.sample_pcd(occluded, n_sample=10)
            sampled_pcd = self.scene_occlusion.world_T_voxel[:3, :3].dot(
                sampled_pcd.T
            ).T
            sampled_pcd += self.scene_occlusion.world_T_voxel[:3, 3]

            if not (obj_id in self.objects):
                mins = np.min(sampled_pcd, axis=0)
                maxs = np.max(sampled_pcd, axis=0)
                if known_models:
                    new_object = ObjectModel(
                        obj_id,
                        self.tracker.name_dict[obj_id],
                        self.tracker.world_T_mjk_dict[obj_id],
                        self.obj_resols,
                    )
                else:
                    new_object = ObjectModel(
                        obj_id,
                        self.obj_resols,
                        np.array([mins, maxs]),
                    )
                self.objects[obj_id] = new_object
                self.tracker.add_new_obj(obj_id, new_object)
            if not self.objects[obj_id].revealed:
                self.objects[obj_id].expand_model(sampled_pcd)
            self.objects[obj_id].update_tsdf(
                seg_depth_img,
                seg_color_img,
                self.extrinsics,
                self.intrinsics,
            )
            # when the object is fully revealed, change its status to revealed
            if not self.objects[obj_id].revealed:
                if len(obj_hiding_dict[obj_id]) == 0:
                    # no objects hiding this one now. It is revealed
                    self.objects[obj_id].revealed = True
        # * update scene occlusion
        occluded = self.scene_occlusion.get_occlusion(
            depth_img,
            rgb_img,
            self.extrinsics,
            self.intrinsics,
        )
        # * update the total occlusion (TODO: do we need label of scenes?)
        # NOTE: a complete approach is to use the later sensed object models to update previous occlusion
        # but this is often not necessary and too much overkill. We just need to obtain the intersection
        # of occlusion
        if self.total_occluded is None:
            self.total_occluded = occluded
        else:
            self.total_occluded &= occluded

        if visualize:
            # visualize the occlusion in the scene
            voxel = visual_utils.visualize_voxel(
                self.scene_occlusion.voxel_x,
                self.scene_occlusion.voxel_y,
                self.scene_occlusion.voxel_z,
                occluded,
                [1, 0, 0],
            )
            frame = visual_utils.visualize_coordinate_frame_centered()
            o3d.visualization.draw_geometries([voxel, frame])

            voxel = visual_utils.visualize_voxel(
                self.scene_occlusion.voxel_x,
                self.scene_occlusion.voxel_y,
                self.scene_occlusion.voxel_z,
                self.total_occluded,
                [1, 0, 0],
            )
            o3d.visualization.draw_geometries([voxel, frame])

    def fit_workspace(self):
        rgb_img, depth_img = self.get_img()
        pcd, pcd_color = putils.depth_color_to_pcd(
            depth_img,
            rgb_img,
            self.intrinsics,
        )

        scene_name = rospy.get_param('scene_name', 'scene1')
        ws_fname = scene_name + '.pkl'
        plane_models, ws_pose, ws_size = ws_fitting.calib_ws(
            pcd,
            pcd_color,
            ws_fname,
            self.extrinsics,
            plane_threshold=0.02,
        )
        return plane_models, ws_pose, ws_size

    def get_img(self):
        rgb_img = rospy.wait_for_message('rgb_image', Image)
        depth_img = rospy.wait_for_message('depth_image', Image)
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, 'passthrough')  # / 255
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, 'passthrough')
        return rgb_img, depth_img


if __name__ == '__main__':
    rospy.init_node('perception')
    rospy.sleep(1.0)
    system = PerceptionSystem(0.01, 0.01, None)
    system.perceive(visualize=False)
