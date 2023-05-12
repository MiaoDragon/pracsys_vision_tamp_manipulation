"""
define the object model
"""
import os
import sys
import pickle
import numpy as np
import open3d as o3d

import rospy
import rospkg
import cv_bridge
import tf as tf_ros
import transformations as tf
from sensor_msgs.msg import Image, CameraInfo

import putils
import utils.visual_utils as visual_utils
"""
NOTE: OpenVDB seems to be a fast and easy way to access voxels.
It allows expanding the voxels dynamically
https://www.openvdb.org/documentation/doxygen/codeExamples.html
"""


class ObjectModel():

    def __init__(self, obj_id, resols, bounds, tsdf_threshold=0.03):
        """
        obj_id: object id
        resols: resolution for the voxel
        bounds: bounding box of the object point cloud (in real space)
                shape: 2x3
        NOTE:
        we don't want to keep changing the transform of the object
        it's better to keep the transform fixed, and have an internal
        relative transform of the voxel and the object.
        We use the flooring ops to find the corresponding voxel of a pcd.
        NOTE:
        when not revealed, expand the object model first, and then update TSDF
        after fully revealed, no need to expand anymore, but only update TSDF
        (the pcd that is passed for update should be masked to only include
         - parts in the object model
         - total occluded region)
        """
        size = bounds[1] - bounds[0]
        shape = np.ceil(size / resols).astype(int)
        self.tsdf = np.zeros(shape)
        self.tsdf_count = np.zeros(shape).astype(int)  # observed times
        self.color_tsdf = np.zeros(shape.tolist() + [3])
        self.voxel_x, self.voxel_y, self.voxel_z = \
                np.indices(self.tsdf.shape).astype(float)
        # * initialize the pose using the lowest bound
        obj_pose = np.eye(4)
        obj_pose[:3, 3] = bounds[0]
        # * set the relative transpose of the object pose to the voxel pose
        obj_T_voxel_base = np.eye(4)  # initially the two frames are the same

        self.size = size  # real-valued size
        self.shape = shape  # integer shape of the voxel
        self.resols = resols
        self.world_T_obj = obj_pose  # object pose in the world
        self.obj_T_world = np.linalg.inv(self.world_T_obj)
        self.obj_T_voxel = obj_T_voxel_base  # voxel pose in the object frame
        self.voxel_T_obj = np.linalg.inv(self.obj_T_voxel)
        self.world_T_voxel = self.world_T_obj.dot(self.obj_T_voxel)
        self.voxel_T_world = np.linalg.inv(self.world_T_voxel)
        self.revealed = False  # if the object is fully revealed
        self.obj_id = obj_id

        self.tsdf_max = tsdf_threshold
        self.tsdf_min = -tsdf_threshold * 1.1

    def update_pose_by_transform(self, rel_transform):
        # when the object is moved, update the transform
        # previous: W T O1
        # delta transform: (W T O2) * (W T O1)^{-1}
        self.update_pose(rel_transform.dot(self.world_T_obj))

    def update_pose(self, new_pose):
        self.world_T_obj = np.array(new_pose)  # object pose in the world
        self.obj_T_world = np.linalg.inv(self.world_T_obj)
        self.world_T_voxel = self.world_T_obj.dot(self.obj_T_voxel)
        self.voxel_T_world = np.linalg.inv(self.world_T_voxel)

    def expand_model(self, pcd_in_world):
        """
        expand the model when new parts are seen given the pcd in world
        NOTE:
        the pcd should be masked to only include parts in the total occluded
        region
        """
        # * transform the point cloud to the voxel frame
        pcd_in_obj = self.obj_T_world[:3, :3].dot(pcd_in_world.T).T
        pcd_in_obj += self.obj_T_world[:3, 3]
        pcd_in_voxel = self.voxel_T_obj[:3, :3].dot(pcd_in_obj.T).T
        pcd_in_voxel += self.voxel_T_obj[:3, 3]

        # * find the new lower bound and upper bound
        idx = np.floor(pcd_in_voxel / self.resols).astype(int)
        idx_min = np.min(idx, axis=0)
        idx_max = np.max(idx, axis=0)
        idx_min[idx_min >= 0] = 0  # make sure we don't have positive min vals
        new_shape = idx_max - idx_min + 1
        new_shape = np.maximum(self.shape, new_shape)
        new_lower = idx_min * self.resols
        new_upper = idx_max * self.resols
        # check if the shape stays the same
        if np.sum(self.shape >= new_shape) == len(self.shape):
            return  # no need to expand since the models are the same
        self.shape = new_shape
        self.size = np.array([new_lower, new_upper])

        # * create new array for the new voxel
        new_tsdf = np.zeros(new_shape)
        new_tsdf_cnt = np.zeros(new_shape).astype(int)
        new_tsdf_col = np.zeros(new_shape.tolist() + [3])
        new_tsdf[-idx_min[0]:-idx_min[0] + self.shape[0],
                 -idx_min[1]:-idx_min[1] + self.shape[1],
                 -idx_min[2]:-idx_min[2] + self.shape[2]] = self.tsdf
        new_tsdf_cnt[-idx_min[0]:-idx_min[0] + self.shape[0],
                     -idx_min[1]:-idx_min[1] + self.shape[1],
                     -idx_min[2]:-idx_min[2] + self.shape[2]] = self.tsdf_count
        new_tsdf_col[-idx_min[0]:-idx_min[0] + self.shape[0],
                     -idx_min[1]:-idx_min[1] + self.shape[1],
                     -idx_min[2]:-idx_min[2] + self.shape[2]] = self.color_tsdf
        self.tsdf = new_tsdf
        self.tsdf_count = new_tsdf_cnt
        self.color_tsdf = new_tsdf_col
        self.voxel_x, self.voxel_y, self.voxel_z = \
                np.indices(self.tsdf.shape).astype(float)

        # * translate the voxel frame to new_lower
        obj_T_voxel = np.array(self.obj_T_voxel)
        obj_T_voxel[:3, 3] += new_lower
        self.obj_T_voxel = obj_T_voxel
        self.voxel_T_obj = np.linalg.inv(self.obj_T_voxel)
        self.world_T_voxel = self.world_T_obj.dot(self.obj_T_voxel)
        self.voxel_T_world = np.linalg.inv(self.world_T_voxel)

    def update_tsdf(
        self,
        depth_img,
        color_img,
        extrinsics,
        intrinsics,
        visualize=False,
    ):
        """
        given the *segmented* depth image belonging to the object, update tsdf
        # NOTE
        notice that if an object/workspace is hidden by this object, we need to fill those locations
        with "free" state. This can be achieved by setting the depth value to be infinite.

        #NOTE
        in this version, we update only the parts which have depth pixels in the segmented image
        since an object might be hidden by others, and we might miss object parts
        """
        # obtain pixel locations for each of the voxels
        voxel_vecs = np.array([self.voxel_x, self.voxel_y, self.voxel_z])
        voxel_vecs = voxel_vecs.transpose((1, 2, 3, 0)) + 0.5
        # voxel_vecs = np.concatenate([self.voxel_x, self.voxel_y, self.voxel_z], axis=3)
        voxel_vecs = voxel_vecs.reshape(-1, 3) * self.resols
        transformed_voxels = self.obj_T_voxel[:3, :3].dot(voxel_vecs.T).T
        transformed_voxels += self.obj_T_voxel[:3, 3]
        transformed_voxels = self.world_T_obj[:3, :3].dot(
            transformed_voxels.T
        ).T
        transformed_voxels += self.world_T_obj[:3, 3]

        # get to the image space
        cam_transform = np.linalg.inv(extrinsics)
        transformed_voxels = cam_transform[:3, :3].dot(transformed_voxels.T).T
        transformed_voxels += cam_transform[:3, 3]

        # cam_to_voxel_dist = np.linalg.norm(transformed_voxels, axis=1)
        cam_to_voxel_depth = np.array(transformed_voxels[:, 2])
        # intrinsics
        cam_intrinsics = intrinsics
        fx = cam_intrinsics[0][0]
        fy = cam_intrinsics[1][1]
        cx = cam_intrinsics[0][2]
        cy = cam_intrinsics[1][2]
        transformed_voxels[:, 0] /= transformed_voxels[:, 2]
        transformed_voxels[:, 0] *= fx
        transformed_voxels[:, 0] += cx
        transformed_voxels[:, 1] /= transformed_voxels[:, 2]
        transformed_voxels[:, 1] *= fy
        transformed_voxels[:, 1] += cy
        transformed_voxels = np.floor(transformed_voxels).astype(int)
        voxel_depth = np.zeros((len(transformed_voxels)))
        valid_mask = (transformed_voxels[:, 0] >= 0)
        valid_mask &= (transformed_voxels[:, 0] < len(depth_img[0]))
        valid_mask &= (transformed_voxels[:, 1] >= 0)
        valid_mask &= (transformed_voxels[:, 1] < len(depth_img))
        voxel_depth[valid_mask] = \
                depth_img[
                    transformed_voxels[valid_mask][:,1],
                    transformed_voxels[valid_mask][:,0]
                ]

        voxel_color = np.zeros((len(transformed_voxels), 3))
        voxel_color[valid_mask] = color_img[
            transformed_voxels[valid_mask][:, 1],
            transformed_voxels[valid_mask][:, 0], :3]
        voxel_color = voxel_color.reshape(list(self.voxel_x.shape) + [3])

        valid_mask = valid_mask.reshape(self.voxel_x.shape)
        voxel_depth = voxel_depth.reshape(self.voxel_x.shape)

        cam_to_voxel_depth = cam_to_voxel_depth.reshape(self.voxel_x.shape)

        # handle valid space
        tsdf = np.zeros(self.tsdf.shape)
        tsdf = (voxel_depth - cam_to_voxel_depth)  # * self.scale
        valid_space = (voxel_depth > 0) & (tsdf > self.tsdf_min) & valid_mask
        if valid_space.astype(int).sum() == 0:
            return

        # visualize the valid space against the entire space
        if visualize:
            vvoxel1 = visual_utils.visualize_voxel(
                self.voxel_x,
                self.voxel_y,
                self.voxel_z,
                voxel_depth > 0,
                [1, 0, 0],
            )
            vvoxel2 = visual_utils.visualize_voxel(
                self.voxel_x,
                self.voxel_y,
                self.voxel_z,
                tsdf > self.tsdf_min,
                [1, 0, 0],
            )
            vvoxel3 = visual_utils.visualize_voxel(
                self.voxel_x,
                self.voxel_y,
                self.voxel_z,
                tsdf > self.tsdf_min * 1.5,
                [1, 0, 0],
            )
            vvoxel4 = visual_utils.visualize_voxel(
                self.voxel_x,
                self.voxel_y,
                self.voxel_z,
                valid_space,
                [1, 0, 0],
            )
            vbox = visual_utils.visualize_bbox(
                self.voxel_x,
                self.voxel_y,
                self.voxel_z,
            )
            frame = visual_utils.visualize_coordinate_frame_centered()

            o3d.visualization.draw_geometries([vvoxel1, vbox, frame])
            o3d.visualization.draw_geometries([vvoxel2, vbox, frame])
            o3d.visualization.draw_geometries([vvoxel3, vbox, frame])
            o3d.visualization.draw_geometries([vvoxel4, vbox, frame])

        # we don't want to update the TSDF value for hidden parts since it will affect previous TSDF value
        self.tsdf[valid_space] *= self.tsdf_count[valid_space]
        self.tsdf[valid_space] += tsdf[valid_space]
        self.tsdf[valid_space] /= self.tsdf_count[valid_space] + 1
        self.color_tsdf[valid_space] *= self.tsdf_count[valid_space].reshape(
            (-1, 1)
        )
        self.color_tsdf[valid_space] += voxel_color[valid_space]
        self.color_tsdf[valid_space] /= self.tsdf_count[valid_space].reshape(
            (-1, 1)
        ) + 1
        self.color_tsdf[self.color_tsdf > 255] = 255.0
        self.color_tsdf[self.color_tsdf < 0] = 0
        self.tsdf_count[valid_space] = self.tsdf_count[valid_space] + 1

        self.tsdf[self.tsdf > self.tsdf_max * 1.1] = self.tsdf_max * 1.1
        self.tsdf[self.tsdf < self.tsdf_min] = self.tsdf_min

        # handle invalid space: don't update
        self.tsdf[self.tsdf_count == 0] = 0.0

        if visualize:
            self.visualize_obj(self.get_conservative_model())
            self.visualize_obj(self.get_optimistic_model())

        del voxel_vecs
        del valid_space
        del tsdf

    def get_optimistic_model(self):
        threshold = 1
        to_return = (self.tsdf_count >= threshold)
        to_return &= (self.tsdf < self.tsdf_max)
        to_return &= (self.tsdf > self.tsdf_min)
        return to_return

    def get_conservative_model(self):
        # unseen parts below to the conservative model
        threshold = 1
        to_return = (self.tsdf_count < threshold)
        to_return |= (
            (self.tsdf_count >= threshold) & (self.tsdf < self.tsdf_max)
        )
        return to_return

    def sample_pcd(self, mask, n_sample=10, color=False):
        # sample voxels in te mask
        # obtain sample in one voxel cell
        grid_sample = np.random.uniform(
            low=[0, 0, 0],
            high=[1, 1, 1],
            size=(n_sample, 3),
        )
        voxel_x = self.voxel_x[mask]
        voxel_y = self.voxel_y[mask]
        voxel_z = self.voxel_z[mask]

        total_sample = np.zeros((len(voxel_x), n_sample, 3))
        total_sample += grid_sample
        total_sample += np.array([
            voxel_x,
            voxel_y,
            voxel_z,
        ]).T.reshape(len(voxel_x), 1, 3)
        total_sample = total_sample.reshape(-1, 3)
        total_indices = np.floor(total_sample).astype(int)
        total_sample *= np.array(self.resols)

        del voxel_x
        del voxel_y
        del voxel_z

        if color:
            total_colors = self.color_tsdf[total_indices[:, 0],
                                           total_indices[:, 1],
                                           total_indices[:, 2]]
            return total_sample, total_colors
        else:
            return total_sample

    def sample_conservative_pcd(self, n_sample=10, color=False):
        # obtain the pcd of the conservative volume
        return self.sample_pcd(self.get_conservative_model(), n_sample, color)

    def sample_optimistic_pcd(self, n_sample=10, color=False):
        # obtain the pcd of the conservative volume
        return self.sample_pcd(self.get_optimistic_model(), n_sample, color)

    def get_surface_normal(self):
        """
        get surface normal from tsdf
        """
        DEBUG = True
        gx, gy, gz = np.gradient(self.tsdf)
        # visualize the gradient for optimistic volume
        if self.sensed:
            mfilter = self.get_conservative_model()
        else:
            mfilter = self.get_optimistic_model()

        filtered_x = self.voxel_x[mfilter]
        filtered_y = self.voxel_y[mfilter]
        filtered_z = self.voxel_z[mfilter]
        filtered_pts = np.array([filtered_x, filtered_y, filtered_z]).T
        filtered_g = np.array(
            [gx[mfilter], gy[mfilter], gz[mfilter]]
        ).T  # pointing from inside to outside (negative value to positive)

        # use the normal to move back the suction point
        filtered_g = filtered_g / np.linalg.norm(
            filtered_g,
            axis=1,
        ).reshape((-1, 1))
        filtered_pts = filtered_pts + filtered_g * 0.5
        # use the TSDF value to shift the suction points. If TSDF > 0, we need to move inside, otherwise outside
        filtered_pts = filtered_pts - filtered_g * self.tsdf[mfilter].reshape(
            (-1, 1)
        )

        if DEBUG:
            pcd_v = visual_utils.visualize_pcd(
                self.sample_pcd(mfilter) / self.resols,
                [0, 0, 0],
            )
            voxel_v = visual_utils.visualize_voxel(
                self.voxel_x,
                self.voxel_y,
                self.voxel_z,
                mfilter,
                [1, 0, 0],
            )

            arrows = []
            for i in range(len(filtered_g)):
                # draw arrows indicating the normal
                if (np.linalg.norm(filtered_g[i]) <= 1e-5):
                    continue

                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=1 / 10,
                    cone_radius=1.5 / 10,
                    cylinder_height=5 / 10,
                    cone_height=4 / 10
                )
                translation = filtered_pts[i]
                z_axis = filtered_g[i] / np.linalg.norm(filtered_g[i])
                x_axis = np.array([-z_axis[2], 0, z_axis[0]])
                y_axis = np.cross(z_axis, x_axis)
                rotation = np.array([x_axis, y_axis, z_axis]).T
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = translation
                arrow.transform(transform)
                arrows.append(arrow)
            o3d.visualization.draw_geometries(arrows + [pcd_v])

        # take only the ones that are on seen TSDFs
        mfilter = (self.tsdf_count[mfilter] > 0)
        filtered_pts = filtered_pts[mfilter]
        filtered_g = filtered_g[mfilter]

        length = np.linalg.norm(filtered_g, axis=1)
        filtered_pts = filtered_pts[length >= 1e-5]
        filtered_g = filtered_g[length >= 1e-5]
        filtered_g = filtered_g / np.linalg.norm(
            filtered_g,
            axis=1,
        ).reshape(-1, 1)

        return filtered_pts * self.resols, -filtered_g

    def visualize_obj(self, mask):
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        pcd = self.sample_pcd(mask) / self.resols
        pcd_ind = np.floor(pcd).astype(int)
        # Get vertex colors
        rgb_vals = self.color_tsdf[pcd_ind[:, 0], pcd_ind[:, 1], pcd_ind[:, 2]]
        rgb_vals /= 255
        pcd = visual_utils.visualize_pcd(pcd, rgb_vals)
        bbox = visual_utils.visualize_bbox(
            self.voxel_x,
            self.voxel_y,
            self.voxel_z,
            color=[1, 0, 0],
        )
        # voxel = visualize_voxel(obj.voxel_x, obj.voxel_y, obj.voxel_z, model, [1,0,0])
        frame = visual_utils.visualize_coordinate_frame_centered()
        o3d.visualization.draw_geometries([pcd, bbox, frame])
