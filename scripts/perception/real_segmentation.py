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
import ros_numpy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point

import cv2
import open3d as o3d

from rearrangement_system.msg import CylinderCenter
from rearrangement_system.srv import RoughPoseEstimation, RoughPoseEstimationResponse
from cv_bridge import CvBridge

import matplotlib.pyplot as plt
import skg
import pickle
from pracsys_perception.srv import SegmentationSrv
import transformations as tf

### This file defines the real camera for the purpose of getting the images of the physical camera

class CylinderSegmentation():

    def __init__(self, camera):
        self.camera = camera  # camera model
        self.bridge = CvBridge()
        self.getDepthIntrinsicInfo()
    def getDepthIntrinsicInfo(self):
        intrinsics = self.camera.info['intrinsics']
        self.depth_camera_info = dict()
        self.depth_camera_info['intrinsics'] = dict()
        self.depth_camera_info['intrinsics']['fx'] = intrinsics[0,0]
        self.depth_camera_info['intrinsics']['fy'] = intrinsics[1,1]
        self.depth_camera_info['intrinsics']['ppx'] = intrinsics[0,2]
        self.depth_camera_info['intrinsics']['ppy'] = intrinsics[1,2]
        self.depth_camera_info['height'] = self.camera.info['img_shape'][0]
        self.depth_camera_info['width'] = self.camera.info['img_shape'][1]

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

    def convert_depth_to_point_cloud(self, depth_numpy_image):
        """
        Args:
            depth_numpy_image (np.array): depthmap, shape [h, w]
        Return:
            point_cloud (np.array): point cloud, shape [h*w, 3] in x-y-z
                The x values of the pointcloud in meters
                The y values of the pointcloud in meters
                The z values of the pointcloud in meters
        """
        [height, width] = depth_numpy_image.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        x = (u.flatten() - self.depth_camera_info['intrinsics']['ppx'])/self.depth_camera_info['intrinsics']['fx']
        y = (v.flatten() - self.depth_camera_info['intrinsics']['ppy'])/self.depth_camera_info['intrinsics']['fy']

        z = depth_numpy_image.flatten() / 1000
        x = np.multiply(x,z)
        y = np.multiply(y,z)

        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]
        return np.stack([x, y, z], axis = -1)

    def convert_depth_color_to_point_cloud(self, depth_numpy_image, color_numpy_image):
        depth = depth_numpy_image / 1000

        i,j = np.indices(depth_numpy_image.shape)
        x = (j - self.depth_camera_info['intrinsics']['ppx'])/self.depth_camera_info['intrinsics']['fx'] * depth
        y = (i - self.depth_camera_info['intrinsics']['ppy'])/self.depth_camera_info['intrinsics']['fy'] * depth
        x = x.reshape(-1)
        y = y.reshape(-1)
        depth = depth.reshape(-1)
        pcd = np.array([x,y,depth]).T

        color = color_numpy_image.reshape((-1,3))
        mask = np.nonzero(pcd[:,2])
        pcd = pcd[mask]
        color = color[mask]
        return pcd, color

    def convert_pcd_to_indices(self, pcd):
        height = self.depth_camera_info['height']
        width = self.depth_camera_info['width']
        pcd = pcd / pcd[:,2].reshape((-1,1))
        x = pcd[:,0] * self.depth_camera_info['intrinsics']['fx'] + self.depth_camera_info['intrinsics']['ppx']
        y = pcd[:,1] * self.depth_camera_info['intrinsics']['fy'] + self.depth_camera_info['intrinsics']['ppy']
        indices = np.array([y,x]).T
        indices = np.floor(indices).astype(int)
        return indices



    def visualize_point_cloud(self, point_cloud_array, pcd_color=None, show_normal=False, mask=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_array)
        if pcd_color is not None:
            pcd_color = np.array(pcd_color)
            # for masked pcd, change color
            if mask is not None:
                pcd_color[mask] = np.array([1,0,0])
            pcd.colors = o3d.utility.Vector3dVector(pcd_color)
            
        if show_normal == True:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        else:
            o3d.visualization.draw_geometries([pcd], point_show_normal=False)
 
    def crop_pcd(self, mask, pcd, pcd_color=None):
        if pcd_color is None:
            return pcd[mask], None
        else:
            return pcd[mask], pcd_color[mask]

    def filter_plane(self, point_cloud, pcd_color):
        """
        filter out the back plane, the side planes and the bottom plane
        """
        ### filter out three largest planes which represent known background 
        point_cloud = np.array(point_cloud)
        pcd_color = np.array(pcd_color)
        plane_models = []
        for i in range(4):
            keep_mask, plane_model = self.filter_largest_plane(point_cloud)
            self.visualize_point_cloud(point_cloud, pcd_color, show_normal=False, mask=~keep_mask)
            point_cloud, pcd_color = self.crop_pcd(keep_mask, point_cloud, pcd_color)
            plane_models.append(plane_model)
        
        # find the back plane normal, which is close to z axis of the camera frame
        plane_models = np.array(plane_models)
        z_axis = np.array([0,0,-1.])
        idx_back = plane_models[:,:3].dot(z_axis).argmax()
        back_model = np.array(plane_models[idx_back])
        # find the side plane normal, which are the closest pairs
        plane_models = np.delete(plane_models, idx_back, 0)

        max_pair1 = -1
        max_pair2 = -1
        max_v = -1
        for i in range(3):
            for j in range(i+1,3):
                dot = np.abs(plane_models[i,:3].dot(plane_models[j,:3]))
                if dot > max_v:
                    max_v = dot
                    max_pair1 = i
                    max_pair2 = j
        
        side_models = np.array(plane_models[[max_pair1, max_pair2]])
        bot_plane = np.delete(plane_models, [max_pair1, max_pair2], 0).reshape(-1)

        # store the plane models
        plane_dict = {"back_plane": back_model, "side_planes": side_models, "bot_plane": bot_plane}
        f = open('plane_models.pkl', 'wb')
        pickle.dump(plane_dict, f)

    def find_cylinders(self, rgb_img, depth_img, num_objects=2, filter_plane=False):
        color_numpy_image = np.array(rgb_img) / 255
        depth_numpy_image = np.array(depth_img)
        # print('depth_numpy_image shape: ')
        # print(depth_numpy_image.shape)

        # print('color_numpy_image shape: ')
        # print(color_numpy_image.shape)
        # input('next')
        # print(depth_numpy_image.dtype)
        depth_numpy_image = np.array(depth_numpy_image).astype(np.float32)
        # TODO: apply filter to the depth image so it is smoother
        kernel_size = 10
        # morphed_depth = cv2.morphologyEx(depth_numpy_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
        depth1 = cv2.bilateralFilter(src=depth_numpy_image, d=11, sigmaColor=0.05, sigmaSpace=0.65)
        # cv2.imshow('morphed_depth', depth1/depth1.max())
        # cv2.waitKey(0)

        # point_cloud = self.convert_depth_to_point_cloud(depth1)
        point_cloud, pcd_color = self.convert_depth_color_to_point_cloud(depth1,color_numpy_image)

        # input("see the whole point cloud without any filtering")
        # self.visualize_point_cloud(point_cloud, show_normal=False)

        #################### filter out some background noise ####################
        point_cloud, pcd_color = self.crop_pcd(point_cloud[:, 2] > 0.9, point_cloud, pcd_color)
        # input("see the point cloud after filtering those within 1.1m")
        # self.visualize_point_cloud(point_cloud, pcd_color, show_normal=False)
        point_cloud, pcd_color = self.crop_pcd(point_cloud[:, 2] < 1.65, point_cloud, pcd_color)
        # input("see the point cloud after filtering based on z axis distance")
        # self.visualize_point_cloud(point_cloud, pcd_color, show_normal=False)

        # point_cloud = point_cloud[point_cloud[:, 1] < 0.25]
        # # input("see the point cloud after filtering based on y axis distance")
        # self.visualize_point_cloud(point_cloud, show_normal=False)

        # point_cloud = point_cloud[point_cloud[:, 0] < 0.65]
        # point_cloud = point_cloud[point_cloud[:, 0] > -0.65]

        # self.visualize_point_cloud(point_cloud, show_normal=False)
        ###########################################################################

        ### filter out three largest planes which represent known background 
        if filter_plane:
            self.filter_plane(point_cloud, pcd_color)
        # use the filtered plane to segment out objects
        f = open('plane_models.pkl', 'rb')
        plane_models = pickle.load(f)
        back_model = plane_models['back_plane']
        side_models = plane_models['side_planes']
        bot_model = plane_models['bot_plane']
        bot_plane = bot_model
        mask = (point_cloud[:,:3].dot(back_model[:3]) + back_model[3] >= 0.04)
        mask &= (point_cloud[:,:3].dot(side_models[0][:3]) + side_models[0][3] >= 0.04)
        mask &= (point_cloud[:,:3].dot(side_models[1][:3]) + side_models[1][3] >= 0.04)
        mask &= (point_cloud[:,:3].dot(bot_model[:3]) + bot_model[3] >= 0.04)
        # self.visualize_point_cloud(point_cloud, pcd_color, show_normal=False, mask=~mask)
        point_cloud, pcd_color = self.crop_pcd(mask, point_cloud, pcd_color)

        # crop the ceiling
        mask = (point_cloud[:,:3].dot(bot_plane[:3]) + bot_plane[3] >= 0.5-0.05)
        mask = ~mask
        # self.visualize_point_cloud(point_cloud, pcd_color, show_normal=False, mask=~mask)
        point_cloud, pcd_color = self.crop_pcd(mask, point_cloud, pcd_color)
        
        # self.visualize_point_cloud(point_cloud, pcd_color, show_normal=False)
        mask = self.outlier_filter_pcd(point_cloud, pcd_color)
        print('after outlier_filter_pcd...')
        point_cloud, pcd_color = self.crop_pcd(mask, point_cloud, pcd_color)


        # TODO: get the point clouds belong to the object
        labels = self.dbscan_from_point_cloud(point_cloud) ### point cloud classification
        sizes_label = np.sum(np.expand_dims(labels, axis = 1) == np.arange(np.max(labels) + 1), axis = 0)
        top_size_label = np.argsort(sizes_label)[::-1] ### sort in descending order
        

        print('number of labels: ')
        # visualize each label
        cylinders = []
        total_mask = np.zeros(labels.shape).astype(bool)

        cylinder_models = []
        
        for i in range(num_objects):
            print('showing label %d/%d...' % (i,len(top_size_label)))
            mask = (labels == top_size_label[i])
            total_mask |= mask

            self.visualize_point_cloud(point_cloud, pcd_color, show_normal=False, mask=mask)

            pcd_i, color_i = self.crop_pcd(mask, point_cloud, pcd_color)
            mid_center, R, height, axis, rot_mat, cylinder = self.fit_cylinder(pcd_i, bot_plane[:3])
            cylinders.append(cylinder)
            cylinder_model = {'mid_center': mid_center, 'radius': R, 'height': height, 'axis': axis, 'transform': rot_mat, 'pcd': pcd_i}
            cylinder_models.append(cylinder_model)

            # input('show next label...')
        # visualize the cylinders
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(point_cloud)
        vis_pcd.colors = o3d.utility.Vector3dVector(pcd_color)
        o3d.visualization.draw_geometries([vis_pcd] + cylinders)

        # * extract the point cloud corresponding to cylinders
        point_cloud, pcd_color = self.crop_pcd(total_mask, point_cloud, pcd_color)
        return cylinder_models

    def find_largest_plane(self, point_cloud_input,
            segment_plane_kwargs = {'distance_threshold': 0.01,
                'ransac_n': 3, 'num_iterations': 1000},
            plane_model_input = None):
        """
        Find The Largest Plane
        Args:
            point_cloud_input (np.array): input point cloud, shape [n, 3] in x-y-z
            segment_plane_kwargs (dictionary): arguments for segment_plane
                check http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Plane-segmentation
            plane_model_input (np.array or None): input plane model
                (𝑎,𝑏,𝑐,𝑑) such that for each point (𝑥,𝑦,𝑧) on the plane we have 𝑎𝑥+𝑏𝑦+𝑐𝑧+𝑑=0
        Return:
            plane_info (dictionary): plane information
                'plane_model' (np.array): (𝑎,𝑏,𝑐,𝑑) such that for each point (𝑥,𝑦,𝑧) on the plane we have 𝑎𝑥+𝑏𝑦+𝑐𝑧+𝑑=0
                'inliers' (list): indices of inliers
        """
        if (plane_model_input is not None):
            distance_to_plane = np.abs(np.sum(point_cloud_input[:, :3]
                * plane_model_input[:3], axis = 1) + plane_model_input[-1]
                ) / np.linalg.norm(plane_model_input[:3])
            inliers = np.where(distance_to_plane
                    <= segment_plane_kwargs['distance_threshold'])[0].tolist()
            plane_model = plane_model_input
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_input)
            plane_model, inliers = pcd.segment_plane(**segment_plane_kwargs)
            # visualize the pcd
            inlier_cloud = pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1,0,0])
            outlier_cloud = pcd.select_by_index(inliers, invert=True)
            outlier_cloud.paint_uniform_color([0,0,0])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        return {'plane_model': plane_model, 'inliers': inliers}

    def filter_largest_plane(self, point_cloud_input,
            segment_plane_kwargs = {'distance_threshold': 0.01,
                'ransac_n': 10, 'num_iterations': 1000},
            plane_model_input = None):

        plane_info = self.find_largest_plane(
                point_cloud_input, segment_plane_kwargs, plane_model_input)
        inliers_mask = np.zeros(point_cloud_input.shape[0]).astype(np.bool)
        inliers_mask[np.array(plane_info['inliers'])] = True

        # mask of points: to remove points near the plane, and also behind it
        # obtain normal vector sign: where most point cloud locate (has a positive inner product)
        [a,b,c,d] = plane_info['plane_model']
        model = np.array([a,b,c,d])
        model = model / np.linalg.norm([a,b,c])  # normalize
        [a,b,c,d] = model
        pcd = point_cloud_input[~inliers_mask]
        v = np.mean(a * pcd[:,0] + b * pcd[:,1] + c * pcd[:,2] + d)
        if v < 0:
            a = -a
            b = -b
            c = -c
            d = -d
        normal = np.array([a,b,c])

        keep_mask = (~inliers_mask)
        keep_mask = keep_mask & (point_cloud_input.dot(normal) >= -d+0.04)
        
        # point_cloud_output = point_cloud_input[inliers_mask]
        return keep_mask, np.array([a,b,c,d])

    def outlier_filter_pcd(self, pcd, pcd_color):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        # pcd_o3d = pcd_o3d.uniform_down_sample(every_k_points=5)
        print("before RSO..")
        cl, ind = pcd_o3d.remove_statistical_outlier(20, 2.0) # segfault?
        # cl, ind = pcd_o3d.remove_radius_outlier(16, 0.02)  # segfault?
        print("after RSO..")

        ind = np.array(ind).astype(int)
        mask = np.zeros(len(pcd)).astype(bool)
        mask[ind] = True

        # self.visualize_point_cloud(pcd, pcd_color, False, ~mask)
        # input('outlier removal')
        return mask
        
    def fit_cylinder(self, point_cloud, bottom_normal):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        pcd_o3d = pcd_o3d.normalize_normals()
        # normalzie normals
        normals = np.asarray(pcd_o3d.normals)

        # * find the cylinder axis by SVD
        if bottom_normal is None:
            X = 0.
            for i in range(len(normals)):
                normal = normals[i].reshape((3,1))
                X += normal.dot(normal.T)
            eigval, eigvec = np.linalg.eigh(X)
            # sort and use the smallest eigen-vector
            vec = eigvec[np.argsort(eigval)[0]]
        
        vec = bottom_normal

        # project the points to the plane
        projected_pcd = point_cloud - point_cloud.dot(vec).reshape((-1,1))*vec.reshape((1,3))
        # project to 2D by finding two axis
        x_2d = np.array([1,0,0])
        x_2d = np.cross(vec, x_2d)
        x_2d = x_2d / np.linalg.norm(x_2d)
        y_2d = np.cross(vec, x_2d)

        projected_2d = np.zeros((len(projected_pcd),2))
        projected_2d[:,0] = projected_pcd.dot(x_2d)
        projected_2d[:,1] = projected_pcd.dot(y_2d)


        # show the axis 
        if False:
            min_v = point_cloud.dot(vec).min()
            max_v = point_cloud.dot(vec).max()

            height = max_v - min_v

            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.1)
            mid_center = projected_pcd[0] + (min_v + max_v)/2 * vec
            rot_mat = np.eye(4)
            rot_mat[:3,0] = x_2d
            rot_mat[:3,1] = y_2d
            rot_mat[:3,2] = vec
            rot_mat[0,3] = mid_center[0]
            rot_mat[1,3] = mid_center[1]
            rot_mat[2,3] = mid_center[2]

            cylinder.transform(rot_mat)
            cylinder.paint_uniform_color([1,0,0])
            pcd_o3d.paint_uniform_color([0,0,0])
            o3d.visualization.draw_geometries([pcd_o3d, cylinder], point_show_normal=False)


        # circle fitting
        import circle_fit
        # xc, yc, R, var = circle_fit.least_squares_circle(projected_pcd)

        R, c = skg.nsphere.nsphere_fit(projected_2d)
        print('center: ')
        print(c)
        xc, yc = c

        # circle_fit.plot_data_circle(projected_2d[:,0], projected_2d[:,1], xc, yc, R)
        # plt.show()
        # transform back to get the cylinder
        center = xc * x_2d + yc * y_2d

        min_v = point_cloud.dot(vec).min()
        max_v = point_cloud.dot(vec).max()

        height = max_v - min_v
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=R, height=height)

        # obtain the transform
        mid_center = center + (min_v + max_v)/2 * vec
        rot_mat = np.eye(4)
        rot_mat[:3,0] = x_2d
        rot_mat[:3,1] = y_2d
        rot_mat[:3,2] = vec
        rot_mat[0,3] = mid_center[0]
        rot_mat[1,3] = mid_center[1]
        rot_mat[2,3] = mid_center[2]

        # rot_mat[:3,0] = xc
        # rot_mat[:3,1] = yc
        # rot_mat[:3,2] = height
        cylinder.transform(rot_mat)

        cylinder.paint_uniform_color([1,0,0])
        pcd_o3d.paint_uniform_color([0,0,0])

        # o3d.visualization.draw_geometries([pcd_o3d, cylinder], point_show_normal=False)
        return mid_center, R, height, vec, rot_mat, cylinder

    def dbscan_from_point_cloud(self, point_cloud,
            dbscan_kwargs = {'eps': 0.01, 'min_samples': 5}):
        """
        Get Bounding Box By DBSCAN From Point Cloud
        Args:
            point_cloud (np.array): point cloud, shape [n, 3]
            dbscan_kwargs (dictionary): arguments for DBSCAN
                check https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        Return:
            label (np.array): label of each point, shape [n]q
        """
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(**dbscan_kwargs).fit(point_cloud)
        if (clustering.labels_.max() < 0):
            return None
        return clustering.labels_



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
                cylinder_i['mid_center'] = mid_center
                cylinder_i['radius'] = radius
                cylinder_i['height'] = height
                cylinder_i['axis'] = axis
                cylinder_i['transform'] = transform
                pts = resp.cylinders[i].pcd.points
                pcd = []
                for j in range(len(pts)):
                    pcd.append([pts[j].x,pts[j].y,pts[j].z])
                pcd = np.array(pcd)
                cylinder_i['pcd'] = pcd
                cylinder_models.append(cylinder_i)

            # seg_img = np.zeros(rgb_img.shape).astype(float)/255
            # rgb_img = cv2.cvtColor(rgb_img.astype(np.float32), cv2.COLOR_BGR2RGB)
            # colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]]
            # for i in range(len(cylinder_models)):
            #     cylinder = cylinder_models[i]
            #     pcd_i = np.array(cylinder['pcd'])
            #     ind_i = self.convert_pcd_to_indices(pcd_i)
            #     # seg_img[ind_i[:,0], ind_i[:,1]] = colors[i]
            #     seg_img[ind_i[:,0], ind_i[:,1]] = rgb_img[ind_i[:,0],ind_i[:,1]]

            poses = [cylinder_models[i]['transform'] for i in range(len(cylinder_models))]
            # cv2.imshow("segmentation", seg_img)
            # cv2.waitKey(0)

            seg_img = np.zeros(self.camera.info['img_shape']).astype(int)-1

            for i in range(len(cylinder_models)):
                cylinder = cylinder_models[i]
                pcd_i = np.array(cylinder['pcd'])
                ind_i = self.convert_pcd_to_indices(pcd_i)
                print('pcd.max: ', ind_i.max(axis=0))

                seg_img[ind_i[:,0], ind_i[:,1]] = i
                
            return seg_img, cylinder_models

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def segmentation_v2(self, rgb_img, depth_img, filter_plane=False):
        """
        segment the point cloud, then transform back to the image to provide the segmentation of objects
        NOTE: this gives the segmentation id, which may be different from the one used in simulation.
        # TODO: visualization: replace each object label with different colors to visualize

        # TODO: check the number of pixels for each pcd, and reject ones that have a small amount of pixels
        """
        num_objs = 3
        cylinder_models = self.find_cylinders(rgb_img, depth_img, num_objs, filter_plane)
        # {'mid_center': mid_center, 'radius': R, 'height': height, 'axis': axis, 'transform': rot_mat}

        # transform the cylinder to the seg_img
        seg_img = np.zeros(rgb_img.shape).astype(float)/255
        rgb_img = cv2.cvtColor(rgb_img.astype(np.float32), cv2.COLOR_BGR2RGB)
        colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]]
        for i in range(len(cylinder_models)):
            cylinder = cylinder_models[i]
            pcd_i = np.array(cylinder['pcd'])
            ind_i = self.convert_pcd_to_indices(pcd_i)
            # seg_img[ind_i[:,0], ind_i[:,1]] = colors[i]
            seg_img[ind_i[:,0], ind_i[:,1]] = rgb_img[ind_i[:,0],ind_i[:,1]]

        poses = [cylinder_models[i]['transform'] for i in range(len(cylinder_models))]
        cv2.imshow("segmentation", seg_img)
        cv2.waitKey(0)
        return seg_img, cylinder_models

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
