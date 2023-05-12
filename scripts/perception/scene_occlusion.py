"""
modeling of the scene occlusion

"""
import cv2
import numpy as np


class SceneOcclusion():

    def __init__(self, pose, size, resols):
        # * init voxel
        pose = np.array(pose)
        size = np.array(size)
        resols = np.array(resols)
        self.occupied = np.ceil(size / resols).astype(int)
        self.occupied = np.zeros(self.occupied).astype(bool)
        self.occluded = np.ceil(size / resols).astype(int)
        self.occluded = np.zeros(self.occluded).astype(bool)
        self.voxel_x, self.voxel_y, self.voxel_z = np.indices(
            self.occluded.shape
        ).astype(float)
        self.pose = pose
        self.size = size  # this is real-valued size of the workspace
        self.resols = resols
        self.world_T_voxel = pose
        self.voxel_T_world = np.linalg.inv(pose)

    def get_occlusion(
        self,
        depth_img,
        color_img,
        camera_extrinsics,
        camera_intrinsics,
    ):
        """
        generate the occlusion for the entire scene
        occlusion includes: 
        - object occupied space (after object is fully reconstructed)
        - occlusion due to known object, 
        - occlusion due to unknown object

        TODO: we might need to consider the seg_img when robot comes in the view
        """
        pt = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0.5, 0.5, 0.5],
            ]
        )
        occluded = np.zeros(self.voxel_x.shape).astype(bool)

        for i in range(len(pt)):
            voxel_vecs = np.array([
                self.voxel_x,
                self.voxel_y,
                self.voxel_z,
            ]).transpose((1, 2, 3, 0)).reshape(-1, 3)
            voxel_vecs = voxel_vecs + pt[i].reshape(
                1, -1
            )  # get the middle point
            voxel_vecs = voxel_vecs * self.resols
            transformed_voxels = self.pose[:3, :3].dot(voxel_vecs.T).T
            transformed_voxels += self.pose[:3, 3]

            # get to the image space
            cam_transform = np.linalg.inv(camera_extrinsics)
            transformed_voxels = cam_transform[:3, :3].dot(
                transformed_voxels.T
            ).T + cam_transform[:3, 3]

            cam_to_voxel_depth = np.array(transformed_voxels[:, 2])
            # intrinsics
            cam_intrinsics = camera_intrinsics
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
            valid_mask = transformed_voxels[:, 0] >= 0
            valid_mask &= transformed_voxels[:, 0] < len(depth_img[0])
            valid_mask &= transformed_voxels[:, 1] >= 0
            valid_mask &= transformed_voxels[:, 1] < len(depth_img)
            voxel_depth[valid_mask] = \
                depth_img[
                    transformed_voxels[valid_mask][:, 1],
                    transformed_voxels[valid_mask][:, 0]
                ]
            valid_mask = valid_mask.reshape(self.voxel_x.shape)
            voxel_depth = voxel_depth.reshape(self.voxel_x.shape)

            cam_to_voxel_depth = cam_to_voxel_depth.reshape(self.voxel_x.shape)
            included = cam_to_voxel_depth - voxel_depth >= 0.
            # depth > 0 in case we might want to mask certain regions
            included &= voxel_depth > 0.
            included &= valid_mask
            occluded |= included

        # print(occluded.astype(int).sum() / valid_mask.astype(int).sum())
        # del cam_to_voxel_depth
        # del voxel_depth
        # del voxel_vecs
        # del transformed_voxels
        # del valid_mask

        return occluded

    def single_object_occlusion(
        self,
        camera_extrinsics,
        camera_intrinsics,
        obj_pose,
        obj_pcd,
    ):
        occupied = np.zeros(self.voxel_x.shape).astype(bool)
        occluded = np.zeros(self.voxel_x.shape).astype(bool)
        R = obj_pose[:3, :3]
        T = obj_pose[:3, 3]

        pcd = R.dot(obj_pcd.T).T + T

        # ** filter out the voxels that correspond to object occupied space
        # map the pcd to voxel space
        pcd_in_voxel = self.world_T_voxel[:3, :3].dot(pcd.T).T
        pcd_in_voxel += self.world_T_voxel[:3, 3]
        pcd_in_voxel /= self.resols
        # the floor of each axis will give us the index in the voxel
        indices = np.floor(pcd_in_voxel).astype(int)
        # extract the ones that are within the limit
        indices = indices[indices[:, 0] >= 0]
        indices = indices[indices[:, 0] < self.voxel_x.shape[0]]
        indices = indices[indices[:, 1] >= 0]
        indices = indices[indices[:, 1] < self.voxel_x.shape[1]]
        indices = indices[indices[:, 2] >= 0]
        indices = indices[indices[:, 2] < self.voxel_x.shape[2]]

        occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

        # ** extract the occlusion by object id
        cam_transform = np.linalg.inv(camera_extrinsics)

        transformed_pcd = cam_transform[:3, :3].dot(pcd.T).T
        transformed_pcd += cam_transform[:3, 3]
        fx = camera_intrinsics[0][0]
        fy = camera_intrinsics[1][1]
        cx = camera_intrinsics[0][2]
        cy = camera_intrinsics[1][2]
        transformed_pcd[:, 0] /= transformed_pcd[:, 2]
        transformed_pcd[:, 0] *= fx
        transformed_pcd[:, 0] += cx
        transformed_pcd[:, 1] /= transformed_pcd[:, 2]
        transformed_pcd[:, 1] *= fy
        transformed_pcd[:, 1] += cy
        depth = transformed_pcd[:, 2]
        sort_indices = np.argsort(-depth)
        transformed_pcd = transformed_pcd[:, :2]
        transformed_pcd = np.floor(transformed_pcd).astype(int)
        transformed_pcd = transformed_pcd[sort_indices]
        depth = depth[sort_indices]
        max_j = transformed_pcd[:, 0].max() + 1
        max_i = transformed_pcd[:, 1].max() + 1
        depth_img = np.zeros((max_i, max_j)).astype(float)
        depth_img[transformed_pcd[:, 1], transformed_pcd[:, 0]] = depth

        # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        # depth_img = cv2.resize(depth_img, ori_shape, interpolation=cv2.INTER_LINEAR)
        # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        depth_img = cv2.medianBlur(np.float32(depth_img), 5)
        # depth_img = cv2.boxFilter(np.float32(depth_img), -1, (5, 5))

        occluded_i = self.get_occlusion(
            depth_img,
            None,
            camera_extrinsics,
            camera_intrinsics,
        )

        occluded = (~occupied) & occluded_i
        # occluded = occluded_i

        del indices
        del pcd_in_voxel
        del transformed_pcd

        return occupied, occluded

    def sample_pcd(self, mask, n_sample=10):
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
        total_sample = total_sample + grid_sample
        total_sample = total_sample + np.array([
            voxel_x,
            voxel_y,
            voxel_z,
        ]).T.reshape(len(voxel_x), 1, 3)
        total_sample = total_sample.reshape(-1, 3) * np.array(self.resols)

        # del voxel_x
        # del voxel_y
        # del voxel_z
        return total_sample

    def occlusion_from_pcd(
        self,
        camera_extrinsics,
        camera_intrinsics,
        img_shape,
        obj_poses,
        obj_pcds,
    ):
        # depth_img = np.zeros(img_shape).astype(float)
        occluded = np.zeros(self.voxel_x.shape).astype(bool)
        for obj_id, obj_pose in obj_poses.items():
            obj_pcd = obj_pcds[obj_id]
            R = obj_pose[:3, :3]
            T = obj_pose[:3, 3]

            pcd = R.dot(obj_pcd.T).T + T

            # ** extract the occlusion by object id
            cam_transform = np.linalg.inv(camera_extrinsics)

            # NOTE: multiple pcds can map to the same depth. We need to use the min value of the depth if this happens
            if len(pcd) == 0:
                continue
            transformed_pcd = cam_transform[:3, :3].dot(pcd.T).T
            transformed_pcd += cam_transform[:3, 3]
            fx = camera_intrinsics[0][0]
            fy = camera_intrinsics[1][1]
            cx = camera_intrinsics[0][2]
            cy = camera_intrinsics[1][2]
            transformed_pcd[:, 0] /= transformed_pcd[:, 2]
            transformed_pcd[:, 0] *= fx
            transformed_pcd[:, 0] += cx
            transformed_pcd[:, 1] /= transformed_pcd[:, 2]
            transformed_pcd[:, 1] *= fy
            transformed_pcd[:, 1] += cy
            depth = transformed_pcd[:, 2]
            transformed_pcd = transformed_pcd[:, :2]
            transformed_pcd = np.floor(transformed_pcd).astype(int)
            max_j = transformed_pcd[:, 0].max() + 1
            max_i = transformed_pcd[:, 1].max() + 1
            valid_filter = (transformed_pcd[:, 0] >= 0)
            valid_filter &= (transformed_pcd[:, 0] < img_shape[1])
            valid_filter &= (transformed_pcd[:, 1] >= 0)
            valid_filter &= (transformed_pcd[:, 1] < img_shape[0])
            transformed_pcd = transformed_pcd[valid_filter]
            depth = depth[valid_filter]

            depth_img = np.zeros((max_i, max_j)).astype(float)
            if len(transformed_pcd) == 0:
                continue

            unique_indices = np.unique(transformed_pcd, axis=0)
            unique_valid = (unique_indices[:, 0] >= 0)
            unique_valid &= (unique_indices[:, 1] >= 0)
            unique_indices = unique_indices[unique_valid]
            unique_depths = np.zeros(len(unique_indices))
            for i in range(len(unique_indices)):
                unique_depths[i] = \
                    depth[
                        (transformed_pcd[:, 0] == unique_indices[i, 0]) &
                        (transformed_pcd[:, 1] == unique_indices[i, 1])
                    ].min()
            depth_img[unique_indices[:, 1], unique_indices[:,0]] = \
                    unique_depths
            # depth_img[transformed_pcd[:,1],transformed_pcd[:,0]] = depth

            # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            # depth_img = cv2.resize(depth_img, ori_shape, interpolation=cv2.INTER_LINEAR)
            # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            depth_img = cv2.medianBlur(np.float32(depth_img), 5)

            occluded_i = self.get_occlusion(
                depth_img,
                None,
                camera_extrinsics,
                camera_intrinsics,
            )
            occluded = occluded | occluded_i
        return occluded
