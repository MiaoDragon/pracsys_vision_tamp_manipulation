import cv2
import numpy as np


def depth_color_to_pcd(depth, color, intrinsics):
    """
    assuming the camera has its forward-vec in the reverse direction of the lookat
    since the depth value is positive (inverted), we need to get the reverse of it
    to find the 
    """
    fx = intrinsics[0, 0]
    ppx = intrinsics[0, 2]
    fy = intrinsics[1, 1]
    ppy = intrinsics[1, 2]

    print('intrinsics: ')
    print(intrinsics)

    # if self.mode == 'real':
    #     depth = np.array(depth_numpy_image) / 1000
    # else:
    depth = np.array(depth)
    i, j = np.indices(depth.shape)
    x = (j - ppx) / fx * depth
    y = (i - ppy) / fy * depth
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    pcd = np.array([x, y, depth]).T
    color = color.reshape((-1, 3))
    mask = np.nonzero(pcd[:, 2])
    pcd = pcd[mask]
    color = color[mask]
    return pcd, color


def pcd_to_depth(pcd, intrinsics, height, width):
    """
    given the pcd in the camera frame, use the intrinsics to generate
    a depth image of the pcd.
    """
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    projected_pcd = np.zeros((len(pcd), 2))
    projected_pcd[:, 0] = pcd[:, 0] / pcd[:, 2] * fx + cx
    projected_pcd[:, 1] = pcd[:, 1] / pcd[:, 2] * fy + cy
    depth = pcd[:, 2]
    projected_pcd = np.floor(projected_pcd).astype(int)
    # mask out outside pts
    valid_mask = (projected_pcd[:, 1] >= 0)
    valid_mask &= (projected_pcd[:, 1] < height)
    valid_mask &= (projected_pcd[:, 0] >= 0)
    valid_mask &= (projected_pcd[:, 0] < width)
    projected_pcd = projected_pcd[valid_mask]
    depth_img = np.zeros((height, width)).astype(float)
    depth_img[projected_pcd[:, 1], projected_pcd[:, 0]] = depth
    depth_img = cv2.boxFilter(np.float32(depth_img), -1, (5, 5))
    return depth_img
