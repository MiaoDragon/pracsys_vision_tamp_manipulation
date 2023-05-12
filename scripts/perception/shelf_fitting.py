import os
import pickle
import pyransac3d
import numpy as np
import open3d as o3d


def calib_ws(pcd, pcd_color, ws_fname, extrinsics, plane_threshold=0.02):
    """
    calibrate the workspace: pose and size
    the x axis of the workspace is in the reverse direction of y axis of the robot
    the y axis of the workspace is in the x axis of the robot
    the z axis of the workspace is the same as the z of the robot

    The axes might not be exact (in real world we have some errors)
    we will just use the fitted cuboid plane model
    """
    if not os.path.exists(ws_fname):
        plane_models = filter_ws(
            pcd,
            pcd_color,
            ws_fname,
            plane_threshold=plane_threshold,
        )
    else:
        f = open(ws_fname, 'rb')
        plane_models = pickle.load(f)
    # since the plane model is in the camera space, transform in the world frame
    plane_models = np.array(plane_models)
    cam_T_wd = np.linalg.inv(extrinsics)
    plane_models = cam_T_wd.T.dot(plane_models.T).T
    plane_models = plane_models / np.linalg.norm(
        plane_models[:, :-1],
        axis=1,
        keepdims=True,
    )

    # find the one in the y axis of the world
    x_axis = np.array([1., 0, 0])
    y_axis = np.array([0., 1, 0])
    z_axis = np.array([0., 0, 1])
    normals = np.array(plane_models)
    normals = normals[:, :3]
    x_idx = np.argmax(normals.dot(-y_axis))
    x_d = plane_models[x_idx, 3]
    y_idx = np.argmax(normals.dot(x_axis))
    y_d = plane_models[y_idx, 3]
    z_idx = np.argmax(normals.dot(z_axis))
    z_d = plane_models[z_idx, 3]

    print(normals)
    print(plane_models)
    # find the intersection
    # [n1T, n2T, n3T]x = [-d1, -d2, -d3]
    mat = np.array([normals[x_idx], normals[y_idx], normals[z_idx]])
    pt = np.linalg.inv(mat).dot(np.array([-x_d, -y_d, -z_d]))
    pose = np.eye(4)
    pose[:3, 0] = normals[x_idx]
    pose[:3, 1] = normals[y_idx]
    pose[:3, 2] = normals[z_idx]
    pose[:3, 3] = pt

    # * find the size
    x__idx = np.argmax(normals.dot(y_axis))
    x__d = -plane_models[x__idx, 3]
    y__idx = np.argmax(normals.dot(-x_axis))
    y__d = -plane_models[y__idx, 3]
    z__idx = np.argmax(normals.dot(-z_axis))
    z__d = -plane_models[z__idx, 3]
    size = np.array([x_d - x__d, y_d - y__d, z_d - z__d])
    size = np.abs(size)

    print('pose: ', pose)
    print('size: ', size)
    return plane_models, pose, size


def filter_ws(point_cloud, pcd_color, ws_fname, plane_threshold=0.02):
    """
    filter out the back plane, the side planes and the bottom plane
    For confined space
    """
    ### filter out three largest planes which represent known background
    point_cloud = np.array(point_cloud)
    pcd_color = np.array(pcd_color)
    plane_models = []
    keep_mask, cubid_models = filter_cuboid(
        point_cloud,
        plane_threshold=plane_threshold,
    )
    plane_models += cubid_models
    visualize_point_cloud(
        point_cloud,
        pcd_color,
        show_normal=True,
        mask=~keep_mask,
    )
    visualize_point_cloud(point_cloud, pcd_color, show_normal=True)

    f = open(ws_fname, 'wb')
    pickle.dump(plane_models, f)
    return plane_models


def filter_cuboid(
    point_cloud_input,
    segment_kwargs={
        'distance_threshold': 0.01,
        'num_iterations': 1000
    },
    plane_threshold=0.04
):
    visualize_point_cloud(point_cloud_input)
    cuboid = pyransac3d.Cuboid()
    best_eq, best_inliers = cuboid.fit(
        point_cloud_input,
        thresh=segment_kwargs['distance_threshold'],
        maxIteration=segment_kwargs['num_iterations']
    )
    # plane_info = find_largest_plane(
    #         point_cloud_input, segment_plane_kwargs, plane_model_input)
    inliers_mask = np.zeros(point_cloud_input.shape[0]).astype(bool)
    inliers_mask[best_inliers] = True
    # mask of points: to remove points near the plane, and also behind it
    # obtain normal vector sign: where most point cloud locate (has a positive inner product)
    # [a,b,c,d] = plane_info['plane_model']
    keep_mask = ~inliers_mask
    models = []
    for i in range(len(best_eq)):
        model = np.array(best_eq[i])
        model = model / np.linalg.norm([model[:-1]])  # normalize
        model[-1] = model[-1]
        # apply the threshold
        pcd = point_cloud_input[~inliers_mask]
        v = np.mean(pcd.dot(model[:3] + model[-1]))
        if v < 0:
            model = -model
        model[-1] -= plane_threshold  # applying the offset for the plane
        keep_mask = keep_mask & (point_cloud_input.dot(model[:3]) >= -model[-1])

        # opposite plane
        offset = np.max(pcd.dot(model[:3]) + model[-1])
        oppo_model = np.array(model)
        oppo_model[-1] -= offset
        oppo_model = -oppo_model
        oppo_model[-1] -= plane_threshold  # applying the offset for the plane
        keep_mask = keep_mask & (
            point_cloud_input.dot(oppo_model[:3]) >= -oppo_model[-1]
        )
        models.append(np.array(model))
        models.append(np.array(oppo_model))

    return keep_mask, models


def visualize_point_cloud(
    point_cloud_array,
    pcd_color=None,
    show_normal=False,
    mask=None,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_array)
    if pcd_color is not None:
        pcd_color = np.array(pcd_color)
        # for masked pcd, change color
        if mask is not None:
            pcd_color[mask] = np.array([1, 0, 0])
        pcd.colors = o3d.utility.Vector3dVector(pcd_color)

    if show_normal == True:
        pcd.estimate_normals(
            search_param=o3d.geometry
            .KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    else:
        o3d.visualization.draw_geometries([pcd], point_show_normal=False)


def crop_pcd_by_plane(pcd, pcd_color, plane_model, plane_threshold=0.):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    keep_mask = (pcd.dot(normal) >= -d + plane_threshold)
    visualize_point_cloud(pcd, pcd_color, mask=~keep_mask)
    return pcd[keep_mask], pcd_color[keep_mask]


def crop_pcd_by_planes(pcd, pcd_color, plane_models, plane_threshold=0.):
    for i in range(len(plane_models)):
        pcd, pcd_color = crop_pcd_by_plane(
            pcd,
            pcd_color,
            plane_models[i],
            plane_threshold,
        )
    return pcd, pcd_color


def crop_pcd(mask, pcd, pcd_color=None):
    if pcd_color is None:
        return pcd[mask], None
    else:
        return pcd[mask], pcd_color[mask]
