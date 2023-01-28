"""
generate a problem for the object retrieval problem under partial observation

objects are randomly placed on the shelf, and ensure stable placing and collision-free
the target object is hidden by other objects
"""

import os
import time
import json
import pickle
import random

import rospkg
import numpy as np
import open3d as o3d
import transformations as tf

from dm_control import mjcf
from utils.visual_utils import *
from utils.transform_utils import *


def load_problem(scene_json, robot_xml, obj_poses, obj_shapes, obj_sizes):
    scene_dict = None
    with open(scene_json, 'r') as f:
        scene_dict = json.load(f)
    if scene_dict is None:
        print("Could not read file:", scene_json)
        return
    world_model = mjcf.from_xml_string(
        """
    <mujoco model="World">
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="0.5 0.5" texuniform="true" specular="0" shininess="0" reflectance="0" emission="1" />
      </asset>
      <worldbody>
        <geom name="floor" size="2 2 .05" type="plane" material="grid" condim="3"/>
        <light directional="true" pos="-0.5 0.5 3" dir="0 0 -1" castshadow="false" diffuse="1 1 1"/>
        <body name="scene" pos="0 0 0">
        </body>
        <body name="phys" pos="-0.5 0 0">
          <freejoint/>
          <geom name="p1" size=".1" type="sphere" rgba=".9 .1 .1 1" pos="0.00 0 1.5"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )
    scene_body = world_model.worldbody.body['scene']
    scene_body.pos = scene_dict['workspace']['pos']
    scene_body.quat = np.roll(scene_dict['workspace']['ori'], 1)
    components = scene_dict['workspace']['components']
    for component_name, component in components.items():
        shape = np.array(component['shape'])
        scene_body.add(
            'geom',
            name=component_name,
            type='box',
            pos=component['pose']['pos'],
            quat=np.roll(component['pose']['ori'], 1),
            size=shape / 2,
            rgba=[1., 0.64, 0.0, 1.0],
            # gap=10,
        )

    num_objs = len(obj_shapes)
    for i in range(num_objs):
        obj_shape = obj_shapes[i]
        color = [*from_color_map(i, num_objs), 1]
        x_size, y_size, z_size = obj_sizes[i]
        x, y, z = obj_poses[i][:3, 3]
        quat = tf.quaternion_from_matrix(obj_poses[i])

        if obj_shape in ('cube', 'wall', 'ontop'):
            obj_shape = 'box'
            sizes = [x_size / 2, y_size / 2, z_size / 2]
        elif obj_shape == 'cylinder':
            sizes = [x_size / 2, z_size / 2]

        world_model.worldbody.add(
            'body',
            name=f'body{i}',
            pos=[x, y, z],
            quat=quat,
        ).add(
            'geom',
            name=f'geom{i}',
            type=obj_shape,
            condim=1,
            size=sizes,
            rgba=color,
        )

    robot = mjcf.from_path(robot_xml)
    world_model.attach(robot)
    return world_model


def random_stacked_problem(scene, level, num_objs, num_hiding_objs):
    """
    generate one random instance of the problem
    last one object is the target object
    """
    # load scene definition file
    pid = p.connect(PYBULLET_MODE)
    f = open(scene, 'r')
    scene_dict = json.load(f)

    rp = rospkg.RosPack()
    package_path = rp.get_path('vbcpm_execution_system')

    base_pos = scene_dict['workspace']['pos']
    workspace_low = np.add(scene_dict['workspace']['region_low'], base_pos)
    workspace_high = np.add(scene_dict['workspace']['region_high'], base_pos)
    padding = scene_dict['workspace']['padding']
    # camera = Camera()

    n_samples = 12000
    if True or level == 1:
        # obj_list = ['cube', 'wall', 'cylinder', 'cylinder', 'ontop', 'ontop']
        obj_list = ['cube', 'wall', 'ontop', 'ontop', 'cylinder']

        pcd_cube = np.random.uniform(
            low=[-0.5, -0.5, -0.5], high=[0.5, 0.5, 0.5], size=(n_samples, 3)
        )

        pcd_cylinder_r = np.random.uniform(low=0, high=0.5, size=n_samples)
        pcd_cylinder_r = np.random.triangular(
            left=0., mode=0.5, right=0.5, size=n_samples
        )
        pcd_cylinder_xy = np.random.normal(
            loc=[0., 0.], scale=[1., 1.], size=(n_samples, 2)
        )
        pcd_cylinder_xy = pcd_cylinder_xy / np.linalg.norm(
            pcd_cylinder_xy, axis=1
        ).reshape(-1, 1)
        pcd_cylinder_xy = pcd_cylinder_xy * pcd_cylinder_r.reshape(-1, 1)

        pcd_cylinder_h = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
        pcd_cylinder_h = pcd_cylinder_h.reshape(-1, 1)
        pcd_cylinder = np.concatenate([pcd_cylinder_xy, pcd_cylinder_h], axis=1)
        # print('pcd cube:')
        # print(pcd_cube)
        # print('pcd cylinder: ')
        # print(pcd_cylinder)
        # basic shape: cube of size 1, cylinder of size 1

        # assuming the workspace coordinate system is at the center of the world
        # * sample random objects on the workspace
        obj_ids = []
        obj_poses = []
        obj_pcds = []
        obj_shapes = []
        obj_sizes = []
        obj_tops = []
        obj_colors = []
        for i in range(num_objs):
            # randomly pick one object shape
            obj_shape = random.choice(obj_list)
            if i == num_hiding_objs:
                obj_shape = 'wall'
            if i == 0:
                obj_shape = 'cube'
            # obj_shape = obj_list[i%len(obj_list)]
            # randomly scale the object
            if obj_shape == 'cube':
                x_scales = np.arange(0.25, 0.40, 0.05) / 10
                y_scales = np.arange(0.25, 0.40, 0.05) / 10
                z_scales = np.arange(0.6, 1.0, 0.05) / 10
            elif obj_shape == 'ontop':
                x_scales = np.arange(0.25, 0.40, 0.05) / 10
                y_scales = np.arange(0.25, 0.40, 0.05) / 10
                z_scales = np.arange(0.6, 1.0, 0.05) / 10
            elif obj_shape == 'cylinder':
                x_scales = np.arange(0.25, 0.40, 0.05) / 10
                y_scales = np.arange(0.25, 0.40, 0.05) / 10
                z_scales = np.arange(1.0, 1.5, 0.05) / 10
            elif obj_shape == 'wall':
                x_scales = np.arange(0.25, 0.40, 0.05) / 10
                y_scales = np.arange(1.0, 2.0, 0.05) / 10
                z_scales = np.arange(1.2, 1.8, 0.05) / 10

            # if i == 0:
            #     color = [1.0, 0., 0., 1]
            # else:
            #     color = [*select_color(i), 1]
            color = [*from_color_map(i, num_objs), 1]

            # scale base object and transform until it satisfies constraints
            while True:
                x_size = x_scales[np.random.choice(len(x_scales))]
                y_size = y_scales[np.random.choice(len(y_scales))]
                z_size = z_scales[np.random.choice(len(z_scales))]
                if obj_shape == 'cylinder':
                    y_size = x_size

                # sample a pose in the workspace
                if i < num_hiding_objs:
                    x_low_offset = (workspace_high[0] - workspace_low[0] - x_size) / 2
                else:
                    x_low_offset = 0

                if obj_shape == 'cube' or obj_shape == 'wall' or obj_shape == 'ontop':
                    pcd = pcd_cube * np.array([x_size, y_size, z_size])
                elif obj_shape == 'cylinder':
                    pcd = pcd_cylinder * np.array([x_size, y_size, z_size])

                if obj_shape == 'ontop':
                    prev_ind = random.randint(0, i - 1)
                    x, y = obj_poses[prev_ind][:2, 3]
                    z = 0.001
                    z += obj_tops[prev_ind] + z_size
                    quat = p.getBasePositionAndOrientation(
                        obj_ids[prev_ind],
                        physicsClientId=pid,
                    )[1]
                    mRot = obj_poses[prev_ind][:3, :3]
                else:
                    x = np.random.uniform(
                        low=workspace_low[0] + x_size / 2 + x_low_offset,
                        high=workspace_high[0] - x_size / 2
                    )
                    y = np.random.uniform(
                        low=workspace_low[1] + y_size / 2,
                        high=workspace_high[1] - y_size / 2
                    )
                    z = 0.001
                    z += workspace_low[2] + z_size
                    quat = p.getQuaternionFromEuler(
                        (0, 0, np.random.uniform(-np.pi, np.pi))
                    )
                    mRot = np.reshape(p.getMatrixFromQuaternion(quat), (3, 3))

                # save top coord for later and adjust current z
                ztop = z
                z -= z_size / 2

                if obj_shape == 'cube' or obj_shape == 'wall' or obj_shape == 'ontop':
                    cid = p.createCollisionShape(
                        shapeType=p.GEOM_BOX,
                        halfExtents=[x_size / 2, y_size / 2, z_size / 2]
                    )
                    vid = p.createVisualShape(
                        shapeType=p.GEOM_BOX,
                        halfExtents=[x_size / 2, y_size / 2, z_size / 2],
                        rgbaColor=color
                    )
                elif obj_shape == 'cylinder':
                    cid = p.createCollisionShape(
                        shapeType=p.GEOM_CYLINDER, height=z_size, radius=x_size / 2
                    )
                    vid = p.createVisualShape(
                        shapeType=p.GEOM_CYLINDER,
                        length=z_size,
                        radius=x_size / 2,
                        rgbaColor=color
                    )
                bid = p.createMultiBody(
                    # baseMass=0.01,
                    baseMass=0.0001,
                    baseCollisionShapeIndex=cid,
                    baseVisualShapeIndex=vid,
                    basePosition=[x, y, z],
                    baseOrientation=quat
                )
                # check collision with scene
                collision = False
                for comp_name, comp_id in workspace.component_id_dict.items():
                    contacts = p.getClosestPoints(
                        bid, comp_id, distance=0., physicsClientId=pid
                    )
                    if len(contacts):
                        collision = True
                        break
                for obj_id in obj_ids:
                    contacts = p.getClosestPoints(
                        bid, obj_id, distance=0., physicsClientId=pid
                    )
                    if len(contacts):
                        collision = True
                        break
                if collision:
                    p.removeBody(bid)
                    continue
                if i == num_hiding_objs and num_hiding_objs > 0:
                    # for the target, need to be hide by other objects
                    # Method 1: use camera segmentation to see if the target is unseen
                    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                        width=camera.info['img_size'],
                        height=camera.info['img_size'],
                        viewMatrix=camera.info['view_mat'],
                        projectionMatrix=camera.info['proj_mat']
                    )
                    # cv2.imshow('camera_rgb', rgb_img)
                    depth_img = depth_img / camera.info['factor']
                    far = camera.info['far']
                    near = camera.info['near']
                    depth_img = far * near / (far - (far - near) * depth_img)
                    depth_img[depth_img >= far] = 0.
                    depth_img[depth_img <= near] = 0.
                    seen_obj_ids = set(np.array(seg_img).astype(int).reshape(-1).tolist())
                    if obj_ids[0] in seen_obj_ids:
                        p.removeBody(bid)
                        continue
                    # Method 2: use occlusion

                obj_ids.append(bid)
                pose = np.zeros((4, 4))
                pose[:3, :3] = mRot  # np.eye(3)
                pose[:3, 3] = np.array([x, y, z])
                obj_poses.append(pose)
                obj_pcds.append(pcd)
                obj_shapes.append(obj_shape)
                obj_sizes.append([x_size, y_size, z_size])
                obj_tops.append(ztop)
                obj_colors.append(color)
                break

    return (
        pid,
        scene,
        robot,
        workspace,
        camera,
        obj_poses,
        obj_pcds,
        obj_ids,
        obj_shapes,
        obj_sizes,
        obj_colors,
        obj_poses[0],
        obj_pcds[0],
        obj_ids[0],
        obj_shapes[0],
        obj_sizes[0],
        obj_colors[0],
    )
