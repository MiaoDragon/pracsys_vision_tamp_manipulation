"""
generate a problem for the object retrieval problem under partial observation

objects are randomly placed on the shelf, and ensure stable placing and collision-free
the target object is hidden by other objects
"""

import os
import re
import sys
import time
import json
import pickle
import random
import xml.etree.ElementTree as ET

import rospkg
import numpy as np
import transformations as tf

import mujoco
import mujoco_viewer
from dm_control import mjcf

import seaborn as sns
import colorcet as cc


def from_color_map(num, size):
    return sns.color_palette(cc.glasbey, size)[num]


def load_problem(scene_json, obj_poses, obj_shapes, obj_sizes):
    scene_dict = None
    with open(scene_json, 'r') as f:
        scene_dict = json.load(f)
    if scene_dict is None:
        print("Could not read file:", scene_json)
        return
    world_model = mjcf.from_xml_string(
        """
    <mujoco model="World">
      <option>
        <flag warmstart="disable" />
      </option>
      <asset>
        <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="0 0 0" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" specular="0" shininess="0" reflectance="0" emission="1" />
      </asset>
      <worldbody>
        <geom name="floor" size="2 2 0.05" type="plane" material="grid" condim="3"/>
        <light directional="true" pos="-0.5 0.5 3" dir="0 0 -1" castshadow="false" diffuse="1 1 1"/>
        <body name="body_cam" pos="0.4 0 1.1" xyaxes="0 -1 0 0 0 1" mocap="true">
          <camera name="cam" fovy="90"/>
          <geom name="geom_cam" size="0.04 0.04 0.01" type="box" rgba="0 0 0 1" contype="2" conaffinity="2"/>
        </body>
        <body name="workspace" pos="0 0 0">
        </body>
      </worldbody>
    </mujoco>
    """
    )
    scene_body = world_model.worldbody.body['workspace']
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
            rgba=[.56, 0.37, 0.29, 1.0],
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

        obj_body = world_model.worldbody.add(
            'body',
            name=f'object_body_{i}',
            pos=[x, y, z],
            quat=quat,
        )
        obj_body.add('freejoint', name=f'joint_{i}')
        obj_body.add(
            'geom',
            name=f'geom_{i}',
            type=obj_shape,
            condim=1,
            size=sizes,
            rgba=color,
            solref="0.004 1",
        )

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


def stabilize(in_xml_name, gui=True):
    '''
    load in simulator and stabalize the object poses
    '''
    # mjcf_model = mjcf.from_path(in_xml_name)
    model = mujoco.MjModel.from_xml_path(in_xml_name)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data) if gui else None

    dif_threshold = 1e-3
    rot_threshold = 1 * np.pi / 180
    prev_pos = []
    prev_ori = []

    n_step = 0
    while True:
        # loop until the change of obj_pose becomes small enough

        if viewer is not None:
            if viewer.is_alive:
                viewer.render()
            else:
                break

        mujoco.mj_step(model, data)
        n_step += 1
        if n_step % 500 != 0:
            continue

        stable = True
        for i in range(model.nbody):
            pos = np.array(data.body(i).xpos)
            ori = np.array(data.body(i).xquat)  # w x y z
            if i >= len(prev_pos):
                prev_pos.append(pos)
                prev_ori.append(ori)
                stable = False
                continue
            diff_pos = pos - prev_pos[i]

            diff_ori = tf.quaternion_matrix(prev_ori[i])
            diff_ori = tf.quaternion_matrix(ori).dot(np.linalg.inv(diff_ori))
            # diff_ori = np.linalg.inv(diff_ori).dot(tf.quaternion_matrix(ori))
            ang, _, _ = tf.rotation_from_matrix(diff_ori)
            print('obj: ', model.body(i).name)
            print('new ori: ', ori)
            print('new pos: ', pos)
            print('prev_ori: ', prev_ori[i])
            print('prev_pos: ', prev_pos[i])
            print('diff_pos: ', np.linalg.norm(diff_pos))
            print('ang: ', np.abs(ang) * 180 / np.pi)
            if np.linalg.norm(diff_pos) > dif_threshold or np.abs(ang) > rot_threshold:
                stable = False
                break

        if stable:
            break

        for i in range(model.nbody):
            print(data.body(i))
            pos = np.array(data.body(i).xpos)
            ori = np.array(data.body(i).xquat)  # w x y z
            prev_pos[i] = np.array(pos)
            prev_ori[i] = np.array(ori)

    print('n_step: ', n_step)

    if viewer is not None:
        viewer.close()

    for i in range(model.nbody):
        model.body(i).pos = data.body(i).xpos
        model.body(i).quat = data.body(i).xquat
    # for i in range(model.ngeom):
    #     model.geom(i).pos = 0

    return model


def updatePath(element, key, in_dir, out_dir):
    chdir = element.get(key)
    if os.path.isabs(chdir):
        chdir = os.path.relpath(chdir, out_dir)
    else:
        chdir = os.path.relpath(in_dir + '/' + chdir, out_dir)
    element.set(key, chdir)


def updateAssetDirs(root, in_dir, out_dir):
    compiler = root.find('compiler')
    for key in ('assetdir', 'meshdir', 'texturedir'):
        if key in compiler.keys():
            updatePath(compiler, key, in_dir, out_dir)


def updateAssets(root, mesh_dir, text_dir, in_dir, out_dir):
    compiler = root.find('compiler')
    m_dir = compiler.get('meshdir', compiler.get('assetdir', './'))
    if os.path.isabs(m_dir):
        m_dir = os.path.relpath(m_dir, in_dir)
    t_dir = compiler.get('texturedir', compiler.get('assetdir', './'))
    if os.path.isabs(t_dir):
        t_dir = os.path.relpath(t_dir, in_dir)
    for asset in root.find('asset'):
        if 'file' in asset.keys():
            if asset.tag == 'mesh':
                asset.set('file', m_dir + '/' + asset.get('file'))
                updatePath(asset, 'file', in_dir, mesh_dir)
            elif asset.tag == 'texture':
                asset.set('file', t_dir + '/' + asset.get('file'))
                updatePath(asset, 'file', in_dir, text_dir)

    if 'assetdir' in compiler.keys():
        compiler.attrib.pop('assetdir')
    compiler.set('meshdir', mesh_dir)
    compiler.set('texturedir', text_dir)


if __name__ == "__main__":

    ## get args ##
    robot_xml = sys.argv[1].strip() if len(sys.argv) > 1 else None
    if robot_xml is None:
        print('Please specify robot xml file.', file=sys.stderr)
        sys.exit(-1)

    scene_desc = sys.argv[2].strip() if len(sys.argv) > 2 else None
    if scene_desc is None:
        print('Please specify json, pkl, or xml file.', file=sys.stderr)
        sys.exit(-1)

    out_filename = sys.argv[3].strip() if len(sys.argv) > 3 else None
    if out_filename is None:
        print('Please specify output filename for generated scene.', file=sys.stderr)
        sys.exit(-1)

    gui = sys.argv[4][0] in ('t', 'T', 'y', 'Y') if len(sys.argv) > 4 else False

    robot_dir = os.path.abspath('/'.join(robot_xml.split('/')[:-1]))
    scene_dir_rel = '/'.join(scene_desc.split('/')[:-1])
    scene_dir = os.path.abspath(scene_dir_rel)
    out_dir = os.path.abspath('/'.join(out_filename.split('/')[:-1]))
    tmp_name = out_dir + '/tmp.mjcf'

    ## load model ##
    if scene_desc.split('.')[-1] == 'pkl':
        with open(scene_desc, 'rb') as f:
            data = pickle.load(f)
            scene_f = data[0]
            obj_poses = data[1]
            obj_pcds = data[2]
            obj_shapes = data[3]
            obj_sizes = data[4]
            # scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_shape, target_obj_size = data
        world_model = load_problem(scene_f, obj_poses, obj_shapes, obj_sizes)
    elif scene_desc.split('.')[-1] == 'json':
        world_model = load_problem(scene_desc, [], [], [])
    elif scene_desc.split('.')[-1] == 'xml':
        root = ET.parse(scene_desc).getroot()
        for child in root:
            for include in child.findall('include'):
                child.remove(include)
        updateAssetDirs(root, scene_dir, out_dir)
        world_model = ET.tostring(root).decode()
    else:
        # TODO generate random scene
        pass

    ## fix scene xml string ###
    if type(world_model) is not str:
        scene_xml_str = world_model.to_xml_string()
        # scene_xml_str = re.sub('-[a-f0-9]+\.', '.', scene_xml_str)
        scene_xml_str = re.sub('\/\/|<default class="\/".+|class="\/"', '', scene_xml_str)
        scene_xml_str = re.sub('<default class="main".+|class="main"', '', scene_xml_str)
    else:
        scene_xml_str = world_model

    ## copy robot.xml to out_filename directory ##
    scene_root = ET.fromstring(scene_xml_str)
    robot_root = ET.parse(robot_xml).getroot()
    comp_scene = scene_root.find('compiler')
    if comp_scene is not None:
        mesh_dir = comp_scene.get('meshdir', comp_scene.get('assetdir', scene_dir_rel))
        text_dir = comp_scene.get('texturedir', comp_scene.get('assetdir', scene_dir_rel))
    else:
        mesh_dir = scene_dir_rel
        text_dir = scene_dir_rel
    print(mesh_dir)
    print(text_dir)
    print(robot_dir)
    print(out_dir)
    updateAssets(robot_root, mesh_dir, text_dir, robot_dir, out_dir)
    robot_fname = robot_root.get('model', f'temp_robot_{scene_desc.split("/")[-1]}')
    robot_fname = out_dir + '/' + robot_fname + '.xml'
    with open(robot_fname, 'wb') as f:
        f.write(ET.tostring(robot_root))

    ## write unstabilized model to xml ##
    with open(tmp_name, 'w') as f:
        f.write(scene_xml_str)

    ## stabilize model ##
    model = stabilize(tmp_name, gui)

    ## save file ##
    mujoco.mj_saveLastXML(tmp_name, model)

    ## include robot.xml in temp xml file ##
    rel_path = os.path.relpath(robot_fname, out_dir)
    include = ET.fromstring(f'<include file="{rel_path}"/>')
    scene_root = ET.parse(tmp_name).getroot()
    scene_root.insert(0, include)
    scene_xml_str = ET.tostring(scene_root).decode()
    with open(tmp_name, 'wb') as f:
        f.write(ET.tostring(scene_root))

    ## combine robot and temp xml into out_filename xml ##
    model = mujoco.MjModel.from_xml_path(tmp_name)
    mujoco.mj_saveLastXML(out_filename, model)

    ## clean up files ##
    os.remove(tmp_name)
    os.remove(robot_fname)
