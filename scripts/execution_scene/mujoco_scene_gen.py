"""
generate an integrated XML file for loading objects, workspaces, and robots
given provided JSON file describing the workspace.
TODO:
1. save JSON files describing the object information (MAY NOT BE NEEDED)
    - name, pos_ratios, orientations, scales
2. generate batch of problem files (such as XML), in the format of 
    xmls/[scene_name]-prob-[NO].xml
"""
import json
import mjcf.elements as e
import rospkg, os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import transformations as tf

import mujoco
import mujoco_viewer

col_group = 2
vis_group = 1

def include_robot(read_folder, export_folder):
    tree = ET.parse(os.path.join(read_folder,'motoman_vc.xml'))
    root = tree.getroot()
    robot_default = root.find('default')
    robot_default_str = ET.tostring(robot_default)

    robot_default_mjcf = e.Default()

    f = open(os.path.join(export_folder,'robot_default.xml'), 'wb')
    f.write(robot_default_str)
    f.close()

    # loop through the assets
    robot_asset = tree.find('asset')
    robot_asset_str = ET.tostring(robot_asset)
    f = open(os.path.join(export_folder, 'robot_asset.xml'), 'wb')
    f.write(robot_asset_str)
    f.close()


    # add the body
    robot_body = tree.find('worldbody')
    robot_body_str = ET.tostring(robot_body)
    f = open(os.path.join(export_folder,'robot_body.xml'), 'wb')
    f.write(robot_body_str)
    f.close()

    # add the acutuator
    robot_actuator = tree.find('actuator')
    robot_actuator_str = ET.tostring(robot_actuator)
    f = open(os.path.join(export_folder,'robot_actuator.xml'), 'wb')
    f.write(robot_actuator_str)
    f.close()

    # add the contact, tendon and equality
    robot_contact = tree.find('contact')
    robot_contact_str = ET.tostring(robot_contact)
    f = open(os.path.join(export_folder,'robot_contact.xml'), 'wb')
    f.write(robot_contact_str)
    f.close()

    robot_tendon = tree.find('tendon')
    robot_tendon_str = ET.tostring(robot_tendon)
    f = open(os.path.join(export_folder,'robot_tendon.xml'), 'wb')
    f.write(robot_tendon_str)
    f.close()


    robot_equality = tree.find('equality')
    robot_equality_str = ET.tostring(robot_equality)
    f = open(os.path.join(export_folder,'robot_equality.xml'), 'wb')
    f.write(robot_equality_str)
    f.close()



    # robot_default = e.Include(os.path.join(export_folder,'robot_default.xml'))
    # robot_asset = e.Include(os.path.join(export_folder,'robot_asset.xml'))
    # robot_body = e.Include(os.path.join(export_folder,'robot_body.xml'))
    # robot_actuator = e.Include(os.path.join(export_folder,'robot_actuator.xml'))

    robot_default = e.Include('robot_default.xml')
    robot_asset = e.Include('robot_asset.xml')
    robot_body = e.Include('robot_body.xml')
    robot_actuator = e.Include('robot_actuator.xml')
    robot_contact = e.Include('robot_contact.xml')
    robot_tendon = e.Include('robot_tendon.xml')
    robot_equality = e.Include('robot_equality.xml')

    robot_default_mjcf.add_child(robot_default)

    # default.add_child(robot_default_mjcf)
    # asset.add_child(robot_asset)
    # worldbody.add_child(robot_body)
    # actuator.add_child(robot_actuator)

    # model_xml = mujoco.xml()

    # f = open('a_cups.xml', 'w')
    # f.write(model_xml)
    # f.close()
    return robot_default_mjcf, robot_asset, robot_body, robot_actuator, robot_contact, robot_tendon, robot_equality

def add_obj(obj_name, package_path, obj_path, obj_pose, obj_scale):
    idx = obj_name.rfind("_")  # find the last one _ which separates the model name and id
    obj_model_name = obj_name[:idx]
    obj_c = obj_model_name+"/collision_meshes/collision/collision.obj"
    obj_v = obj_model_name+"/meshes/visual/visual.obj"
    # find the texture filename
    files = os.listdir(os.path.join(package_path, obj_path, obj_model_name, 'meshes/visual'))
    for f in files:
        if '.png' in f:
            texture_fname = f
            break
    

    asset_list = []
    if idx > 0:
        obj_t = obj_model_name+'/meshes/visual/'+texture_fname

        col_mesh = e.Mesh(name=obj_model_name+"_col", file=os.path.join(obj_path, obj_c), scale=obj_scale.tolist())
        vis_mesh = e.Mesh(name=obj_model_name+'_vis', file=os.path.join(obj_path, obj_v), scale=obj_scale.tolist())
        vis_text = e.Texture(name=obj_model_name+'_vis', file=os.path.join(obj_path, obj_t), type='2d')
        vis_mat = e.Material(name=obj_model_name+'_vis', texture=obj_model_name+'_vis')
        asset_list = [col_mesh, vis_mesh, vis_text, vis_mat]
    # asset_obj.add_children([col_mesh, vis_mesh, vis_text, vis_mat])
    # col_group = 2
    # vis_group = 1
    pos = obj_pose[:3,3].tolist()
    ori = tf.quaternion_from_matrix(obj_pose)  # w x y z
    ori = ori.tolist()
    obj_body = e.Body(name="object_"+obj_name, pos=pos, quat=ori)  # we use prefix "object_" for convenience
    obj_col_geom = e.Geom(mesh=obj_model_name+"_col", type='mesh', group=col_group)#, contype=0, conaffinity=0)
    obj_vis_geom = e.Geom(mesh=obj_model_name+"_vis", contype=0, conaffinity=0, type='mesh', material=obj_model_name+'_vis', group=vis_group)

    obj_joint = e.Joint(type="free")
    obj_body.add_children([obj_col_geom, obj_vis_geom, obj_joint])

    return asset_list, obj_body



def load_scene(scene_json, scene_name):
    rp = rospkg.RosPack()
    package_path = rp.get_path('pracsys_vbsr')
    read_folder = os.path.join(package_path, 'motoman_model')
    export_folder = os.path.join(package_path, 'xmls')


    obj_path = 'data/models/objects/ocrtoc/'

    # * read in JSON file
    scene_dict = None
    with open(scene_json, 'r') as f:
        scene_dict = json.load(f)
    if scene_dict is None:
        print("Could not read file:", scene_json)
        return

    # * prepare basic & common components
    root = e.Mujoco(model=scene_name)
    compiler = e.Compiler(meshdir=package_path,
                        texturedir=package_path,
                        autolimits=True,
                        angle="radian",
                        inertiafromgeom=True,
                        balanceinertia=True)
    option = e.Option(integrator="RK4")
    warmstart_flag = e.Flag(warmstart="disable")
    option.add_child(warmstart_flag)
    default = e.Default()
    asset = e.Asset()
    worldbody = e.Worldbody()
    actuator = e.Actuator()
    contact = e.Contact()
    tendon = e.Tendon()
    equality = e.Equality()

    # create a default for workspace
    global_geom_default = e.Geom(solref=[.004,1])
    default.add_child(global_geom_default)
    ws_default = e.Default('workspace')
    ws_geom_default = e.Geom(friction=[1, 0.005, 0.0001])
    ws_default.add_child(ws_geom_default)
    default.add_child(ws_default)

    root.add_children([compiler, option, default, asset, 
                       worldbody, actuator, contact, tendon, equality])

    floor_texture = e.Texture(name='grid', type='2d', builtin='checker',
                              width='512', height='512', 
                              rgb1=[.1, .2, .3], rgb2=[.2, .3, .4])
    floor_mat = e.Material(name='grid', texture='grid', texrepeat=[.5, .5],
                           texuniform='true', specular=0, shininess=0,
                           reflectance=0, emission=1)
    asset.add_children([floor_texture, floor_mat])



    floor_geom = e.Geom(name='floor', size=[2,2,.05], type='plane', material='grid', condim=3)
    floor_geom_vis = e.Geom(name='floor_vis', size=[2,2,.05], type='plane', material='grid', contype=0, conaffinity=0,
                        group=vis_group)

    light = e.Light(directional=True, pos=[-.5,.5,3], dir=[0,0,-1], castshadow=False, diffuse=[1,1,1])
    worldbody.add_children([floor_geom, floor_geom_vis, light])

    # * construct world 
    pos = scene_dict['workspace']['pos']
    ori = scene_dict['workspace']['ori']
    ori = np.roll(ori, 1).tolist()  # x,y,z,w -> w,x,y,z
    workspace = e.Body(name='workpace', pos=pos, quat=ori)  # quat: [w,x,y,z]
    worldbody.add_child(workspace)

    components = scene_dict['workspace']['components']
    for component_name, component in components.items():
        shape = np.array(component['shape'])
        mjcf_component = e.Geom(class_='workspace',
                                name=component_name, type='box',
                                pos=component['pose']['pos'],
                                quat=np.roll(component['pose']['ori'],1).tolist(),
                                size=(shape/2).tolist(),
                                rgba=[1., 0.64, 0.0, 1.0])
        mjcf_component_vis = e.Geom(name=component_name+'_vis', type='box',
                                pos=component['pose']['pos'],
                                quat=np.roll(component['pose']['ori'],1).tolist(),
                                size=(shape/2).tolist(),
                                rgba=[1., 0.64, 0.0, 1.0],
                                contype=0, conaffinity=0, group=vis_group)

        workspace.add_children([mjcf_component, mjcf_component_vis])
    
    # * add camera
    body_cam = e.Body(name='body_cam', pos=[0.4, 0, 1.1], xyaxes=[0, -1, 0, 0, 0, 1], mocap=True)
    cam = e.Camera(name='cam', fovy=50)
    geom_cam = e.Geom(name='geom_cam', size=[0.04, 0.04, 0.01], type='box', rgba=[0,0,0,1], contype=2, conaffinity=2)
    body_cam.add_children([cam, geom_cam])  # TODO: add camera from XML file
    worldbody.add_child(body_cam)

    # * add robot
    robot_default_mjcf, robot_asset, robot_body, robot_actuator, \
        robot_contact, robot_tendon, robot_equality = include_robot(read_folder, export_folder)
    root.add_child(robot_default_mjcf)
    asset.add_child(robot_asset)
    worldbody.add_child(robot_body)
    actuator.add_child(robot_actuator)
    contact.add_child(robot_contact)
    tendon.add_child(robot_tendon)
    equality.add_child(robot_equality)

    # * add objects
    ws_low = np.array(scene_dict['workspace']['region_low'])
    ws_high = np.array(scene_dict['workspace']['region_high'])



    model_xml = root.xml()
    fname = os.path.join(export_folder, scene_name+'.xml')
    f = open(fname, 'w')
    f.write(model_xml)
    f.close()


def load_problem(scene_json, scene_name, prob_name, obj_names, obj_pos_ratios, obj_oris, obj_scales=np.ones((3))):
    rp = rospkg.RosPack()
    package_path = rp.get_path('pracsys_vbsr')
    read_folder = os.path.join(package_path, 'motoman_model')
    export_folder = os.path.join(package_path, 'xmls')


    obj_path = 'data/models/objects/ocrtoc/'

    # * read in JSON file
    scene_dict = None
    with open(scene_json, 'r') as f:
        scene_dict = json.load(f)
    if scene_dict is None:
        print("Could not read file:", scene_json)
        return

    # * prepare basic & common components
    root = e.Mujoco(model=scene_name+'-'+prob_name)
    compiler = e.Compiler(meshdir=package_path,
                        texturedir=package_path,
                        autolimits=True,
                        angle="radian",
                        inertiafromgeom=True,
                        balanceinertia=True)
    option = e.Option()
    warmstart_flag = e.Flag(warmstart="disable")
    option.add_child(warmstart_flag)
    default = e.Default()
    asset = e.Asset()
    worldbody = e.Worldbody()
    actuator = e.Actuator()
    contact = e.Contact()
    tendon = e.Tendon()
    equality = e.Equality()

    # create a default for workspace
    global_geom_default = e.Geom(solref=[.004,1])
    default.add_child(global_geom_default)
    ws_default = e.Default('workspace')
    ws_geom_default = e.Geom(friction=[1, 0.005, 0.0001])
    ws_default.add_child(ws_geom_default)
    default.add_child(ws_default)

    root.add_children([compiler, option, default, asset, 
                       worldbody, actuator, contact, tendon, equality])

    floor_texture = e.Texture(name='grid', type='2d', builtin='checker',
                              width='512', height='512', 
                              rgb1=[.1, .2, .3], rgb2=[.2, .3, .4])
    floor_mat = e.Material(name='grid', texture='grid', texrepeat=[.5, .5],
                           texuniform='true', specular=0, shininess=0,
                           reflectance=0, emission=1)
    asset.add_children([floor_texture, floor_mat])



    floor_geom = e.Geom(name='floor', size=[2,2,.05], type='plane', material='grid', condim=3)
    floor_geom_vis = e.Geom(name='floor_vis', size=[2,2,.05], type='plane', material='grid', contype=0, conaffinity=0,
                        group=vis_group)

    light = e.Light(directional=True, pos=[-.5,.5,3], dir=[0,0,-1], castshadow=False, diffuse=[1,1,1])
    worldbody.add_children([floor_geom, floor_geom_vis, light])

    # * construct world 
    pos = scene_dict['workspace']['pos']
    ori = scene_dict['workspace']['ori']
    ori = np.roll(ori, 1).tolist()  # x,y,z,w -> w,x,y,z
    workspace = e.Body(name='workspace', pos=pos, quat=ori)  # quat: [w,x,y,z]
    worldbody.add_child(workspace)

    components = scene_dict['workspace']['components']
    for component_name, component in components.items():
        shape = np.array(component['shape'])
        mjcf_component = e.Geom(class_='workspace',
                                name=component_name, type='box',
                                pos=component['pose']['pos'],
                                quat=np.roll(component['pose']['ori'],1).tolist(),
                                size=(shape/2).tolist(),
                                rgba=[1., 0.64, 0.0, 1.0])
        mjcf_component_vis = e.Geom(name=component_name+'_vis', type='box',
                                pos=component['pose']['pos'],
                                quat=np.roll(component['pose']['ori'],1).tolist(),
                                size=(shape/2).tolist(),
                                rgba=[1., 0.64, 0.0, 1.0],
                                contype=0, conaffinity=0, group=vis_group)

        workspace.add_children([mjcf_component, mjcf_component_vis])
    
    # * add camera
    body_cam = e.Body(name='body_cam', pos=[0.4, 0, 1.1], xyaxes=[0, -1, 0, 0, 0, 1], mocap=True)
    cam = e.Camera(name='cam', fovy=90)
    geom_cam = e.Geom(name='geom_cam', size=[0.04, 0.04, 0.01], type='box', rgba=[0,0,0,1], contype=2, conaffinity=2)
    body_cam.add_children([cam, geom_cam])  # TODO: add camera from XML file
    worldbody.add_child(body_cam)

    # * add robot
    robot_default_mjcf, robot_asset, robot_body, robot_actuator, \
        robot_contact, robot_tendon, robot_equality = include_robot(read_folder, export_folder)
    root.add_child(robot_default_mjcf)
    asset.add_child(robot_asset)
    worldbody.add_child(robot_body)
    actuator.add_child(robot_actuator)
    contact.add_child(robot_contact)
    tendon.add_child(robot_tendon)
    equality.add_child(robot_equality)

    # * add objects
    ws_low = np.array(scene_dict['workspace']['region_low'])
    ws_high = np.array(scene_dict['workspace']['region_high'])

    obj_mjcf_nodes = []
    added_asset_names = []
    for i in range(len(obj_names)):
        obj_name = obj_names[i]
        obj_pos = np.array(obj_pos_ratios[i])
        obj_ori = np.array(obj_oris[i])  # w x y z
        print('ws_high-ws_low: ', ws_high-ws_low)
        print('ws_low: ', ws_low)
        obj_pos = obj_pos * (ws_high-ws_low) + ws_low + np.array(scene_dict['workspace']['pos'])
        # obj_pose[2,3] += ws_low[2]  # offset by the workspace
        obj_pose = tf.quaternion_matrix(obj_ori)
        obj_pose[:3,3] = obj_pos
        
        obj_scale = np.array(obj_scales[i])

        new_assets, new_obj = add_obj(obj_name, package_path, obj_path, obj_pose, obj_scale)        

        if new_assets[0].name not in added_asset_names:
            added_asset_names.append(new_assets[0].name)
            asset.add_children(new_assets)

        worldbody.add_child(new_obj)
        obj_mjcf_nodes.append(new_obj)


    model_xml = root.xml()
    fname = os.path.join(export_folder, scene_name+'-'+prob_name+'.xml')
    f = open(fname, 'w')
    f.write(model_xml)
    f.close()

    # * load in simulator and stabalize the object poses. Then get the final one to export
    model = mujoco.MjModel.from_xml_path(fname)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    obj_names = ['object_'+name for name in obj_names]

    # for _ in range(10000):
    dif_threshold = 1e-3
    rot_threshold = 1 * np.pi / 180
    vel_ang_threshold = 70 * np.pi / 180
    vel_pos_threshold = 2e-2
    acc_ang_threshold = 9000 * np.pi / 180
    acc_pos_threshold = 10
    prev_pos = []
    prev_ori = []

    n_step = 0
    while True:
        # loop until the change of obj_pose becomes small enough
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            n_step += 1
            viewer.render()
        else:
            break
        if n_step % 1000 != 0:
            continue
        stable = True
        for i in range(len(obj_names)):
            pos = np.array(data.body(obj_names[i]).xpos)
            ori = np.array(data.body(obj_names[i]).xquat)  # w x y z
            dofadr = model.body(obj_names[i]).dofadr[0]
            dofnum = model.body(obj_names[i]).dofnum[0]
            vels = np.array(data.qvel)
            accs = np.array(data.qacc)
            dofvel = vels[dofadr:dofadr+dofnum]
            dofacc = accs[dofadr:dofadr+dofnum]
            vel = np.array(data.body(obj_names[i]).cvel)  # COM vel: 6 dim [3d rot, 3d tran]
            acc = np.array(data.body(obj_names[i]).cacc)
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
            print('obj: ', obj_names[i])
            print('new ori: ', ori)
            print('new pos: ', pos)
            print('prev_ori: ', prev_ori[i])
            print('prev_pos: ', prev_pos[i])
            print('diff_pos: ', np.linalg.norm(diff_pos))
            print('ang: ', np.abs(ang)*180/np.pi)
            # print('acc: ', acc)
            print('acc from data directly: ')
            print(data.cacc[data.body(obj_names[i]).id])
            if np.linalg.norm(diff_pos) > dif_threshold or np.abs(ang) > rot_threshold:
                stable = False
                break
            # check velocity
            # print('dofvel: ', dofvel)
            # print('dofacc: ', dofacc)

            print('velocity angle: ')
            print(np.linalg.norm(vel[:3])*180/np.pi)
            print('velocity pos: ')
            print(np.linalg.norm(vel[3:]))
            print('dof velocity angle: ')
            print(np.linalg.norm(dofvel[3:])*180/np.pi)
            print('dof velocity pos: ')
            print(np.linalg.norm(dofvel[:3]))
            print('dof acc angle: ')
            print(np.linalg.norm(dofacc[3:])*180/np.pi)
            print('dof acc pos: ')
            print(np.linalg.norm(dofacc[:3]))

            print('acceleration angle: ')
            print(np.linalg.norm(acc[:3]*180/np.pi))
            print('acceleration pos: ')
            print(np.linalg.norm(acc[3:]))

            if np.linalg.norm(dofvel[3:]) > vel_ang_threshold:
                print('vel_ang_threshold unstable')
                stable = False
                break
            if np.linalg.norm(dofvel[:3]) > vel_pos_threshold:
                print('vel_pos_threshold unstable')
                stable = False
                break
            if np.linalg.norm(dofacc[3:]) > acc_ang_threshold:
                print('acc_ang_threshold unstable')
                stable = False
                break
            if np.linalg.norm(dofacc[:3]) > acc_pos_threshold:
                print('acc_pos_threshold unstable')
                stable = False
                break

        if stable:
            break
        for i in range(len(obj_names)):
            pos = np.array(data.body(obj_names[i]).xpos)
            ori = np.array(data.body(obj_names[i]).xquat)  # w x y z
            prev_pos[i] = np.array(pos)
            prev_ori[i] = np.array(ori)

    print('n_step: ', n_step)
    
    viewer.close()


    # change the poses
    for i in range(len(obj_names)):
        obj_mjcf_nodes[i].pos = prev_pos[i].tolist()
        obj_mjcf_nodes[i].quat = prev_ori[i].tolist()

    # * export
    model_xml = root.xml()
    fname = os.path.join(export_folder, scene_name+'-'+prob_name+'.xml')
    f = open(fname, 'w')
    f.write(model_xml)
    f.close()

if __name__ == '__main__':
    rp = rospkg.RosPack()
    package_path = rp.get_path('pracsys_vbsr')
    scene = os.path.join(package_path, 'data/configs/prob1.json')

    obj_names = ['a_cups_0', "b_cups_0", "bleach_cleanser_0"]
    obj_pos_ratios = [[0.1,0.1,0.2], [0.3,0.3,0.2], [0.4,0.7,0.3]]
    airplane_ori = tf.quaternion_about_axis(90*np.pi/180, [0,0,1])  # w x y z
    obj_oris = [[1,0,0,0],[1,0,0,0], airplane_ori]
    obj_scales = [[1,1,1],[1,1,1],[1,1,1]]

    # load_scene(scene, 'scene1')
    load_problem(scene, 'scene1', 'prob1', obj_names, obj_pos_ratios, obj_oris, obj_scales)