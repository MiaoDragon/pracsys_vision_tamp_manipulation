"""
simulated execution node using Mujoco.

arguments:
- scene name (MJCF file)
-----------------------------------------------------
-----------------provided interface------------------
-----------------------------------------------------
ROS Service:
- execute_trajectory
    input:
        geometric paths containing a list of joint positions and names
    function:
        conduct time parameterization to compute the velocity and time
        format the result as a JointTrajectory msg
        pass the JointTrajectory containing position, velocity, and time to the lower-level tracker
- ee_control
    input:
        control
- (TODO) reset_scene
-----------------------------------------------------
ROS message:
- RGBD images
- seg image (fake perception)
-----------------------------------------------------
"""

import os
import sys
import json
import glfw
import pickle
import numpy as np
import time

import mujoco
import mujoco_viewer

import cv2
import rospy
import rospkg
import tf2_ros
import actionlib
import transformations as tf
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, Pose
from sensor_msgs.msg import Image, JointState, CameraInfo
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from rosgraph_msgs.msg import Clock
from pracsys_vision_tamp_manipulation.msg import ObjectGroundTruthState
from pracsys_vision_tamp_manipulation.srv import FakeObjectControl, FakeObjectControlResponse


class ExecutionNode():
    joint_names = []
    init_joints = []

    def __init__(self, xml_file, gui=True):
        """
        xml_file: mjcf format description of the scene
        Required field:
        # - joint_names: to be defined for specific robots.
        # - ee_names: to be defined for specific end effectors.
        """
        if os.path.exists(xml_file):
            prob_path = xml_file
        else:
            rp = rospkg.RosPack()
            package_path = rp.get_path('pracsys_vision_tamp_manipulation')
            prob_path = os.path.join(package_path, 'xmls', xml_file)
        self.ws_xml_path = os.path.abspath(prob_path)
        self.load_ros_param()

        # * set up mujoco
        self.model = mujoco.MjModel.from_xml_path(prob_path)
        self.data = mujoco.MjData(self.model)
        self.gui = gui
        self.set_viewer()
        self.renderer = mujoco.Renderer(self.model, 480, 640)

        # * for tracking arm and control ee asynchronously
        self.arm_trajectory = []
        self.ee_trajectory = []

        self.cam_intrinsics = None
        self.cam_extrinsics = None
        self.joint_limit = self.get_joint_limit(self.joint_names)
        # self.set_joint_angle(self.init_joints)

        # * set up ROS interfaces
        # ** services
        rospy.Service(
            "/ground_truth/fake_obj_control", FakeObjectControl,
            self.fake_obj_control
        )

        # ** messages
        self.bridge = CvBridge()
        self.rgb_cam_pub = rospy.Publisher('/rgb_image', Image, queue_size=5)
        self.depth_cam_pub = rospy.Publisher(
            '/depth_image', Image, queue_size=5
        )
        self.seg_cam_pub = rospy.Publisher(
            '/ground_truth/seg_image', Image, queue_size=5
        )
        self.cam_info_pub = rospy.Publisher(
            '/camera_info', CameraInfo, queue_size=5
        )
        self.gt_obj_pub = rospy.Publisher(
            '/ground_truth/object_state', ObjectGroundTruthState, queue_size=5
        )
        # this only publishes the object poses that are visible from the segmentation image

        # ** TF
        self.br = tf2_ros.TransformBroadcaster()

        # ** clock
        self.clock_pub = rospy.Publisher('/clock', Clock, queue_size=5)

        self.set_ros_param()

    def get_objq_indices(self, obj_name):
        jnt = self.model.joint(self.model.body(obj_name).jntadr[0])
        qpos_inds = np.array(
            range(jnt.qposadr[0], jnt.qposadr[0] + len(jnt.qpos0))
        )
        return qpos_inds

    def get_qpos_indices(self, joints):
        qpos_inds = np.array([self.model.joint(j).qposadr[0] for j in joints])
        return qpos_inds

    def get_qvel_indices(self, joints):
        qvel_inds = np.array([self.model.joint(j).dofadr[0] for j in joints])
        return qvel_inds

    def get_ctrl_indices(self, joints, prefix='', replace=''):
        ctrl_name = lambda j: prefix + j.replace('_joint', replace)
        ctrl_inds = [self.model.actuator(ctrl_name(j)).id for j in joints]
        return np.array(ctrl_inds)

    def get_act_indices(self, joints, prefix='', replace=''):
        act_name = lambda j: prefix + j.replace('_joint', replace)
        act_inds = [self.model.actuator(act_name(j)).actadr[0] for j in joints]
        return np.array(act_inds)

    def get_jnt_indices(self, joints):
        jnt_inds = np.array([self.model.joint(j).id for j in joints])
        return jnt_inds

    def load_ros_param(self):
        """
        load from ros param:
        - viewer-related info
        """
        # * set viewer
        viewer_lookat = rospy.get_param(
            'execution_scene/viewer/look_at', [1.07, 0, 1.0]
        )
        viewer_distance = rospy.get_param(
            'execution_scene/viewer/distance', 3.55
        )
        viewer_elevation = rospy.get_param(
            'execution_scene/viewer/elevation', -30.0
        )
        viewer_azimuth = rospy.get_param('execution_scene/viewer/azimuth', 0)
        self.viewer_lookat = np.array(viewer_lookat)
        self.viewer_distance = viewer_distance
        self.viewer_elevation = viewer_elevation
        self.viewer_azimuth = viewer_azimuth

    def set_ros_param(self):
        """
        set the ROS parameter for important static values
        """
        rospy.set_param('execution/joint_names', self.joint_names)
        rospy.set_param('execution/joint_limits', self.joint_limit.tolist())
        rospy.set_param('use_sim_time', True)
        rospy.set_param('ws_xml_path', self.ws_xml_path)

    def set_viewer(self):
        if self.gui:
            self.viewer = mujoco_viewer.MujocoViewer(
                self.model,
                self.data,
                # mode="offscreen"
            )
            self.viewer.cam.lookat = self.viewer_lookat
            self.viewer.cam.elevation = self.viewer_elevation
            self.viewer.cam.distance = self.viewer_distance
            self.viewer.cam.azimuth = self.viewer_azimuth

    def get_joint_state(self, joint_names):
        iqpos = self.get_qpos_indices(joint_names)
        iqvel = self.get_qvel_indices(joint_names)
        position = self.data.qpos[iqpos]
        velocity = self.data.qvel[iqvel]
        return position, velocity

    def get_joint_limit(self, joint_names):
        ijidx = self.get_jnt_indices(joint_names)
        return np.array(self.model.jnt_range[ijidx, :])

    def set_joint_angle(self, joints):
        if type(joints) == dict:
            joint_names = list(joints.keys())
            joint_vals = np.array(list(joints.values()))
        else:
            joint_names = self.joint_names
            joint_vals = np.array(joints)
        iqpos = self.get_qpos_indices(joint_names)
        self.data.qpos[iqpos] = joint_vals
        pctrl = self.get_ctrl_indices(self.joint_names)
        # vctrl = self.get_ctrl_indices(self.joint_names, replace='_v')
        intvctrl = self.get_ctrl_indices(self.joint_names, replace='_intv')
        intact = self.get_act_indices(self.joint_names, replace='_intv')

        self.data.ctrl[pctrl] = joint_vals
        # self.data.ctrl[vctrl] = 0
        self.data.act[intact] = joint_vals
        self.data.ctrl[intvctrl] = 0
        print('setting joint angle...')

        mujoco.mj_forward(self.model, self.data)

    def publish_camera_info(self, camera_id):
        """
        publish the camera intrinsics matrix and extrinsics tf
        ref: https://github.com/deepmind/dm_control/blob/774f46182140106e22725914aad3c6299ed91edd/dm_control/mujoco/engine.py#L41
        """
        if self.cam_intrinsics is None:
            # compute intrinsics and extrinsics
            pos = self.data.cam_xpos[camera_id]
            rot = self.data.cam_xmat[camera_id].reshape(3, 3)
            fov = self.model.cam_fovy[camera_id]

            # * notice that the mujoco camera pose is pointing in the opposite direction of the look-at
            # to match it into the right-hand coordinate system, and to match the depth value in point cloud
            # we will change the sign of z axis and y axis (up vector)
            rot[:, 1] = -rot[:, 1]
            rot[:, 2] = -rot[:, 2]

            # Translation matrix (4x4).
            transform = np.eye(4)
            transform[0:3, 3] = pos
            # Rotation matrix (4x4).
            transform[0:3, 0:3] = rot
            # Focal transformation matrix (3x4).
            # focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * self.renderer.height / 2.0
            # focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

            # Translation matrix (4x4).
            translation = np.eye(4)
            # translation[0:3, 3] = -pos
            # Rotation matrix (4x4).
            rotation = np.eye(4)
            # rotation[0:3, 0:3] = rot
            # Focal transformation matrix (3x4).
            focal_scaling = (
                1. / np.tan(np.deg2rad(fov) / 2.)
            ) * self.renderer.height / 2.0
            focal = np.diag([focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
            # NOTE: issue in rendering: by default the fx is negative
            # mujoco doesn't seem to support having different focal length for x and y axes

            # Image matrix (3x3).
            image = np.eye(3)
            image[0, 2] = (self.renderer.width - 1) / 2.0
            image[1, 2] = (self.renderer.height - 1) / 2.0
            cam_intrinsics = image @ focal @ rotation @ translation  # 3x4

            print('extrinsics: ', transform)
            print('intrinsics: ', cam_intrinsics)
            self.cam_intrinsics = cam_intrinsics
            self.cam_extrinsics = transform

        msg = CameraInfo()
        msg.header.frame_id = 'camera_link'
        msg.height = self.renderer.height
        msg.width = self.renderer.width
        msg.P = self.cam_intrinsics.flatten()
        self.cam_info_pub.publish(msg)

        # publish camera in world
        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base"
        t.child_frame_id = 'camera_link'
        t.transform.translation.x = self.cam_extrinsics[0, 3]
        t.transform.translation.y = self.cam_extrinsics[1, 3]
        t.transform.translation.z = self.cam_extrinsics[2, 3]
        q = tf.transformations.quaternion_from_matrix(self.cam_extrinsics)
        t.transform.rotation.w = q[0]
        t.transform.rotation.x = q[1]
        t.transform.rotation.y = q[2]
        t.transform.rotation.z = q[3]

        self.br.sendTransform(t)

    def get_rgb_img(self):
        self.renderer.update_scene(self.data, camera=0)
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        rgb_img = self.renderer.render()
        rgb_img = np.array(rgb_img)
        return rgb_img

    def get_depth_img(self):
        self.renderer.update_scene(self.data, camera=0)
        self.renderer.enable_depth_rendering()
        depth_img = self.renderer.render()
        depth_img = np.array(depth_img)
        far = 1.2
        depth_img[depth_img >= far] = far
        return depth_img

    def get_seg_img(self):
        self.renderer.update_scene(self.data, camera=0)
        self.renderer.enable_segmentation_rendering()
        seg_img = self.renderer.render()
        # Display the contents of the first channel, which contains object
        # IDs. The second channel, seg[:, :, 1], contains object types.
        geom_ids = seg_img[:, :, 0]
        new_geom_ids = np.array(geom_ids)
        # print out the root id
        for gid in list(set(geom_ids.flatten().tolist())):
            if gid != -1:
                body_id = self.model.body(self.model.geom(gid).bodyid).rootid[0]
                # print('segmented object: ', self.model.body(body_id).name, gid, body_id)

                # if the object is workspace or robot, change the segid to -1
                if self.model.body(body_id).name in ['motoman_base']:
                    new_geom_ids[geom_ids == gid] = -1
                elif self.model.body(body_id).name in ['world', 'workspace']:
                    new_geom_ids[geom_ids == gid] = -2
                else:
                    # unify for each object
                    new_geom_ids[geom_ids == gid] = body_id
            else:
                new_geom_ids[geom_ids == gid] = -2  # simulation background
        geom_ids = new_geom_ids
        return geom_ids

    def publish_image(self):
        """
        obtain image from mujoco and publish
        ref: https://github.com/deepmind/mujoco/blob/main/python/tutorial.ipynb
        """
        #TODO: get image from mujoco sensor
        # maxwidth = self.model.vis.global_.offwidth
        # maxheight = self.model.vis.global_.offheight
        # This could be specified in XML, '<visual> <global offwidth="..."/> </visual>
        rgb_img = self.get_rgb_img()
        depth_img = self.get_depth_img()
        geom_ids = self.get_seg_img()

        # swap red and blue
        b = np.array(rgb_img[:, :, 0])
        r = np.array(rgb_img[:, :, 2])
        rgb_img[:, :, 0] = r
        rgb_img[:, :, 2] = b

        if self.gui:
            # Infinity is mapped to -1
            geom_ids_vis = geom_ids.astype(int) + 1
            import seaborn as sns
            palette = sns.color_palette(None, geom_ids_vis.max() + 10)
            palette = np.array(palette)
            # print(palette.shape)
            pixels = palette[geom_ids_vis, :]

            cv2.imshow("rgb_img", rgb_img)
            cv2.imshow("depth_img", depth_img)
            cv2.imshow("seg_img", pixels)

            cv2.waitKey(1)

        msg = self.bridge.cv2_to_imgmsg(rgb_img, 'passthrough')
        msg.header.frame_id = 'camera_link'
        msg.header.stamp = rospy.Time.now()
        self.rgb_cam_pub.publish(msg)

        msg = self.bridge.cv2_to_imgmsg(depth_img, 'passthrough')
        msg.header.frame_id = 'camera_link'
        msg.header.stamp = rospy.Time.now()
        self.depth_cam_pub.publish(msg)

        msg = self.bridge.cv2_to_imgmsg(geom_ids, 'passthrough')
        msg.header.frame_id = 'camera_link'
        msg.header.stamp = rospy.Time.now()
        self.seg_cam_pub.publish(msg)

    def publish_joint_state(self):
        pass

    def publish_ground_truth_state(self):
        names = []
        poses = []
        ids = []
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name[:7] == 'object_':
                # get qpos indices for joint
                iqpos = self.get_objq_indices(name)
                pos_quat = self.data.qpos[iqpos]
                pose = Pose()
                pose.position.x = pos_quat[0]
                pose.position.y = pos_quat[1]
                pose.position.z = pos_quat[2]
                pose.orientation.w = pos_quat[3]
                pose.orientation.x = pos_quat[4]
                pose.orientation.y = pos_quat[5]
                pose.orientation.z = pos_quat[6]
                names.append(name)
                poses.append(pose)
                ids.append(i)

        msg = ObjectGroundTruthState()
        msg.header.stamp = rospy.Time.now()
        msg.name = names
        msg.pose = poses
        msg.id = ids
        self.gt_obj_pub.publish(msg)

    def fake_obj_control(self, req):
        """
        *fake perception for unit testing*
        directly apply force:torque to the object through data.xfrc_applied
        for a set time
        """
        obj_id = req.obj_id
        duration = req.duration
        force = [req.wrench.force.x, req.wrench.force.y, req.wrench.force.z]
        torque = [req.wrench.torque.x, req.wrench.torque.y, req.wrench.torque.z]
        applied_wrench = force + torque
        applied_wrench = np.array(applied_wrench)
        start_time = time.time()
        while time.time() - start_time < duration:
            self.data.xfrc_applied[obj_id, :] = applied_wrench
        return FakeObjectControlResponse()

    def publish_clock(self):
        msg = Clock()
        msg.clock = rospy.Time.from_sec(self.data.time)
        self.clock_pub.publish(msg)

    def publish_all(self, clock_param):
        self.publish_image()
        self.publish_joint_state()
        self.publish_camera_info(0)
        self.publish_ground_truth_state()

    def step(self, render=True):
        self.do_traj()
        mujoco.mj_step(self.model, self.data)
        if render and self.gui and self.viewer is not None and self.viewer.is_alive:
            glfw.make_context_current(self.viewer.window)
            self.viewer.render()

    def do_traj(self):
        """
        track the trajectory if there are ones inside the saved list
        """
        # * controlling arm
        if len(self.arm_trajectory) == 0:
            iqpos = self.get_qpos_indices(self.joint_names)
            # pctrl = self.get_ctrl_indices(self.joint_names)
            # vctrl = self.get_ctrl_indices(self.joint_names, replace='_v')
            intact = self.get_act_indices(self.joint_names, replace='_intv')
            intvctrl = self.get_ctrl_indices(self.joint_names, replace='_intv')
            # self.data.ctrl[pctrl] = self.data.qpos[iqpos]
            # self.data.ctrl[vctrl] = 0
            self.data.ctrl[intvctrl] = 0
        else:
            joint_names, position, velocity, time = self.arm_trajectory.pop(0)
            iqpos = self.get_qpos_indices(joint_names)
            pctrl = self.get_ctrl_indices(joint_names)
            # vctrl = self.get_ctrl_indices(joint_names, replace='_v')
            intact = self.get_act_indices(joint_names, replace='_intv')
            intvctrl = self.get_ctrl_indices(joint_names, replace='_intv')
            self.data.ctrl[pctrl] = position
            # self.data.ctrl[vctrl] = velocity
            self.data.act[intact] = position
            self.data.ctrl[intvctrl] = velocity

        # * controlling end effector
        if len(self.ee_trajectory) > 0:
            joint_names, position = self.ee_trajectory.pop(0)
            ctrl = self.get_ctrl_indices(joint_names)
            self.data.ctrl[ctrl] = position

    def run(self):
        """
        keep spinning and publishing to the ROS topics
        """
        timer = rospy.Timer(rospy.Duration(0.1), self.publish_all)
        render_fps = 20  # number of frames every simulation time
        # NOTE: the simulation time may not match the real time.
        render_step_freq = int(1 / render_fps / self.model.opt.timestep)
        mujoco.mj_step1(self.model, self.data)
        render_step = 0
        while not rospy.is_shutdown():
            if render_step % render_step_freq == 0:
                self.step(render=True)
                render_step = 0
            else:
                self.step(render=False)
            self.publish_clock()
            render_step += 1


if __name__ == "__main__":
    rospy.init_node("execution_node")
    # rospy.on_shutdown(lambda: os.system('pkill -9 -f execution_node'))
    # rospy.sleep(1.0)
    if len(sys.argv) == 3:
        scene_name = sys.argv[1].strip()
        prob_name = sys.argv[2].strip()
    else:
        scene_name = sys.argv[1].strip()
        prob_name = None

    scene_xml = sys.argv[1].strip() if len(sys.argv) > 1 else None
    gui = sys.argv[2][0] in ('t', 'T', 'y', 'Y') if len(sys.argv) > 2 else False

    execution_node = ExecutionNode(scene_xml, gui)
    execution_node.run()
