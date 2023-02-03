"""
simumlated execution scene. Some of it overlaps with the sim_scene defined in scene folder.
"""
import os
import re
import sys
import json
import glob
import pickle
import numpy as np
import transformations as tf

import rospy
import rospkg
from cv_bridge import CvBridge
from moveit_commander.conversions import *
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle

import mujoco_viewer
from dm_control import mujoco
import problem_generation as prob_gen

from pracsys_vision_tamp_manipulation.msg import ObjectGroundTruthState, RobotState, PercievedObject
from pracsys_vision_tamp_manipulation.srv import ExecuteTrajectory, ExecuteTrajectoryResponse, AttachObject, AttachObjectResponse


class ExecutionSystem():

    def __init__(self, robot_xml, scene=None, trial=None, gui=True):
        self.robot_model_name = 'sda10f'
        self.joint_names = [
            "torso_joint_b1",
            "arm_left_joint_1_s",
            "arm_left_joint_2_l",
            "arm_left_joint_3_e",
            "arm_left_joint_4_u",
            "arm_left_joint_5_r",
            "arm_left_joint_6_b",
            "arm_left_joint_7_t",
            "arm_right_joint_1_s",
            "arm_right_joint_2_l",
            "arm_right_joint_3_e",
            "arm_right_joint_4_u",
            "arm_right_joint_5_r",
            "arm_right_joint_6_b",
            "arm_right_joint_7_t",
        ]
        self.mj_joint_names = [
            self.robot_model_name + '/' + jn for jn in self.joint_names
        ]
        if trial is not None:
            with open(trial, 'rb') as f:
                data = pickle.load(f)
                scene_f = data[0]
                obj_poses = data[1]
                obj_pcds = data[2]
                obj_shapes = data[3]
                obj_sizes = data[4]
                # scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_shape, target_obj_size = data
            world_model = prob_gen.load_problem(
                scene_f, robot_xml, obj_poses, obj_shapes, obj_sizes
            )

        else:
            scene_f = scene

        ASSETS = dict()
        assets_dir = '/'.join(robot_xml.split('/')[:-1]) + '/meshes/'
        for fname in glob.glob(assets_dir + '*.stl'):
            with open(fname, 'rb') as f:
                ASSETS[fname] = f.read()
        fixed_xml_str = re.sub('-[a-f0-9]+.stl', '.stl', world_model.to_xml_string())
        self.physics = mujoco.Physics.from_xml_string(fixed_xml_str, ASSETS)
        self.model = self.physics.model._model
        self.data = self.physics.data._data
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data) if gui else None

        # self.camera = camera
        self.bridge = CvBridge()

        # * initialize ROS services
        # - robot trajectory tracker
        rospy.Service("execute_trajectory", ExecuteTrajectory, self.execute_trajectory)
        rospy.Service("attach_object", AttachObject, self.attach_object)

        # * initialize ROS pubs and subs
        # - camera
        # - robot_state_publisher
        # self.rgb_img, self.depth_img, self.seg_img = self.camera.sense()

        self.rgb_cam_pub = rospy.Publisher('rgb_image', Image, queue_size=5)
        self.depth_cam_pub = rospy.Publisher('depth_image', Image, queue_size=5)
        self.seg_cam_pub = rospy.Publisher('seg_image', Image, queue_size=5)
        self.js_pub = rospy.Publisher('joint_state', JointState, queue_size=5)
        self.gt_obj_pub = rospy.Publisher(
            'object_ground_truth_state', ObjectGroundTruthState, queue_size=5
        )

        self.obj_pub = rospy.Publisher('object_state', PercievedObject, queue_size=5)
        self.rs_pub = rospy.Publisher('robot_state_publisher', RobotState, queue_size=5)

        init_joints = [0]
        init_joints += [1.75, 0.8, 0.0, -0.66, 0.0, 0.0, 0.0]  # left
        init_joints += [1.75, 0.8, 0.0, -0.66, 0.0, 0.0, 0.0]  # right
        self.data.qpos[self.get_qpos_indices(self.mj_joint_names)] = init_joints
        self.data.ctrl[self.get_ctrl_indices(self.mj_joint_names)] = init_joints

    def get_objq_indices(self, obj_name):
        jnt = self.model.joint(self.model.body(obj_name).jntadr[0])
        qpos_inds = np.array(range(jnt.qposadr[0], jnt.qposadr[0] + len(jnt.qpos0)))
        return qpos_inds

    def get_qpos_indices(self, joints):
        qpos_inds = np.array([self.model.joint(j).qposadr[0] for j in joints])
        return qpos_inds

    def get_qvel_indices(self, joints):
        qvel_inds = np.array([self.model.joint(j).dofadr[0] for j in joints])
        return qvel_inds

    def get_ctrl_indices(self, joints):
        ctrl_inds = np.array(
            [self.model.actuator(j.replace('joint_', '')).id for j in joints]
        )
        return ctrl_inds

    def colliding_body_pairs(self):
        pairs = [
            (
                self.model.body(self.model.geom(c.geom1).bodyid[0]).name,
                self.model.body(self.model.geom(c.geom2).bodyid[0]).name
            ) for c in self.data.contact
        ]
        return pairs

    def step(self):
        # mujoco.mj_step(self.model, self.data)
        self.physics.step()
        if self.viewer is not None and self.viewer.is_alive:
            self.viewer.render()

    def execute_trajectory(self, req):
        traj = req.trajectory  # sensor_msgs/JointTrajectory
        points = traj.points

        step_sz = 1 * np.pi / 180
        interpolated_pts = [points[0].positions]
        for i in range(len(points) - 1):
            pos1 = np.array(points[i].positions)
            pos2 = np.array(points[i + 1].positions)
            abs_change = np.abs(pos2 - pos1)
            n_steps = int(np.ceil(abs_change.max() / step_sz))
            interpolated_pts += np.linspace(pos1, pos2, n_steps + 1)[1:].tolist()

        epsilon = 2**-5
        joint_names = [self.robot_model_name + '/' + name for name in traj.joint_names]
        print(joint_names, self.mj_joint_names)
        print(joint_names == self.mj_joint_names)
        iqpos = self.get_qpos_indices(joint_names)
        ctrls = self.get_ctrl_indices(joint_names)
        print(ctrls)
        print(len(joint_names), len(self.mj_joint_names), len(ctrls))
        num_collision = 0
        for i in range(len(interpolated_pts)):
            pos = interpolated_pts[i]
            self.data.ctrl[ctrls] = pos  # position control
            while np.linalg.norm(self.data.qpos[iqpos] - pos) > epsilon:
                print(np.linalg.norm(self.data.qpos[iqpos] - pos) > epsilon)
                # self.step()
                # rospy.sleep(0.01)
                if len(self.data.contact) > 1:
                    num_collision += 1
                    print("collision!:", self.colliding_body_pairs())

        # self.rgb_img, self.depth_img, self.seg_img = self.camera.sense()
        return ExecuteTrajectoryResponse(num_collision, True)

    def attach_object(self, req):
        """
        close or open the gripper
        """
        if req.attach == True:
            # gripper is index 1 suction is 0
            self.data.ctrl[1] = 255
        else:
            self.data.ctrl[1] = 0

        return AttachObjectResponse(True)

    def publish_image(self):
        """
        obtain image from mujoco and publish
        """

        self.rgb_img = self.physics.render(camera_id=0)
        self.depth_img = self.physics.render(camera_id=0, depth=True)
        seg_img = self.physics.render(camera_id=0, segmentation=True)
        self.seg_img = seg_img[:, :, 0]

        msg = self.bridge.cv2_to_imgmsg(self.rgb_img, 'passthrough')
        msg.header.stamp = rospy.Time.now()
        self.rgb_cam_pub.publish(msg)

        msg = self.bridge.cv2_to_imgmsg(self.depth_img, 'passthrough')
        msg.header.stamp = rospy.Time.now()
        self.depth_cam_pub.publish(msg)

        msg = self.bridge.cv2_to_imgmsg(self.seg_img, 'passthrough')
        msg.header.stamp = rospy.Time.now()
        self.seg_cam_pub.publish(msg)

    def publish_robot_state(self):
        """
        obtain joint state from Mujoco and publish
        """

        iqpos = self.get_qpos_indices(self.mj_joint_names)
        iqvel = self.get_qvel_indices(self.mj_joint_names)
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.data.qpos[iqpos]
        msg.velocity = self.data.qvel[iqvel]
        msg.header.stamp = rospy.Time.now()
        # self.js_pub.publish(msg)

        rs_msg = RobotState()
        rs_msg.attached_obj = -1
        rs_msg.joint_state = msg
        rs_msg.header.stamp = rospy.Time.now()
        self.rs_pub.publish(rs_msg)

    def publish_ground_truth_state(self):
        names = []
        poses = []
        for i in range(self.model.njnt):
            jnt = self.model.jnt(i)
            name = self.model.jnt(i).name
            if name[:5] == 'joint':
                # get qpos indices for joint
                iqpos = np.array(range(jnt.qposadr[0], jnt.qposadr[0] + len(jnt.qpos0)))
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

        msg = ObjectGroundTruthState()
        msg.header.stamp = rospy.Time.now()
        msg.id = names
        msg.pose = poses
        self.gt_obj_pub.publish(msg)

    def publish_objects(self):
        for i in range(self.model.njnt):
            jnt = self.model.jnt(i)
            name = self.model.jnt(i).name
            if name[:5] == 'joint':
                obj_msg = self.obj2msg(i)
                self.obj_pub.publish(obj_msg)

    def obj2msg(self, obj_id):
        obj_msg = PercievedObject()
        obj_msg.header.frame_id = 'world'
        obj_msg.header.stamp = rospy.get_rostime()
        obj_msg.name = f'{obj_id}'

        jnt = self.model.jnt(obj_id)
        iqpos = np.array(range(jnt.qposadr[0], jnt.qposadr[0] + len(jnt.qpos0)))
        pos_quat = self.data.qpos[iqpos]
        obj_msg.pose = Pose()
        obj_msg.pose.position.x = pos_quat[0]
        obj_msg.pose.position.y = pos_quat[1]
        obj_msg.pose.position.z = pos_quat[2]
        obj_msg.pose.orientation.w = pos_quat[3]
        obj_msg.pose.orientation.x = pos_quat[4]
        obj_msg.pose.orientation.y = pos_quat[5]
        obj_msg.pose.orientation.z = pos_quat[6]
        # obj_msg.mesh = Mesh()
        obj_msg.solid = SolidPrimitive()
        obj_msg.solid.dimensions = [0]

        geom = self.model.geom(self.model.body(jnt.bodyid[0]).geomadr[0])
        shape = geom.type[0]
        size = geom.size

        if shape == mujoco.mjtGeom.mjGEOM_MESH:
            print(
                "Element %s with geometry type %s not supported. Ignored." %
                (object_id, shape)
            )
            return None
        else:
            SCALE = 1.0
            obj_msg.type = PercievedObject.SOLID_PRIMITIVE
            if shape == mujoco.mjtGeom.mjGEOM_BOX:
                obj_msg.solid.type = SolidPrimitive.BOX
                obj_msg.solid.dimensions = np.multiply(SCALE, 2 * size).tolist()
            elif shape == mujoco.mjtGeom.mjGEOM_CYLINDER:
                obj_msg.solid.type = SolidPrimitive.CYLINDER
                obj_msg.solid.dimensions = np.multiply(SCALE, 2 * size).tolist()
            elif shape == mujoco.mjtGeom.mjGEOM_SPHERE:
                obj_msg.solid.type = SolidPrimitive.SPHERE
                obj_msg.solid.dimensions = np.multiply(SCALE, 2 * size).tolist()
            elif shape == mujoco.mjtGeom.mjGEOM_CAPSULE:
                print(
                    "Element %s with geometry type %s not supported. Ignored." %
                    (object_id, shape)
                )
                return None
            elif shape == mujoco.mjtGeom.mjGEOM_PLANE:
                print(
                    "Element %s with geometry type %s not supported. Ignored." %
                    (object_id, shape)
                )
                return None

        return obj_msg

    def publish_all(self, timer_event):
        self.publish_image()
        self.publish_objects()
        self.publish_robot_state()

    def run(self):
        """
        keep spinning and publishing to the ROS topics
        """

        timer = rospy.Timer(rospy.Duration(0.1), self.publish_all)
        while not rospy.is_shutdown():
            self.step()
        timer.shutdown()


if __name__ == "__main__":
    rospy.init_node("execution_system")
    rospy.on_shutdown(lambda: os.system('pkill -9 -f sim_scene'))
    # rospy.sleep(1.0)
    robot_xml = sys.argv[1].strip() if len(sys.argv) > 1 else None
    if robot_xml is None:
        print('Please specify robot xml file.', file=sys.stderr)
        sys.exit(-1)
    scene_or_trial = sys.argv[2].strip() if len(sys.argv) > 2 else None
    gui = sys.argv[3][0] in ('t', 'T', 'y', 'Y') if len(sys.argv) > 3 else False
    trial = scene_or_trial if scene_or_trial.split('.')[-1] == 'pkl' else None
    scene = scene_or_trial if scene_or_trial.split('.')[-1] == 'json' else None
    if scene is None and trial is None:
        print('Please specify json or pkl file.', file=sys.stderr)
        sys.exit(-1)
    execution_system = ExecutionSystem(
        robot_xml,
        scene,
        trial,
        gui,
    )
    execution_system.run()
