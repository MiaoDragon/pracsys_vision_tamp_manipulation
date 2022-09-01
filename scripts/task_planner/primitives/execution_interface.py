"""
interface from planning system to execution system
"""
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pracsys_vision_tamp_manipulation.msg import RobotState, PercievedObject
# from pracsys_vision_tamp_manipulation.srv import ExecuteTrajectory, AttachObject

from uniform_object_rearrangement.srv import ExecuteTrajectory, AttachObject, ExecuteTrajectoryRequest, AttachObjectRequest


import message_filters

import time
import rospy
import numpy as np
import pybullet as p
import transformations as tf
from threading import Thread, Lock

from utils.visual_utils import *
from perception.object_belief import ObjectBelief


class ExecutionInterface():
    """
    this handles in/out interface to communicate with the execution scene
    """

    def __init__(self, scene, perception):
        self.bridge = CvBridge()
        self.attached_obj = None
        self.attached_pose = None
        self.ros_time = 0
        self.execution_calls = 0
        self.num_executed_actions = 0
        self.num_collision = 0
        self.scene = scene
        self.perception = perception
        self.object_local_id_dict = {}
        self.shape_type_dict = {
            SolidPrimitive.BOX: p.GEOM_BOX,
            SolidPrimitive.CYLINDER: p.GEOM_CYLINDER,
            SolidPrimitive.SPHERE: p.GEOM_SPHERE,
        }
        self.object_state_msg = {}
        self.debug_texts = {}
        # self.target_obj_id = -1

        # for updating information
        self.current_robot_state = self.scene.robot.joint_dict

        self.depth_topic = '/camera/aligned_depth_to_color/image_raw'
        self.color_topic = '/camera/color/image_raw'
        color_sub = message_filters.Subscriber(self.color_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        seg_sub = message_filters.Subscriber('seg_image', Image)
        state_sub = message_filters.Subscriber('robot_state_publisher', RobotState)
        # obj_sub = message_filters.Subscriber('object_state', PercievedObject)

        # ts = message_filters.ApproximateTimeSynchronizer(
        #     [color_sub, depth_sub, seg_sub, state_sub],
        #     5,
        #     0.01,
        # )

        # ts =message_filters.TimeSynchronizer([color_sub, depth_sub, seg_sub, state_sub],5)

        # ts.registerCallback(self.info_cb)

        # self.timer = rospy.Timer(rospy.Duration(0.01), self.timer_cb)

        # add a lock
        # self.lock = Lock()

        # ts2 = message_filters.ApproximateTimeSynchronizer(
        #     [obj_sub],
        #     1,
        #     0.001,
        #     allow_headerless=True,
        # )
        # ts2.registerCallback(self.update_object_state)
        # state_sub = rospy.Subscriber('robot_state_publisher', RobotState, self.get_robot_state)


    def joint_dict_list_to_val_list(self, joint_dict_list):
        pass

    def execute_traj(self, joint_dict_list):
        # convert joint_val_list to ROS message
        arm = 'Right'
        msg = ExecuteTrajectoryRequest()
        joint_states = []
        joint_val_list = self.scene.robot.joint_dict_list_to_val_list(joint_dict_list)
        # msg.arm_trajectory = JointState()
        for i in range(len(joint_val_list)):
            joint_state = JointState()
            for j in range(len(joint_val_list[i])):
                joint_state.name.append(self.scene.robot.joint_names[j])
                joint_state.position.append(joint_val_list[i][j])
            joint_states.append(joint_state)
        msg.arm_trajectory.trajectory = joint_states
        msg.arm_trajectory.armType = arm

        rospy.wait_for_service("execute_trajectory")
        try:
            execute_traj_srv = rospy.ServiceProxy('execute_trajectory', ExecuteTrajectory)
            resp = execute_traj_srv(msg)
            self.scene.robot.set_joint_from_dict(joint_dict_list[-1])
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)



    def execute_traj_sim(self, joint_dict_list, ignored_obj_id=-1, duration=0.001):
        """
        call execution_system to execute the trajectory
        if an object has been attached, update the object model transform at the end
        """
        if len(joint_dict_list) == 0 or len(joint_dict_list) == 1:
            return

        start_time = time.time()

        # convert joint_dict_list to JointTrajectory
        traj = JointTrajectory()
        traj.joint_names = list(joint_dict_list[0].keys())
        points = []
        for i in range(len(joint_dict_list)):
            point = JointTrajectoryPoint()
            positions = []
            for name in traj.joint_names:
                if name in joint_dict_list[i]:
                    positions.append(joint_dict_list[i][name])
                else:
                    positions.append(joint_dict_list[i - 1][name])
                    joint_dict_list[i][name] = joint_dict_list[i - 1][name]
            point.positions = positions
            # point.time_from_start = i *
            points.append(point)
        traj.points = points

        rospy.wait_for_service('execute_trajectory', timeout=10)
        try:
            execute_trajectory = rospy.ServiceProxy(
                'execute_trajectory', ExecuteTrajectory
            )
            # resp1 = execute_trajectory(traj, ignored_obj_id)
            resp1 = execute_trajectory(traj)

            self.num_collision += resp1.num_collision
            # print('number of collision: ', self.num_collision)
            # update object pose using the last joint angle if an object is attached

            # * since we are using real-time perception, the robot state is updated
            # continuously in a separate thread.

            # rospy.sleep(1)  # wait for update of the robot state monitor
            # self.scene.robot.set_joint_from_dict_data(self.current_robot_state)

            # self.scene.robot.set_joint_from_dict_data(joint_dict_list[-1])
            # if self.attached_obj is not None:
            #     # TODO: maybe we should use stand-alone FK solver instead of PyBullet
            #     obj_transform = self.scene.robot.get_tip_link_pose_urdfpy(joint_dict_list[-1]).dot(self.scene.robot.attached_obj_rel_pose)
            #     self.perception.objects[self.attached_obj].update_transform(obj_transform)

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        self.ros_time += time.time() - start_time
        self.execution_calls += 1


    def attach_obj(self, obj_id):
        """
        call execution_system to attach the object
        """
        start_time = time.time()
        rospy.wait_for_service('attach_object', timeout=10)

        msg = AttachObjectRequest()
        msg.armType = 'Right'
        msg.attach = True
        
        try:
            attach_object = rospy.ServiceProxy('attach_object', AttachObject)
            resp1 = attach_object(msg)
            # ee_pose = self.scene.robot.get_tip_link_pose()
            # obtain object pose in ee link
            # rel_pose = np.linalg.inv(ee_pose).dot(self.perception.objects[obj_id].transform)
            # self.scene.robot.attach(obj_id, rel_pose)
            # self.attached_obj = obj_id

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        rospy.sleep(1)  # wait for update of the robot state monitor
        self.ros_time += time.time() - start_time
        self.execution_calls += 1

    def detach_obj(self):
        """
        call execution_system to detach the object
        UPDATE April 14, 2022:
        each action is finished with a detach action. So we count
        how many detach is called, this will indicate how many actions
        are performed
        """
        start_time = time.time()

        rospy.wait_for_service('attach_object', timeout=10)

        msg = AttachObjectRequest()
        msg.armType = 'Right'
        msg.attach = False
        

        try:
            attach_object = rospy.ServiceProxy('attach_object', AttachObject)
            resp1 = attach_object(msg)
            # self.attached_obj = None
            # self.num_executed_actions += 1
            # self.scene.robot.detach()

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        rospy.sleep(1)  # wait for update of the robot state monitor
        self.ros_time += time.time() - start_time
        self.execution_calls += 1

    def get_image(self):
        # print('waiting for message...')
        # start_time = time.time()
        # rospy.sleep(0.2)

        # color_img = rospy.wait_for_message('rgb_image', Image, timeout=10)
        # depth_img = rospy.wait_for_message('depth_image', Image, timeout=10)
        # seg_img = rospy.wait_for_message('seg_image', Image, timeout=10)

        # color_img = self.bridge.imgmsg_to_cv2(color_img, 'passthrough')
        # depth_img = self.bridge.imgmsg_to_cv2(depth_img, 'passthrough')
        # seg_img = self.bridge.imgmsg_to_cv2(seg_img, 'passthrough')

        # self.ros_time += time.time() - start_time
        # self.execution_calls += 1
        # return color_img, depth_img, seg_img
        # visualize the images
        # cv2.imshow('img', color_img)
        # print('Press space...')
        # cv2.waitKey()
        # return self.color_img, self.depth_img, self.seg_img
        pass

    def joint_dict_from_joint_state(self, msg: JointState):
        # extract the joint dict from joint state ROS message
        names = msg.name
        position = msg.position
        return dict(zip(names, position))

    def get_robot_state(self, msg: RobotState):
        """
        continuously update robot state obtained from execution_scene
        """
        self.current_robot_state = self.joint_dict_from_joint_state(msg.joint_state)
        self.scene.robot.set_joint_from_dict_data(self.current_robot_state)
        if msg.attached_obj == -1:
            attached_obj = -1
        else:
            attached_obj = self.perception.data_assoc.obj_ids[msg.attached_obj]
        # if attached, update object transform
        if (self.attached_obj is None and attached_obj != -1) or \
            (self.attached_obj is not None and attached_obj != -1 and self.attached_obj != attached_obj):
            # new attached object
            self.attached_obj = attached_obj
            self.attached_pose = np.array(
                self.perception.objects[self.attached_obj].transform
            )
            delta_transform = tf.quaternion_matrix(
                [
                    msg.delta_transform.rotation.w,
                    msg.delta_transform.rotation.x,
                    msg.delta_transform.rotation.y,
                    msg.delta_transform.rotation.z,
                ]
            )
            delta_transform[0, 3] = msg.delta_transform.translation.x
            delta_transform[1, 3] = msg.delta_transform.translation.y
            delta_transform[2, 3] = msg.delta_transform.translation.z
            self.perception.objects[self.attached_obj].update_transform(
                delta_transform.dot(self.attached_pose)
            )
        elif (self.attached_obj == attached_obj):
            # previous attached object
            delta_transform = tf.quaternion_matrix(
                [
                    msg.delta_transform.rotation.w,
                    msg.delta_transform.rotation.x,
                    msg.delta_transform.rotation.y,
                    msg.delta_transform.rotation.z,
                ]
            )
            delta_transform[0, 3] = msg.delta_transform.translation.x
            delta_transform[1, 3] = msg.delta_transform.translation.y
            delta_transform[2, 3] = msg.delta_transform.translation.z
            self.perception.objects[self.attached_obj].update_transform(
                delta_transform.dot(self.attached_pose)
            )
        else:
            # no attached object
            self.attached_obj = None
            self.attached_pose = None

    def update_object_state(self, msg: PercievedObject):
        self.object_state_msg[msg.name] = msg
        position = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        orientation = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        if msg.name not in self.object_local_id_dict:
            shape_type = self.shape_type_dict[msg.solid.type]
            # print(
            #     shape_type,
            #     msg.solid.dimensions,
            #     position,
            #     orientation,
            # )
            oid = self.spawn_object(
                shape_type,
                msg.solid.dimensions,
                position,
                orientation,
            )
            self.object_local_id_dict[msg.name] = oid
        else:
            p.resetBasePositionAndOrientation(
                self.object_local_id_dict[msg.name],
                position,
                orientation,
                physicsClientId=self.scene.robot.pybullet_id,
            )
        ### debug visualization ###
        # p_name = str(self.perception.data_assoc.obj_ids.get(int(msg.name), 'H'))
        p_name = msg.name
        # if self.object_local_id_dict[msg.name] == self.target_obj_id:
        # target is red...
        if msg.color[0] > 200 and msg.color[1] < 100 and msg.color[2] < 100:
            p_name = 'T.' + p_name
        x_pos = 0.5 * msg.solid.dimensions[self.r_index(
            self.shape_type_dict[msg.solid.type]
        )]
        x_pos += 0.01
        y_pos = 0.5 * msg.solid.dimensions[1]
        y_pos += 0.01
        # print(self.debug_texts)
        if msg.name not in self.debug_texts:
            self.debug_texts[msg.name] = (
                p_name,
                position,
                p.addUserDebugText(
                    p_name,
                    np.add(position, [x_pos, y_pos, 0.0]),
                    textColorRGB=[1, 0, 0],
                    textSize=2.0,
                    physicsClientId=self.scene.robot.pybullet_id,
                ),
            )
        else:
            prev_p_name, prev_position, prev_pyb_id = self.debug_texts[msg.name]
            if p_name != prev_p_name or prev_position != position:
                self.debug_texts[msg.name] = (
                    p_name,
                    position,
                    p.addUserDebugText(
                        p_name,
                        np.add(position, [x_pos, y_pos, 0.0]),
                        textColorRGB=[1, 0, 0],
                        textSize=2.0,
                        replaceItemUniqueId=prev_pyb_id,
                        physicsClientId=self.scene.robot.pybullet_id,
                    ),
                )
        ### end debug visualization ###

    def construct_obj_msg(self, obj: ObjectBelief):
        msg = PercievedObject()
        msg.name = str(obj.pybullet_id)
        msg.color = obj.color
        msg.type = PercievedObject.SOLID_PRIMITIVE
        if obj.obj_shape == 'cylinder':
            msg.solid.type = SolidPrimitive.CYLINDER
            msg.solid.dimensions = [obj.obj_model['height'], obj.obj_model['radius'], 0]
        transform = np.array(obj.transform)
        position = transform[:3, 3]
        qw, qx, qy, qz = tf.quaternion_from_matrix(transform)
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]

        msg.pose.orientation.w = qw
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz

        return msg

    def update_object_state_from_perception(self, obj: ObjectBelief):
        msg = self.construct_obj_msg(obj)
        self.update_object_state(msg)

    def r_index(self, stype):
        return {p.GEOM_CYLINDER: 1}.get(stype, 0)

    def spawn_object(
        self,
        shape_type,
        dimensions,
        position=[0, 0, 2],
        orientation=[0, 0, 0, 1],
    ):
        # don't worry about passing extra parameters not for
        # a specific shape type, they are ignored
        cuid = p.createCollisionShape(
            shape_type,
            halfExtents=np.multiply(0.5, dimensions),  # array of half dimensions if BOX
            radius=dimensions[self.r_index(shape_type)],  # [0] if SPHERE, [1] if CYLINDER
            height=dimensions[0],  # [0] if CYLINDER
            physicsClientId=self.scene.robot.pybullet_id,
        )
        mass = 0  # static box
        oid = p.createMultiBody(
            mass,
            cuid,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self.scene.robot.pybullet_id,
        )
        return oid

    def timer_cb(self, timer):
        color_msg = rospy.wait_for_message(self.color_topic, Image, timeout=10)
        depth_msg = rospy.wait_for_message(self.depth_topic, Image, timeout=10)
        print('obtained color and ddepth')
        # seg_msg = rospy.wait_for_message('seg_image', Image, timeout=10)
        print('obtained seg')
        # state_msg = rospy.wait_for_message(
        #     'robot_state_publisher', RobotState, timeout=10
        # )
        print('obtained state')

        self.color_img = self.bridge.imgmsg_to_cv2(color_msg, 'passthrough')
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough') / 1000 # /1000 when using real system
        # self.seg_img = self.bridge.imgmsg_to_cv2(seg_msg, 'passthrough')
        # self.get_robot_state(state_msg)
        print('got robot state')
        attached_obj = self.attached_obj
        # ct = time.strftime("%Y%m%d-%H%M%S")
        # cv2.imwrite(ct + "_img.png", self.color_img)

        # perform perception
        print('before pipeline_sim...')
        self.perception.pipeline_sim(self.color_img, self.depth_img)
        if attached_obj is not None:
            obj = self.perception.objects[self.attached_obj]
            pcd = obj.sample_conservative_pcd() / obj.resol
            pcd_ind = np.floor(pcd).astype(int)
            # Get vertex colors
            rgb_vals = np.zeros(pcd_ind.shape)
            # rgb_vals = obj.color_tsdf[pcd_ind[:, 0], pcd_ind[:, 1], pcd_ind[:, 2]] / 255

            pcd = visualize_pcd(pcd, rgb_vals)
            bbox = visualize_bbox(obj.voxel_x, obj.voxel_y, obj.voxel_z)
            # voxel = visualize_voxel(obj.voxel_x, obj.voxel_y, obj.voxel_z, model, [1,0,0])

            center = [
                obj.voxel_y.shape[0] / 2, obj.voxel_y.shape[1] / 2,
                obj.voxel_y.shape[2] / 2
            ]  # look_at target
            eye = [
                -obj.voxel_y.shape[0] * 1, obj.voxel_y.shape[1] / 2,
                obj.voxel_y.shape[2] * 2
            ]  # camera position
            up = [0, 0, 1]  # camera orientation

            render = setup_render(center, eye, up)
            # Show the original coordinate axes for comparison.
            # X is red, Y is green and Z is blue.
            render.scene.show_axes(True)
            # Define a simple unlit Material.
            # (The base color does not replace the arrows' own colors.)
            # mtl = create_material([1.0, 1.0, 1.0, 0.01], 'defaultUnlit')
            mtl2 = create_material([1.0, 1.0, 1.0, 1], 'defaultUnlit')
            # render.scene.add_geometry("voxel", voxel, mtl)
            render.scene.add_geometry("pcd", pcd, mtl2)
            render.scene.add_geometry("bbox", bbox, mtl2)
            # Read the image into a variable
            img_o3d = render.render_to_image()
            img_o3d = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
            # cv2.imwrite(ct + "_recon.png", img_o3d)

        # if object is attached, then sense the object
        # if attached_obj is not None:
        #     self.perception.sense_object(attached_obj, self.color_img, self.depth_img, self.seg_img,
        #                                 self.scene.camera, [self.scene.robot.robot_id], self.scene.workspace.component_ids)

    def info_cb(self, rgb_msg, depth_msg, seg_msg, state_msg):  #, obj_msg):

        self.color_img = self.bridge.imgmsg_to_cv2(rgb_msg, 'passthrough')
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        self.seg_img = self.bridge.imgmsg_to_cv2(seg_msg, 'passthrough')
        self.get_robot_state(state_msg)
        # self.update_object_state(obj_msg)
        attached_obj = self.attached_obj
        # ct = time.strftime("%Y%m%d-%H%M%S")
        # cv2.imwrite(ct+"_img.png", self.color_img)
        # if attached_obj is not None:
        #     np.save(ct+"_obj_transform.npy", self.perception.objects[self.attached_obj].transform)

        # perform perception
        self.perception.pipeline_sim(
            self.color_img,
            self.depth_img,
            self.seg_img,
            self.scene.camera,
            [self.scene.robot.robot_id],
            self.scene.workspace.component_ids,
        )
