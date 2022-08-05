"""
simumlated execution scene. Some of it overlaps with the sim_scene defined in scene folder.
"""
import sys

# import problem_generation as prob_gen
import problem_generation_stacked as prob_gen
from utils.visual_utils import *
from pracsys_vision_tamp_manipulation.srv import AttachObject, ExecuteTrajectory, \
                                    AttachObjectResponse, ExecuteTrajectoryResponse
from pracsys_vision_tamp_manipulation.msg import RobotState

import rospy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import transformations as tf
import pickle
import json
import os, rospkg

from threading import Lock


class ExecutionSystem():

    def __init__(self, load=None, scene_name='scene1'):
        # if load is None, then randomly generate a problem instance
        # otherwise use the load as a filename to load the previously generated problem
        # obj_type = input('using simple geometry? y or n: ').strip()
        obj_type = sys.argv[3].strip()

        if load is not None:
            print('loading file: ', load)
            # load previously generated object
            f = open(load, 'rb')
            data = pickle.load(f)
            f.close()
            if obj_type == 'y':
                scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_shape, target_obj_size = data
                data = prob_gen.load_problem_level(
                    scene_f,
                    obj_poses,
                    obj_pcds,
                    obj_shapes,
                    obj_sizes,
                    target_pose,
                    target_pcd,
                    target_obj_shape,
                    target_obj_size,
                )
                pid, scene_f, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, target_pose, target_pcd, target_obj_id = data
                f = open(scene_f, 'r')
                scene_dict = json.load(f)
                print('loaded')
            else:
                scene_f, obj_poses, obj_shapes, target_pose, target_obj_shape = data
                data = prob_gen.load_problem_ycb(
                    scene_f, obj_poses, obj_shapes, target_pose, target_obj_shape
                )
                pid, scene_f, robot, workspace, camera, obj_poses, obj_ids, target_pose, target_obj_id = data
                f = open(scene_f, 'r')
                scene_dict = json.load(f)
                print('loaded')
                obj_pcds = None
                obj_sizes = None
                target_pcd = None
                target_obj_size = None
        else:
            rp = rospkg.RosPack()
            package_path = rp.get_path('pracsys_vision_tamp_manipulation')
            scene_f = os.path.join(package_path, 'scenes/' + scene_name + '.json')
            # scene_f = scene_name+'.json'
            level = input(
                'Input the difficulty level: 1 - simple, 2 - medium, 3 - hard...'
            )
            level = int(level)
            num_objs = input('Input the number of objects: ')
            num_objs = int(num_objs)
            if obj_type == 'y':
                # data = prob_gen.random_one_problem_level(scene=scene_f, level=level, num_objs=num_objs, num_hiding_objs=1)
                data = prob_gen.random_stacked_problem(
                    scene=scene_f, level=level, num_objs=num_objs, num_hiding_objs=1
                )
                pid, scene_f, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_id, target_obj_shape, target_obj_size = data
                data = (
                    scene_f,
                    obj_poses,
                    obj_pcds,
                    obj_shapes,
                    obj_sizes,
                    target_pose,
                    target_pcd,
                    target_obj_shape,
                    target_obj_size,
                )
                save = input('save current scene? 0 - no, 1 - yes...')
                f = open(scene_f, 'r')
                scene_dict = json.load(f)
                f.close()
                if int(save) == 1:
                    save_f = input('save name: ').strip()
                    f = open(save_f + '.pkl', 'wb')
                    pickle.dump(data, f)
                    f.close()
                    print('saved')
            else:
                data = prob_gen.random_one_problem_ycb(
                    scene=scene_f, level=level, num_objs=num_objs, num_hiding_objs=1
                )
                pid, scene_f, robot, workspace, camera, obj_poses, obj_ids, obj_shapes, target_pose, target_obj_id, target_obj_shape = data
                data = (scene_f, obj_poses, obj_shapes, target_pose, target_obj_shape)
                save = input('save current scene? 0 - no, 1 - yes...')
                f = open(scene_f, 'r')
                scene_dict = json.load(f)
                f.close()
                if int(save) == 1:
                    save_f = input('save name: ').strip()
                    f = open(save_f + '.pkl', 'wb')
                    pickle.dump(data, f)
                    f.close()
                    print('saved')
                obj_pcds = None
                obj_sizes = None
                target_pcd = None
                target_obj_size = None

        self.pid = pid
        self.scene_dict = scene_dict
        self.robot = robot
        self.workspace = workspace
        self.camera = camera
        self.obj_poses = obj_poses
        self.obj_pcds = obj_pcds
        self.obj_ids = obj_ids
        self.obj_shapes = obj_shapes
        self.obj_sizes = obj_sizes
        self.target_pose = target_pose
        self.target_pcd = target_pcd
        self.target_obj_id = target_obj_id
        self.target_obj_shape = target_obj_shape
        self.target_obj_size = target_obj_size

        self.total_obj_ids = self.obj_ids + [self.target_obj_id]
        self.bridge = CvBridge()

        self.attached_obj_id = None
        self.attached_obj_pose = None
        self.ee_transform = None
        self.obj_delta_transform = None
        lock = Lock()
        # * initialize ROS services
        # - robot trajectory tracker
        rospy.Service("execute_trajectory", ExecuteTrajectory, self.execute_trajectory)
        rospy.Service("attach_object", AttachObject, self.attach_object)

        # * initialize ROS pubs and subs
        # - camera
        # - robot_state_publisher

        self.rgb_img, self.depth_img, self.seg_img = self.camera.sense()
        self.robot_state = self.robot.joint_dict

        self.rgb_cam_pub = rospy.Publisher('rgb_image', Image)
        self.depth_cam_pub = rospy.Publisher('depth_image', Image)
        self.seg_cam_pub = rospy.Publisher('seg_image', Image)
        self.rs_pub = rospy.Publisher('robot_state_publisher', RobotState)
        self.js_pub = rospy.Publisher('joint_states', JointState)

        self.timer = rospy.Timer(rospy.Duration(0.01), self.publish_joint_state)

        # self.done_sub = rospy.Subscriber('done_msg', Int32, self.done_callback)

        # construct_occlusion_graph(obj_ids, obj_poses, camera, pid)

    # def done_callback(self, msg):
    #     if msg.data == 1:
    #         # finished
    #         exit(1)

    def check_collision(self, ignored_obj_id):
        """
        check collision between objects, workspace and robots
        """
        # check collision between objects and robot
        for obj_id in self.obj_ids:
            if self.attached_obj_id is not None and obj_id == self.attached_obj_id:
                continue
            if obj_id == ignored_obj_id:
                continue
            distance = -0.02  # seems the PyBullet collision geometry is very conservative. The moveit planned path
            # sometimes collide with the environment
            contacts = p.getClosestPoints(
                obj_id, self.robot.robot_id, distance=distance, physicsClientId=self.pid
            )
            if len(contacts) > 0:
                print('collision happened between robot and object ', obj_id)
                # input('press Enter')
                return True
        for c_name, cid in self.workspace.component_id_dict.items():
            distance = -0.02  # seems the PyBullet collision geometry is very conservative. The moveit planned path
            # sometimes collide with the environment
            contacts = p.getClosestPoints(
                cid, self.robot.robot_id, distance=distance, physicsClientId=self.pid
            )
            if len(contacts) > 0:
                print('collision happened between robot and ', c_name)
                # input('press Enter')
                return True
        # check for attached object
        if self.attached_obj_id is not None:
            for obj_id in self.obj_ids:
                if obj_id == self.attached_obj_id:
                    continue
                if obj_id == ignored_obj_id:
                    continue
                distance = -0.02  # seems the PyBullet collision geometry is very conservative. The moveit planned path
                # sometimes collide with the environment
                contacts = p.getClosestPoints(
                    obj_id,
                    self.attached_obj_id,
                    distance=distance,
                    physicsClientId=self.pid
                )
                if len(contacts) > 0:
                    print(
                        'collision happened between attached object and object ', obj_id
                    )
                    # input('press Enter')
                    return True
            for c_name, cid in self.workspace.component_id_dict.items():
                if c_name == 'shelf_bottom':
                    distance = -0.02
                else:
                    distance = -0.02  # seems the PyBullet collision geometry is very conservative. The moveit planned path
                    # sometimes collide with the environment
                contacts = p.getClosestPoints(
                    cid,
                    self.attached_obj_id,
                    distance=distance,
                    physicsClientId=self.pid
                )
                if len(contacts) > 0:
                    print('collision happened between attached object and ', c_name)
                    # input('press Enter')
                    return True
        return False

    def execute_trajectory(self, req):
        """
        PyBullet:
        if object attached, move arm and object at the same time
        """
        traj = req.trajectory  # sensor_msgs/JointTrajectory
        ignored_obj_id = req.ignored_obj_id
        # joint_dict_list = []
        joint_names = traj.joint_names
        points = traj.points
        num_collision = 0
        # TODO: add a tracker for more realistic movements
        # interpolation of trajectory with small step size
        received_pts = []
        for i in range(len(points)):
            received_pts.append(points[i].positions)
        np.savetxt('received-pts.txt', np.array(received_pts))

        step_sz = 1 * np.pi / 180
        interpolated_pts = [traj.points[0].positions]

        for i in range(len(points) - 1):
            pos1 = np.array(points[i].positions)
            pos2 = np.array(points[i + 1].positions)
            abs_change = np.abs(pos2 - pos1)
            n_steps = int(np.ceil(abs_change.max() / step_sz))
            interpolated_pts += np.linspace(
                start=pos1, stop=pos2, num=n_steps + 1
            )[1:].tolist()

        # print('interpolated points: ')
        # print(interpolated_pts)
        np.savetxt('interpolated-pts.txt', np.array(interpolated_pts))

        for i in range(len(interpolated_pts)):
            pos = interpolated_pts[i]
            # time_from_start = points[i].time_from_start
            joint_dict = {joint_names[j]: pos[j] for j in range(len(pos))}
            self.robot.set_joint_from_dict(joint_dict)

            if self.attached_obj_id is not None:
                transform = self.robot.get_tip_link_pose()
                new_obj_transform = transform.dot(self.robot.attached_obj_rel_pose)
                quat = tf.quaternion_from_matrix(new_obj_transform)  # w x y z
                p.resetBasePositionAndOrientation(
                    self.attached_obj_id,
                    new_obj_transform[:3, 3], [quat[1], quat[2], quat[3], quat[0]],
                    physicsClientId=self.robot.pybullet_id
                )
                if self.check_collision(ignored_obj_id):
                    num_collision += 1
                    print('collision happened.')

            # update images and robot state
            self.rgb_img, self.depth_img, self.seg_img = self.camera.sense()
            self.robot_state = self.robot.joint_dict
            if self.attached_obj_id is not None:
                self.ee_transform = transform
                self.obj_delta_transform = new_obj_transform.dot(
                    np.linalg.inv(self.attached_obj_pose)
                )

            rospy.sleep(0.01)

        # input('waiting...')
        # rospy.sleep(0.03)
        return ExecuteTrajectoryResponse(num_collision, True)

    def get_body_transform(self, bid):
        # obtain the transformation matrix of the body bid: world T bid
        link_state = p.getBasePositionAndOrientation(
            bid, physicsClientId=self.robot.pybullet_id
        )
        pos = link_state[0]
        ori = link_state[1]
        transform = tf.quaternion_matrix([ori[3], ori[0], ori[1], ori[2]])
        transform[:3, 3] = pos
        return transform

    def attach_object(self, req):
        """
        attach the object closest to the robot
        """
        if req.attach == True:
            obj_transform = self.get_body_transform(req.obj_id)
            self.attached_obj_pose = obj_transform
            # obtain relative transform to robot ee
            ee_transform = self.robot.get_tip_link_pose()
            obj_rel_transform = np.linalg.inv(ee_transform).dot(obj_transform)
            self.robot.attach(req.obj_id, obj_rel_transform)

            # initialize delta_transform
            self.ee_transform = ee_transform
            self.obj_delta_transform = np.eye(4)
            self.attached_obj_id = req.obj_id

            # compute the ee T obj (pose)

        else:
            self.robot.detach()
            self.attached_obj_id = None
            self.attached_obj_pose = None
            self.ee_transform = None
            self.obj_delta_transform = None

        return AttachObjectResponse(True)

    def publish_image(self):
        """
        obtain image from PyBullet and publish
        """
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
        obtain joint state from PyBullet and publish
        """

        msg = RobotState()
        for name, val in self.robot_state.items():
            msg.joint_state.name.append(name)
            msg.joint_state.position.append(val)
        if self.attached_obj_id is None:
            msg.attached_obj = -1
        else:
            msg.attached_obj = self.attached_obj_id
            # ee_transform = self.robot.get_tip_link_pose_urdfpy()
            # attached_obj_pose = ee_transform.dot(self.robot.attached_obj_rel_pose)
            delta_transform = self.obj_delta_transform
            # delta_transform = attached_obj_pose.dot(np.linalg.inv(self.attached_obj_pose))
            qw, qx, qy, qz = tf.quaternion_from_matrix(delta_transform)
            x = delta_transform[0, 3]
            y = delta_transform[1, 3]
            z = delta_transform[2, 3]
            msg.delta_transform.rotation.w = qw
            msg.delta_transform.rotation.x = qx
            msg.delta_transform.rotation.y = qy
            msg.delta_transform.rotation.z = qz
            msg.delta_transform.translation.x = x
            msg.delta_transform.translation.y = y
            msg.delta_transform.translation.z = z
        msg.header.stamp = rospy.Time.now()
        self.rs_pub.publish(msg)

    def publish_joint_state(self, timer):
        msg = JointState()
        for name, val in self.robot_state.items():
            msg.name.append(name)
            msg.position.append(val)
        msg.header.stamp = rospy.Time.now()
        self.js_pub.publish(msg)

    def run(self):
        """
        keep spinning and publishing to the ROS topics
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publish_robot_state()
            self.publish_image()
            rate.sleep()


def main():
    rospy.init_node("execution_system")
    rospy.sleep(1.0)
    scene_name = 'scene2'

    if int(sys.argv[1]) > 0:
        load = True
        # load = input("enter the problem name for loading: ").strip()
        load = sys.argv[2].strip()

        load = load + '.pkl'
    else:
        load = None
    execution_system = ExecutionSystem(load, scene_name)
    # start_time = time.time()
    # res = execution_system.robot.get_tip_link_pose()
    # print('pybullet takes time: ', time.time() - start_time)
    # print(res)

    # start_time = time.time()
    # res = execution_system.robot.get_tip_link_pose_urdfpy()
    # print('urdfpy takes time: ', time.time() - start_time)
    # print(res)
    print('pid: ', execution_system.pid)
    execution_system.run()


if __name__ == "__main__":
    main()
