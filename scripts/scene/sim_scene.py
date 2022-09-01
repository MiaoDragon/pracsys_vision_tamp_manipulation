"""
A basic structure of a simulated scene. It can be used for setting up planning scene
or the execution scene.
"""
import sys

sys.path.insert(0,'/home/pracsys/Documents/ylmiao/motoman_ws/src/pracsys_perception/scripts')

from .robot import Robot
from .workspace import Workspace
from .camera import Camera
import pybullet as p
import rospkg
import os
import tf
import transformations
import rospy
import pickle
import numpy as np
from sensor_msgs.msg import CameraInfo

def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order  

    output: 2 points on line of intersection, np.arrays, shape (3,)

    https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

# could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]



class SimScene():
    def __init__(self, scene_dict):
        # pid = p.connect(p.DIRECT)
        pid = p.connect(p.GUI)

        listener = tf.TransformListener()

        rp = rospkg.RosPack()
        package_path = rp.get_path('pracsys_vision_tamp_manipulation')
        urdf_path = os.path.join(package_path,scene_dict['robot']['urdf'])
        print(urdf_path)
        joints = [0.] * 16

        ll = [-1.58, \
            -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13, \
            -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13]# +  \
            # [0.0, -0.8757, 0.0, 0.0, -0.8757, 0.0]
        ### upper limits for null space
        ul = [1.58, \
            3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13, \
            3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13]# + \
            # [0.8, 0.0, 0.8757, 0.81, 0.0, 0.8757]
        ### joint ranges for null space
        jr = [1.58*2, \
            6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26, \
            6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26] + \
            [0.8, 0.8757, 0.8757, 0.81, 0.8757, 0.8757]

        # tip_link_name = {'left': 'motoman_left_ee', 'right': 'motoman_right_ee'}
        tip_link_name = 'motoman_right_ee'
        # tip_joint_name = {'left': 'arm_left_joint_7_t', 'right': 'arm_right_joint_7_t'}
        tip_joint_name = 'arm_right_joint_7_t'


        robot = Robot(urdf_path, scene_dict['robot']['pose']['pos'], scene_dict['robot']['pose']['ori'], 
                        ll, ul, jr, tip_link_name, tip_joint_name, 0.3015, pid, [1] + [0] * 7 + [1] * 7)


        # joints = [0,
        #         0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0,  # left (suction)
        #         1.75, 0.8, 0.0, -0.66, 0.0, 0.0 ,0.0,  # right
        #         ]
        # joints = [0,
        #         1.75, 0.8, 0.0, -0.66, 0.0, 0.0 ,0.0,  # left
        #         0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0  # right
        #         ]
        joints = [0,
                1.75, 0.8, 0.0, -0.66, 0.0, 0.0 ,0.0,  # left
                1.75, 0.8, 0.0, -0.66, 0.0, 0.0 ,0.0,  # right
                ]

        robot.set_joints(joints)  # 
        robot.set_init_joints(joints)


        # camera: using information from transformation
        rospy.sleep(1.0)
        (trans,rot) = listener.lookupTransform('/base', '/camera_color_optical_frame', rospy.Time(0))  # transformation of camera in base, i.e.  R T C
        # (trans,rot) = listener.lookupTransform('/camera_color_optical_frame', '/base_link', rospy.Time(0))  # transformation of camera in base, i.e.  R T C

        qx,qy,qz,qw = rot
        rot_mat = transformations.quaternion_matrix([qw,qx,qy,qz])
        rot_mat[:3,3] = trans
        
        print('trans: ')
        print(rot_mat)

        look_at_vec = rot_mat[:3,2]  # z axis

        # right_vector: R, up_vector: G, direction: B
        # camera_color_optical_frame to vectors:
        # up_vector: -G axis of the frame
        up_vector = -rot_mat[:3,1]
        
        # obtain information of camera
        camera_info = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        height = camera_info.height
        width = camera_info.width
        diag = np.sqrt(height**2+width**2)
        K = camera_info.K
        K = np.array(K).reshape((3,3))
        fx = K[0,0]
        fy = K[1,1]
        fov0 = 2*np.arctan(width/(2*fx))  # in radius
        fov1 = 2*np.arctan(height/(2*fy))
        # convert to angle
        fov0 = fov0 * 180 / np.pi
        fov1 = fov1 * 180 / np.pi
        
        camera = Camera(cam_pos=trans, look_at=trans+look_at_vec, up_vec=up_vector, far=1.5, fov=[fov0,fov1], img_size=[height, width], camera_intrinsics=K)
        camera.sense()
        print('extrinsics:')
        print(camera.info['extrinsics'])  # this follows the TF of real state
        # * workspace: generate from vision
        # left, right walls
        # bottom, top walls
        # back front walls
        plane_models = os.path.join(package_path, 'scripts/plane_models.pkl')
        plane_models = open(plane_models, 'rb')
        plane_dict = pickle.load(plane_models)
        back_plane = plane_dict['back_plane']
        bot_plane = plane_dict['bot_plane']
        side_planes = plane_dict['side_planes']
        padding = 0.04
        height = 0.54 #0.57

        back_plane[3] -= padding
        bot_plane[3] -= padding
        side_planes[0][3] -= padding
        side_planes[1][3] -= padding

        top_plane = np.array(bot_plane)
        top_plane[3] -= height
        top_plane = -top_plane  # the normal always point toward inside the workspace

        # above are in the frame of camera. Transform to world
        rot_mat = np.linalg.inv(rot_mat).T
        back_plane = rot_mat.dot(back_plane)
        bot_plane = rot_mat.dot(bot_plane)
        top_plane = rot_mat.dot(top_plane)
        side_planes[0] = rot_mat.dot(side_planes[0])
        side_planes[1] = rot_mat.dot(side_planes[1])

        back_plane = back_plane / np.linalg.norm(back_plane[:3])
        # back plane x value is negative
        if back_plane[0] > 0:
            back_plane = -back_plane
        # bot_plane z is positive
        bot_plane = bot_plane / np.linalg.norm(bot_plane[:3])
        if bot_plane[2] < 0:
            bot_plane = -bot_plane
        # top_plane z value is negative
        top_plane = top_plane / np.linalg.norm(top_plane[:3])
        if top_plane[2] > 0:
            top_plane = -top_plane
        
        side_planes[0] = side_planes[0] / np.linalg.norm(side_planes[0][:3])        
        side_planes[1] = side_planes[1] / np.linalg.norm(side_planes[1][:3])        
        # determine which one is left or right
        # right has positive y value
        if side_planes[0][1] < 0:
            side_planes[0] = -side_planes[0]
        if side_planes[1][1] < 0:
            side_planes[1] = -side_planes[1]
        # left plane has a higher -d value
        if side_planes[0][3] < side_planes[1][3]:
            left_plane = side_planes[0]
            right_plane = side_planes[1]
        else:
            left_plane = side_planes[1]
            right_plane = side_planes[0]

        print('side_planes: ')
        print(side_planes)

        # find the closest y-z plane for back plane
        back_x = back_plane[3]  # sign because of normal direction
        front_x = back_x - 0.65 - 0.1 - 0.08
        # find the closest x-y plane for bottom plane
        print('bot_plane: ', bot_plane)
        print('back_plane: ', back_plane)
        bot_z = -bot_plane[3]
        top_z = top_plane[3]
        right_y = -right_plane[3]
        left_y = -left_plane[3]

        print('front_x: ', front_x)
        print('back_x: ', back_x)
        print('left_y: ', left_y)
        print('right_y: ', right_y)
        print('top_z: ', top_z)
        print('bot_z: ', bot_z)



        # construct the workspace
        workspace = Workspace(front_x, back_x, left_y, right_y, top_z, bot_z, trans, pid)

        self.robot = robot
        self.workspace = workspace
        self.camera = camera
        self.pid = pid

        self.objects = []  # a list of object models
        self.obj_id_to_objects = {}  # a dict mapping from object id to object
    # def __init__(self, workspace, robot, camera, pid):
    #     self.workspace = workspace
    #     self.robot = robot
    #     self.camera = camera
    #     self.pid = pid
    def add_object(self, obj_model):
        self.objects.append(obj_model)
        self.obj_id_to_objects[obj_model.obj_id] = obj_model


def test():
    # test the module
    import json
    rp = rospkg.RosPack()
    scene_name = "scene2"
    package_path = rp.get_path('pracsys_vision_tamp_manipulation')
    scene_f = os.path.join(package_path,'scenes/'+scene_name+'.json')
    f = open(scene_f, 'rb')
    scene_dict = json.load(f)
    scene = SimScene(scene_dict)
    input('Enter...')
if __name__ == "__main__":
    test()