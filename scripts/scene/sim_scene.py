"""
A basic structure of a simulated scene. It can be used for setting up planning scene
or the execution scene.
"""
from .robot import Robot
from .workspace import Workspace
from .camera import Camera
import pybullet as p
import rospkg
import os


class SimScene():

    def __init__(self, scene_dict):
        # pid = p.connect(p.DIRECT)
        pid = p.connect(p.GUI)

        rp = rospkg.RosPack()
        package_path = rp.get_path('vbcpm_execution_system')
        urdf_path = os.path.join(package_path, scene_dict['robot']['urdf'])
        # joints = [0.] * 16

        ll = [-1.58, \
                -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13, \
                -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13] +  \
                [0.0, -0.8757, 0.0, 0.0, -0.8757, 0.0]
        ### upper limits for null space
        ul = [1.58, \
                3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13, \
                3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13] + \
                [0.8, 0.0, 0.8757, 0.81, 0.0, 0.8757]
        ### joint ranges for null space
        jr = [1.58*2, \
                6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26, \
                6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26] + \
                [0.8, 0.8757, 0.8757, 0.81, 0.8757, 0.8757]

        robot = Robot(
            urdf_path,
            scene_dict['robot']['pose']['pos'],
            scene_dict['robot']['pose']['ori'],
            ll,
            ul,
            jr,
            'motoman_right_ee',
            # 0.3015, # suction distance
            0.044,  # gripper width
            pid,
            [1] + [0] * 7 + [1] * 7,
        )

        joints = [0]
        joints += [1.75, 0.8, 0.0, -0.66, 0.0, 0.0, 0.0]  # left
        joints += [1.75, 0.8, 0.0, -0.66, 0.0, 0.0, 0.0]  # right
        robot.set_joints(joints)
        robot.set_init_joints()

        workspace_low = scene_dict['workspace']['region_low']
        workspace_high = scene_dict['workspace']['region_high']
        padding = scene_dict['workspace']['padding']
        workspace = Workspace(scene_dict['workspace']['pos'], scene_dict['workspace']['ori'], \
                scene_dict['workspace']['components'], workspace_low, workspace_high, padding, \
                pid)
        workspace_low = workspace.region_low
        workspace_high = workspace.region_high
        # camera: using information published from execution_scene
        camera = Camera()

        self.robot = robot
        self.workspace = workspace
        self.camera = camera
        self.pid = pid

    # def __init__(self, workspace, robot, camera, pid):
    #     self.workspace = workspace
    #     self.robot = robot
    #     self.camera = camera
    #     self.pid = pid
