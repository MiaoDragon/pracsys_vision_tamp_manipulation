"""
implement the primitive to be used by the task planner and pre-condition checking
"""
"""
Provide PyBullet planning scene to check IK, collisions, etc.
and implementations of task actions
"""

import gc
import copy
import time

import cv2
import std_msgs
import numpy as np
import pybullet as p

from utils.visual_utils import *
from utils.transform_utils import *
from .rearrangement import Rearrangement
from . import utils, obj_pose_generation
from motion_planner.motion_planner import MotionPlanner
from task_planner.primitives.execution_interface import ExecutionInterface

class PrimitivePlanner():

    def __init__(self, scene, perception_system, execution: ExecutionInterface):
        """
        Create a PyBullet scene including workspace, robot and camera
        """
        # load scene definition file
        motion_planner = MotionPlanner(scene.robot, scene.workspace)
        self.scene = scene
        self.perception = perception_system
        self.execution = execution
        self.motion_planner = motion_planner
        self.rearrange_planner = Rearrangement()
        self.scene.robot.set_motion_planner(motion_planner)

        self.perception_time = 0.0
        self.motion_planning_time = 0.0
        self.pose_generation_time = 0.0
        self.ros_time = 0.0  # communication to execution scene
        self.rearrange_time = 0.0

        self.perception_calls = 0
        self.motion_planning_calls = 0
        self.pose_generation_calls = 0
        self.execution_calls = 0
        self.rearrange_calls = 0
        self.num_executed_actions = 0
        self.num_collision = 0

    def TryMoveOneObject(self, obj, pre_grasp_dist=0.02, pre_place_dist=0.08):
        robot = self.execution.scene.robot
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        time_info = {"success": False}
        total0 = time.time()

        ## Generate Grasps ##
        filteredPoses = obj_pose_generation.geometric_gripper_grasp_pose_generation(
            obj_local_id,
            robot,
            self.scene.workspace,
            offset2=(0, 0, -pre_grasp_dist),
        )

        ## Generate Placements ##
        placements = obj_pose_generation.generate_placements(
            obj,
            robot,
            self.execution,
            self.perception,
            self.scene.workspace,
            display=True,
        )

        # planning for each grasp until success
        for poseInfo in filteredPoses:
            if len(poseInfo['collisions']) != 0:
                break
            ## Set Collision Space ##
            self.set_collision_env_with_models()

            pick_joint_dict = robot.joint_vals_to_dict(poseInfo['dof_joints'])
            pick_joint_dict_list = self.motion_planner.ee_approach_plan(
                robot.joint_dict,
                # eof_poses,
                pick_joint_dict,
                # robot,
                disp_dist=pre_grasp_dist,
                disp_dir=(0, 0, 1),
                is_pre_dir_abs=False,
                attached_acos=[],
            )

            ## Plan Lift ##
            new_start_joint_dict = dict(pick_joint_dict_list[-1])
            pick_tip_pose = robot.get_tip_link_pose(new_start_joint_dict)
            lift_tip_pose = np.eye(4)
            lift_tip_pose[:3, 3] = np.array([0, 0, 0.04])  # lift up by 0.04

            lift_joint_dict_list = self.motion_planner.straight_line_motion(
                new_start_joint_dict,
                pick_tip_pose,
                lift_tip_pose,
                robot,
                collision_check=False,
                workspace=self.scene.workspace,
                display=False
            )

            ## Place ##
            new_start_joint_dict, grasp_joint_dict = (
                lift_joint_dict_list[-1],
                pick_joint_dict_list[-1],
            )
            # get gripper to object matrix
            obj_transform = translation_quaternion2homogeneous(
                *p.getBasePositionAndOrientation(obj_local_id, robot.pybullet_id)
            )
            ee_transform = robot.get_tip_link_pose(grasp_joint_dict)
            obj_rel_transform = np.linalg.inv(ee_transform).dot(obj_transform)
            obj2gripper = np.linalg.inv(obj_rel_transform)
            for sample_pos in reversed(placements):
                # get gripper transform at placement
                obj_transform[:3, 3] = sample_pos
                gripper_transform = obj_transform.dot(obj2gripper)
                pos, rot = homogeneous2translation_quaternion(gripper_transform)

                # check IK
                valid, jointPoses = robot.get_ik(
                    robot.tip_link_name,
                    pos,
                    rot,
                    robot.init_joint_vals,
                )
                if not valid:
                    robot.set_joints_without_memorize(robot.joint_vals)
                    continue

                # check collision
                ignore_ids = [robot.robot_id]
                collisions = set()
                for i in range(p.getNumBodies(physicsClientId=robot.pybullet_id)):
                    obj_pid = p.getBodyUniqueId(i, physicsClientId=robot.pybullet_id)
                    if obj_pid in ignore_ids:
                        continue
                    contacts = p.getClosestPoints(
                        robot.robot_id,
                        obj_pid,
                        distance=0.,
                        physicsClientId=robot.pybullet_id,
                    )
                    if len(contacts):
                        collisions.add(obj_pid)
                robot.set_joints_without_memorize(robot.joint_vals)
                if len(collisions) > 0:
                    continue

                ## Plan Place ##
                place_joint_dict = robot.joint_vals_to_dict(jointPoses)
                aco = self.attach_known(obj, robot, grasp_joint_dict)
                place_joint_dict_list = self.motion_planner.ee_approach_plan(
                    new_start_joint_dict,
                    place_joint_dict,
                    disp_dist=pre_place_dist,
                    disp_dir=(0, 0, -1),
                    is_pre_dir_abs=True,
                    attached_acos=[aco],
                )
                self.detach_known(obj)
                if not place_joint_dict_list:

                    continue

                ## Plan Lift ##
                new_start_joint_dict2 = dict(place_joint_dict_list[-1])
                place_tip_pose = robot.get_tip_link_pose(new_start_joint_dict2)
                lift_tip_pose = np.eye(4)
                lift_tip_pose[:3, 3] = np.array([0, 0, 0.06])  # lift up by 0.06

                lift_joint_dict_list2 = self.motion_planner.straight_line_motion(
                    new_start_joint_dict2,
                    place_tip_pose,
                    lift_tip_pose,
                    robot,
                    collision_check=False,
                    workspace=self.scene.workspace,
                    display=False
                )
                ## Execute ##
                print("Succeded to plan move for Object {obj.obj_id}!")
                self.execution.detach_obj()
                self.execution.execute_traj(pick_joint_dict_list)
                self.execution.attach_obj(obj.obj_id)
                self.execution.execute_traj(lift_joint_dict_list)
                self.execution.execute_traj(place_joint_dict_list)
                self.execution.detach_obj()
                self.execution.execute_traj(lift_joint_dict_list2)
                return time_info
        return time_info

    def attach_known(self, obj, robot, grasp_joint_dict):
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        ee_transform_now = robot.get_tip_link_pose(
            {key: 0.0
             for key in grasp_joint_dict.keys()}
        )
        ee_transform_grasp = robot.get_tip_link_pose(grasp_joint_dict)
        obj_transform = translation_quaternion2homogeneous(
            *p.getBasePositionAndOrientation(obj_local_id, robot.pybullet_id)
        )
        obj_rel_transform = np.linalg.inv(ee_transform_grasp).dot(obj_transform)
        obj_abs_transform = ee_transform_now.dot(obj_rel_transform)
        pose = homogeneous2pose_stamped_msg(obj_abs_transform)
        mins, maxs = p.getAABB(obj_local_id, physicsClientId=robot.pybullet_id)
        size = [cmax - cmin for cmin, cmax in zip(mins, maxs)]
        return self.motion_planner.attach_known(
            str(obj.pybullet_id), pose=pose, size=size
        )

    def detach_known(self, obj):
        self.motion_planner.detach_known(str(obj.pybullet_id))
        # self.motion_planner.scene_interface.remove_world_object("TEMP_ATTACHED")

    def set_collision_env_with_models(self):
        ## Set Collision Space ##
        obs_msgs = []
        # for obs_id in self.execution.object_state_msg.keys():
        for obs in self.perception.objects.values():
            obs_id = obs.pybullet_id
            print(obs_id)
            # print(self.execution.object_state_msg[str(obs_id)].name)
            obs_msgs.append(self.execution.object_state_msg[str(obs_id)])
        self.motion_planner.set_collision_env_with_models(obs_msgs)

    def pipeline_sim(self):
        # sense & perceive
        # wait for image to update
        print('pipeline_sim...')
        self.execution.timer_cb(None)

        v_pcds = []
        for obj_id, obj in self.perception.objects.items():
            v_pcd = obj.sample_conservative_pcd()
            v_pcd = obj.transform[:3, :3].dot(v_pcd.T).T + obj.transform[:3, 3]
            # v_pcd = occlusion.world_in_voxel_rot.dot(v_pcd.T).T + occlusion.world_in_voxel_tran
            # v_pcd = v_pcd / occlusion.resol
            v_color = np.zeros(v_pcd.shape)
            v_color[:, 0] = 1
            v_color[:, 1] = 0
            v_color[:, 2] = 0
            v_pcds.append(visualize_pcd(v_pcd, v_color))
        o3d.visualization.draw_geometries(v_pcds)
