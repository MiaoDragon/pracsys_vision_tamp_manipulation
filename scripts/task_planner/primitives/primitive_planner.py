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
from random import shuffle

import cv2
import rospy
import std_msgs
import numpy as np
import pybullet as p

from utils.visual_utils import *
from utils.transform_utils import *
from .rearrangement import Rearrangement
from . import utils, obj_pose_generation
from motion_planner.motion_planner import MotionPlanner

from sensor_msgs.msg import Image
from pracsys_vision_tamp_manipulation.msg import RobotState


class PrimitivePlanner():

    def __init__(self, scene, perception_system, execution, dep_graph):
        """
        Create a PyBullet scene including workspace, robot and camera
        """
        # load scene definition file
        motion_planner = MotionPlanner(scene.robot, scene.workspace)
        self.scene = scene
        self.perception = perception_system
        self.execution = execution
        self.dep_graph = dep_graph
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

        self.intermediate_joint_dict = self.execution.scene.robot.joint_vals_to_dict(
            [0] * 15
            # (
            #     0.014463338682204323, -4.0716884372437e-05, -0.00056967948338301,
            #     -3.003771794272945e-05, -0.001129114022435695, -1.1408804411786207e-05,
            #     0.0004592057758004012, 3.0809339998184584e-05, -0.18604427248646432,
            #     0.9614386497656244, -0.10507968438009953, -1.702685750483462,
            #     -0.017805293683614262, -0.5223902790606496, 3.461813038728145e-05
            # )
        )

    def TryMoveOne(self, sinks, probs, pre_grasp_dist=0.02, pre_place_dist=0.08):
        time_infos = []
        for obj_id in np.random.choice(sinks, len(sinks), replace=False, p=probs):
            obj = self.perception.objects[obj_id]
            success, info = self.TryMoveOneObject(obj, pre_grasp_dist, pre_place_dist)
            time_infos.append(info)
            if success:
                return True, time_infos
        return False, time_infos

    def MoveOrPlaceback(
        self,
        obj,
        pre_grasp_dist=0.02,
        lift_height=0.5,
        pre_place_dist=0.08,
    ):
        robot = self.execution.scene.robot
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        time_info = {"success": False, "action": "MoveOrPlaceback", "object": obj.obj_id}
        did_uncover = False
        total0 = time.time()

        ## Generate Grasps ##
        t0 = time.time()
        if self.dep_graph and self.dep_graph.grasps and obj_local_id in self.dep_graph.grasps:
            filteredPoses = self.dep_graph.grasps[obj_local_id]
        else:
            filteredPoses = obj_pose_generation.geometric_gripper_grasp_pose_generation(
                obj_local_id,
                robot,
                self.scene.workspace,
                offset2=(0, 0, -pre_grasp_dist),
            )
        t1 = time.time()
        time_info['grasps_gen'] = t1 - t0
        print("Grasp Generation Time: ", time_info['grasps_gen'])

        # planning for each grasp until success
        for poseInfo in filteredPoses:
            if len(poseInfo['collisions']) != 0:
                break

            ## Set Collision Space ##
            self.set_collision_env_with_models(obj.obj_id)

            tpk0 = time.time()
            ## Plan Pick ##
            t0 = time.time()
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
            t1 = time.time()
            add2dict(time_info, 'pick_plan', [t1 - t0])
            print("Pick Plan Time: ", time_info['pick_plan'][-1])
            if not pick_joint_dict_list:
                tpk1 = time.time()
                add2dict(time_info, 'total_pick', tpk1 - tpk0)
                continue

            ## Plan Lift ##
            new_start_joint_dict = dict(pick_joint_dict_list[-1])
            pick_tip_pose = robot.get_tip_link_pose(new_start_joint_dict)
            lift_tip_pose = np.eye(4)
            lift_tip_pose[:3, 3] = np.array([0, 0, lift_height])

            lift_joint_dict_list = self.motion_planner.straight_line_motion(
                new_start_joint_dict,
                pick_tip_pose,
                lift_tip_pose,
                robot,
                collision_check=False,
                workspace=self.scene.workspace,
                display=False
            )
            tpk1 = time.time()
            add2dict(time_info, 'total_pick', tpk1 - tpk0)
            print("Total Pick Time: ", time_info['total_pick'])

            ## Plan Intermediate ##
            t0 = time.time()
            aco = self.attach_known(obj, robot, new_start_joint_dict)
            inter_joint_dict_list = self.motion_planner.joint_dict_motion_plan(
                lift_joint_dict_list[-1],
                self.intermediate_joint_dict,
                attached_acos=[aco],
            )
            self.detach_known(obj)
            t1 = time.time()
            add2dict(time_info, 'inter_plan', [t1 - t0])
            print("Intermediate Plan Time: ", time_info['inter_plan'][-1])
            if not inter_joint_dict_list:
                tpk1 = time.time()
                add2dict(time_info, 'total_pick', tpk1 - tpk0)
                continue

            ## Execute Pick ##
            print(f"Succeded to plan move for Object {obj.obj_id}!")
            t0 = time.time()
            self.execution.detach_obj()
            self.execution.execute_traj(pick_joint_dict_list)
            self.execution.attach_obj(obj.obj_id)
            self.execution.execute_traj(lift_joint_dict_list)
            self.execution.execute_traj(inter_joint_dict_list)
            t1 = time.time()
            time_info['execute_pick'] = t1 - t0
            print("Execute pick time: ", time_info['execute_pick'])

            print("Occluded Before:", self.perception.filtered_occluded.sum())
            prev_volume = self.perception.filtered_occluded.sum()
            ## Update Perception ##
            print("** Perception Started... **")
            t0 = time.time()
            self.pipeline_sim()
            t1 = time.time()
            time_info['perception'] = t1 - t0
            print("** Perception Done! (", time_info['perception'][-1], ") **")
            print("Occluded After:", self.perception.filtered_occluded.sum())
            volume = self.perception.filtered_occluded.sum()
            print("Change:", volume - prev_volume)
            if volume < prev_volume:
                did_uncover = True
            else:
                did_uncover = False

            ## Generate Placements ##
            t0 = time.time()
            placements = obj_pose_generation.generate_placements(
                obj,
                robot,
                self.execution,
                self.perception,
                self.scene.workspace,
                # display=True,
            )
            t1 = time.time()
            time_info['placements_gen'] = t1 - t0
            print("Placement Generation Time: ", time_info['placements_gen'])

            ## Place ##
            new_start_joint_dict, grasp_joint_dict = (
                inter_joint_dict_list[-1],
                inter_joint_dict_list[-1],
                # lift_joint_dict_list[-1],
                # pick_joint_dict_list[-1],
            )
            ## random version ##
            # max_iters = 100
            # count = 0
            # while count < max_iters:
            #     count += 1

            #     # sample placement postition
            #     t0 = time.time()
            #     sample_pos = obj_pose_generation.generate_random_placement(
            #         obj,
            #         robot,
            #         self.execution,
            #         self.perception,
            #         self.scene.workspace,
            #     )
            #     t1 = time.time()
            #     print("Placement Sample Time: ", t1 - t0)

            # get gripper to object matrix
            obj_transform = translation_quaternion2homogeneous(
                *p.getBasePositionAndOrientation(obj_local_id, robot.pybullet_id)
            )
            ee_transform = robot.get_tip_link_pose(grasp_joint_dict)
            obj_rel_transform = np.linalg.inv(ee_transform).dot(obj_transform)
            obj2gripper = np.linalg.inv(obj_rel_transform)
            shuffle(placements)
            for sample_pos in placements:
                tpl0 = time.time()

                # get gripper transform at placement
                place_obj_transform = translation_quaternion2homogeneous(
                    sample_pos,
                    [0, 0, 0, 1],
                )
                gripper_transform = place_obj_transform.dot(obj2gripper)
                pos, rot = homogeneous2translation_quaternion(gripper_transform)

                # check IK
                t0 = time.time()
                valid, jointPoses = robot.get_ik(
                    robot.tip_link_name,
                    pos,
                    rot,
                    robot.init_joint_vals,
                )
                t1 = time.time()
                add2dict(time_info, 'place_ik', [t1 - t0])
                print("Place IK Time: ", time_info['place_ik'][-1])
                if not valid:
                    robot.set_joints_without_memorize(robot.joint_vals)
                    tpl1 = time.time()
                    add2dict(time_info, 'total_place', tpl1 - tpl0)
                    continue

                # check collision
                t0 = time.time()
                ignore_ids = [0, robot.robot_id, obj_local_id]
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
                t1 = time.time()
                add2dict(time_info, 'place_cc', [t1 - t0])
                print("Place Collision Check Time: ", time_info['place_cc'][-1])
                robot.set_joints_without_memorize(robot.joint_vals)
                if len(collisions) > 0:
                    # print("ik failed b/c of collisions:", collisions)
                    tpl1 = time.time()
                    add2dict(time_info, 'total_place', tpl1 - tpl0)
                    continue

                # self.set_collision_env_with_models(obj.obj_id)
                ## Plan Place ##
                t0 = time.time()
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
                t1 = time.time()
                add2dict(time_info, 'place_plan', [t1 - t0])
                print("Place Plan Time: ", time_info['place_plan'][-1])
                if not place_joint_dict_list:
                    tpl1 = time.time()
                    add2dict(time_info, 'total_place', tpl1 - tpl0)
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

                tpl1 = time.time()
                add2dict(time_info, 'total_place', tpl1 - tpl0)
                print("Total Place Time: ", time_info['total_place'])

                time_info['placement'] = sample_pos
                ## Execute Place ##
                print(f"Succeded to plan move for Object {obj.obj_id}!")
                t0 = time.time()
                self.execution.execute_traj(place_joint_dict_list)
                self.execution.detach_obj()
                self.execution.execute_traj(lift_joint_dict_list2)
                t1 = time.time()
                time_info['execute_place'] = t1 - t0
                print("Execute time: ", time_info['execute_place'])
                time_info["success"] = True
                total1 = time.time()
                time_info['total'] = total1 - total0
                print("Total time: ", time_info['total'])
                return True, time_info

            time_info['placement'] = p.getBasePositionAndOrientation(
                obj_local_id, robot.pybullet_id
            )[0]
            ## Execute Reverse Pick ##
            print(f"Failed to plan place for {obj.obj_id}! Putting it back...")
            t0 = time.time()
            self.execution.execute_traj(list(reversed(inter_joint_dict_list)))
            self.execution.execute_traj(list(reversed(lift_joint_dict_list)))
            self.execution.detach_obj()
            self.execution.execute_traj(list(reversed(pick_joint_dict_list)))
            t1 = time.time()
            time_info['execute_place'] = t1 - t0
            print("Execute time: ", time_info['execute_place'])
            time_info["success"] = True
            total1 = time.time()
            time_info['total'] = total1 - total0
            print("Total time: ", time_info['total'])
            return True, time_info

        total1 = time.time()
        time_info['total'] = total1 - total0
        print("Total time: ", time_info['total'])
        return False, time_info

    def TryMoveOneObject(self, obj, pre_grasp_dist=0.02, pre_place_dist=0.08):
        robot = self.execution.scene.robot
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        time_info = {"success": False, "action": "TryMoveOneObject", "object": obj.obj_id}
        total0 = time.time()

        ## Generate Grasps ##
        t0 = time.time()
        if self.dep_graph and self.dep_graph.grasps and obj_local_id in self.dep_graph.grasps:
            filteredPoses = self.dep_graph.grasps[obj_local_id]
        else:
            filteredPoses = obj_pose_generation.geometric_gripper_grasp_pose_generation(
                obj_local_id,
                robot,
                self.scene.workspace,
                offset2=(0, 0, -pre_grasp_dist),
            )
        t1 = time.time()
        time_info['grasps_gen'] = t1 - t0
        print("Grasp Generation Time: ", time_info['grasps_gen'])

        ## Generate Placements ##
        t0 = time.time()
        placements = obj_pose_generation.generate_placements(
            obj,
            robot,
            self.execution,
            self.perception,
            self.scene.workspace,
            # display=True,
        )
        t1 = time.time()
        time_info['placements_gen'] = t1 - t0
        print("Placement Generation Time: ", time_info['placements_gen'])

        # planning for each grasp until success
        for poseInfo in filteredPoses:
            if len(poseInfo['collisions']) != 0:
                break

            ## Set Collision Space ##
            self.set_collision_env_with_models(obj.obj_id)

            tpk0 = time.time()
            ## Plan Pick ##
            t0 = time.time()
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
            t1 = time.time()
            add2dict(time_info, 'pick_plan', [t1 - t0])
            print("Pick Plan Time: ", time_info['pick_plan'][-1])
            if not pick_joint_dict_list:
                tpk1 = time.time()
                add2dict(time_info, 'total_pick', tpk1 - tpk0)
                continue

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
            tpk1 = time.time()
            add2dict(time_info, 'total_pick', tpk1 - tpk0)
            print("Total Pick Time: ", time_info['total_pick'])

            if self.dep_graph and self.dep_graph.target_id and self.dep_graph.target_id in self.dep_graph.graph:
                print("** Target Seen, Skipping Intermediate **")
                new_start_joint_dict = lift_joint_dict_list[-1]
            else:
                ## Plan Intermediate ##
                t0 = time.time()
                aco = self.attach_known(obj, robot, new_start_joint_dict)
                inter_joint_dict_list = self.motion_planner.joint_dict_motion_plan(
                    lift_joint_dict_list[-1],
                    self.intermediate_joint_dict,
                    attached_acos=[aco],
                )
                self.detach_known(obj)
                t1 = time.time()
                add2dict(time_info, 'inter_plan', [t1 - t0])
                print("Intermediate Plan Time: ", time_info['inter_plan'][-1])
                if not inter_joint_dict_list:
                    tpk1 = time.time()
                    add2dict(time_info, 'total_pick', tpk1 - tpk0)
                    continue
                new_start_joint_dict = inter_joint_dict_list[-1]
            ## Place ##
            grasp_joint_dict = pick_joint_dict_list[-1]
            ## random version ##
            # max_iters = 100
            # count = 0
            # while count < max_iters:
            #     count += 1

            #     # sample placement postition
            #     t0 = time.time()
            #     sample_pos = obj_pose_generation.generate_random_placement(
            #         obj,
            #         robot,
            #         self.execution,
            #         self.perception,
            #         self.scene.workspace,
            #     )
            #     t1 = time.time()
            #     print("Placement Sample Time: ", t1 - t0)

            # get gripper to object matrix
            obj_transform = translation_quaternion2homogeneous(
                *p.getBasePositionAndOrientation(obj_local_id, robot.pybullet_id)
            )
            ee_transform = robot.get_tip_link_pose(grasp_joint_dict)
            obj_rel_transform = np.linalg.inv(ee_transform).dot(obj_transform)
            obj2gripper = np.linalg.inv(obj_rel_transform)
            shuffle(placements)
            for sample_pos in placements:
                tpl0 = time.time()

                # get gripper transform at placement
                # obj_transform[:3, 3] = sample_pos
                # gripper_transform = obj_transform.dot(obj2gripper)
                place_obj_transform = translation_quaternion2homogeneous(
                    sample_pos,
                    [0, 0, 0, 1],
                )
                gripper_transform = place_obj_transform.dot(obj2gripper)
                pos, rot = homogeneous2translation_quaternion(gripper_transform)

                # check IK
                t0 = time.time()
                valid, jointPoses = robot.get_ik(
                    robot.tip_link_name,
                    pos,
                    rot,
                    robot.init_joint_vals,
                )
                t1 = time.time()
                add2dict(time_info, 'place_ik', [t1 - t0])
                print("Place IK Time: ", time_info['place_ik'][-1])
                if not valid:
                    robot.set_joints_without_memorize(robot.joint_vals)
                    tpl1 = time.time()
                    add2dict(time_info, 'total_place', tpl1 - tpl0)
                    continue

                # check collision
                t0 = time.time()
                ignore_ids = [0, robot.robot_id, obj_local_id]
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
                t1 = time.time()
                add2dict(time_info, 'place_cc', [t1 - t0])
                print("Place Collision Check Time: ", time_info['place_cc'][-1])
                robot.set_joints_without_memorize(robot.joint_vals)
                if len(collisions) > 0:
                    # print("ik failed b/c of collisions:", collisions)
                    tpl1 = time.time()
                    add2dict(time_info, 'total_place', tpl1 - tpl0)
                    continue

                ## Plan Place ##
                t0 = time.time()
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
                t1 = time.time()
                add2dict(time_info, 'place_plan', [t1 - t0])
                print("Place Plan Time: ", time_info['place_plan'][-1])
                if not place_joint_dict_list:
                    tpl1 = time.time()
                    add2dict(time_info, 'total_place', tpl1 - tpl0)
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

                tpl1 = time.time()
                add2dict(time_info, 'total_place', tpl1 - tpl0)
                print("Total Place Time: ", time_info['total_place'])

                time_info['placement'] = sample_pos
                ## Execute ##
                print(f"Succeded to plan move for Object {obj.obj_id}!")
                t0 = time.time()
                self.execution.detach_obj()
                self.execution.execute_traj(pick_joint_dict_list)
                self.execution.attach_obj(obj.obj_id)
                self.execution.execute_traj(lift_joint_dict_list)

                ## Update Perception ##
                if self.dep_graph and self.dep_graph.target_id and self.dep_graph.target_id in self.dep_graph.graph:
                    print("** Target Seen, Skipping Perception **")
                else:
                    self.execution.execute_traj(inter_joint_dict_list)
                    rospy.sleep(0.001)
                    print("** Perception Started... **")
                    t0 = time.time()
                    self.pipeline_sim()
                    t1 = time.time()
                    time_info['perception'] = t1 - t0
                    print("** Perception Done! (", time_info['perception'][-1], ") **")

                self.execution.execute_traj(place_joint_dict_list)
                self.execution.detach_obj()
                self.execution.execute_traj(lift_joint_dict_list2)
                t1 = time.time()
                time_info['execute'] = t1 - t0
                print("Execute time: ", time_info['execute'])
                time_info["success"] = True
                total1 = time.time()
                time_info['total'] = total1 - total0
                print("Total time: ", time_info['total'])
                return True, time_info

        total1 = time.time()
        time_info['total'] = total1 - total0
        print("Total time: ", time_info['total'])
        return False, time_info

    def grasp_test(self, obj):
        # robot = self.scene.robot
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        robot = self.execution.scene.robot
        # print("***", self.scene == self.execution.scene, "***")
        # print(list(self.perception.filtered_occluded_dict.keys()))

        t0 = time.time()
        filteredPoses = obj_pose_generation.geometric_gripper_grasp_pose_generation(
            obj_local_id,
            robot,
            self.scene.workspace,
            offset2=(0, 0, -0.03),
        )
        t1 = time.time()

        print("Grasp Time: ", t1 - t0)
        for poseInfo in filteredPoses:
            pose = poseInfo['all_joints']
            sparse_pose = poseInfo['dof_joints']
            cols = poseInfo['collisions']
            input("Next?")
            # for iters in range(1000):
            #     robot.setMotors(sparse_pose)
            #     p.stepSimulation()
            # print(sparse_pose)
            robot.set_joints_without_memorize(sparse_pose)
            # print(cols)
        input("Done...")
        robot.set_joints_without_memorize(robot.joint_vals)

    def pick(self, obj, pre_grasp_dist=0.02):
        robot = self.execution.scene.robot
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        time_info = {"success": False, "action": "Retrieve", "object": obj.obj_id}
        total0 = time.time()

        ## Generate Grasps ##
        t0 = time.time()
        if self.dep_graph and self.dep_graph.grasps and obj_local_id in self.dep_graph.grasps:
            filteredPoses = self.dep_graph.grasps[obj_local_id]
        else:
            filteredPoses = obj_pose_generation.geometric_gripper_grasp_pose_generation(
                obj_local_id,
                robot,
                self.scene.workspace,
                offset2=(0, 0, -pre_grasp_dist),
            )
        t1 = time.time()
        time_info['grasps_gen'] = t1 - t0
        print("Grasp Generation Time: ", time_info['grasps_gen'])

        # planning for each grasp until success
        for poseInfo in filteredPoses:
            if len(poseInfo['collisions']) != 0:
                break

            ## Set Collision Space ##
            self.set_collision_env_with_models(obj.obj_id)

            tpk0 = time.time()
            ## Plan Pick ##
            t0 = time.time()
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
            t1 = time.time()
            add2dict(time_info, 'pick_plan', [t1 - t0])
            print("Pick Plan Time: ", time_info['pick_plan'][-1])
            if not pick_joint_dict_list:
                tpk1 = time.time()
                add2dict(time_info, 'total_pick', tpk1 - tpk0)
                continue

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
            tpk1 = time.time()
            add2dict(time_info, 'total_pick', tpk1 - tpk0)
            print("Total Pick Time: ", time_info['total_pick'])
            ## Execute ##
            print(f"Succeded to plan move for Object {obj.obj_id}!")
            t0 = time.time()
            self.execution.detach_obj()
            self.execution.execute_traj(pick_joint_dict_list)
            self.execution.attach_obj(obj.obj_id)
            self.execution.execute_traj(lift_joint_dict_list)
            t1 = time.time()
            time_info['execute'] = t1 - t0
            print("Execute time: ", time_info['execute'])
            time_info["success"] = True
            total1 = time.time()
            time_info['total'] = total1 - total0
            print("Total time: ", time_info['total'])
            return True, time_info

        total1 = time.time()
        time_info['total'] = total1 - total0
        print("Total time: ", time_info['total'])
        return False, time_info

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
        # size = [cmax - cmin for cmin, cmax in zip(mins, maxs)]
        shape = p.getCollisionShapeData(obj_local_id, -1, robot.pybullet_id)[0]
        if shape[2] == p.GEOM_BOX:
            size_x = shape[3][0]
            size_y = shape[3][1]
            size_z = shape[3][2]
        elif shape[2] in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
            size_x = 2 * shape[3][1]
            size_y = size_x
            size_z = shape[3][0]
        else:
            print("Unimplemented Shape!")
        size = (size_x, size_y, size_z)
        return self.motion_planner.attach_known(
            str(obj.pybullet_id), pose=pose, size=size
        )

    def detach_known(self, obj):
        self.motion_planner.detach_known(str(obj.pybullet_id))
        # self.motion_planner.scene_interface.remove_world_object("TEMP_ATTACHED")

    def set_collision_env_with_models(self, obj_id):
        ## cheat for now ##
        self.motion_planner.set_collision_env_with_models(
            self.execution.object_state_msg.values()
        )
        return
        ## Set Collision Space ##
        obs_msgs = []
        # for obs_id in self.execution.object_state_msg.keys():
        for obs in self.perception.objects.values():
            obs_id = obs.pybullet_id
            # print(obs_id)
            # print(self.execution.object_state_msg[str(obs_id)].name)
            obs_msgs.append(self.execution.object_state_msg[str(obs_id)])
        self.motion_planner.set_collision_env_with_models(obs_msgs)
        self.motion_planner.clear_octomap()
        # print("keys:", list(self.perception.filtered_occluded_dict.keys()))
        # print("keys2:", list(self.perception.objects.keys()))
        # print("id?:",obj_id)
        self.set_collision_env(
            list(self.perception.objects.keys()),
            [],
            # list(self.perception.filtered_occluded_dict.keys()),
            [obj_id],
            padding=6,
        )

    def place(self, obj, start_joint_dict, grasp_joint_dict, pre_place_dist=0.08):
        robot = self.execution.scene.robot
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]

        ## Set Collision Space ##
        self.set_collision_env_with_models(obj.obj_id)

        # print("Equal?", self.perception == self.execution.perception)
        t0 = time.time()
        placements = obj_pose_generation.generate_placements(
            obj,
            robot,
            self.execution,
            self.perception,
            self.scene.workspace,
        )
        t1 = time.time()
        print("Placement Sample Time: ", t1 - t0)

        ## Place ##

        ## random version ##
        # max_iters = 100
        # count = 0
        # while count < max_iters:
        #     count += 1

        #     # sample placement postition
        #     t0 = time.time()
        #     sample_pos = obj_pose_generation.generate_random_placement(
        #         obj,
        #         robot,
        #         self.execution,
        #         self.perception,
        #         self.scene.workspace,
        #     )
        #     t1 = time.time()
        #     print("Placement Sample Time: ", t1 - t0)
        for sample_pos in reversed(placements):
            print(sample_pos)

            # get gripper to object matrix
            obj_transform = translation_quaternion2homogeneous(
                *p.getBasePositionAndOrientation(obj_local_id, robot.pybullet_id)
            )
            ee_transform = robot.get_tip_link_pose(grasp_joint_dict)
            obj_rel_transform = np.linalg.inv(ee_transform).dot(obj_transform)

            # get gripper transform at placement
            # print(obj_transform)
            obj_transform[:3, 3] = sample_pos
            # print(obj_transform)
            obj2gripper = np.linalg.inv(obj_rel_transform)
            gripper_transform = obj_transform.dot(obj2gripper)
            pos, rot = homogeneous2translation_quaternion(gripper_transform)

            # check IK
            # robot.set_joints_without_memorize(robot.init_joint_vals)
            valid, jointPoses = robot.get_ik(
                robot.tip_link_name,
                pos,
                rot,
                robot.init_joint_vals,
            )
            if not valid:
                robot.set_joints_without_memorize(robot.joint_vals)
                continue
            ignore_ids = [0, robot.robot_id, obj_local_id]
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
            print("ik failed b/c of collisions:", collisions)
            if len(collisions) > 0:
                robot.set_joints_without_memorize(robot.joint_vals)
                continue

            robot.set_joints_without_memorize(robot.joint_vals)

            place_joint_dict = robot.joint_vals_to_dict(jointPoses)

            ## Plan Place ##
            t0 = time.time()
            aco = self.attach_known(obj, robot, grasp_joint_dict)
            # print(aco)
            # input("?")
            place_joint_dict_list = self.motion_planner.ee_approach_plan(
                start_joint_dict,
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
            new_start_joint_dict = dict(place_joint_dict_list[-1])
            place_tip_pose = robot.get_tip_link_pose(new_start_joint_dict)
            lift_tip_pose = np.eye(4)
            lift_tip_pose[:3, 3] = np.array([0, 0, 0.06])  # lift up by 0.05

            lift_joint_dict_list = self.motion_planner.straight_line_motion(
                new_start_joint_dict,
                place_tip_pose,
                lift_tip_pose,
                robot,
                collision_check=False,
                workspace=self.scene.workspace,
                display=False
            )
            # lift_joint_dict_list = self.motion_planner.straight_line_motion2(
            #     start_joint_dict,
            #     direction=(0, 0, 1),
            #     magnitude=0.06,
            # )
            t1 = time.time()
            print("Plan Time: ", t1 - t0)

            ## Execute ##
            print("Succeded to plan to place!")
            # self.execution.execute_traj(place_joint_dict_list)
            # self.execution.detach_obj()
            # self.execution.execute_traj(lift_joint_dict_list)
            return place_joint_dict_list, lift_joint_dict_list

        print("Failed to plan to place!")
        return [], []

    def plan_to_suction_pose(
        self,
        obj,
        suction_pose_in_obj,
        suction_joint,
        start_joint_dict,
    ):
        # self.motion_planner = motion_planner.MotionPlanner(self.scene.robot, self.scene.workspace)
        suction_joint_dict_list = self.motion_planner.suction_plan(
            start_joint_dict,
            obj.transform.dot(suction_pose_in_obj),
            suction_joint,
            self.scene.robot,
            workspace=self.scene.workspace,
            display=False
        )
        if len(suction_joint_dict_list) == 0:
            return [], []
            # return [], []
        # lift up
        relative_tip_pose = np.eye(4)
        relative_tip_pose[:3, 3] = np.array([0, 0, 0.05])  # lift up by 0.05
        # print('#######################################')
        # print('start joint angle: ')
        # print(self.scene.robot.joint_dict_to_vals(suction_joint_dict_list[-1]))
        joint_dict_list = self.motion_planner.straight_line_motion(
            suction_joint_dict_list[-1],
            obj.transform.dot(suction_pose_in_obj),
            relative_tip_pose,
            self.scene.robot,
            workspace=self.scene.workspace,
            display=False
        )
        # print('straight-line motion, len(joint_dict_list): ', len(joint_dict_list))
        # input('waiting...')
        if len(joint_dict_list) <= 0:
            return [], []

        return suction_joint_dict_list, joint_dict_list

    def plan_to_intermediate_pose(
        self,
        move_obj_idx,
        obj,
        suction_pose_in_obj,
        x_dist,
        intermediate_obj_pose,
        intermediate_joint,
        start_joint_dict,
    ):

        intermediate_joint_dict_list_1 = [start_joint_dict]
        if not obj.sensed:
            # first time to move the object
            current_tip_pose = self.scene.robot.get_tip_link_pose(start_joint_dict)
            relative_tip_pose = np.eye(4)

            relative_tip_pose[:3, 3] = -x_dist
            # relative_tip_pose[:3,3] = tip_pose[:3,3] - current_tip_pose[:3,3]
            relative_tip_pose[1:3, 3] = 0  # only keep the x value

            # self.motion_planner.clear_octomap()
            joint_dict_list = self.motion_planner.straight_line_motion(
                start_joint_dict,
                current_tip_pose,
                relative_tip_pose,
                self.scene.robot,
                collision_check=False,
                workspace=self.scene.workspace,
                display=False
            )

            intermediate_joint_dict_list_1 = joint_dict_list

        # reset collision env: to remove the object to be moved

        # self.set_collision_env(list(self.prev_occluded_dict.keys()), [move_obj_idx], [move_obj_idx], padding=3)

        if len(intermediate_joint_dict_list_1) == 0:
            return []
        joint_dict_list = self.motion_planner.suction_with_obj_plan(
            intermediate_joint_dict_list_1[-1],
            suction_pose_in_obj,
            intermediate_joint,
            self.scene.robot,
            obj,
        )

        gc.collect()

        if len(joint_dict_list) == 0:
            return []
        intermediate_joint_dict_list = intermediate_joint_dict_list_1 + joint_dict_list

        return intermediate_joint_dict_list

    def obj_sense_plan(self, obj, joint_angles, tip_pose_in_obj, start_joint_dict=None):
        if start_joint_dict is None:
            joint_dict_list = self.motion_planner.suction_with_obj_plan(
                self.scene.robot.joint_dict,
                tip_pose_in_obj,
                joint_angles,
                self.scene.robot,
                obj,
            )
        else:
            joint_dict_list = self.motion_planner.suction_with_obj_plan(
                start_joint_dict,
                tip_pose_in_obj,
                joint_angles,
                self.scene.robot,
                obj,
            )
        return joint_dict_list

    def plan_to_placement_pose(
        self,
        obj,
        tip_pose_in_obj,
        intermediate_joint,
        intermediate_joint_dict_list,
        lift_up_joint_dict_list,
        suction_joint_dict_list,
        start_joint_dict,
    ):
        # ** loop until the object is put back
        object_put_back = False
        while True:
            placement_joint_dict_list = []
            reset_joint_dict_list = []
            # if move_obj_transform is None:
            if True:
                # input('valid pose is not found...')
                # * step 1: plan a path to go to the intermediate pose
                # obtain the start tip transform

                # do a motion planning to current sense pose

                joint_dict_list = self.motion_planner.suction_with_obj_plan(
                    start_joint_dict, tip_pose_in_obj, intermediate_joint,
                    self.scene.robot, obj
                )
                if len(joint_dict_list) == 0:
                    continue
                placement_joint_dict_list = joint_dict_list
                placement_joint_dict_list += intermediate_joint_dict_list[::-1]
                placement_joint_dict_list += lift_up_joint_dict_list[::-1]
                reset_joint_dict_list = suction_joint_dict_list[::-1]
                object_put_back = True
                break
        return placement_joint_dict_list, reset_joint_dict_list

    def obtain_straight_blocking_mask(self, target_obj):
        target_pcd = target_obj.sample_conservative_pcd()
        obj_transform = target_obj.transform
        transform = self.perception.occlusion.transform
        transform = np.linalg.inv(transform)
        target_pcd = obj_transform[:3, :3].dot(target_pcd.T).T + obj_transform[:3, 3]
        target_pcd = transform[:3, :3].dot(target_pcd.T).T + transform[:3, 3]
        target_pcd = target_pcd / self.perception.occlusion.resol
        target_pcd = np.floor(target_pcd).astype(int)

        blocking_mask = np.zeros(self.perception.occlusion.voxel_x.shape).astype(bool)
        valid_filter = (target_pcd[:,0]>=0) & (target_pcd[:,0]<blocking_mask.shape[0]) & \
                        (target_pcd[:,1]>=0) & (target_pcd[:,1]<blocking_mask.shape[1]) & \
                        (target_pcd[:,2]>=0) & (target_pcd[:,2]<blocking_mask.shape[2])
        target_pcd = target_pcd[valid_filter]

        blocking_mask[target_pcd[:, 0], target_pcd[:, 1], target_pcd[:, 2]] = 1
        blocking_mask = blocking_mask[::-1, :, :].cumsum(axis=0)
        blocking_mask = blocking_mask[::-1, :, :] > 0

        # remove interior of target_pcd
        blocking_mask = utils.mask_pcd_xy_with_padding(
            blocking_mask,
            target_pcd,
            padding=1,
        )

        del target_pcd
        del valid_filter

        return blocking_mask

    def obtain_visibility_blocking_mask(self, target_obj):
        start_time = time.time()
        camera_extrinsics = self.scene.camera.info['extrinsics']
        cam_transform = np.linalg.inv(camera_extrinsics)
        camera_intrinsics = self.scene.camera.info['intrinsics']
        occlusion = self.perception.occlusion

        pcd = target_obj.sample_conservative_pcd()
        obj_transform = target_obj.transform
        pcd = obj_transform[:3, :3].dot(pcd.T).T + obj_transform[:3, 3]

        transformed_pcd = cam_transform[:3, :3].dot(pcd.T).T + cam_transform[:3, 3]
        fx = camera_intrinsics[0][0]
        fy = camera_intrinsics[1][1]
        cx = camera_intrinsics[0][2]
        cy = camera_intrinsics[1][2]
        transformed_pcd[:, 0] = transformed_pcd[:, 0] / transformed_pcd[:, 2] * fx + cx
        transformed_pcd[:, 1] = transformed_pcd[:, 1] / transformed_pcd[:, 2] * fy + cy
        depth = transformed_pcd[:, 2]
        transformed_pcd = transformed_pcd[:, :2]
        transformed_pcd = np.floor(transformed_pcd).astype(int)
        max_j = transformed_pcd[:, 0].max() + 1
        max_i = transformed_pcd[:, 1].max() + 1

        vis_mask = np.zeros(self.perception.occlusion.voxel_x.shape).astype(bool)
        if max_i <= 0 or max_j <= 0:
            # not in the camera view
            self.perception_time += time.time() - start_time
            self.perception_calls += 1
            del pcd
            del transformed_pcd
            del depth
            return np.zeros(occlusion.voxel_x.shape).astype(bool)

        unique_indices = np.unique(transformed_pcd, axis=0)
        unique_valid = (unique_indices[:, 0] >= 0) & (unique_indices[:, 1] >= 0)
        unique_indices = unique_indices[unique_valid]
        unique_depths = np.zeros(len(unique_indices))
        for i in range(len(unique_indices)):
            unique_depths[i] = depth[ \
                    (transformed_pcd[:, 0] == unique_indices[i, 0]) & \
                    (transformed_pcd[:, 1] == unique_indices[i,1]) \
                ].min()
        depth_img = np.zeros((max_i, max_j)).astype(float)
        depth_img[unique_indices[:, 1], unique_indices[:, 0]] = unique_depths
        depth_img = cv2.medianBlur(np.float32(depth_img), 5)

        # find voxels that can project to the depth
        pt = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0.5, 0.5, 0.5],
            ]
        )

        for i in range(len(pt)):
            voxel_vecs = np.array(
                [occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z]
            ).transpose((1, 2, 3, 0)).reshape(-1, 3)
            voxel_vecs = voxel_vecs + pt[i].reshape(1, -1)  # get the middle point
            voxel_vecs = voxel_vecs * occlusion.resol
            transformed_voxels = occlusion.transform[:3, :3].dot(
                voxel_vecs.T
            ).T + occlusion.transform[:3, 3]
            # get to the image space
            transformed_voxels = cam_transform[:3, :3].dot(transformed_voxels.T
                                                           ).T + cam_transform[:3, 3]

            # cam_to_voxel_dist = np.linalg.norm(transformed_voxels, axis=1)
            cam_to_voxel_depth = np.array(transformed_voxels[:, 2])
            # intrinsics
            cam_intrinsics = camera_intrinsics
            fx = cam_intrinsics[0][0]
            fy = cam_intrinsics[1][1]
            cx = cam_intrinsics[0][2]
            cy = cam_intrinsics[1][2]
            transformed_voxels[:,0] = \
                    transformed_voxels[:,0] / transformed_voxels[:, 2] * fx + cx
            transformed_voxels[:,1] = \
                    transformed_voxels[:,1] / transformed_voxels[:, 2] * fy + cy
            transformed_voxels = np.floor(transformed_voxels).astype(int)
            voxel_depth = np.zeros((len(transformed_voxels)))
            valid_mask = (transformed_voxels[:,0] >= 0) & \
                    (transformed_voxels[:,0] < len(depth_img[0])) & \
                    (transformed_voxels[:,1] >= 0) & \
                    (transformed_voxels[:,1] < len(depth_img))
            voxel_depth[valid_mask] = depth_img[transformed_voxels[valid_mask][:, 1],
                                                transformed_voxels[valid_mask][:, 0]]
            valid_mask = valid_mask.reshape(occlusion.voxel_x.shape)
            voxel_depth = voxel_depth.reshape(occlusion.voxel_x.shape)

            cam_to_voxel_depth = cam_to_voxel_depth.reshape(occlusion.voxel_x.shape)
            vis_mask = vis_mask | (
                (cam_to_voxel_depth - voxel_depth <= 0.) & (voxel_depth > 0.) & valid_mask
            )

        self.perception_calls += 1
        self.perception_time += time.time() - start_time
        # print(occluded.astype(int).sum() / valid_mask.astype(int).sum())
        del cam_to_voxel_depth
        del voxel_depth
        del valid_mask
        del transformed_voxels
        del voxel_vecs
        del pcd
        del transformed_pcd
        del depth

        return vis_mask

    def set_collision_env(
        self,
        occlusion_obj_list,
        ignore_occlusion_list,
        ignore_occupied_list,
        padding=0,
    ):
        """
        providing the object list to check collision and the ignore list, set up the collision environment
        """

        start_time = time.time()
        # occlusion_filter = np.zeros(self.prev_occluded.shape).astype(bool)
        occupied_filter = np.zeros(self.perception.occupied_label_t.shape).astype(bool)

        # occlusion_filter = self.prev_occluded
        occlusion_filter = np.array(self.perception.filtered_occluded)

        for id_o in occlusion_obj_list:
            # if id == move_obj_idx:
            #     continue
            # should include occlusion induced by this object

            if id_o not in ignore_occlusion_list:
                occlusion_filter = occlusion_filter | (
                    self.perception.filtered_occluded_dict[id_o]
                )
            if id_o not in ignore_occupied_list:
                occupied_filter = occupied_filter | (
                    self.perception.occupied_dict_t[id_o]
                )

        # mask out the ignored obj
        if padding > 0:

            for id_i in ignore_occupied_list:
                pcd = self.perception.objects[id_i].sample_conservative_pcd()
                obj_transform = self.perception.objects[id_i].transform
                pcd = obj_transform[:3, :3].dot(pcd.T).T + obj_transform[:3, 3]
                transform = self.perception.occlusion.transform
                transform = np.linalg.inv(transform)
                pcd = transform[:3, :3].dot(pcd.T).T + transform[:3, 3]
                pcd = pcd / self.perception.occlusion.resol

                pcd = np.floor(pcd).astype(int)

                occlusion_filter = utils.mask_pcd_xy_with_padding(
                    occlusion_filter,
                    pcd,
                    padding,
                )
                occupied_filter = utils.mask_pcd_xy_with_padding(
                    occupied_filter,
                    pcd,
                    padding,
                )
                del pcd

        start_time = time.time()
        self.motion_planner.set_collision_env(
            self.perception.occlusion,
            occlusion_filter,
            occupied_filter,
        )
        self.motion_planning_time += time.time() - start_time
        # self.motion_planning_calls += 1
        del occlusion_filter
        del occupied_filter

        gc.collect()

        end_time = time.time()
        print('set_collision_env takes time: ', end_time - start_time)

    def set_collision_env_with_mask(
        self, mask, ignore_obj_list, ignore_obj_pose_list, padding=0
    ):
        """
        given a mask of the occlusion space create collision env. Mask out the pcd in ignore_pcd_list with padding
        """
        # mask out the ignored obj
        if padding > 0:
            for i in range(len(ignore_obj_list)):
                ignore_obj = ignore_obj_list[i]
                pcd = self.perception.objects[ignore_obj].sample_conservative_pcd()
                obj_transform = ignore_obj_pose_list[i]
                pcd = obj_transform[:3, :3].dot(pcd.T).T + obj_transform[:3, 3]
                transform = self.perception.occlusion.transform
                transform = np.linalg.inv(transform)
                pcd = transform[:3, :3].dot(pcd.T).T + transform[:3, 3]
                pcd = pcd / self.perception.occlusion.resol
                pcd = np.floor(pcd).astype(int)
                mask = utils.mask_pcd_xy_with_padding(mask, pcd, padding)
        start_time = time.time()
        self.motion_planner.set_collision_env(
            self.perception.occlusion,
            mask,
            np.zeros(mask.shape).astype(bool),
        )
        self.motion_planning_time += time.time() - start_time

    def sense_object(self, obj_id, camera, robot_ids, component_ids):
        # sense & perceive
        # wait for image to update
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

            mask = (obj.tsdf_count == 0)
            unseen_pcd = obj.sample_pcd(mask)
            unseen_pcd = obj.transform[:3, :3].dot(unseen_pcd.T).T + obj.transform[:3, 3]
            # v_pcd = occlusion.world_in_voxel_rot.dot(v_pcd.T).T + occlusion.world_in_voxel_tran
            # v_pcd = v_pcd / occlusion.resol
            unseen_color = np.zeros(unseen_pcd.shape)
            unseen_color[:, 0] = 0
            unseen_color[:, 1] = 1
            unseen_color[:, 2] = 0
            v_pcds.append(visualize_pcd(unseen_pcd, unseen_color))

        o3d.visualization.draw_geometries(v_pcds)
        # input('press to start...')
        pass
        # start_time = time.time()
        # color_img, depth_img, seg_img = self.execution.get_image()
        # self.perception.sense_object(obj_id, color_img, depth_img, seg_img,
        #                             self.scene.camera, [self.scene.robot.robot_id], self.scene.workspace.component_ids)
        # self.perception_time += time.time() - start_time
        # self.perception_calls += 1

    def pre_move_compute_valid_joints(self, target_obj_i, moved_objects, blocking_mask):
        target_obj = self.perception.objects[target_obj_i]
        occlusion = self.perception.occlusion
        start_time = time.time()
        valid_pts, valid_orientations, valid_joints = obj_pose_generation.grasp_pose_generation(
            target_obj,
            self.scene.robot,
            self.scene.workspace,
            self.perception.occlusion.transform,
            self.perception.filtered_occlusion_label > 0,
            self.perception.occlusion.resol,
            sample_n=20,
        )
        self.pose_generation_time += time.time() - start_time
        self.pose_generation_calls += 1
        print('number of grasp poses obtained: ')
        print(len(valid_pts))
        # * check each sampled pose to see if it's colliding with any objects. Create blocking object set
        total_blocking_objects = []
        total_blocking_object_nums = []
        joint_indices = []
        transformed_rpcds = []

        res_pts = []
        res_orientations = []
        res_joints = []

        # # visualize the perception
        # v_pcds = []
        # for obj_id, obj in self.perception.objects.items():
        #     v_pcd = obj.sample_conservative_pcd()
        #     v_pcd = obj.transform[:3,:3].dot(v_pcd.T).T + obj.transform[:3,3]
        #     v_pcd = occlusion.world_in_voxel_rot.dot(v_pcd.T).T + occlusion.world_in_voxel_tran
        #     v_pcd = v_pcd / occlusion.resol
        #     v_pcds.append(visualize_pcd(v_pcd, [1,0,0]))
        # o3d.visualization.draw_geometries(v_pcds)

        for i in range(len(valid_orientations)):
            # obtain robot pcd at the given joint
            rpcd = self.scene.robot.get_pcd_at_joints(valid_joints[i])
            # robot pcd in the occlusion
            transformed_rpcd = occlusion.world_in_voxel_rot.dot(rpcd.T).T
            transformed_rpcd += occlusion.world_in_voxel_tran
            transformed_rpcd = transformed_rpcd / occlusion.resol
            # trasnformed_rpcd_before_floor = transformed_rpcd

            transformed_rpcd = np.floor(transformed_rpcd).astype(int)
            valid_filter = (transformed_rpcd[:,0] >= 0) & (transformed_rpcd[:,0] < occlusion.voxel_x.shape[0]) & \
                            (transformed_rpcd[:,1] >= 0) & (transformed_rpcd[:,1] < occlusion.voxel_x.shape[1]) & \
                            (transformed_rpcd[:,2] >= 0) & (transformed_rpcd[:,2] < occlusion.voxel_x.shape[2])
            transformed_rpcd = transformed_rpcd[valid_filter]
            transformed_rpcds.append(transformed_rpcd)
            valid = True
            # for vis:
            occupied = np.zeros(self.perception.occupied_label_t.shape).astype(bool)
            occluded = np.zeros(self.perception.occupied_label_t.shape).astype(bool)
            if len(transformed_rpcd) == 0:
                blocking_objects = []
                valid = True
                for obj_i, obj in self.perception.objects.items():
                    if obj_i == target_obj_i:
                        continue

                    occlusion_i = self.perception.filtered_occluded_dict[obj_i]
                    occupied_i = self.perception.occupied_dict_t[obj_i]
                    occupied = occupied | occupied_i  # for vis
                    occluded = occluded | occlusion_i  # for vis

            else:
                # check if colliding with any objects
                blocking_objects = []
                for obj_i, obj in self.perception.objects.items():
                    if obj_i == target_obj_i:
                        continue
                    occlusion_i = self.perception.filtered_occluded_dict[obj_i]
                    occupied_i = self.perception.occupied_dict_t[obj_i]
                    occupied = occupied | occupied_i  # for vis
                    occluded = occluded | occlusion_i  # for vis

                    if occupied_i[transformed_rpcd[:, 0], transformed_rpcd[:, 1],
                                  transformed_rpcd[:, 2]].sum() > 0:
                        blocking_objects.append(obj_i)
                        valid = False
                        # print('blocking with object ', self.perception.data_assoc.obj_ids_reverse[obj_i])
                        # if obj_i in moved_objects:
                        #     print('blocking object is moved before')
                        # else:
                        #     print('blocking object has not been moved')

                # v_voxel = visualize_voxel(self.perception.occlusion.voxel_x, self.perception.occlusion.voxel_y, self.perception.occlusion.voxel_z,
                #                 occupied, [0,0,1])
                # v_pcd = visualize_pcd(transformed_rpcd, [1,0,0])
                # o3d.visualization.draw_geometries([v_voxel, v_pcd])

            # * make sure there is no object in the straight-line path
            for obj_i, obj in self.perception.objects.items():
                if obj_i == target_obj_i:
                    continue
                occupied_i = self.perception.occupied_dict_t[obj_i]
                if (occupied_i & blocking_mask).sum() > 0:
                    blocking_objects.append(obj_i)
                    valid = False
            # * also make sure there is no object in the visibility constraint

            if valid:
                res_pts.append(valid_pts[i])
                res_orientations.append(valid_orientations[i])
                res_joints.append(valid_joints[i])

            # if the blocking objects contain unmoved objects, then give up on this one
            blocking_objects = list(set(blocking_objects))
            if len(set(blocking_objects) - set(moved_objects)) == 0:
                total_blocking_objects.append(blocking_objects)
                total_blocking_object_nums.append(len(blocking_objects))
                joint_indices.append(i)

        if len(res_orientations) > 0:
            return res_pts, res_orientations, res_joints, 1, \
                joint_indices, total_blocking_object_nums, total_blocking_objects, transformed_rpcds

        if len(total_blocking_objects) == 0:
            # failure since all blocking object sets include at least one unmoved objects
            return [], [], [], 0, [], [], [], []

        return valid_pts, valid_orientations, valid_joints, 0, \
            joint_indices, total_blocking_object_nums, total_blocking_objects, transformed_rpcds

    def pre_move(self, target_obj_i, moved_objects):
        """
        before moving the object, check reachability constraints. Rearrange the blocking objects
        """
        # * check reachability constraints by sampling grasp poses
        target_obj = self.perception.objects[target_obj_i]
        occlusion = self.perception.occlusion

        # get the target object pcd
        target_pcd = target_obj.sample_optimistic_pcd()
        target_pcd = target_obj.transform[:3, :3].dot(target_pcd.T).T
        target_pcd += target_obj.transform[:3, 3]
        target_pcd = occlusion.world_in_voxel_rot.dot(target_pcd.T).T
        target_pcd += occlusion.world_in_voxel_tran
        target_pcd = target_pcd / occlusion.resol

        # vis_pcd = visualize_pcd(target_pcd, [0,1,0])

        target_pcd = np.floor(target_pcd).astype(int)
        valid_filter = (target_pcd[:,0]>=0) & (target_pcd[:,0]<occlusion.voxel_x.shape[0]) & \
                        (target_pcd[:,1]>=0) & (target_pcd[:,1]<occlusion.voxel_x.shape[1]) & \
                        (target_pcd[:,2]>=0) & (target_pcd[:,2]<occlusion.voxel_x.shape[2])
        target_pcd = target_pcd[valid_filter]
        # obtain the mask for objects that are hiding in front of target
        # x <= pcd[:,0], y == pcd[:,1], z == pcd[:,2]
        blocking_mask = np.zeros(occlusion.voxel_x.shape).astype(bool)
        blocking_mask[target_pcd[:, 0], target_pcd[:, 1], target_pcd[:, 2]] = 1
        blocking_mask = blocking_mask[::-1, :, :].cumsum(axis=0)
        blocking_mask = blocking_mask[::-1, :, :] > 0

        # * add visibility blocking constraint to the mask
        # NOTE: Update: we need to compute the visibility blocking mask no matter what, because we need
        # to ensure other objects won't be rearranged and block the target object again

        blocking_mask = blocking_mask | self.obtain_visibility_blocking_mask(target_obj)

        target_pcd = target_obj.sample_conservative_pcd()
        target_pcd = target_obj.transform[:3, :3].dot(target_pcd.T).T
        target_pcd += target_obj.transform[:3, 3]
        target_pcd = occlusion.world_in_voxel_rot.dot(target_pcd.T).T
        target_pcd += occlusion.world_in_voxel_tran
        target_pcd = target_pcd / occlusion.resol
        target_pcd = np.floor(target_pcd).astype(int)
        valid_filter = (target_pcd[:,0]>=0) & (target_pcd[:,0]<occlusion.voxel_x.shape[0]) & \
                        (target_pcd[:,1]>=0) & (target_pcd[:,1]<occlusion.voxel_x.shape[1]) & \
                        (target_pcd[:,2]>=0) & (target_pcd[:,2]<occlusion.voxel_x.shape[2])
        target_pcd = target_pcd[valid_filter]
        # remove interior of target_pcd
        blocking_mask = utils.mask_pcd_xy_with_padding(
            blocking_mask,
            target_pcd,
            padding=1,
        )

        valid_pts, \
                valid_orientations, \
                valid_joints, \
                status, \
                joint_indices, \
                total_blocking_object_nums, \
                total_blocking_objects, \
                transformed_rpcds = \
        self.pre_move_compute_valid_joints(
            target_obj_i,
            moved_objects,
            blocking_mask,
        )

        if (status == 1) or (len(valid_pts) == 0):
            return valid_pts, valid_orientations, valid_joints

        # * find the set of blocking objects with the minimum # of objects
        idx = np.argmin(total_blocking_object_nums)
        blocking_objects = total_blocking_objects[idx]
        valid_pt = valid_pts[joint_indices[idx]]
        valid_orientation = valid_orientations[joint_indices[idx]]
        valid_joint = valid_joints[joint_indices[idx]]
        transformed_rpcd = transformed_rpcds[joint_indices[idx]]

        moveable_objs = set(moved_objects) - set(blocking_objects)
        moveable_objs = list(moveable_objs)

        # * construct collision region
        collision_voxel = np.zeros(occlusion.voxel_x.shape).astype(bool)
        for obj_i, obj in self.perception.objects.items():
            if obj_i in moved_objects:
                continue
            collision_voxel = collision_voxel | \
                    self.perception.filtered_occluded_dict[obj_i]

        transform = self.perception.occlusion.transform
        transform = np.linalg.inv(transform)
        voxel_x, voxel_y, voxel_z = np.indices(collision_voxel.shape).astype(int)
        # remove start poses of objects in collision voxel
        for i in range(len(blocking_objects)):
            pcd = self.perception.objects[blocking_objects[i]].sample_conservative_pcd()
            obj_start_pose = self.perception.objects[blocking_objects[i]].transform
            transformed_pcd = obj_start_pose[:3, :3].dot(pcd.T).T + obj_start_pose[:3, 3]
            transformed_pcd = transform[:3, :3].dot(transformed_pcd.T).T
            transformed_pcd += transform[:3, 3]
            transformed_pcd = transformed_pcd / self.perception.occlusion.resol
            transformed_pcd = np.floor(transformed_pcd).astype(int)
            collision_voxel = utils.mask_pcd_xy_with_padding(
                collision_voxel, transformed_pcd, padding=0
            )

            # robot_collision_voxel[transformed_pcd[:,0],transformed_pcd[:,1],transformed_pcd[:,2]] = 0  # mask out
        for i in range(len(moveable_objs)):
            pcd = self.perception.objects[moveable_objs[i]].sample_conservative_pcd()
            obj_start_pose = self.perception.objects[moveable_objs[i]].transform
            transformed_pcd = obj_start_pose[:3, :3].dot(pcd.T).T + obj_start_pose[:3, 3]
            transformed_pcd = transform[:3, :3].dot(transformed_pcd.T).T
            transformed_pcd += transform[:3, 3]
            transformed_pcd = transformed_pcd / self.perception.occlusion.resol
            transformed_pcd = np.floor(transformed_pcd).astype(int)
            collision_voxel = utils.mask_pcd_xy_with_padding(
                collision_voxel,
                transformed_pcd,
                padding=0,
            )

        robot_collision_voxel = np.array(collision_voxel).astype(bool)
        robot_collision_voxel[transformed_rpcd[:, 0], transformed_rpcd[:, 1],
                              transformed_rpcd[:, 2]] = 1

        # add collision to make sure we can extract the object in straight-line
        blocking_mask = self.obtain_straight_blocking_mask(target_obj)
        robot_collision_voxel = robot_collision_voxel | blocking_mask

        # add visibility "collision" to make sure the goal does not hide potential object to be revealed
        blocking_mask = self.obtain_visibility_blocking_mask(target_obj)
        robot_collision_voxel = robot_collision_voxel | blocking_mask

        # * rearrange the blocking objects
        res = self.rearrange(
            blocking_objects,
            moveable_objs,
            collision_voxel,
            robot_collision_voxel,
        )
        del collision_voxel
        del robot_collision_voxel
        del target_pcd
        del valid_filter

        if res:
            # update the occlusion and occupied space so motion planning won't be messed up
            self.pipeline_sim()
            # do another check for blocking objects since we may reveal more grasp poses
            valid_pts, valid_orientations, valid_joints, status, \
                joint_indices, total_blocking_object_nums, \
                total_blocking_objects, transformed_rpcds \
                    = self.pre_move_compute_valid_joints(target_obj_i, moved_objects, blocking_mask)
            del blocking_mask
            if status == 1:
                return [valid_pt] + valid_pts, \
                        [valid_orientation] + valid_orientations, \
                        [valid_joint] + valid_joints
            else:
                return [valid_pt], [valid_orientation], [valid_joint]
        else:
            del blocking_mask
            return [], [], []

    def rearrange(
        self,
        obj_ids,
        moveable_obj_ids,
        collision_voxel,
        robot_collision_voxel,
    ):
        """
        rearrange the blocking objects
        """
        obj_pcds = [
            self.perception.objects[obj_i].sample_conservative_pcd() for obj_i in obj_ids
        ]
        moveable_obj_pcds = [
            self.perception.objects[obj_i].sample_conservative_pcd()
            for obj_i in moveable_obj_ids
        ]
        obj_start_poses = [self.perception.objects[obj_i].transform for obj_i in obj_ids]
        moveable_obj_start_poses = [
            self.perception.objects[obj_i].transform for obj_i in moveable_obj_ids
        ]
        moved_objs = [self.perception.objects[obj_i] for obj_i in obj_ids]
        moveable_objs = [self.perception.objects[obj_i] for obj_i in moveable_obj_ids]

        # plt.ion()
        # plt.figure(figsize=(10,10))
        start_time = time.time()
        searched_objs, transfer_trajs, searched_trajs, reset_traj = self.rearrange_planner.rearrangement_plan(
            moved_objs,
            obj_pcds,
            obj_start_poses,
            moveable_objs,
            moveable_obj_pcds,
            moveable_obj_start_poses,
            collision_voxel,
            robot_collision_voxel,
            self.perception.occlusion.transform,
            self.perception.occlusion.resol,
            self.scene.robot,
            self.scene.workspace,
            self.perception.occlusion,
            self.motion_planner,
            n_iter=10,
        )
        self.rearrange_time += time.time() - start_time
        self.rearrange_calls += 1

        success = False
        if searched_objs is not None:
            total_obj_ids = obj_ids + moveable_obj_ids
            # execute the rearrangement plan
            for i in range(len(searched_objs)):
                move_obj_idx = total_obj_ids[searched_objs[i]]
                self.execution.execute_traj(
                    transfer_trajs[i],
                    self.perception.data_assoc.obj_ids_reverse[move_obj_idx],
                    duration=0.3
                )
                self.execution.attach_obj(move_obj_idx)
                self.execution.execute_traj(searched_trajs[i])
                self.execution.detach_obj()
            # reset
            self.execution.execute_traj(
                reset_traj,
                self.perception.data_assoc.obj_ids_reverse[move_obj_idx],
            )
            success = True
        else:
            success = False
        # input('after rearrange...')

        del obj_pcds
        del moveable_obj_pcds
        del obj_start_poses
        del moveable_obj_start_poses
        del moved_objs
        del moveable_objs

        return success

    def pipeline_sim(self, visualize=False):
        color_msg = rospy.wait_for_message('rgb_image', Image, timeout=10)
        depth_msg = rospy.wait_for_message('depth_image', Image, timeout=10)
        seg_msg = rospy.wait_for_message('seg_image', Image, timeout=10)
        state_msg = rospy.wait_for_message(
            'robot_state_publisher', RobotState, timeout=10
        )
        self.perception.pipeline_sim(
            self.execution.color_img,
            self.execution.depth_img,
            self.execution.seg_img,
            self.execution.scene.camera,
            [self.execution.scene.robot.robot_id],
            self.execution.scene.workspace.component_ids,
        )
        if visualize:
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

    def move_and_sense_precheck(self, move_obj_idx, moved_objects):
        start_time = time.time()
        self.motion_planner.clear_octomap()
        self.motion_planning_time += time.time() - start_time
        # self.motion_planning_calls += 1
        print(
            'handling object: ',
            self.perception.data_assoc.obj_ids_reverse[move_obj_idx],
        )
        start_time = time.time()
        _, suction_poses_in_obj, suction_joints = self.pre_move(
            move_obj_idx,
            moved_objects,
        )
        end_time = time.time()
        print('pre_move takes time: ', end_time - start_time)
        print('number of generated suction poses: ', len(suction_poses_in_obj))
        if len(suction_joints) == 0:  # no valid suction joint now
            return False, None

        start_obj_pose = self.perception.objects[move_obj_idx].transform

        planning_info = dict()
        planning_info['obj_i'] = move_obj_idx

        planning_info['objects'] = self.perception.objects
        planning_info['occlusion'] = self.perception.occlusion
        planning_info['workspace'] = self.scene.workspace
        planning_info['gripper_tip_poses_in_obj'] = suction_poses_in_obj
        planning_info['suction_joints'] = suction_joints

        planning_info['robot'] = self.scene.robot
        # planning_info['motion_planner'] = self.motion_planner
        planning_info['occluded_label'] = self.perception.filtered_occlusion_label
        planning_info['occupied_label'] = self.perception.occupied_label_t
        planning_info['seg_img'] = self.perception.seg_img == move_obj_idx
        planning_info['camera'] = self.scene.camera

        start_time = time.time()
        intermediate_pose, suction_poses_in_obj, suction_joints, intermediate_joints = \
            obj_pose_generation.generate_intermediate_poses(**planning_info)
        end_time = time.time()
        self.pose_generation_time += end_time - start_time
        self.pose_generation_calls += 1
        del planning_info
        # generate intermediate pose for the obj with valid suction pose
        print('generate_intermediate_poses takes time: ', end_time - start_time)

        # set collision environment and reuse afterwards
        # (plan to suction pose, intermediate pose and sense pose do not change collision env)

        # self.set_collision_env_with_mask(collision_voxel, [move_obj_idx],
        #                                 [self.perception.objects[move_obj_idx].transform], padding=3)
        self.set_collision_env(
            list(self.perception.filtered_occluded_dict.keys()),
            [move_obj_idx],
            [move_obj_idx],
            padding=3,
        )

        print('number of suction_poses_in_obj: ', len(suction_poses_in_obj))
        if len(suction_poses_in_obj) == 0:
            return False, None

        for i in range(len(suction_poses_in_obj)):
            suction_pose_in_obj = suction_poses_in_obj[i]
            suction_joint = suction_joints[i]
            intermediate_joint = intermediate_joints[i]

            start_time = time.time()
            pick_joint_dict_list, lift_joint_dict_list = self.plan_to_suction_pose(
                self.perception.objects[move_obj_idx],
                suction_pose_in_obj,
                suction_joint,
                self.scene.robot.joint_dict,
            )  # internally, plan_to_pre_pose, pre_to_suction, lift up

            end_time = time.time()
            self.motion_planning_time += end_time - start_time
            self.motion_planning_calls += 1
            print('plan_to_suction_pose takes time: ', end_time - start_time)

            if len(pick_joint_dict_list) == 0:
                continue

            start_time = time.time()
            # * compute the straight-line extraction distance
            occupied_filter = self.perception.occupied_dict_t[move_obj_idx]
            x_dist = self.perception.occlusion.voxel_x[occupied_filter].max() + 1
            x_dist = x_dist * self.perception.occlusion.resol[0]

            retreat_joint_dict_list = self.plan_to_intermediate_pose(
                move_obj_idx,
                self.perception.objects[move_obj_idx],
                suction_pose_in_obj,
                x_dist,
                intermediate_pose,
                intermediate_joint,
                lift_joint_dict_list[-1],
            )
            end_time = time.time()
            self.motion_planning_time += end_time - start_time
            self.motion_planning_calls += 1
            print('plan_to_intermediate_pose takes time: ', end_time - start_time)

            if len(retreat_joint_dict_list) == 0:
                continue
            # found one valid plan. Record necessary information for future planning
            planning_info = dict()
            planning_info['tip_pose_in_obj'] = suction_pose_in_obj

            planning_info['intermediate_joint'] = intermediate_joint
            planning_info['intermediate_joint_dict_list'] = retreat_joint_dict_list
            planning_info['lift_up_joint_dict_list'] = lift_joint_dict_list
            planning_info['suction_joint_dict_list'] = pick_joint_dict_list
            planning_info['obj'] = self.perception.objects[move_obj_idx]

            return True, planning_info
        return False, None

    def move_and_sense(self, move_obj_idx, action_info):
        """
        move the valid object out of the workspace, sense it and the environment, and place back
        """
        start_time = time.time()
        self.motion_planner.clear_octomap()
        self.motion_planning_time += time.time() - start_time
        planning_info = action_info
        suction_pose_in_obj = planning_info['tip_pose_in_obj']

        retreat_joint_dict_list = planning_info['intermediate_joint_dict_list']
        lift_joint_dict_list = planning_info['lift_up_joint_dict_list']
        pick_joint_dict_list = planning_info['suction_joint_dict_list']
        self.execution.execute_traj(
            pick_joint_dict_list,
            self.perception.data_assoc.obj_ids_reverse[move_obj_idx],
            duration=0.3
        )

        self.execution.attach_obj(move_obj_idx)
        self.execution.execute_traj(lift_joint_dict_list + retreat_joint_dict_list)

        self.pipeline_sim()  # sense the environmnet

        last_joint_dict = retreat_joint_dict_list[-1]
        for k in range(6):
            planning_info = dict()
            planning_info['obj_i'] = move_obj_idx
            planning_info['objects'] = self.perception.objects
            planning_info['occlusion'] = self.perception.occlusion
            planning_info['workspace'] = self.scene.workspace
            planning_info['selected_tip_in_obj'] = suction_pose_in_obj
            planning_info['joint_dict'] = last_joint_dict

            planning_info['robot'] = self.scene.robot
            planning_info['occluded_label'] = self.perception.filtered_occlusion_label
            planning_info['occupied_label'] = self.perception.occupied_label_t
            # TODO: segmentation now does not have access to PyBullet seg id
            planning_info['camera'] = self.scene.camera
            # self.motion_planner.clear_octomap()
            start_time = time.time()
            print('sample_sense_pose...')

            sense_pose, selected_tip_in_obj, tip_pose, start_joint_angles, joint_angles = \
                obj_pose_generation.sample_sense_pose(**planning_info)
            print('after sample_sense_pose...')

            self.pose_generation_time += time.time() - start_time
            self.pose_generation_calls += 1
            print('sample sense pose takes time: ', time.time() - start_time)
            start_time = time.time()
            obj_sense_joint_dict_list = self.obj_sense_plan(
                self.perception.objects[move_obj_idx],
                joint_angles,
                suction_pose_in_obj,
                last_joint_dict,
            )
            end_time = time.time()

            self.motion_planning_time += end_time - start_time
            self.motion_planning_calls += 1
            print('obj_sense_plan takes time: ', end_time - start_time)

            self.execution.execute_traj(obj_sense_joint_dict_list)

            start_time = time.time()
            self.sense_object(
                move_obj_idx,
                self.scene.camera,
                [self.scene.robot.robot_id],
                self.scene.workspace.component_ids,
            )
            end_time = time.time()
            print('sense_object takes time: ', end_time - start_time)

            if len(obj_sense_joint_dict_list) == 0:
                continue

            # rotate the object 360 degrees so we get a better sensing
            last_joint_dict = obj_sense_joint_dict_list[-1]
            complete = self.perception.objects[move_obj_idx].check_complete()
            if complete:
                break

            ul = self.scene.robot.upper_lim[7]
            ll = self.scene.robot.lower_lim[7]
            current_angle = last_joint_dict[self.scene.robot.joint_names[7]]
            # current_angle = self.scene.robot.joint_vals[7]
            waypoint1 = current_angle + np.pi / 2
            waypoint2 = current_angle + np.pi
            waypoint3 = current_angle - np.pi / 2

            # make sure the waypoints are within limit
            # first put in the range of -pi to pi
            waypoint1 = utils.wrap_angle(waypoint1, ll, ul)
            waypoint2 = utils.wrap_angle(waypoint2, ll, ul)
            waypoint3 = utils.wrap_angle(waypoint3, ll, ul)

            # generate rotation trajectory
            traj1 = utils.generate_rot_traj(
                self.scene.robot.joint_names[7],
                last_joint_dict,
                waypoint1,
            )
            self.execution.execute_traj(traj1)
            # self.pipeline_sim()
            self.sense_object(
                move_obj_idx,
                self.scene.camera,
                [self.scene.robot.robot_id],
                self.scene.workspace.component_ids,
            )

            last_joint_dict = traj1[-1]
            complete = self.perception.objects[move_obj_idx].check_complete()
            if complete:
                break

            traj2 = utils.generate_rot_traj(
                self.scene.robot.joint_names[7], traj1[-1], waypoint2
            )
            self.execution.execute_traj(traj2)
            # self.pipeline_sim()
            self.sense_object(
                move_obj_idx,
                self.scene.camera,
                [self.scene.robot.robot_id],
                self.scene.workspace.component_ids,
            )
            last_joint_dict = traj2[-1]
            complete = self.perception.objects[move_obj_idx].check_complete()
            if complete:
                break

            traj3 = utils.generate_rot_traj(
                self.scene.robot.joint_names[7],
                traj2[-1],
                waypoint3,
            )
            self.execution.execute_traj(traj3)
            # self.pipeline_sim()
            self.sense_object(
                move_obj_idx,
                self.scene.camera,
                [self.scene.robot.robot_id],
                self.scene.workspace.component_ids,
            )
            last_joint_dict = traj3[-1]
            complete = self.perception.objects[move_obj_idx].check_complete()
            if complete:
                break

            traj4 = utils.generate_rot_traj(
                self.scene.robot.joint_names[7],
                traj3[-1],
                current_angle,
            )
            self.execution.execute_traj(traj4)
            last_joint_dict = traj4[-1]
            complete = self.perception.objects[move_obj_idx].check_complete()
            if complete:
                break
        self.perception.objects[move_obj_idx].set_sensed()

        planning_info = dict()
        planning_info['tip_pose_in_obj'] = action_info['tip_pose_in_obj']

        planning_info['intermediate_joint'] = action_info['intermediate_joint']
        planning_info['intermediate_joint_dict_list'] = \
                action_info['intermediate_joint_dict_list']
        planning_info['lift_up_joint_dict_list'] = action_info['lift_up_joint_dict_list']
        planning_info['suction_joint_dict_list'] = action_info['suction_joint_dict_list']
        planning_info['obj'] = action_info['obj']
        planning_info['start_joint_dict'] = last_joint_dict

        start_time = time.time()
        placement_joint_dict_list, reset_joint_dict_list = self.plan_to_placement_pose(
            **planning_info
        )
        end_time = time.time()
        self.motion_planning_time += end_time - start_time
        self.motion_planning_calls += 1
        print('plan_to_placement_pose takes time: ', end_time - start_time)

        self.execution.execute_traj(placement_joint_dict_list)
        self.execution.detach_obj()

        self.execution.execute_traj(
            reset_joint_dict_list,
            self.perception.data_assoc.obj_ids_reverse[move_obj_idx],
        )
        return True
