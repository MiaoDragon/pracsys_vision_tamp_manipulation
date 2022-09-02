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
import std_msgs
import numpy as np
import pybullet as p

from utils.visual_utils import *
from utils.transform_utils import *
from .rearrangement import Rearrangement
from . import utils, obj_pose_generation
from motion_planner.motion_planner import MotionPlanner
from task_planner.primitives.execution_interface import ExecutionInterface
from perception.object_belief import ObjectBelief

class PrimitivePlanner():

    def __init__(
        self, scene, perception_system, execution: ExecutionInterface, dep_graph
    ):
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
            (
                0.014463338682204323, -4.0716884372437e-05, -0.00056967948338301,
                -3.003771794272945e-05, -0.001129114022435695, -1.1408804411786207e-05,
                0.0004592057758004012, 3.0809339998184584e-05, -0.18604427248646432,
                0.9614386497656244, -0.10507968438009953, -1.702685750483462,
                -0.017805293683614262, -0.5223902790606496, 3.461813038728145e-05
            )
        )

    def MoveOrPlaceback(
        self,
        obj: ObjectBelief,
        pre_grasp_dist=0.02,
        lift_height=0.5,
        pre_place_dist=0.08,
    ):
        robot = self.execution.scene.robot
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        time_info = {"success": False}
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
            
            # * reset the grasped object pose at the beginning
            self.motion_planner.move_collision_object(str(obj.pybullet_id), obj.transform)



            if len(poseInfo['collisions']) != 0:
                break

            ## Set Collision Space ##
            self.set_collision_env_with_models()
            input('after setting collision...')
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
            print("Total Place Time: ", time_info['total_pick'])

            ## Plan Intermediate ##
            t0 = time.time()


            # - clear the collision for the object to attach. In moveit the object id is the perception objectbelief id
            # TODO: maybe update attach_known and detach_known so that they handle the object?
            # aco = self.attach_known(obj, robot, new_start_joint_dict)
            self.motion_planner.scene_interface.remove_world_object(str(obj.pybullet_id))
            
            # compute the relative pose of object in robot ee

            # - add the attached object geometry to the planner
            obj_in_gripper = np.linalg.inv(pick_tip_pose).dot(obj.transform)
            aco = self.motion_planner.collision_msg_from_perceive_obj(self.execution.object_state_msg[str(obj.pybullet_id)], 
                                                                      obj.transform)

            inter_joint_dict_list = self.motion_planner.joint_dict_motion_plan(
                lift_joint_dict_list[-1],
                self.intermediate_joint_dict,
                attached_acos=[aco],
            )

            # # - add the object back
            # new_tip_pose = self.scene.robot.get_tip_link_pose(inter_joint_dict_list[-1])
            # obj_pose = new_tip_pose.dot(obj_in_gripper)
            # self.motion_planner.add_cylinder(str(obj.pybullet_id), obj_pose, obj.obj_model['height'], obj.obj_model['radius'])

            # self.detach_known(obj)



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
            tip_pose = self.scene.robot.get_tip_link_pose(pick_joint_dict_list[-1])
            obj_rel_pose = np.linalg.inv(tip_pose).dot(obj.transform)
            self.execution.execute_traj(lift_joint_dict_list)
            self.execution.execute_traj(inter_joint_dict_list)
            t1 = time.time()
            time_info['execute_pick'] = t1 - t0
            print("Execute pick time: ", time_info['execute_pick'])

            ## Update Perception ##
            print("** Perception Started... **")
            t0 = time.time()
            self.pipeline_sim()
            # self.perception.pipeline_sim(
            #     self.execution.color_img,
            #     self.execution.depth_img,
            #     self.execution.seg_img,
            #     self.execution.scene.camera,
            #     [self.execution.scene.robot.robot_id],
            #     self.execution.scene.workspace.component_ids,
            # )
            t1 = time.time()
            time_info['perception'] = t1 - t0
            print("** Perception Done! (", time_info['perception'], ") **")

            ## Generate Placements ##
            t0 = time.time()
            placements = obj_pose_generation.generate_placements(
                obj,
                robot,
                self.execution,
                self.perception,
                self.scene.workspace,
                display=True,
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

                # # reset the object pose at the beginning
                # self.motion_planner.move_collision_object(str(obj.pybullet_id), obj.transform)


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

                ## Plan Place ##
                t0 = time.time()
                place_joint_dict = robot.joint_vals_to_dict(jointPoses)

                # TODO: reset object pose in the while loop before planning
                self.motion_planner.scene_interface.remove_world_object(str(obj.pybullet_id))
                # compute the relative pose of object in robot ee

                # - add the attached object geometry to the planner
                new_robot_pose = self.scene.robot.get_tip_link_pose(new_start_joint_dict)
                aco = self.motion_planner.collision_msg_from_perceive_obj(self.execution.object_state_msg[str(obj.pybullet_id)], 
                                                                        new_robot_pose.dot(obj_in_gripper))

                # aco = self.attach_known(obj, robot, grasp_joint_dict)
                place_joint_dict_list = self.motion_planner.ee_approach_plan(
                    new_start_joint_dict,
                    place_joint_dict,
                    disp_dist=pre_place_dist,
                    disp_dir=(0, 0, -1),
                    is_pre_dir_abs=True,
                    attached_acos=[aco],
                )

                # add object back
                new_robot_pose = self.scene.robot.get_tip_link_pose(place_joint_dict_list[-1])
                
                self.motion_planner.add_cylinder(str(obj.pybullet_id), new_robot_pose.dot(obj_in_gripper), 
                                                obj.obj_model['height'], obj.obj_model['radius'])                
                # self.detach_known(obj)


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

                ## Execute Place ##
                print(f"Succeded to plan move for Object {obj.obj_id}!")
                t0 = time.time()
                self.execution.execute_traj(place_joint_dict_list)
                tip_pose = self.scene.robot.get_tip_link_pose(place_joint_dict_list[-1])
                obj_new_pose = tip_pose.dot(obj_rel_pose)
                obj.update_transform(obj_new_pose)
                # update the object mesh model
                self.execution.update_object_state_from_perception(obj)
                self.execution.detach_obj()
                self.execution.execute_traj(lift_joint_dict_list2)
                t1 = time.time()
                time_info['execute_place'] = t1 - t0
                print("Execute time: ", time_info['execute_place'])
                total1 = time.time()
                time_info['total'] = total1 - total0
                print("Total time: ", time_info['total'])
                return time_info

            ## Execute Reverse Pick ##
            print(f"Failed to plan place for {obj.obj_id}! Putting it back...")
            t0 = time.time()
            self.execution.execute_traj(list(reversed(inter_joint_dict_list)))
            reversed_lift_dict_list = list(reversed(lift_joint_dict_list))
            self.execution.execute_traj(reversed_lift_dict_list)
            tip_pose = self.scene.robot.get_tip_link_pose(reversed_lift_dict_list[-1])
            obj_new_pose = tip_pose.dot(obj_rel_pose)
            obj.update_transform(obj_new_pose)
            # update the object mesh model
            self.execution.update_object_state_from_perception(obj)
            self.execution.detach_obj()
            self.execution.execute_traj(list(reversed(pick_joint_dict_list)))
            t1 = time.time()
            time_info['execute_place'] = t1 - t0
            print("Execute time: ", time_info['execute_place'])
            total1 = time.time()
            time_info['total'] = total1 - total0
            print("Total time: ", time_info['total'])
            return time_info

        total1 = time.time()
        time_info['total'] = total1 - total0
        print("Total time: ", time_info['total'])
        return time_info

    def TryMoveOneObject(self, obj, pre_grasp_dist=0.02, pre_place_dist=0.08):
        robot = self.execution.scene.robot
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        time_info = {"success": False}
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
                offset1=(0, 0, 0.02), # padding for grasping pose
                offset2=(0, 0, -pre_grasp_dist),
            )
        t1 = time.time()
        time_info['grasps_gen'] = t1 - t0
        print("Grasp Generation Time: ", time_info['grasps_gen'])

        ## Generate Placements ##
        placements = obj_pose_generation.generate_placements(
            obj,
            robot,
            self.execution,
            self.perception,
            self.scene.workspace,
            # display=True,
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
            shuffle(placements)
            for sample_pos in placements:
            
                # * reset the grasped object pose at the beginning
                self.motion_planner.move_collision_object(str(obj.pybullet_id), obj.transform)

                print('trying sample...')
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
                robot.set_joints_without_memorize(robot.joint_vals)
                if len(collisions) > 0:
                    continue

                ## Plan Place ##
                place_joint_dict = robot.joint_vals_to_dict(jointPoses)


                self.motion_planner.scene_interface.remove_world_object(str(obj.pybullet_id))
                
                # compute the relative pose of object in robot ee

                # - add the attached object geometry to the planner
                obj_in_gripper = np.linalg.inv(pick_tip_pose).dot(obj.transform)
                aco = self.motion_planner.collision_msg_from_perceive_obj(self.execution.object_state_msg[str(obj.pybullet_id)], 
                                                                        obj.transform)

                # aco = self.attach_known(obj, robot, grasp_joint_dict)
                place_joint_dict_list = self.motion_planner.ee_approach_plan(
                    new_start_joint_dict,
                    place_joint_dict,
                    disp_dist=pre_place_dist,
                    disp_dir=(0, 0, -1),
                    is_pre_dir_abs=True,
                    attached_acos=[aco],
                )

                new_tip_pose = self.scene.robot.get_tip_link_pose(place_joint_dict_list[-1])
                obj_pose = new_tip_pose.dot(obj_in_gripper)
                self.motion_planner.add_cylinder(str(obj.pybullet_id), obj_pose, obj.obj_model['height'], obj.obj_model['radius'])

                # self.detach_known(obj)


                input('after place_joint_dict_list...')

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
                print(f"Succeded to plan move for Object {obj.obj_id}!")
                self.execution.execute_traj(pick_joint_dict_list)
                self.execution.attach_obj(obj.obj_id)
                tip_pose = self.scene.robot.get_tip_link_pose(pick_joint_dict_list[-1])
                obj_rel_pose = np.linalg.inv(tip_pose).dot(obj.transform)
                # self.scene.robot.attach(obj.obj_id, obj_rel_pose)  # obj_rel_pose: object in gripper
                self.execution.execute_traj(lift_joint_dict_list)
                self.execution.execute_traj(place_joint_dict_list)
                tip_pose = self.scene.robot.get_tip_link_pose(place_joint_dict_list[-1])
                obj_new_pose = tip_pose.dot(obj_rel_pose)
                obj.update_transform(obj_new_pose)
                # update the object mesh model
                self.execution.update_object_state_from_perception(obj)
                self.execution.detach_obj()
                self.execution.execute_traj(lift_joint_dict_list2)
                


                return time_info
        return time_info

    def attach_known(self, obj, robot, grasp_joint_dict):
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        # ee_transform_now = robot.get_tip_link_pose(
        #     {key: 0.0
        #      for key in grasp_joint_dict.keys()}
        # )
        ee_transform_now = robot.get_tip_link_pose()
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

    def set_collision_env_with_models(self):
        ## Set Collision Space ##
        obs_msgs = []
        # for obs_id in self.execution.object_state_msg.keys():
        for obs in self.perception.objects.values():
            obs_id = obs.pybullet_id
            # print(obs_id)
            # print(self.execution.object_state_msg[str(obs_id)].name)
            obs_msgs.append(self.execution.object_state_msg[str(obs_id)])
            print('msg for object with id ', obs_id)
            print(self.execution.object_state_msg[str(obs_id)])

        self.motion_planner.set_collision_env_with_models(obs_msgs)

    def pipeline_sim(self, visualize=True):
        # sense & perceive
        # wait for image to update
        print('pipeline_sim...')
        self.execution.timer_cb(None)

        # add the object to pybullet if it is new
        for obj_id, obj in self.perception.objects.items():
            self.execution.update_object_state_from_perception(
                self.perception.objects[obj_id]
            )
            # self.perception.objects[obj_id].pybullet_id = pid

        if visualize:
            v_pcds = []
            for obj_id, obj in self.perception.objects.items():
                obj = self.perception.objects[obj_id]
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
