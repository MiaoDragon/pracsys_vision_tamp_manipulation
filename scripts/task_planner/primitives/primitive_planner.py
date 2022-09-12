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


    def TryMoveOne(self, sinks, probs, pre_grasp_dist=0.02, pre_place_dist=0.08):
        time_infos = []
        for obj_id in np.random.choice(sinks, len(sinks), replace=False, p=np.array(probs)/np.sum(probs)):
            obj = self.perception.objects[obj_id]
            success, info = self.TryMoveOneObject(obj, pre_grasp_dist, pre_place_dist)
            time_infos.append(info)
            if success:
                return True, time_infos
        return False, time_infos

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
                collision_ignored=[obj.pybullet_id]
            )
        t1 = time.time()
        time_info['grasps_gen'] = t1 - t0
        print("Grasp Generation Time: ", time_info['grasps_gen'])

        # planning for each grasp until success
        for poseInfo in filteredPoses:
            # * reset the grasped object pose at the beginning
            if len(poseInfo['collisions']) != 0:
                break

            ## Set Collision Space ##
            self.set_collision_env_with_models(obj.obj_id)
            # input('after setting collision...')
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
            # get gripper to object matrix
            obj_transform = translation_quaternion2homogeneous(
                *p.getBasePositionAndOrientation(obj_local_id, robot.pybullet_id)
            )
            ee_transform = robot.get_tip_link_pose(grasp_joint_dict)
            obj_rel_transform = np.linalg.inv(ee_transform).dot(obj_transform)
            obj2gripper = np.linalg.inv(obj_rel_transform)
            shuffle(placements)


            # remove the object from the scene before attaching
            self.motion_planner.scene_interface.remove_world_object(str(obj.pybullet_id))            


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

                ## Plan Place ##
                t0 = time.time()
                place_joint_dict = robot.joint_vals_to_dict(jointPoses)

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



                t1 = time.time()
                add2dict(time_info, 'place_plan', [t1 - t0])
                print("Place Plan Time: ", time_info['place_plan'][-1])
                if not place_joint_dict_list:
                    tpl1 = time.time()
                    add2dict(time_info, 'total_place', tpl1 - tpl0)
                    continue
                else:
                    break
            if len(place_joint_dict_list) > 0:
                # add object back
                new_robot_pose = self.scene.robot.get_tip_link_pose(place_joint_dict_list[-1])
                
                self.motion_planner.add_cylinder(str(obj.pybullet_id), new_robot_pose.dot(obj_in_gripper), 
                                                obj.obj_model['height'], obj.obj_model['radius'])                
                # self.detach_known(obj)


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


            # otherwise, it is safe to reset the object back to its original place
            self.motion_planner.add_cylinder(str(obj.pybullet_id), obj.transform, 
                                            obj.obj_model['height'], obj.obj_model['radius'])     

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

    def TryMoveOneObject(self, obj, pre_grasp_dist=0.07, pre_place_dist=0.08):
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
                collision_ignored=[obj.pybullet_id]
            )
        t1 = time.time()
        time_info['grasps_gen'] = t1 - t0
        print("Grasp Generation Time: ", time_info['grasps_gen'])

        ## Generate Placements ##
        print('before generating grasp...')

        placements = obj_pose_generation.generate_placements(
            obj,
            robot,
            self.execution,
            self.perception,
            self.scene.workspace,
            # display=True,
        )
        print('after generating grasp...')


        # planning for each grasp until success
        for poseInfo in filteredPoses:
            print('trying poseInfo:')
            if len(poseInfo['collisions']) != 0:
                print('collision happened!')
                break
            ## Set Collision Space ##
            self.set_collision_env_with_models(obj.obj_id)

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

            if len(pick_joint_dict_list) == 0:
                return False, time_info

            ## Plan Lift ##
            new_start_joint_dict = dict(pick_joint_dict_list[-1])
            pick_tip_pose = robot.get_tip_link_pose(new_start_joint_dict)
            lift_tip_pose = np.eye(4)
            lift_tip_pose[:3, 3] = np.array([0, 0, 0.06])  # lift up by 0.04

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


            # input('after picking pose...')
            # * remove object from the scene before attaching it later during placement plan
            self.motion_planner.scene_interface.remove_world_object(str(obj.pybullet_id))                

            # * compute the object pose in the gripper. The object is located at the lifted location
            obj_in_gripper = np.linalg.inv(pick_tip_pose).dot(obj.transform)
            tip_pose = robot.get_tip_link_pose(new_start_joint_dict)
            obj_start_pose = tip_pose.dot(obj_in_gripper)


            for sample_pos in placements:
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


                # compute the relative pose of object in robot ee

                # - add the attached object geometry to the planner: the transform should be the latest transform: i.e. after lifting it up
                aco = self.motion_planner.collision_msg_from_perceive_obj(self.execution.object_state_msg[str(obj.pybullet_id)], 
                                                                        obj_start_pose, link_name=robot.tip_link_name,
                                                                        touch_links=robot.touch_links)

                # aco = self.attach_known(obj, robot, grasp_joint_dict)
                print('entering ee approach_plan...')
                place_joint_dict_list = self.motion_planner.ee_approach_plan(
                    new_start_joint_dict,
                    place_joint_dict,
                    disp_dist=pre_place_dist,
                    disp_dir=(0, 0, -1),
                    is_pre_dir_abs=True,
                    attached_acos=[aco],
                )
                # self.detach_known(obj)

                if len(place_joint_dict_list) == 0:
                    print('place planning failed...')
                    continue
                break
            # input('after placing pose...')

            if len(place_joint_dict) > 0:
                # plan is done
                new_tip_pose = self.scene.robot.get_tip_link_pose(place_joint_dict_list[-1])
                obj_pose = new_tip_pose.dot(obj_in_gripper)
                self.motion_planner.add_cylinder(str(obj.pybullet_id), obj_pose, obj.obj_model['height'], obj.obj_model['radius'])
                # add back the cylinder when the plan succeeded
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


                new_start_joint_dict3 = dict(lift_joint_dict_list2[-1])
                place_tip_pose2 = robot.get_tip_link_pose(new_start_joint_dict3)
                retreat_tip_pose = np.eye(4)
                retreat_tip_pose[:3, 3] = np.array([-0.05, 0, 0.])  # retreat


                lift_joint_dict_list3 = self.motion_planner.straight_line_motion(
                    new_start_joint_dict3,
                    place_tip_pose2,
                    retreat_tip_pose,
                    robot,
                    collision_check=True,
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
                print('after detaching object...')
                self.execution.execute_traj(lift_joint_dict_list2)
                self.execution.execute_traj(lift_joint_dict_list3)
                print('after lifting trajectory...')
                return True, time_info
            # otherwise, reset the object back to its origial place
            self.motion_planner.add_cylinder(str(obj.pybullet_id), obj.transform, obj.obj_model['height'], obj.obj_model['radius'])

        return False, time_info

    def reset(self, joint_dict=None):
        if joint_dict is None:
            joint_dict = self.execution.scene.robot.joint_dict
        
        plan_reset = self.motion_planner.joint_dict_motion_plan(
            joint_dict,
            self.execution.scene.robot.init_joint_dict
        )
        # input('before reset...')
        if len(plan_reset) == 0:
            return False
        self.execution.execute_traj(plan_reset)
        return True

    def pick(self, obj, pre_grasp_dist=0.02):
        robot = self.execution.scene.robot
        obj_local_id = self.execution.object_local_id_dict[str(obj.pybullet_id)]
        ## Grasp ##
        t0 = time.time()
        filteredPoses = obj_pose_generation.geometric_gripper_grasp_pose_generation(
            obj_local_id,
            robot,
            self.scene.workspace,
            offset2=(0, 0, -pre_grasp_dist),
        )
        t1 = time.time()
        print("Grasp Time: ", t1 - t0)
        eof_poses = [
            x['eof_pose_offset'] for x in filteredPoses if len(x['collisions']) == 0
        ]

        ## Set Collision Space ##
        self.set_collision_env_with_models(obj.obj_id)

        # pick poses
        for poseInfo in filteredPoses:
            if len(poseInfo['collisions']) != 0:
                break

            pick_joint_dict = robot.joint_vals_to_dict(poseInfo['dof_joints'])

            ## Plan Pick ##
            t0 = time.time()
            # pick_joint_dict_list = self.motion_planner.joint_dict_motion_plan(
            #     robot.joint_dict,
            #     pick_joint_dict,
            #     robot,
            # )
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
            start_joint_dict = dict(pick_joint_dict_list[-1])
            pick_tip_pose = robot.get_tip_link_pose(start_joint_dict)
            lift_tip_pose = np.eye(4)
            lift_tip_pose[:3, 3] = np.array([0, 0, 0.04])  # lift up by 0.05

            lift_joint_dict_list = self.motion_planner.straight_line_motion(
                start_joint_dict,
                pick_tip_pose,
                lift_tip_pose,
                robot,
                collision_check=False,
                workspace=self.scene.workspace,
                display=False
            )
            # lift_joint_dict_list = self.motion_planner.straight_line_motion2(
            #     start_joint_dict,
            #     direction=(0, 0, 1),
            #     magnitude=0.04,
            # )
            t1 = time.time()
            print("Plan Time: ", t1 - t0)

            ## Execute ##
            print("Succeded to plan to grasp!")
            # self.execution.detach_obj()
            # self.execution.execute_traj(pick_joint_dict_list)
            # self.execution.attach_obj(obj.obj_id)
            # self.execution.execute_traj(lift_joint_dict_list)
            return pick_joint_dict_list, lift_joint_dict_list

        print("Failed to plan to grasp!")
        return [], []



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

    def set_collision_env_with_models_bk(self):
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



    def set_collision_env_with_models(self, obj_id):
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
        print("keys:", list(self.perception.filtered_occluded_dict.keys()))
        print("keys2:", list(self.perception.objects.keys()))
        print("id?:",obj_id)
        self.set_collision_env(
            list(self.perception.objects.keys()),
            [],
            # list(self.perception.filtered_occluded_dict.keys()),
            [obj_id],
            padding=2,
        )



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
            # o3d.visualization.draw_geometries(v_pcds)
