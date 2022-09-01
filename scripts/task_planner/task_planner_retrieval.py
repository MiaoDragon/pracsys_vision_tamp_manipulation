"""
The high-level symbolic task planner, where actions are implemented by
primitives.
Since the task planner does not deal with details of implementations, an
abstract search process can proceed which can generate a skeleton of the
task, later to be verified by lower-level the primitive planner.
"""
import os
import gc
import sys
import time
import json

import cv2
import rospy
import rospkg
import numpy as np
import transformations as tf
import matplotlib.pyplot as plt

from dep_graph import DepGraph
from utils.visual_utils import *
from scene.sim_scene import SimScene
from primitives.primitive_planner import PrimitivePlanner
from perception.perception_system import PerceptionSystem
from primitives.execution_interface import ExecutionInterface


class TaskPlanner():

    def __init__(self, scene_name, prob_id):
        self.prob_id = prob_id
        rp = rospkg.RosPack()
        package_path = rp.get_path('pracsys_vision_tamp_manipulation')
        scene_f = os.path.join(package_path, 'scenes/' + scene_name + '.json')
        f = open(scene_f, 'r')
        scene_dict = json.load(f)

        self.scene = SimScene(scene_dict)

        workspace = self.scene.workspace
        workspace_low = workspace.region_low
        workspace_high = workspace.region_high
        resol = np.array([0.01, 0.01, 0.01])
        world_x = workspace_high[0] - workspace_low[0]
        world_y = workspace_high[1] - workspace_low[1]
        world_z = workspace_high[2] - workspace_low[2]
        x_base = workspace_low[0]
        y_base = workspace_low[1]
        z_base = workspace_low[2]
        x_vec = np.array([1.0, 0., 0.])
        y_vec = np.array([0., 1, 0.])
        z_vec = np.array([0, 0, 1.])

        occlusion_params = {
            'world_x': world_x,
            'world_y': world_y,
            'world_z': world_z,
            'x_base': x_base,
            'y_base': y_base,
            'z_base': z_base,
            'resol': resol,
            'x_vec': x_vec,
            'y_vec': y_vec,
            'z_vec': z_vec
        }
        object_params = {
            'resol': resol,
            'scale': 0.01
        }  # scale is used to scale the depth, not the voxel
        target_params = {'target_pybullet_id': None}

        perception_system = PerceptionSystem(
            occlusion_params, object_params, target_params, self.scene
        )

        execution = ExecutionInterface(self.scene, perception_system)

        dep_graph = DepGraph(perception_system, execution)

        planner = PrimitivePlanner(self.scene, perception_system, execution, dep_graph)

        self.perception = perception_system
        self.execution = execution
        self.planner = planner
        self.dep_graph = dep_graph

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

        self.pipeline_sim()
        self.num_executed_actions = 0
        self.num_collision = 0

    def pipeline_sim(self):
        print("** Perception Started... **")
        self.planner.pipeline_sim()
        # self.perception.pipeline_sim(
        #     self.execution.color_img,
        #     self.execution.depth_img,
        #     self.execution.seg_img,
        #     self.execution.scene.camera,
        #     [self.execution.scene.robot.robot_id],
        #     self.execution.scene.workspace.component_ids,
        # )
        print("** Perception Done! **")

    def run_pipeline(self, ):
        self.dep_graph.first_run()
        # self.execution.target_obj_id = self.dep_graph.target_id
        # self.dep_graph.draw_graph()
        # self.dep_graph.draw_graph(True)

        ### Grasp Sampling Test ###
        print("* Grasp Test *")
        pose_ind = 'q'  # input("Please Enter Object Id: ")
        while pose_ind != 'q':
            try:
                obj_id = int(pose_ind)
                obj = self.perception.objects[obj_id]
            except (IndexError, ValueError, KeyError):
                pose_ind = input("Please Enter Object Id: ")
                continue

            self.planner.grasp_test(obj)
            pose_ind = input("Please Enter Object Id: ")
        ### Grasp Sampling Test End ###

        self.dep_graph.draw_graph()

        ### Pick & Place Test ###
        print("* Pick & Place Test *")
        pose_ind = 'start'
        while pose_ind != 'q':
            pose_ind = input("Please Enter Object Id: ")
            try:
                obj_id = int(pose_ind)
                obj = self.perception.objects[obj_id]
            except (IndexError, ValueError, KeyError):
                continue

            func_choice = input("t -> TryMoveOneObject, m -> MoveOrPlaceback: ")
            if func_choice == 't':
                time_info = self.planner.TryMoveOneObject(obj)
            else:
                time_info = self.planner.MoveOrPlaceback(obj)
            print("\n\nDone:")
            for tt, tm in time_info.items():
                if type(tm) == list:
                    print(f'{tt}: avg={np.average(tm)} std={np.std(tm)} num={len(tm)}')
                else:
                    print(f'{tt}: {tm}')
            plan_reset = self.planner.motion_planner.joint_dict_motion_plan(
                self.execution.scene.robot.joint_dict,
                self.execution.scene.robot.init_joint_dict
            )
            if len(plan_reset) == 0:
                continue
            self.execution.execute_traj(plan_reset)
            if func_choice == 't':
                self.pipeline_sim()
            self.dep_graph.rerun()
            self.dep_graph.draw_graph()
        ### Pick Test End ###


def main():
    rospy.init_node("task_planner")
    rospy.on_shutdown(lambda: os.system('pkill -9 -f task_planner'))
    # rospy.sleep(1.0)
    scene_name = 'scene_table'
    prob_id = sys.argv[1]
    # trial_num = int(sys.argv[2])
    task_planner = TaskPlanner(scene_name, prob_id)
    # input('ENTER to start planning...')
    print('pid: ', task_planner.scene.pid)
    task_planner.run_pipeline()


if __name__ == "__main__":
    main()
