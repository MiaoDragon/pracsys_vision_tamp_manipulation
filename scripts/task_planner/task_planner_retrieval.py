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
from perception.perception_system import PerceptionSystem
from primitives.primitive_planner import PrimitivePlanner
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
            occlusion_params,
            object_params,
            target_params,
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

        self.num_executed_actions = 0
        self.num_collision = 0

        self.pipeline_sim()
        self.dep_graph.first_run()
        self.execution.target_obj_id = self.dep_graph.target_id
        # self.dep_graph.draw_graph(True)
        # self.dep_graph.draw_graph()

    def pipeline_sim(self):
        print("** Perception Started... **")
        self.perception.p2l_pid = self.execution.object_local_id_dict
        self.planner.pipeline_sim()
        print("** Perception Done! **")

    def reset(self):
        plan_reset = self.planner.motion_planner.joint_dict_motion_plan(
            self.execution.scene.robot.joint_dict,
            self.execution.scene.robot.init_joint_dict
        )
        if len(plan_reset) > 0:
            self.execution.execute_traj(plan_reset)

    def run_pipeline(self, ):
        ### Grasp Sampling Test ###
        print("* Grasp Test *")
        pose_ind = input("Please Enter Object Id: ")
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
                success, time_info = self.planner.TryMoveOneObject(obj)
            else:
                success, time_info = self.planner.MoveOrPlaceback(obj)
            print("\n\nDone:")
            for tt, tm in time_info.items():
                if type(tm) == list:
                    print(f'{tt}: avg={np.average(tm)} std={np.std(tm)} num={len(tm)}')
                else:
                    print(f'{tt}: {tm}')
            self.dep_graph.rerun()
            # self.dep_graph.draw_graph()
        ### Pick Test End ###

    def alg_pipeline(self, timeout=20 * 60):
        TryMoveOne = self.planner.TryMoveOne
        MoveOrPlaceback = self.planner.MoveOrPlaceback
        Retrieve = self.planner.pick

        time_infos = []

        t0 = time.time()
        num_seen = len(self.perception.objects)
        failure = False
        while not failure:
            self.dep_graph.rerun()
            self.dep_graph.draw_graph(
                path=None  #f"{self.prob_id}_{time.strftime('%Y-%m-%d_%H:%M:%S')}.png"
            )
            sinks, probs = self.dep_graph.sinks(lambda x: x**3 + 0.001)
            # sinks, probs = self.dep_graph.sinks(lambda x: x * 0 + 1)
            target = self.dep_graph.target_pid
            print("target?", target, sinks, probs)
            if len(sinks) == 0:
                failure = True
                break
            if target in sinks:
                # self.dep_graph.draw_graph()
                failure = False
                break
            success, info = TryMoveOne(sinks, probs)
            time_infos += info
            if not success:
                failure = True
                for sink in sinks:
                    num_seen = len(self.perception.objects)
                    obj = self.perception.objects[sink]
                    success, info = MoveOrPlaceback(obj)
                    time_infos.append(info)
                    if not success:
                        continue
                    # self.dep_graph.rerun()
                    # new_sinks, new_probs = self.dep_graph.sinks()
                    # self.dep_graph.draw_graph()
                    if len(self.perception.objects) > num_seen:
                        failure = time.time() - t0 > timeout
                        break
                    # ind_ignore = sinks.index(sink)
                    # success, info = TryMoveOne(
                    #     [s for i, s in enumerate(sinks) if i != ind_ignore],
                    #     [p for i, p in enumerate(probs) if i != ind_ignore]
                    # )
                    success, info = TryMoveOne(sinks, probs)
                    time_infos += info
                    if not success:
                        continue
                    failure = time.time() - t0 > timeout
                    break
            if not failure:
                self.planner.MoveEndEffectorAndPercieve()

        if failure:
            time_infos.append({'timed_out': time.time() - t0 > timeout, 'success': False})
            return False, time_infos
        else:
            obj = self.perception.objects[target]
            Retrieve(obj)
            time_infos.append({'timed_out': time.time() - t0 > timeout, 'success': True})
            return True, time_infos

    def alg_random(self, timeout=20 * 60):
        MoveOrPlaceback = self.planner.MoveOrPlaceback
        Retrieve = self.planner.pick

        time_infos = []

        t0 = time.time()
        failure = True
        while time.time() - t0 < timeout:
            self.dep_graph.rerun()
            sinks, probs = self.dep_graph.sinks(lambda x: x * 0 + 1)
            target = self.dep_graph.target_pid
            print("target?", target, sinks, probs)
            if len(sinks) == 0:
                failure = True
                break
            if target in sinks:
                failure = False
                break
            sink = np.random.choice(sinks)
            print(sink)
            obj = self.perception.objects[sink]
            print(obj)
            success, info = MoveOrPlaceback(obj)
            time_infos.append(info)

        if failure:
            time_infos.append({'timed_out': True, 'success': False})
            return False, time_infos
        else:
            obj = self.perception.objects[target]
            Retrieve(obj)
            time_infos.append({'timed_out': False, 'success': True})
            return True, time_infos

    def save_stats(self, time_infos, fname):
        with open(fname, 'w') as f:
            json.dump(time_infos, f)


def main():
    rospy.init_node("task_planner")
    rospy.on_shutdown(lambda: os.system('pkill -9 -f task_planner'))
    scene_name = 'scene_table'
    prob_id = sys.argv[1]
    task_planner = TaskPlanner(scene_name, prob_id)
    print('pid: ', task_planner.scene.pid)
    # task_planner.run_pipeline()

    success, stats = task_planner.alg_pipeline()
    # task_planner.save_stats(stats, prob_id + '_pipeline3.json')
    # task_planner.save_stats(stats, prob_id + '_uniform.json')
    # success, stats = task_planner.alg_random()
    # task_planner.save_stats(stats, prob_id + '_random.json')


if __name__ == "__main__":
    main()
