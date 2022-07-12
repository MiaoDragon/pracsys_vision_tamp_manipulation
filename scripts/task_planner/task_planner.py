"""
The high-level symbolic task planner, where actions are implemented by
primitives.
Since the task planner does not deal with details of implementations, an 
abstract search process can proceed which can generate a skeleton of the
task, later to be verified by lower-level the primitive planner.
"""
import sys
sys.path.insert(0,'/home/yinglong/Documents/research/task_motion_planning/infrastructure/motoman_ws/src/pracsys_vision_tamp_manipulation/scripts')

# import rearrangement_plan
from utils.visual_utils import *

import numpy as np
import transformations as tf
import matplotlib.pyplot as plt
import time
import gc
import cv2
import rospy

import json
from scene.sim_scene import SimScene
from perception.perception_system import PerceptionSystem
from primitives.primitive_planner import PrimitivePlanner
from primitives.execution_interface import ExecutionInterface
import rospkg, os
class TaskPlanner():
    def __init__(self, scene_name, prob_id):
        self.prob_id = prob_id
        rp = rospkg.RosPack()
        package_path = rp.get_path('pracsys_vision_tamp_manipulation')
        scene_f = os.path.join(package_path,'scenes/'+scene_name+'.json')
        f = open(scene_f, 'r')
        scene_dict = json.load(f)

        self.scene = SimScene(scene_dict)


        workspace = self.scene.workspace
        workspace_low = workspace.region_low
        workspace_high = workspace.region_high
        resol = np.array([0.01,0.01,0.01])
        world_x = workspace_high[0]-workspace_low[0]
        world_y = workspace_high[1]-workspace_low[1]
        world_z = workspace_high[2]-workspace_low[2]
        x_base = workspace_low[0]
        y_base = workspace_low[1]
        z_base = workspace_low[2]
        x_vec = np.array([1.0,0.,0.])
        y_vec = np.array([0.,1,0.])
        z_vec = np.array([0,0,1.])

        occlusion_params = {'world_x': world_x, 'world_y': world_y, 'world_z': world_z, 'x_base': x_base, 'y_base': y_base,
                            'z_base': z_base, 'resol': resol, 'x_vec': x_vec, 'y_vec': y_vec, 'z_vec': z_vec}
        object_params = {'resol': resol, 'scale': 0.01}  # scale is used to scale the depth, not the voxel
        target_params = {'target_pybullet_id': None}

        perception_system = PerceptionSystem(occlusion_params, object_params, target_params)

        execution = ExecutionInterface(self.scene, perception_system)

        planner = PrimitivePlanner(self.scene, perception_system, execution)

        self.perception = perception_system
        self.planner = planner
        self.execution = execution


        self.perception_time = 0.0
        self.motion_planning_time = 0.0
        self.pose_generation_time = 0.0
        self.ros_time = 0.0 # communication to execution scene
        self.rearrange_time = 0.0

        self.perception_calls = 0
        self.motion_planning_calls = 0
        self.pose_generation_calls = 0
        self.execution_calls = 0
        self.rearrange_calls = 0

        self.pipeline_sim()
        input('press to start...')
        self.num_executed_actions = 0
        self.num_collision = 0
    
    def pipeline_sim(self):
        self.planner.pipeline_sim()


    def move_and_sense_precheck(self, move_obj_idx, moved_objects):
        """
        check for reconstruction plan
        """
        return self.planner.move_and_sense_precheck(move_obj_idx, moved_objects)

    def move_and_sense(self, move_obj_idx, action_info):
        """
        move the valid object out of the workspace, sense it and the environment, and place back
        """
        return self.planner.move_and_sense(move_obj_idx, action_info)

    def run_pipeline(self, iter_n=10):
        # select object
        #TODO
        moved_objects = []
        iter_i = 0

        valid_objects = []
        start_time = time.time()
        while True:
            print('iteration: ', iter_i)
            gc.collect()
            # select object: active objects but not moved
            active_objs = []
            for obj_id, obj in self.perception.objects.items():
                # check if the object becomes active when the hiding object have all been moved
                # if len(obj.obj_hide_set - set(moved_objects)) == 0:
                #     obj.set_active()
                # UPDATE: we are now using image to decide if an object is active or not

                obj_hide_list = list(obj.obj_hide_set)
                print('object ', self.perception.data_assoc.obj_ids_reverse[obj_id], ' hiding list: ')
                for k in range(len(obj_hide_list)):
                    print(self.perception.data_assoc.obj_ids_reverse[obj_hide_list[k]])
                if obj.active:
                    active_objs.append(obj_id)

                    if (obj_id not in valid_objects) and (obj_id not in moved_objects):
                        valid_objects.append(obj_id)  # a new object that becomes valid
            # valid_objects = set(active_objs) - set(moved_objects)
            # valid_objects = list(valid_objects)

            # move_obj_idx = np.random.choice(valid_objects)
            print('valid object list: ')
            for k in range(len(valid_objects)):
                print(self.perception.data_assoc.obj_ids_reverse[valid_objects[k]])
            # Terminated when there are no valid_objects left
            if len(valid_objects) == 0:
                running_time = time.time() - start_time
                print('#############Finished##############')
                print('number of reconstructed objects: ', len(moved_objects))
                print('number of executed actions: ', self.num_executed_actions)
                print('running time: ', time.time() - start_time, 's')

                import pickle
                f = open(self.prob_name + '-trial-' + str(self.trial_num) + '-result.pkl', 'wb')
                res_dict = {}
                res_dict['num_reconstructed_objs'] = len(moved_objects)
                res_dict['num_collision'] = self.num_collision
                res_dict['running_time'] = running_time
                res_dict['num_executed_actions'] = self.num_executed_actions
                res_dict['perception_time'] = self.perception_time
                res_dict['motion_planning_time'] = self.motion_planning_time
                res_dict['pose_generation_time'] = self.pose_generation_time
                res_dict['rearrange_time'] = self.rearrange_time
                res_dict['ros_time'] = self.ros_time
                res_dict['perception_calls'] = self.perception_calls
                res_dict['motion_planning_calls'] = self.motion_planning_calls
                res_dict['pose_generation_calls'] = self.pose_generation_calls
                res_dict['rearrange_calls'] = self.rearrange_calls
                res_dict['execution_calls'] = self.execution_calls
                # res_dict['rearrange_motion_planning_time'] = self.rearrange_planner.motion_planning_time
                # res_dict['rearrange_pose_generation_time'] = self.rearrange_planner.pose_generation_time
                # res_dict['rearrange_motion_planning_calls'] = self.rearrange_planner.motion_planning_calls
                # res_dict['rearrange_pose_generation_calls'] = self.rearrange_planner.pose_generation_calls

                pickle.dump(res_dict, f)
                f.close()
                # from std_msgs.msg import Int32
                # done_pub = rospy.Publisher('done_msg', Int32)
                # done_pub.publish(Int32(1))

                return

            move_obj_idx = valid_objects[0]

            # if iter_i < len(orders):
            #     move_obj_idx = orders[iter_i]

            # move_obj_idx = 0
            iter_i += 1
            status, planning_info = self.move_and_sense_precheck(move_obj_idx, moved_objects)
            if status:
                status = self.move_and_sense(move_obj_idx, planning_info)
            if status == True:
                moved_objects.append(move_obj_idx)
                valid_objects.pop(0)
            else:
                # put the first element at the back so we can try again
                valid_objects.pop(0)
                valid_objects.append(move_obj_idx)
            self.pipeline_sim()



def main():
    rospy.init_node("task_planner")
    rospy.sleep(1.0)
    scene_name = 'scene2'
    prob_id = sys.argv[1]
    # trial_num = int(sys.argv[2])
    
    task_planner = TaskPlanner(scene_name, prob_id)
    # input('ENTER to start planning...')
    print('pid: ', task_planner.scene.pid)
    task_planner.run_pipeline()

if __name__ == "__main__":
    main()