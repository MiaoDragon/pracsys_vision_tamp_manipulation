"""
associate the sensed objects with the recorded objects in the simulator
segmentation_id -> obj_id
this will be based on the pose of the object predicted in the simulator.
TODO: is it possible that we move some objects from the shelf and sense again? Then we need to rule out objects
that are not inside
"""

import numpy as np
from scene.sim_scene import SimScene
from perception.object_belief import ObjectBelief
class CylinderDataAssociation():
    def __init__(self, obj_params: dict):
        self.obj_params = obj_params  # for creating new objects
    def data_association(self, seg_img, sensed_obj_models, existing_obj_models):
        """
        sensed_pose: a list of object poses
        obj_ids: ids of objects that have already been sensed
        obj_poses: pose of objects that have already been sensed
        TODO: segment robot parts and add robot id to seg_img
        """
        seg_ids = list(set(seg_img.reshape(-1).tolist()))
        print('segmentation ids: ')
        print(seg_ids)
        obj_poses = [existing_obj_models[i].transform for i in range(len(existing_obj_models))]  # assuming the object id is indexed as [0,1,2,...]
        obj_poses = np.array(obj_poses)
        obj_ids = np.linspace(0, len(obj_poses), len(obj_poses), endpoint=False).astype(int)
        # new_seg_img = np.zeros(seg_img).astype(int) - 1
        new_seg_img = np.array(seg_img).astype(int)

        new_obj_id = len(obj_ids)  # the id the first new object is going to take

        mapped_obj_models = {}  # it is a dictionary since some objects may be out of view, and
                                # sensed_obj_models may not include all objects

        for i in range(len(seg_ids)):
            if seg_ids[i] == -1:
                continue
            
            # if no objects have been sensed
            if len(existing_obj_models) == 0:
                obj_id = new_obj_id
                new_obj_id += 1
            else:
                sensed_pose = sensed_obj_models[seg_ids[i]]['transform']
                # find the closest one in the scene
                sensed_pose = np.array(sensed_pose)
                diff = sensed_pose[:3,3].reshape((1,-1)) - obj_poses[:,:3,3]
                diff = np.linalg.norm(diff, axis=1)
                print('diff: ')
                print(diff)
                argmin = np.argmin(diff)
                # if the distance is larger than some threshold, then this is a new object
                dis_threshold = 0.05
                print('diff_argmin: ', diff[argmin])
                if diff[argmin] >= dis_threshold:
                    # new object. create a new entry in the object list
                    obj_id = new_obj_id
                    new_obj_id += 1
                else:
                    obj_id = obj_ids[argmin]
            new_seg_img[seg_img==seg_ids[i]] = obj_id
            mapped_obj_models[obj_id] = sensed_obj_models[seg_ids[i]]
        return new_seg_img, mapped_obj_models

