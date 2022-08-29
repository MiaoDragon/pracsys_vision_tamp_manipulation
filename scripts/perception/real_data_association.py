"""
associate the sensed objects with the recorded objects in the simulator
segmentation_id -> obj_id
this will be based on the pose of the object predicted in the simulator.
TODO: is it possible that we move some objects from the shelf and sense again? Then we need to rule out objects
that are not inside
"""

import numpy as np
from scene.sim_scene import SimScene
class CylinderDataAssociation():
    def __init__(self, scene: SimScene):
        self.scene = scene  # the scene includes the object poses, which is updated to match the current scene
    def data_association(self, seg_img, sensed_poses):
        """
        sensed_pose: a list of object poses
        TODO: segment robot parts and add robot id to seg_img
        """
        seg_ids = list(set(seg_img.reshape(-1).tolist()))
        print('segmentation ids: ')
        print(seg_ids)
        obj_ids = []
        obj_poses = []
        for i in range(len(self.scene.objects)):
            # obj_ids.append(self.scene.objects[i].obj_id)
            obj_ids.append(i)

            obj_poses.append(self.scene.objects[i].transform)
        obj_poses = np.array(obj_poses)

        # new_seg_img = np.zeros(seg_img).astype(int) - 1
        new_seg_img = np.array(seg_img).astype(int)

        for i in range(len(seg_ids)):
            if seg_ids[i] == -1:
                continue
            sensed_pose = sensed_poses[seg_ids[i]]
            # find the closest one in the scene
            sensed_pose = np.array(sensed_pose)
            diff = sensed_pose[:3,3].reshape((1,-1)) - obj_poses[:,:3,3]
            diff = np.linalg.norm(diff, axis=1)
            print('diff: ')
            print(diff)
            argmin = np.argmin(diff)
            obj_id = obj_ids[argmin]
            new_seg_img[seg_img==seg_ids[i]] = obj_id
        return new_seg_img

