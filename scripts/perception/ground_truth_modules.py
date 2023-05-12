"""
This code provides functions that deal with fake perceptions in simulations.
"""
import rospy
import cv_bridge
import numpy as np
import transformations as tf
from sensor_msgs.msg import Image
from pracsys_vision_tamp_manipulation.msg import ObjectGroundTruthState


class GroundTruthSegmentation():

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()

    def segment_img(self, rgb_img, depth_img):
        seg_img = rospy.wait_for_message('ground_truth/seg_image', Image)
        seg_img = self.bridge.imgmsg_to_cv2(seg_img, 'passthrough')
        # the seg_img gives a picture of geom_ids
        seg_img = np.array(seg_img).astype(int)
        return seg_img


class GroundTruthTracker():
    """
    record the relative transform of the mujoco pose and the object_model pose
    use this relative transform to then obtain the correct pose at each time
    Assuming the tracker obtains a tracked pose world_T_mjk. This later is used
    to update the object belief pose
    """

    def __init__(self):
        # obj_id -> relative transform (NOTE: this is not delta transform)
        self.mjk_T_obj_dict = {}
        # this records the latest pose of objects
        self.world_T_mjk_dict = {}
        self.name_dict = {}

    def track_poses(self, rgb_img, depth_img, obj_dict):
        obj_state = rospy.wait_for_message(
            '/ground_truth/object_state',
            ObjectGroundTruthState,
        )
        zip_state = zip(obj_state.id, obj_state.name, obj_state.pose)
        for oid, name, pose in zip_state:
            self.name_dict[oid] = name
            pos = [
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ]
            ori = [
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ]
            mjk_pose = tf.quaternion_matrix(ori)
            mjk_pose[:3, 3] = np.array(pos)
            self.world_T_mjk_dict[oid] = mjk_pose

    def update_poses(self, obj_dict):
        for oid, world_T_mjk in self.world_T_mjk_dict.items():
            if oid in self.mjk_T_obj_dict:
                # update the pose based on the recorded value
                obj_pose = world_T_mjk.dot(self.mjk_T_obj_dict[oid])
                obj_dict[oid].update_pose(obj_pose)
            else:
                if oid not in obj_dict:
                    continue  # object belief model hasn't been created
                mjk_T_obj = np.linalg.inv(world_T_mjk).dot(
                    obj_dict[oid].world_T_obj
                )
                self.mjk_T_obj_dict[oid] = mjk_T_obj

    def add_new_obj(self, obj_id, obj):
        assert obj_id in self.world_T_mjk_dict  # the new obj should be sensed
        world_T_mjk = self.world_T_mjk_dict[obj_id]
        mjk_T_obj = np.linalg.inv(world_T_mjk).dot(obj.world_T_obj)
        self.mjk_T_obj_dict[id] = mjk_T_obj
