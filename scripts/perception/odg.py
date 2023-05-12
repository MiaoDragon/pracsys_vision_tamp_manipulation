"""
code related to building the Occlusion Dependency Graph
"""
import numpy as np


class OcclusionDependencyGraph():

    def __init__(self):
        pass

    def partial_odg_from_img(self, depth_img, seg_img):
        """
        given the RGBD images and segmented image, obtain the partial ODG from
        the img. construct a dependency graph for existing objects.
        return a dict:  obj_id -> objects that are hiding it
        """
        obj_ids = set(seg_img.flatten().tolist())
        obj_ids = list(obj_ids)
        # determine hiding relation: the target object shouldn't be hidden and inactive
        # hidden: at least one depth value is larger than a neighboring object depth value

        # determine where there are objects in the segmented img

        # UPDATE: we want to consider robot hiding as well
        obj_seg_filter = np.ones(seg_img.shape).astype(bool)
        obj_seg_filter[seg_img == -1] = 0
        obj_seg_filter[seg_img == -2] = 0

        hiding_objs = {}  # obj_id -> objects that are hiding it
        for obj_id in obj_ids:
            hiding_set = set()
            seged_depth_img = np.zeros(depth_img.shape)
            seged_depth_img[seg_img == obj_id] = depth_img[seg_img == obj_id]
            # obtain indices of the segmented object
            img_i, img_j = np.indices(seg_img.shape)

            def hiding_set_at_boundary(dx, dy):
                valid = (img_i + dx >= 0)
                valid &= (img_i + dx < seg_img.shape[0])
                valid &= (img_j + dy >= 0)
                valid &= (img_j + dy < seg_img.shape[1])
                valid &= (seg_img == obj_id)
                # the neighbor object should be
                # 1. an object (can be robot)
                # 2. not the current object
                filter1 = obj_seg_filter[img_i[valid] + dx, img_j[valid] + dy]
                filter2 = (
                    seg_img[img_i[valid] + dx, img_j[valid] + dy] != obj_id
                )
                depth_filter = depth_img[img_i[valid] + dx, img_j[valid] + dy]
                depth_filter = depth_filter[filter1 & filter2]
                seg_depth_f = depth_img[img_i[valid] + dx, img_j[valid] + dy]
                seg_depth_f = seg_depth_f[filter1 & filter2]
                seg_filter = seg_img[img_i[valid] + dx, img_j[valid] + dy]
                seg_filter = seg_filter[filter1 & filter2]
                hiding_seg_obj_filtered = seg_filter[depth_filter < seg_depth_f]
                return set(hiding_seg_obj_filtered.tolist())

            hiding_set = hiding_set.union(hiding_set_at_boundary(1, 0))
            hiding_set = hiding_set.union(hiding_set_at_boundary(-1, 0))
            hiding_set = hiding_set.union(hiding_set_at_boundary(0, 1))
            hiding_set = hiding_set.union(hiding_set_at_boundary(0, -1))

            # NOTE: hiding_set stores seg_ids, which are pybullet ids instead of obj_id
            # we need to convert them
            hiding_set = list(hiding_set)
            hiding_set = [oid for oid in hiding_set]
            hiding_objs[obj_id] = hiding_set

        return hiding_objs

    def odg_from_obj(self, obj_dict, camera_extrinsics, camera_intrinsics):
        """
        given the current obs of the scene, build an ODG from the objects.
        Pseudo-code:
        - render depth image from the obj pcd for each obj
        - for each obj, check within its depth values, if there are other
          objs that have larger depth. connect to the immediate one to
          build the direct occlusion edge
          (TODO: refer to previous demo video of how I build the graph)
        - build a graph DS
        """
        pass

    def prune_odg_by_obj(self, obj_dict):
        """
        prune the occlusion dependency graph by setting the revealed objects
        as sinks.
        """
        pass
