"""
Implement the graph representing relationships between objects or regions
"""
import time
import copy
from random import choice

import numpy as np
import networkx as nx
from matplotlib.pyplot import show

import cv2
import pybullet as p
import open3d as o3d

from utils.transform_utils import *
from primitives import obj_pose_generation


def perturbation(rmin, rmax, amin=0, amax=2 * np.pi):
    rad = np.random.uniform(rmin, rmax)
    ang = np.random.uniform(amin, amax)
    return np.array([rad * np.cos(ang), rad * np.sin(ang)])


class DepGraph():

    def __init__(
        self,
        perception,
        execution,
    ):

        self.execution = execution
        self.perception = perception
        self.pybullet_id = self.execution.scene.robot.pybullet_id
        self.target_id = None
        self.target_pid = -1
        self.temp_target_id = None
        self.gt_graph = None
        self.graph = None
        self.grasps = None

    def first_run(self):
        self.gen_graph()
        self.select_target()
        self.gen_grasps()
        self.update_belief()

    def rerun(self):
        self.gen_graph()
        self.gen_grasps()
        self.update_belief()

    def select_target(self):
        if self.gt_graph is None:
            print("Generate dep-graph before selecting target.")
            return

        candidates = list(
            filter(
                lambda n: n[1] == 'H',
                self.gt_graph.nodes(data="dname"),
            )
        )
        candidates = candidates if candidates else list(self.gt_graph.nodes(data="dname"))
        # sort by number of depdendencies
        candidates = sorted(
            candidates,
            key=lambda n: len(nx.descendants(self.gt_graph, n[0])),
            reverse=True
        )
        self.target_id, target_name = candidates[0]
        self.gt_graph.nodes[self.target_id]['dname'] = 'T.' + target_name
        try:
            self.target_pid = int(target_name)
        except:
            self.target_pid = -1

    def gen_graph(self):
        self.local2perception = {
            v: str(self.perception.data_assoc.obj_ids.get(int(k), 'H'))
            for k, v in self.execution.object_local_id_dict.items()
        }

        self.gt_graph = nx.DiGraph()
        self.graph = nx.DiGraph()

        num_obj_ignore = 2  # 0 (robot), 1 (table)
        for i in range(num_obj_ignore, p.getNumBodies(physicsClientId=self.pybullet_id)):
            obj_i = p.getBodyUniqueId(i, physicsClientId=self.pybullet_id)
            obj_pi = self.local2perception[obj_i]
            # print(obj_pi)
            if obj_pi != 'H':
                self.graph.add_node(obj_i, dname=obj_pi)
            self.gt_graph.add_node(obj_i, dname=obj_pi)
            for j in range(i + 1, p.getNumBodies(physicsClientId=self.pybullet_id)):
                obj_j = p.getBodyUniqueId(j, physicsClientId=self.pybullet_id)
                obj_pj = self.local2perception[obj_j]
                if obj_pj != 'H':
                    self.graph.add_node(obj_j, dname=obj_pj)
                self.gt_graph.add_node(obj_j, dname=obj_pj)
                contacts = p.getClosestPoints(
                    obj_i,
                    obj_j,
                    distance=0.002,
                    physicsClientId=self.pybullet_id,
                )
                if not contacts:
                    continue
                # print(obj_pi, obj_pj, contacts[0][7][2])
                if obj_pi != 'H' and obj_pj != 'H':
                    if contacts[0][7][2] < -0.999:
                        self.graph.add_edge(obj_i, obj_j, etype="below", w=1)
                    elif contacts[0][7][2] > 0.999:
                        self.graph.add_edge(obj_j, obj_i, etype="below", w=1)
                if contacts[0][7][2] < -0.999:
                    self.gt_graph.add_edge(obj_i, obj_j, etype="below", w=1)
                elif contacts[0][7][2] > 0.999:
                    self.gt_graph.add_edge(obj_j, obj_i, etype="below", w=1)

        if self.target_id:
            self.gt_graph.nodes[self.target_id]['dname'] = \
                    f"T.{self.gt_graph.nodes[self.target_id]['dname']}"
            if self.target_id in self.graph.nodes:
                try:
                    self.target_pid = int(self.local2perception[self.target_id])
                except:
                    self.target_pid = -1
                self.graph.nodes[self.target_id]['dname'] = \
                        f"T.{self.graph.nodes[self.target_id]['dname']}"

    def gen_grasps(self, pre_grasp_dist=0.02):
        robot = self.execution.scene.robot
        time_info = {}

        # pre compute object grasps for visible objects
        self.grasps = {}
        for obj_local_id, obj_perc_id in self.local2perception.items():
            if obj_perc_id == 'H':
                continue
            t0 = time.time()
            grasp_poses = obj_pose_generation.geometric_gripper_grasp_pose_generation(
                obj_local_id,
                robot,
                self.execution.scene.workspace,
                offset2=(0, 0, -pre_grasp_dist),
            )
            t1 = time.time()
            add2dict(time_info, 'grasps_gen', [t1 - t0])
            print("Grasp Generation Time: ", time_info['grasps_gen'][-1])
            self.grasps[obj_local_id] = grasp_poses

            # continue if grasps failed. How would this happen?
            if len(grasp_poses) == 0:
                print("No grasps found for object", obj_perc_id, "‽")
                continue

            # continue if there are no blocking objects
            if len(grasp_poses[0]['collisions']) == 0:
                continue

            # add edges for each blocking object
            edges_to_add = {}
            total = 0
            for poseInfo in grasp_poses:
                for obj_col_id in poseInfo['collisions']:
                    # ignore hidden object collisions
                    if obj_col_id not in self.graph.nodes:
                        continue
                    add2dict(edges_to_add, (obj_local_id, obj_col_id), 1)
                    # print(obj_local_id, obj_col_id)
                total += 1
            for edge, weight in edges_to_add.items():
                # print(edge, weight)
                if edge in self.graph.edges:
                    # print("edge exists")
                    continue
                self.graph.add_edge(*edge, etype="grasp blocked by", w=weight / total)

    def update_belief(self):
        # target visible
        if self.target_id and self.target_id in self.graph.nodes:
            return

        # make sure temp node id doesn't conflict with other nodes
        if self.temp_target_id is None:
            self.temp_target_id = max(self.gt_graph.nodes) + 1

        # remove previous node/edges if any
        if self.temp_target_id in self.graph.nodes:
            self.graph.remove_node(self.temp_target_id)
        self.graph.add_node(self.temp_target_id, dname='T.H')

        total = 0
        for v, n in list(self.graph.nodes(data="dname")):
            # only make edges to source nodes
            if self.graph.in_degree(v) > 0 or v == self.temp_target_id:
                continue

            weight = self.heuristic_volume(v, n)
            if weight == 0:
                continue
            total += weight
            self.graph.add_edge(self.temp_target_id, v, etype="hidden by", w=weight)

        # normalize weights and represent as reciprocal of probability
        update_weights = {}
        for v, n in list(self.graph.nodes(data="dname")):
            edge = (self.temp_target_id, v)
            if edge not in self.graph.edges:
                continue
            w = self.graph.edges[edge]['w']
            update_weights[edge] = {'w': w / total}
        nx.set_edge_attributes(self.graph, update_weights)

    def heuristic_volume(self, v, n, visualize=False):
        '''
        v <- node id
        n <- node name
        '''
        total_occluded = self.perception.filtered_occlusion_label == int(n) + 1
        if visualize:
            print(n, total_occluded.sum(), total_occluded.any(2).sum())
            free_x, free_y = np.where(total_occluded.any(2))
            shape = self.perception.occlusion.occlusion.shape
            img = 255 * np.ones(shape[0:2]).astype('uint8')
            img[free_x, free_y] = 0
            cv2.imshow("Test", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        for x in nx.descendants(self.graph, v):
            o = self.graph.nodes[x]['dname']
            if o == 'H':
                print("Error‽")
            total_occluded |= self.perception.filtered_occlusion_label == int(o) + 1

        if visualize:
            print(n, total_occluded.sum(), total_occluded.any(2).sum())
            free_x, free_y = np.where(total_occluded.any(2))
            shape = self.perception.occlusion.occlusion.shape
            img = 255 * np.ones(shape[0:2]).astype('uint8')
            img[free_x, free_y] = 0
            cv2.imshow("Test", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return total_occluded.sum()

    def pick_order(self, pick_node):
        order = nx.dfs_postorder_nodes(self.graph, source=pick_node)
        ind2name = dict(self.graph.nodes(data="dname"))
        return [ind2name[v] for v in order]

    def sinks(self, curv=lambda x: x + 0.001):
        '''
        return a list of sinks and probabilities for sampling them.
        curv adjusts the shape of the probability distribution of a sinks.
        the 'probability' of a sink means how likely it is to be the optimal choice.
        (this probability is estimated as the sum of probabilities of each simple path to the target, where
        the probability of each path is the product of the belief of each edge)
        '''
        target_id = self.target_id if self.target_id in self.graph.nodes else self.temp_target_id
        sinks = []
        probs = []
        for v, n in list(self.graph.nodes(data="dname")):
            # only look at sink nodes
            if self.graph.out_degree(v) > 0 or v == self.temp_target_id:
                continue
            if 'H' in n:
                print("What the... ‽")
                continue

            # return target if its a sink
            if v == self.target_id:
                return [self.target_pid], [1]

            sum_of_prod = 0
            for paths in nx.all_simple_edge_paths(self.graph, target_id, v):
                prod = 1
                for edge in paths:
                    prod *= self.graph.edges[edge]['w']
                sum_of_prod += prod

            if sum_of_prod > 0:
                sinks.append(int(n))
                probs.append(sum_of_prod)

        print("Probs", probs, sum(probs))
        probs = curv(np.array(probs))  # curve distribution
        probs = probs / sum(probs)  # re-normalize
        return sinks, probs

    def draw_graph(self, ground_truth=False, label="dname"):
        if ground_truth:
            graph = self.gt_graph
        else:
            graph = self.graph

        pos = nx.nx_pydot.graphviz_layout(
            graph,
            # 'fdp',
            # k=1 / len(graph.nodes),
            # pos=pos,
            # fixed=[v for v, d in graph.out_degree if v > 0 and d == 0],
            # iterations=100
        )
        # colors = [
        #     color if color is not None else [1.0, 1.0, 1.0, 1.0]
        #     for color in dict(graph.nodes(data="color")).values()
        # ]
        nx.draw(graph, pos)  # , node_color=colors)
        nx.draw_networkx_labels(graph, pos, dict(graph.nodes(data=label)))
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            {(i, j): t
             for i, j, t in graph.edges(data="etype")},
        )
        nx.draw_networkx_edge_labels(
            graph,
            {k: (v[0], v[1] - 10)
             for k, v in pos.items()},
            {
                (i, j): "" if w is None else np.round(w, 4)
                for i, j, w in graph.edges(data="w")
            },
        )
        show()
