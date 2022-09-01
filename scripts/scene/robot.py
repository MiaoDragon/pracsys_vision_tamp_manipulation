"""
This script defines a robot object
"""
import pybullet as p
import numpy as np
import transformations as tf
from urdfpy import URDF


class Robot():

    def __init__(
        self,
        urdf,
        pos,
        ori,
        lower_lim,
        upper_lim,
        joint_range,
        tip_link_name,
        tip_joint_name,
        ee_params,
        pybullet_id,
        active_joint_mask
    ):
        """
        given the URDF file, the pose of the robot base, and the joint angles,
        initialize a robot model in the scene

        NOTE: lower_lim and upper_lim contain finger tip joints too due to PyBullet bug in IK
        """
        robot_id = p.loadURDF(
            urdf,
            pos,
            ori,
            useFixedBase=True,
            flags=p.URDF_USE_IMPLICIT_CYLINDER | p.URDF_INITIALIZE_SAT_FEATURES,
            physicsClientId=pybullet_id,
        )
        # get the number of active joints
        num_joints = p.getNumJoints(robot_id, pybullet_id)
        # joint_dict = {}
        joint_names = []
        joint_indices = []
        joint_name_ind_dict = {}
        joint_vals = []  # for the DOF
        joint_dict = {}
        total_joint_name_ind_dict = {}
        total_link_name_ind_dict = {}
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i, pybullet_id)
            name = info[1].decode("utf-8")
            joint_type = info[2]
            total_joint_name_ind_dict[name] = i
            total_link_name_ind_dict[info[12].decode("utf-8")] = i
            if joint_type == p.JOINT_FIXED:
                # fixed joint does not belong to DOF
                continue
            if 'arm' not in name and 'torso_joint_b1' != name:
                # not arm joints
                continue
            # joint_dict[name] =
            joint_names.append(name)
            joint_indices.append(i)
            joint_name_ind_dict[name] = i

            state = p.getJointState(robot_id, i, pybullet_id)
            joint_vals.append(state[0])
            joint_dict[name] = state[0]

        # print('joint names: ')
        # print(joint_names)
        self.robot_id = robot_id
        self.num_joints = num_joints
        self.joint_names = joint_names
        self.pybullet_id = pybullet_id
        self.suction_length = ee_params
        self.gripper_width = ee_params

        self.total_link_name_ind_dict = total_link_name_ind_dict
        self.total_joint_name_ind_dict = total_joint_name_ind_dict
        self.joint_name_to_idx = joint_name_ind_dict
        self.joint_indices = joint_indices
        self.joint_names = joint_names
        self.joint_vals = joint_vals
        self.joint_dict = joint_dict
        self.init_joint_vals = list(joint_vals)
        self.init_joint_dict = dict(joint_dict)

        self.init_joint_vals_np = np.array(list(joint_vals))
        self.active_joint_mask = np.array(active_joint_mask).astype(
            bool
        )  
        self.lower_lim = np.array(lower_lim)
        self.upper_lim = np.array(upper_lim)

        self.sample_lower_lim = np.array(lower_lim)[:len(active_joint_mask)]
        self.sample_upper_lim = np.array(upper_lim)[:len(active_joint_mask)]


        self.jr = joint_range

        self.tip_link_name = tip_link_name

        self.transform = tf.quaternion_matrix([ori[3], ori[0], ori[1], ori[2]])
        self.transform[:3, 3] = pos

        self.world_in_robot = np.linalg.inv(self.transform)

        # obtain the geometry information from URDF
        # store the pcd of each component, with transform relative to each link
        # to get the actual pcd in the world, multiply the pcd by the link transform
        robot = URDF.load(urdf)
        """
        to check collision of robot with voxel, it might be more efficient to check pcd vs. voxel
        """
        pcd_link_dict = {}
        link_dict = {}
        for link in robot.links:
            # print('link name: ', link.name)
            collisions = link.collisions
            link_dict[link.name] = link

            if len(collisions) == 0:
                continue
            link_pcd = []  # one link may have several collision objects
            for collision in collisions:
                origin = collision.origin  # this is the relative transform to get the pose of the geometry (mesh)
                # pose of the trimesh:
                # pose of link * origin * scale * trimesh_obj
                geometry = collision.geometry.geometry
                # print('geometry: ', geometry)
                # print('tag: ', geometry._TAG)
                # geometry.scale: give us the scale for the mesh

                meshes = geometry.meshes
                for mesh in meshes:
                    # print('mesh vertices: ')
                    # print(mesh.vertices)
                    # mesh.sample()
                    pcd = mesh.sample(len(mesh.vertices) * 5)
                    if collision.geometry.mesh is not None:
                        if collision.geometry.mesh.scale is not None:
                            pcd = pcd * collision.geometry.mesh.scale
                    pcd = origin[:3, :3].dot(pcd.T).T + origin[:3, 3]
                    link_pcd.append(pcd)
            link_pcd = np.concatenate(link_pcd, axis=0)
            # print('link_pcd shape: ', link_pcd.shape)
            pcd_link_dict[link.name] = link_pcd
        self.pcd_link_dict = pcd_link_dict
        self.link_dict = link_dict  # urdfpy object
        self.urdfpy_robot = robot

        # setup attached object
        self.attached_obj_id = None
        self.attached_obj_rel_pose = None  # ee T obj



    def set_init_joints(self, joints=None):
        if joints is None:
            self.init_joint_vals = list(self.joint_vals)
            self.init_joint_dict = dict(self.joint_dict)

        if isinstance(joints, dict):
            self.init_joint_dict = dict(joints)
            self.init_joint_vals = self.joint_dict_to_vals(joints)
            self.init_joint_vals_np = np.array(list(self.init_joint_vals))
        else:
            self.init_joint_vals = np.array(joints)
            self.init_joint_dict = self.joint_vals_to_dict(joints)
            self.init_joint_vals_np = np.array(joints)


    def attach(self, obj_id, obj_rel_pose):
        self.attached_obj_id = obj_id
        self.attached_obj_rel_pose = obj_rel_pose

    def detach(self):
        self.attached_obj_id = None
        self.attached_obj_rel_pose = None

    def attached(self):
        # check whether the robot is attached or not
        return (self.attached_obj_id is not None)

    def get_link_pcd_at_joints(self, joints, link_name):
        """
        useful if we want to only extract part of the pcd
        """
        pass

    def get_pcd_at_joints(self, joints):
        """
        given joint value list, return the pcd of the robot
        """
        # set pybullet joints to the joint values
        self.set_joints_without_memorize(joints)
        total_pcd = []
        for link_name, link_idx in self.total_link_name_ind_dict.items():
            if link_name not in self.pcd_link_dict:
                # print('link name ', link_name, 'not in pcd_link_dict')
                continue
            T = self.get_link_pose(link_name)
            total_pcd.append(T[:3, :3].dot(self.pcd_link_dict[link_name].T).T + T[:3, 3])
        total_pcd = np.concatenate(total_pcd, axis=0)
        self.set_joints_without_memorize(self.joint_vals)
        return total_pcd

    def set_motion_planner(self, motion_planner):
        self.motion_planner = motion_planner

    def set_joints_without_memorize(self, joints):
        if isinstance(joints, dict):
            # dict
            for i in range(len(self.joint_indices)):
                joint_idx = self.joint_indices[i]
                joint_name = self.joint_names[i]
                if joint_name in joints:
                    p.resetJointState(
                        self.robot_id, joint_idx, joints[joint_name], 0., self.pybullet_id
                    )
        else:
            # list or numpy array
            for i in range(len(self.joint_indices)):
                joint_idx = self.joint_indices[i]
                p.resetJointState(
                    self.robot_id, joint_idx, joints[i], 0., self.pybullet_id
                )

    def set_joints(self, joints):
        for i in range(len(self.joint_indices)):
            joint_idx = self.joint_indices[i]
            p.resetJointState(self.robot_id, joint_idx, joints[i], 0., self.pybullet_id)
            self.joint_dict[self.joint_names[i]] = joints[i]
        # reset joints
        self.joint_vals = joints

    def set_joint_from_dict(self, joint_dict):
        joints = []
        for i in range(len(self.joint_indices)):
            joint_idx = self.joint_indices[i]
            joint_name = self.joint_names[i]
            if joint_name in joint_dict:
                p.resetJointState(
                    self.robot_id, joint_idx, joint_dict[joint_name], 0., self.pybullet_id
                )
                joints.append(joint_dict[joint_name])
                self.joint_dict[joint_name] = joint_dict[joint_name]
            else:
                joints.append(self.joint_vals[i])

        self.joint_vals = joints

    def set_joint_from_dict_data(self, joint_dict):
        # only update the data, but not updating the pybullet scene
        joints = []
        for i in range(len(self.joint_indices)):
            joint_name = self.joint_names[i]
            if joint_name in joint_dict:
                joints.append(joint_dict[joint_name])
                self.joint_dict[joint_name] = joint_dict[joint_name]
            else:
                joints.append(self.joint_vals[i])


    def joint_dict_list_to_val_list(self, joint_dict_list):
        joint_vals = [self.joint_dict_to_vals(joint_dict_list[i]) for i in range(len(joint_dict_list))]
        return joint_vals

    def joint_dict_to_vals(self, joint_dict):
        joint_vals = []
        for i in range(len(self.joint_names)):
            if self.joint_names[i] in joint_dict:
                joint_vals.append(joint_dict[self.joint_names[i]])
            else:
                joint_vals.append(self.joint_dict[self.joint_names[i]])
        return joint_vals

    # def joint_dict_to_vals(self, joint_dict):
    #     joint_vals = []
    #     for i in range(len(self.joint_names)):
    #         joint_vals.append(joint_dict[self.joint_names[i]])
    #     return joint_vals

    def joint_vals_to_dict(self, joint_vals):
        joint_dict = {}
        for i in range(len(self.joint_names)):
            joint_dict[self.joint_names[i]] = joint_vals[i]
        return joint_dict

    def standarize_joint_vals(self, joint_vals):
        """
        given joint val, convert them to [-pi,pi]
        """
        joint_vals = [
            (joint_val + np.pi) % (2 * np.pi) - np.pi for joint_val in joint_vals
        ]
        return joint_vals

    def get_link_pose(self, link_name):
        link_idx = self.total_link_name_ind_dict[link_name]
        link_state = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=link_idx)
        pos = link_state[4]
        ori = link_state[5]  # x y z w
        transform = tf.transformations.quaternion_matrix([ori[3], ori[0], ori[1], ori[2]])
        transform[:3, 3] = pos
        return transform

    def get_tip_link_pose_urdfpy(self, joints=None):
        if joints is None:
            # checking with current joint
            joints = self.joint_dict
        fk = self.urdfpy_robot.link_fk(cfg=joints)
        return fk[self.link_dict[self.tip_link_name]]
        link_idx = self.total_link_name_ind_dict[self.tip_link_name]
        link_state = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=link_idx)
        pos = link_state[4]
        ori = link_state[5]  # x y z w
        transform = tf.transformations.quaternion_matrix([ori[3], ori[0], ori[1], ori[2]])
        transform[:3, 3] = pos

        if joints is not None:
            # reset
            self.set_joints_without_memorize(self.joint_dict)
        return transform

    def get_tip_link_pose(self, joints=None):
        if joints is not None:
            # checking with current joint
            self.set_joints_without_memorize(joints)
        else:
            self.set_joints_without_memorize(self.joint_dict)

        link_idx = self.total_link_name_ind_dict[self.tip_link_name]
        link_state = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=link_idx)
        pos = link_state[4]
        ori = link_state[5]  # x y z w
        transform = tf.transformations.quaternion_matrix([ori[3], ori[0], ori[1], ori[2]])
        transform[:3, 3] = pos

        if joints is not None:
            # reset
            self.set_joints_without_memorize(self.joint_dict)
        return transform

    def get_tip_link_pos_ori(self, joints=None):
        if joints is not None:
            self.set_joints_without_memorize(joints)

        link_idx = self.total_link_name_ind_dict[self.tip_link_name]
        link_state = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=link_idx)
        pos = np.array(link_state[4])
        ori = link_state[5]  # x y z w
        ori = np.array([ori[3], ori[0], ori[1], ori[2]])  # w x y z

        if joints is not None:
            # reset
            self.set_joints_without_memorize(self.joint_dict)
        return pos, ori

    def set_suction_length(self, suction_length=0.3015):
        self.suction_length = suction_length

    def get_ik(
        self,
        link_name,
        position,
        orientation,
        rest_pose,
        lower_lim=None,
        upper_lim=None,
        jr=None,
        collision_check=False,
        workspace=None,
        visualize=False
    ):
        if lower_lim is None:
            lower_lim = self.lower_lim
        if upper_lim is None:
            upper_lim = self.upper_lim
        if jr is None:
            jr = self.jr
        dof_joint_vals = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.total_link_name_ind_dict[link_name],
            targetPosition=position,
            targetOrientation=orientation,
            lowerLimits=lower_lim,
            upperLimits=upper_lim,
            jointRanges=jr,
            restPoses=rest_pose[:len(self.jr)],
            maxNumIterations=2000,
            residualThreshold=0.001,
            physicsClientId=self.pybullet_id
        )
        # print('len(dof_joint_vals): ', len(dof_joint_vals))
        # print('joint names: ', self.joint_names)
        # print("Get_ik", dof_joint_vals, rest_pose, len(rest_pose), len(self.jr))
        dof_joint_vals = self.standarize_joint_vals(dof_joint_vals)
        # print('after IK: ')
        # print(dof_joint_vals)
        # check whether the computed IK solution achieves the link pose with some threshold
        valid = self.check_ik(dof_joint_vals, link_name, position, orientation, visualize)
        # print('dof_joint_vals > self.upper_lim: ', dof_joint_vals > self.upper_lim)
        # print('sum: ', (dof_joint_vals > self.upper_lim).sum())

        # print('dof_joint_vals < self.lower_lim: ', dof_joint_vals < self.lower_lim)
        # print('sum: ', (dof_joint_vals < self.lower_lim).sum())

        if (dof_joint_vals > upper_lim[:len(dof_joint_vals)]).sum() > 0 or \
                (dof_joint_vals < lower_lim[:len(dof_joint_vals)]).sum() > 0:
            # print('IK joint out of limit')
            # print('ik found: ')
            # print(dof_joint_vals)
            # print('upper limit: ')
            # print(self.upper_lim[:len(dof_joint_vals)])
            # print('lower limit: ')
            # print(self.lower_lim[:len(dof_joint_vals)])
            valid = False
        if not valid:
            return valid, dof_joint_vals
        # check collision
        if collision_check:
            joint_dict = self.joint_vals_to_dict(dof_joint_vals)
            robot_state = self.motion_planner.get_robot_state_from_joint_dict(joint_dict)
            result = self.motion_planner.get_state_validity(
                robot_state, group_name="right_arm"
            )
            if not result.valid:
                # print('state_validity failed')
                valid = False
                # show the result
                if visualize:
                    self.motion_planner.display_robot_state(
                        robot_state, group_name='right_arm'
                    )
                    input('show next...')
                    # self.set_joints_without_memorize(joint_dict)
                    # input('show next...')
                    # self.set_joints_without_memorize(self.joint_vals)

            # result = self.robot_in_collision(dof_joint_vals, workspace)
            # if result:
            #     valid = False
        return valid, dof_joint_vals

    def quat_distance(self, q1, q2):
        # compare two angles in quat
        # take the shorter angle: when the dot product < 0, the angle lies in [pi,2pi]
        # note that this is taken care of by the conditional code
        ang_dist = np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))
        # clipping so the dot product is valid
        ang_dist = min(ang_dist, 1.0)
        ang_dist = max(ang_dist, -1.0)
        theta = 2 * np.arccos(ang_dist)
        if theta > np.pi:
            theta = theta - 2 * np.pi
        # theta[theta>np.pi] = theta[theta>np.pi] - 2*np.pi
        return np.abs(theta)

    def swing_distance(self, q1, q2):
        """
        twist around the ee_link direction is fine, as it does not affect the suction cup.
        We need to consider the swing along this direction
        """
        pass

    def check_ik(self, dof_joint_vals, link_name, position, orientation, visualize=False):

        for i in range(len(self.joint_indices)):
            joint_idx = self.joint_indices[i]
            p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                targetValue=dof_joint_vals[i],
                physicsClientId=self.pybullet_id
            )

        # check the pose of the link
        link_idx = self.total_link_name_ind_dict[link_name]
        link_state = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=link_idx)
        pos = link_state[4]
        ori = link_state[5]
        pos_threshold = 2e-2
        ori_threshold = 3 * np.pi / 180
        # print('dof_joint_vals: ')
        # print(dof_joint_vals)
        valid = False

        if (np.linalg.norm(np.array(pos) - np.array(position)) <= pos_threshold) and \
                (self.quat_distance(np.array(ori), np.array(orientation)) <= ori_threshold):
            valid = True
            return valid

        if not valid:
            # print('target pos: ', position, ' target orientation: ', orientation)
            # print('FK pos: ', pos, ' FK ori: ', ori)
            # print('orientation distance: ', self.quat_distance(np.array(ori), np.array(orientation)))
            # print('position distance: ', np.linalg.norm(np.array(pos)-np.array(position)))
            if visualize:
                print('target pos: ', position, ' target orientation: ', orientation)
                print('FK pos: ', pos, ' FK ori: ', ori)
                print(
                    'orientation distance: ',
                    self.quat_distance(np.array(ori), np.array(orientation))
                )
                joint_dict = self.joint_vals_to_dict(dof_joint_vals)
                robot_state = self.motion_planner.get_robot_state_from_joint_dict(
                    joint_dict
                )
                self.motion_planner.display_robot_state(
                    robot_state, group_name='right_arm'
                )

                input("checking ik")

            self.set_joints_without_memorize(self.joint_vals)

        return valid

    def robot_in_collision(self, joint_vals, workspace):
        self.set_joints_without_memorize(joint_vals)
        # self-collision

        # contacts = p.getClosestPoints(self.robot_id, self.robot_id, distance=0.,physicsClientId=self.pybullet_id)
        # if len(contacts):
        #     self.set_joints_without_memorize(self.joint_vals)
        #     return True

        collision = False
        for comp_name, comp_id in workspace.component_id_dict.items():
            contacts = p.getClosestPoints(
                self.robot_id, comp_id, distance=0., physicsClientId=self.pybullet_id
            )
            if len(contacts):
                collision = True
                self.set_joints_without_memorize(self.joint_vals)
                break
        return collision
