import numpy as np
import copy
def mask_pcd_with_padding(occ_filter, pcd_indices, padding=1):
    """
    given the transformed pcd indices in occlusion transform, add padding to the pcd and mask it as valid
    in occ_filter
    """
    valid_filter = (pcd_indices[:,0] >= padding) & (pcd_indices[:,0] < occ_filter.shape[0]-padding) & \
                    (pcd_indices[:,1] >= padding) & (pcd_indices[:,1] < occ_filter.shape[1]-padding) & \
                    (pcd_indices[:,2] >= 0) & (pcd_indices[:,2] < occ_filter.shape[2])
    pcd_indices = pcd_indices[valid_filter]
    if len(pcd_indices) == 0:
        return occ_filter
    masked_occ_filter = np.array(occ_filter)
    masked_occ_filter[pcd_indices[:,0]-1,pcd_indices[:,1],pcd_indices[:,2]] = 0
    masked_occ_filter[pcd_indices[:,0]+1,pcd_indices[:,1],pcd_indices[:,2]] = 0
    masked_occ_filter[pcd_indices[:,0],pcd_indices[:,1]-1,pcd_indices[:,2]] = 0
    masked_occ_filter[pcd_indices[:,0],pcd_indices[:,1]+1,pcd_indices[:,2]] = 0
    masked_occ_filter[pcd_indices[:,0]-1,pcd_indices[:,1]-1,pcd_indices[:,2]] = 0
    masked_occ_filter[pcd_indices[:,0]+1,pcd_indices[:,1]-1,pcd_indices[:,2]] = 0
    masked_occ_filter[pcd_indices[:,0]-1,pcd_indices[:,1]+1,pcd_indices[:,2]] = 0
    masked_occ_filter[pcd_indices[:,0]+1,pcd_indices[:,1]+1,pcd_indices[:,2]] = 0

    return masked_occ_filter

def mask_pcd_xy_with_padding(occ_filter, pcd_indices, padding=1):
    """
    given the transformed pcd indices in occlusion transform, add padding to the pcd and mask it as valid
    in occ_filter
    filter out all z axis since we assume objects won't be stacked on top of each other
    """
    masked_occ_filter = np.array(occ_filter)
    valid_filter = (pcd_indices[:,0] >= 0) & (pcd_indices[:,0] < occ_filter.shape[0]) & \
                    (pcd_indices[:,1] >= 0) & (pcd_indices[:,1] < occ_filter.shape[1])
    pcd_indices = pcd_indices[valid_filter]
    masked_occ_filter[pcd_indices[:,0],pcd_indices[:,1],:] = 0

    valid_filter = (pcd_indices[:,0] >= padding) & (pcd_indices[:,0] < occ_filter.shape[0]-padding) & \
                    (pcd_indices[:,1] >= padding) & (pcd_indices[:,1] < occ_filter.shape[1]-padding)
    pcd_indices = pcd_indices[valid_filter]
    if len(pcd_indices) == 0:
        return masked_occ_filter
    for padding_i in range(0,padding+1):
        for padding_j in range(0,padding+1):
            masked_occ_filter[pcd_indices[:,0]-padding_i,pcd_indices[:,1]-padding_j,:] = 0
            masked_occ_filter[pcd_indices[:,0]-padding_i,pcd_indices[:,1]+padding_j,:] = 0
            masked_occ_filter[pcd_indices[:,0]+padding_i,pcd_indices[:,1]-padding_j,:] = 0
            masked_occ_filter[pcd_indices[:,0]+padding_i,pcd_indices[:,1]+padding_j,:] = 0

    del valid_filter
    del pcd_indices

    return masked_occ_filter
def wrap_angle(angle, ll, ul):
    angle = angle % (np.pi*2)
    if angle > np.pi:
        angle = angle - np.pi*2
    if angle < ll:
        angle = ll
    if angle > ul:
        angle = ul
    return angle

def generate_rot_traj(joint_name, start_joint_dict, waypoint, dtheta=5*np.pi/180):
    start_angle = start_joint_dict[joint_name]
    change = waypoint - start_angle
    ntheta = int(np.ceil(np.abs(change) / dtheta))
    dtheta = change / ntheta

    traj = []
    joint_dict = copy.deepcopy(start_joint_dict)
    traj.append(joint_dict)
    angle = start_angle
    for i in range(ntheta):
        joint_dict = copy.deepcopy(joint_dict)
        joint_dict[joint_name] += dtheta
        traj.append(joint_dict)

    return traj