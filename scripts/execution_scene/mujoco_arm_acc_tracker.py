"""
implement a controller that does the following:
    a_target -> system -> a
the implementation is sepcific to hinge-joint robots, and under go the following procedure:
    a_target -> u -> system -> a
and make sure a_target = a

This is because if we only specify the torque, the gravity and other forces may affect
the behavior of the robot so the acceleration is not what we desired.

since it's hinge-joint robot, we assume nv=nu.
"""
import mujoco as mj
import numpy as np
def arm_acc_tracker(a_target, model: mj.MjModel, data: mj.MjData):
    """
    a_target: an array of nv, that specifies the desired acceleration for each joint
    """
    data.qacc = np.array(a_target)
    mj.mj_inverse(model, data)  # compute the inverse force
    f = np.array(data.qfrc_inverse)  # this is in the joint space (nv)
    joint_ids = model.actuator_trnid[:,0]  # this is a mapping from nu to nv
    # transform the joint space force to control space (nu)
    p = np.zeros(model.nu)
    p = f[joint_ids]    
    u = np.zeros(p.shape)
    # assuming affine
    biasprm = np.array(model.actuator_biasprm)
    gainprm = np.array(model.actuator_gainprm)
    u = u - biasprm[1]*data.actuator_length - biasprm[2]*data.actuator_velocity - biasprm[0]
    u = u / gainprm[0]
    
    data.ctrl = u
    return u