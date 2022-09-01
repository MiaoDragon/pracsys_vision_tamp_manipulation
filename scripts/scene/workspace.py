"""
This script defines the workspace object that remains static, and should be avoided
during planning for collision.
"""
import pybullet as p
import numpy as np
import transformations as tf

def create_box(shape, pos, ori, pid):
    col_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=shape/2, physicsClientId=pid)
    vis_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=shape/2, rgbaColor=[160/255, 107/255, 84/255, 1], physicsClientId=pid)
    comp_id = p.createMultiBody(baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                basePosition=pos, baseOrientation=ori, physicsClientId=pid)
    return comp_id

class Workspace():
    def __init__(self, front_x, back_x, left_y, right_y, top_z, bot_z, cam_pos, pid):
        # components is a list of geometric parts
        self.components = {}
        self.component_id_dict = {}
        self.component_ids = []
        bbox_lls = {}
        bbox_uls = {}
        transforms = {}

        width = 0.06
        extend_x = back_x - front_x
        extend_y = left_y-right_y
        extend_z = top_z-bot_z+width*2+0.04
        mid_x = (front_x+back_x)/2
        mid_y = (right_y+left_y)/2
        mid_z = (bot_z+top_z)/2


        def add_component(cid, pos, ori, shape, component_name):
            self.component_id_dict[component_name] = cid
            self.component_ids.append(cid)
            self.components[component_name] = {'pose': {'pos': pos, 'ori': ori}, 'shape': shape}
            rot_mat = tf.quaternion_matrix([ori[3], ori[0], ori[1], ori[2]])
            rot_mat[:3,3] = pos
            transforms[component_name] = rot_mat
            bbox_lls[component_name] = -shape/2
            bbox_uls[component_name] = shape/2

        # create back wall
        component_name = 'back'
        shape = np.array([width, extend_y, extend_z])
        pos = np.array([back_x, mid_y, mid_z])
        ori = [0,0,0,1]
        cid = create_box(shape, pos, ori, pid)


        add_component(cid, pos, ori, shape, component_name)


        # create left wall
        component_name = 'left'
        shape = np.array([extend_x, width, extend_z])
        pos = np.array([mid_x, left_y, mid_z])
        ori = [0,0,0,1]
        cid = create_box(shape, pos, ori, pid)
        add_component(cid, pos, ori, shape, component_name)
        # create right wall
        component_name = 'left_padding'
        padding = 0.35
        shape = np.array([extend_x, padding, extend_z])
        pos = np.array([mid_x, left_y+padding/2, mid_z])
        ori = [0,0,0,1]
        cid = create_box(shape, pos, ori, pid)
        add_component(cid, pos, ori, shape, component_name)


        # create right wall
        component_name = 'right'
        shape = np.array([extend_x, width, extend_z])
        pos = np.array([mid_x, right_y, mid_z])
        ori = [0,0,0,1]
        cid = create_box(shape, pos, ori, pid)
        add_component(cid, pos, ori, shape, component_name)

        # create right wall
        component_name = 'right_padding'
        shape = np.array([extend_x, padding, extend_z])
        pos = np.array([mid_x, right_y-padding/2, mid_z])
        ori = [0,0,0,1]
        cid = create_box(shape, pos, ori, pid)
        add_component(cid, pos, ori, shape, component_name)


        # create top wall
        component_name = 'top'
        shape = np.array([extend_x, extend_y, width])
        pos = np.array([mid_x, mid_y, top_z])
        ori = [0,0,0,1]
        cid = create_box(shape, pos, ori, pid)
        add_component(cid, pos, ori, shape, component_name)

        # create bot wall
        component_name = 'bottom'
        bot_retreat = 0.1
        shape = np.array([extend_x-bot_retreat, extend_y, width])
        pos = np.array([mid_x+bot_retreat/2, mid_y, bot_z])
        ori = [0,0,0,1]
        cid = create_box(shape, pos, ori, pid)
        add_component(cid, pos, ori, shape, component_name)

        # create behind wall
        component_name = 'camera_table'
        shape = np.array([0.1, extend_y, top_z])
        pos = np.array([cam_pos[0]-0.1-0.2, mid_y, top_z/2])
        ori = [0,0,0,1]
        cid = create_box(shape, pos, ori, pid)
        add_component(cid, pos, ori, shape, component_name)
        component_name = 'camera_link'
        cam_padding=0.1
        link_length=0.3
        shape = np.array([cam_padding+link_length, 0.2, 0.2])
        pos = np.array([cam_pos[0]+cam_padding/2-link_length/2,cam_pos[1],cam_pos[2]])
        ori = [0,0,0,1]
        cid = create_box(shape, pos, ori, pid)
        add_component(cid, pos, ori, shape, component_name)




        # # create ceiling padding
        # shape = np.array([extend_x, extend_y, width])
        # pos = np.array([mid_x, mid_y, bot_z])
        # ori = [0,0,0,1]
        # cid = create_box(shape, pos, ori, pid)
        # add_component(cid, pos, ori, shape, component_name)



        self.pos = np.array([0,0,0.])
        self.ori = np.array([1.,0,0,0])

        # modify the workspace_low and high by using the padding
        workspace_low = np.array([front_x,right_y,bot_z])
        workspace_high = np.array([back_x,left_y,top_z])

        workspace_low[1] = workspace_low[1]# + padding[1]
        workspace_high[0] = workspace_high[0]# - padding[0]
        workspace_high[1] = workspace_high[1]# - padding[1]

        self.region_low = np.array(workspace_low)# + np.array(base_pos)  # the bounding box of the valid regions in the workspace
        self.region_high = np.array(workspace_high)# + np.array(base_pos)
        
        self.bbox_lls = bbox_lls
        self.bbox_uls = bbox_uls
        self.transforms = transforms