# message that represents the conservative volume of an object
Header header
uint32[] data
geometry_msgs/Pose pose
# pose of the (0,0,0) and the coordinate system
# For objects: this is the obj_T_voxel
# For scene occlusion: this is world_T_voxel
geometry_msgs/Vector3 resols
uint32 size_x
uint32 size_y
uint32 size_z