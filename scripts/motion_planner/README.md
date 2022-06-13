handles motion planning request:
- plan a trajectory from point A to point B w/wo objects attached, given collision env (which can be modeled by pcd or voxel, implemented in the scene folder)
- cartesian plan from point A to point B. This can be achieved by calling IK to get configs or differential IK
- inverse kinematics to obtain configs from poses. This can use PyBullet implemented methods or other methods.
- collision checking: self collision of robot, collision of robot with static scene, collision of robot and objects, collision of attached object with other objects.
These methods may be better implemented by functions, unless specific data structure is to be used such as roadmap.