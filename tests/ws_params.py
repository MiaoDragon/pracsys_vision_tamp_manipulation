import sys
import rospy

if sys.argv[1] == 'shelf':
    rospy.set_param(
        '/workspace/pose', [
            [0, 1, 0, 1.15],
            [-1, 0, 0, 0.4],
            [0, 0, 1, 1.0],
            [0, 0, 0, 1],
        ]
    )
    rospy.set_param('/workspace/size', [0.8, 0.3, 0.3])
elif sys.argv[1] == 'table':
    rospy.set_param(
        '/workspace/pose', [
            [0, 1, 0, 0.62],
            [-1, 0, 0, 0.4],
            [0, 0, 1, 0.9],
            [0, 0, 0, 1],
        ]
    )
    rospy.set_param('/workspace/size', [0.8, 0.5, 0.3])
