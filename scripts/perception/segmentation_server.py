"""
provide a ROS server to do segmentation work
"""
from perception.real_segmentation_imp import CylinderSegmentation
from pracsys_vision_tamp_manipulation.srv import SegmentationSrv,SegmentationSrvResponse
from pracsys_vision_tamp_manipulation.msg import Cylinder
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import rospy
import transformations as tf
from scene.camera import Camera

class SegmentationROSServer:
    def __init__(self):
        # camera = Camera()
        self.seg = CylinderSegmentation()
        self.srv = rospy.Service('segmentation',SegmentationSrv, self.segmentation_cb)


    def segmentation_cb(self, req):
        cylinder_models = self.seg.estimate(req.num_objs)

        cylinders = []
        for i in range(len(cylinder_models)):
            cylinder_i = Cylinder()
            mid_center = cylinder_models[i]['mid_center']
            cylinder_i.mid_center.x = mid_center[0]
            cylinder_i.mid_center.y = mid_center[1]
            cylinder_i.mid_center.z = mid_center[2]
            cylinder_i.radius = cylinder_models[i]['radius'] + 0.01
            cylinder_i.height = cylinder_models[i]['height'] + 0.02
            axis = cylinder_models[i]['axis']
            cylinder_i.axis.x = axis[0]
            cylinder_i.axis.y = axis[1]
            cylinder_i.axis.z = axis[2]
            transform = cylinder_models[i]['transform']
            qw,qx,qy,qz = tf.quaternion_from_matrix(transform)
            cylinder_i.transform.rotation.w = qw
            cylinder_i.transform.rotation.x = qx
            cylinder_i.transform.rotation.y = qy
            cylinder_i.transform.rotation.z = qz
            cylinder_i.transform.translation.x = transform[0,3]
            cylinder_i.transform.translation.y = transform[1,3]
            cylinder_i.transform.translation.z = transform[2,3]
            cylinder_i.color = [
                int(cylinder_models[i]['color'][0]*255),
                int(cylinder_models[i]['color'][1]*255),
                int(cylinder_models[i]['color'][2]*255),
            ]

            cylinder_i.pcd.points
            pts = []
            print('pcd shape: ')
            print(cylinder_models[i]['pcd'].shape)
            pcd = cylinder_models[i]['pcd']
            for j in range(len(pcd)):
                pt = Point32()
                pt.x = cylinder_models[i]['pcd'][j,0]
                pt.y = cylinder_models[i]['pcd'][j,1]
                pt.z = cylinder_models[i]['pcd'][j,2]
                pts.append(pt)
            cylinder_i.pcd.points = pts
            cylinders.append(cylinder_i)

        response = SegmentationSrvResponse()
        response.cylinders = cylinders
        return response

if __name__ == "__main__":
    rospy.init_node("segmentation")
    print("inited service.")
    seg_srv = SegmentationROSServer()
    rospy.spin()
