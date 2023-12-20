#!/usr/bin/env python3.9

import rospy
import time
import hardcoded_bridge as wtf
from detector import FaceDetector
from std_msgs.msg import Image
from msg.ImageArray import ImageArray

class face_detector_node():
    def __init__(self):
        self.subscriber = rospy.Subscriber('/maqui/camera/front/image_raw', Image, self._callback)
        self.publisher = rospy.Publisher('/maqui/interactions/face_detection', ImageArray, queue_size=10, latch=True)
        self.face_detector = FaceDetector(keep_top_k = 3)

    def _callback(self, data):
        
        cv_image = wtf.imgmsg_to_cv2(data)
        self.face_detector.detect(cv_image)
        self.face_detector.cut_faces(save_face = True)

        output = ImageArray()
        output.header.stamp = rospy.Time.now()
        try:
            for face in self.face_detector.faces:
                imgmsg = wtf.cv2_to_imgmsg(face)
                output.data.append(imgmsg)
            print(f'Sending face image')
            self.publisher.publish(output)
            time.sleep(5)
        except:
            with Exception as e:
                print(e)
        
def main():
    rospy.init_node('face_detector_node')
    face_detector_node()
    time.sleep(0.5)
    rospy.spin()

if __name__ == '__main__':
    main()