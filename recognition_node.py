#!/usr/bin/env python3.9
import rospy
import time
import hardcoded_bridge as wtf
from facerecognizer import FaceRecognizer
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class face_recognition_node():
    def __init__(self):
        self.subscriber = rospy.Subscriber('/maqui/interactions/face_detection', Image, self._callback)
        self.publisher = rospy.Publisher('/maqui/interactions/face_recognition', Image, queue_size=10, latch=True)
        self.face_recognizer = FaceRecognizer()

    def _callback(self, data):
        
        cv_image = wtf.imgmsg_to_cv2(data)

        face = self.face_detector.get_proba(cv_image)
        output = Image()
        
        try:
            if face is not None:
                output.data = self.br.cv2_to_imgmsg(face)
                print(f'Sending face image')
                self.publisher.publish(output)
                time.sleep(5)
            else:
                pass

        except:
            with Exception as e:
                print(e)

def main():
    rospy.init_node('face_recognition_node')
    face_recognition_node()
    time.sleep(0.5)
    rospy.spin()

if __name__ == '__main__':
    main()