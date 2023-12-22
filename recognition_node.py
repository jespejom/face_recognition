#!/usr/bin/env python3.9

import rospy
import time
import hardcoded_bridge as wtf
from std_msgs.msg import String
from uchile_msgs.msg import ImageArray, StringArray
from recognizer.config import get_config
from recognizer.recognize import FaceRecognizer

class face_recognition_node():
    def __init__(self):
        conf = get_config(training = False, mobile = True)
        self.face_recognizer = FaceRecognizer(conf)

        self.subscriber = rospy.Subscriber('/maqui/interactions/face_detection', ImageArray, self._callback)
        self.publisher = rospy.Publisher('/maqui/interactions/face_recognition', StringArray, queue_size=5, latch=True)

    def _callback(self, data):
        #print('data:', data)
        cv_images = [wtf.imgmsg_to_cv2(image_msg) for image_msg in data.data]
        #print('imgs:',cv_images)
        names = self.face_recognizer.recognize_faces(cv_images)
        output = StringArray()
        output.header.stamp = rospy.Time.now()
        try:
            for name in names:
                name = String()
                output.data.append(name)
            
            print(f'Sending name of recognized person')
            self.publisher.publish(output)
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