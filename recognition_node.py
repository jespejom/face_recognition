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
        self.subscriber_topic = '/maqui/interactions/face_detection'
        self.subscriber = None
        self.publisher = rospy.Publisher('/maqui/interactions/face_recognition', StringArray, queue_size=5, latch=True)

    def _callback(self, data):

        cv_images = [wtf.imgmsg_to_cv2(image_msg) for image_msg in data.data]
        names = self.face_recognizer.recognize_faces(cv_images)
        output = StringArray()
        output.header.stamp = rospy.Time.now()
        try:
            for name in names:
                name = String()
                output.data.append(name)
            self.publisher.publish(output)
        except:
            with Exception as e:
                print(e)

    def start_callback(self):
        if self.subscriber is None:
            self.subscriber = rospy.Subscriber(self.subscriber_topic, ImageArray, self._callback)

    def stop_callback(self):
        if self.subscriber is not None:
            self.subscriber.unregister()
            self.subscriber = None

    def add_face_to_facebank(self, face, name):
        self.face_recognizer.save_identities(names)

def main():
    rospy.init_node('face_recognition_node')
    node = face_recognition_node()
    rospy.on_shutdown(node.stop_callback)  # Registra el método stop_callback() para que se llame cuando el nodo se apague
    time.sleep(0.5)
    node.start_callback()  # Inicia el callback
    rospy.spin()

if __name__ == '__main__':
    main()