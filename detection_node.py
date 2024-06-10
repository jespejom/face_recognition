#!/usr/bin/env python3.9

import rospy
import time
import hardcoded_bridge as wtf
from sensor_msgs.msg import Image
from uchile_msgs.msg import ImageArray
from detector.detect import FaceDetector

class face_detector_node():
    def __init__(self):
        print('Nodo creado')
        self.subscriber_topic = '/maqui/camera/front/image_raw'

        self.subscriber = None
        self.publisher = rospy.Publisher('/maqui/interactions/face_detection', ImageArray, queue_size=5, latch=True)
        self.face_detector = FaceDetector(keep_top_k = 3)

    def _callback(self, data):
        cv_image = wtf.imgmsg_to_cv2(data)
        self.face_detector.detect(cv_image)
        self.face_detector.cut_faces()

        output = ImageArray()
        output.header.stamp = rospy.Time.now()
        try:
            for face in self.face_detector.faces:
                imgmsg = wtf.cv2_to_imgmsg(face)
                output.data.append(imgmsg)
                print(f'Sending face image')
            # V2: publicar solo cuando el buffer del detector este lleno
            self.publisher.publish(output)
        except:
            with Exception as e:
                print(e)

    def start_callback(self):
        if self.subscriber is None:
            self.subscriber = rospy.Subscriber(self.subscriber_topic, Image, self._callback)

    def stop_callback(self):
        if self.subscriber is not None:
            self.subscriber.unregister()
            self.subscriber = None
def main():
    rospy.init_node('face_detector_node')
    node = face_detector_node()
    rospy.on_shutdown(node.stop_callback)  # This will call the stop_callback method when the node is stopped
    time.sleep(0.5)
    node.start_callback()
    rospy.spin()

if __name__ == '__main__':
    main()