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
        #TODO: cambiar a Bender
        self.subscriber_topic = '/maqui/camera/front/image_raw'

        self.subscriber = None
        self.publisher = rospy.Publisher('/maqui/interactions/face_detection', ImageArray, queue_size=5, latch=True)
        self.detector = FaceDetector(keep_top_k = 3, buff_size = 10)

    def _callback(self, data):
        cv_image = wtf.imgmsg_to_cv2(data)
        self.detector.process_img(cv_image)

        if self.detector.is_buffer_full():
            output = StringArray()
            output.header.stamp = rospy.Time.now()
            try:
                output.data.append(str(self.buffer_faces))
                print(f'Sending face image')
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