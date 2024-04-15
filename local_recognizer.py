from detector.detect import FaceDetector
from recognizer.recognize import FaceRecognizer
import cv2
import os

if __name__ == '__main__':
    recognizer = FaceRecognizer()
    detector = FaceDetector(keep_top_k = 4)
    #detect faces on camera, opencvqq
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('read error')
            break
        detector.detect(frame)
        detector.cut_faces()
        
        # for face in detector.faces:
            
        #     cv2.imshow('face Capture', face)
        #     cv2.waitKey(1)
            #num_imgs = len([path for path in os.listdir(dir_path) if '.jpg' in path])
            #cv2.imwrite('data/facebank/img.jpg', frame)
        
        names = recognizer.recognize_faces(detector.faces)
        print(names)
