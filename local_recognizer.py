from detector.detect import FaceDetector
from recognizer.recognize import FaceRecognizer
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import time

if __name__ == '__main__':
    
    detector = FaceDetector(keep_top_k = 4)
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)

    buff_size = 10
    while True:
        detector.prepare_buffer()
        while len(detector.buffer_faces) < buff_size:
            
            ret, frame = cap.read()
            if not ret:
                print('read error')
                break
            detector.process_img(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.2)

        buffer = detector.buffer_faces
        recognizer.update_facebank()
        recognizer.buffer_faces = buffer
        names = recognizer.recognize_faces_by_location()
        print('names', names)