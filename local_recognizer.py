from detector.detect import FaceDetector
from recognizer.recognize import FaceRecognizer
import cv2
import os

if __name__ == '__main__':
    recognizer = FaceRecognizer()
    detector = FaceDetector(keep_top_k = 4)
    #detect faces on camera, opencvqq
    cap = cv2.VideoCapture(0)
    buffer = []

    while len(detector.buffer_faces) < 10:
        ret, frame = cap.read()
        if not ret:
            print('read error')
            break
        detector.detect(frame)
    buffer = detector.buffer_faces

    for t in range(len(buffer)):
        faces = [item['face'] for item in buffer[t]]
        names = recognizer.recognize_faces(faces)
        for cara in range(len(buffer[t])):
            buffer[t][cara].update({'name': names[cara]})
    