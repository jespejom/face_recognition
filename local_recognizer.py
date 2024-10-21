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

            # img = cv2.imread('./data/img2.jpg', cv2.IMREAD_COLOR)
            # detector.process_img(img)
            # cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.2)

        buffer = detector.buffer_faces
        recognizer.update_facebank()
        recognizer.buffer_faces = buffer
        names = recognizer.recognize_faces_by_location()
        print('names', names)
        if len(names) == 0:
            print('No faces found')
        else:
            for name in names:
                # add name to facebank
                print(buffer)
                if name['name'] == 'Unknown':
                    new_name = input('Ingrese nombre de la persona: ')
                    for face in buffer:
                        #check if is an empyt list
                        if face == []:
                            continue
                        else:
                            for f_i in face:
                                if f_i['name'] == 'Unknown':
                                    recognizer.save_identities(face[0]['face'], new_name)
                    # recognizer.save_identities(buffer[0][0], new_name)