from detector.detect import FaceDetector
from recognizer.recognize import FaceRecognizer
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    recognizer = FaceRecognizer()
    detector = FaceDetector(keep_top_k = 4)
    #detect faces on camera, opencvqq
    cap = cv2.VideoCapture(0)
    cap = cv2.imread('./data/img2.jpg', cv2.IMREAD_COLOR)

    while len(detector.buffer_faces) < 10:
        #ret, frame = cap.read()
        frame = cap
        # if not ret:
        #     print('read error')
        #     break
        detector.detect(frame)
    buffer = detector.buffer_faces
    unique_names = []
    #print('buffer', buffer)
    for t in range(len(buffer)):
        faces = [item['face'] for item in buffer[t]]
        names = recognizer.recognize_faces(faces)
        unique_names.append(list(set(names)))
        for cara in range(len(buffer[t])):
            buffer[t][cara].update({'name': names[cara]})
    img_size = buffer[0][0]['face'].size
    map_names = np.zeros((img_size[0], img_size[1], len(unique_names)), np.uint8)
    print(buffer)
    for name in unique_names:
        for t in buffer:
            for cara in t:
                if cara['name'] in name:
                    map_names[cara['pos'][1]:cara['pos'][3], cara['pos'][0]:cara['pos'][2], unique_names.index(name)] +=1
    plt.imshow(map_names)
    plt.savefig('map_names.png')

    