from detector.detect import FaceDetector
from recognizer.recognize import FaceRecognizer
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import time

if __name__ == '__main__':
    
    detector = FaceDetector(keep_top_k = 4)
    cap = cv2.VideoCapture(0)
    # cap = cv2.imread('./data/img2.jpg', cv2.IMREAD_COLOR)
    buff_size = 10
    while True:
        while len(detector.buffer_faces) < buff_size:
            ret, frame = cap.read()
            if not ret:
                print('read error')
                break
            detector.process_img(frame)
            print(len(detector.buffer_faces))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.2)

        buffer = detector.buffer_faces
        recognizer = FaceRecognizer()
        unique_names = []
        for t in range(len(buffer)):
            faces = [item['face'] for item in buffer[t]]
            names = recognizer.recognize_faces(faces)
            unique_names.extend(names)
            for cara in range(len(buffer[t])):
                buffer[t][cara].update({'name': names[cara]})
        img_size = frame.shape
        unique_names = list(set(unique_names))
        map_names = np.zeros((img_size[0], img_size[1], len(unique_names)), np.uint8)

        # Crear map_names usando operaciones vectorizadas
        for id, name in enumerate(unique_names):
            for t in buffer:
                for cara in t:
                    if cara['name'] in name:
                        pos = cara['pos']
                        map_names[pos[1]:pos[3], pos[0]:pos[2], id] += 1

        valid = []
        for id, name in enumerate(unique_names):
            det_img = map_names[:, :, id]
            print('np.max',np.max(det_img))
            if np.max(det_img) >= int(0.5 * buff_size):
                # Calcular posición promedio ponderada
                rows, cols = np.indices(det_img.shape)
                total_sum = np.sum(det_img)
                x_pos = np.sum(rows * det_img) / total_sum if total_sum > 0 else 0
                y_pos = np.sum(cols * det_img) / total_sum if total_sum > 0 else 0
                valid.append({'pos': [x_pos, y_pos], 'name': name})

        print('valid:', valid)
        detector.buffer_faces = []
        
