import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import FaceRecognizer
from utils import load_facebank, draw_box_name, prepare_facebank


if __name__ == '__main__':

    update = True
    threshold = 1.54
    tta = True
    score = True

    # Configurar mobile o ir-se
    conf = get_config(training = False, mobile = True)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = FaceRecognizer(conf, True)
    learner.threshold = threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # inital camera
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:            
            try:
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:,:-1]
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice    
                results, score = learner.infer(conf, faces, targets)
                for idx, bbox in enumerate(bboxes):
                    name = names[results[idx] + 1]
                    frame = draw_box_name(bbox, name, frame)
            except:
                pass
            cv2.imshow('face Capture', frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
           break
    cap.release()
    cv2.destroyAllWindows()    