from detector.detect import FaceDetector
from recognizer.recognize import FaceRecognizer
from recognizer.config import get_config
import cv2

if __name__ == '__main__':
    conf = get_config(training = False, mobile = True)
    recognizer = FaceRecognizer(conf)
    detector = FaceDetector(keep_top_k = 4)
    img = cv2.imread('./data/img2.jpg', cv2.IMREAD_COLOR)
    if img is None:
        print('read error')
        exit(1)
    detector.detect(img)
    detector.cut_faces()
    faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in detector.faces]
    
    names = recognizer.recognize_faces(faces)
    print(names)
