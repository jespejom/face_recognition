from deepface import DeepFace
import cv2
import datetime
import time
import numpy as np
"""
idx_person = 0
know_faces = []

def create_face(face_dict, time):
    global idx_person
    face_dict['name'] = None
    face_dict['last_seen'] = time
    face_dict['id'] = idx_person
    know_faces.append(face_dict)
    idx_person += 1
    return face_dict['id']

def is_kwown_face(img1, know_faces):
    for i in range(len(know_faces)):
        img2 = know_faces[i]['face']
        result = DeepFace.verify(img1_path = img1, img2_path = img2)
        if result['verified']:
            know_faces[i]['last_seen'] = current_time
            idx_identify = know_faces[i]['id']
            return idx_identify
    return None

def get_vert_face(face_area):
    print(face_area)
    xi = int(face_area['x'])
    xf = int(face_area['x']+face_area['h'])

    yi = int(face_area['y'])
    yf = int(face_area['y']+face_area['w'])

    return np.array([xi, xf, yi, yf], dtype=np.int32)

list_imgs = ['img1.jpg', 'img2.jpg']

for img in list_imgs:
    current_time = datetime.datetime.now()
    idx = is_kwown_face(img, know_faces)
    if idx is None:
        faces = DeepFace.extract_faces(img, detector_backend = "retinaface")
        for dic_face in faces:
            vert = get_vert_face(dic_face['facial_area'])
            print(vert)
            imagen = cv2.imread(img)
            face_img = imagen[vert[0]:vert[1], vert[2]:vert[3]]
            cv2.imwrite('face.png', face_img)
            new_face = create_face(dic_face, current_time)

print(know_faces)
"""

    

# Detection
s = time.time()
img = cv2.imread("img1.jpg")

r1 = DeepFace.extract_faces("img1.jpg", detector_backend = "retinaface")
s2 = time.time()
r2 = DeepFace.extract_faces(img, detector_backend = "retinaface")
s3 = time.time()
#dict1 = r2[0]
#print(dict1.keys())
print(s, s2, s3)
#dict1 = create_face(dict1)
#print(dict1.keys())
cv2.imshow('bla', r2[0]['face']) 
cv2.waitKey(0) 
  
s = time.time()
DeepFace.verify("img1.jpg", "img2.jpg", model_name = "ArcFace", detector_backend = "retinaface")
s2 = time.time()
DeepFace.verify("img1.jpg", "img2.jpg", model_name = "ArcFace", detector_backend = "retinaface")
s3 = time.time()
print(s2-s, s3-s2)
# closing all open windows 
cv2.destroyAllWindows()

#embedding_objs = DeepFace.represent(resized)
#print('embedding_objs')
#print(len(embedding_objs))
#embedding = embedding_objs[0]["embedding"]
#print(len(embedding))


#cv2.imshow("img1", r)

"""
result = DeepFace.verify(img1_path = resized, img2_path = "img2.jpg")
print(result)

#cv2.waitKey(4)"""