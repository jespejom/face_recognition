#pip3 install opencv-python
import cv2 
from retinaface import RetinaFace
import time
# init with 'normal' accuracy option (resize width or height to 800 )
# or you can choice 'speed' (resize to 320)
# or you can initiate with no parameter for running with original image size


# same with cv2.imread,cv2.cvtColor 
#rgb_image = detector.read("faces.png")
print(time.time())
resp = RetinaFace.detect_faces("faces.png")
#faces = detector.predict(rgb_image)
# faces is list of face dictionary
# each face dictionary contains x1 y1 x2 y2 left_eye right_eye nose left_lip right_lip
# faces=[{"x1":20,"y1":32, ... }, ...]
print(time.time())
resp = RetinaFace.detect_faces("faces.png")
print(time.time())
#result_img = detector.draw(rgb_image,faces)

# save ([...,::-1] : rgb -> bgr )
cv2.imwrite("result_img.jpg",resp)

# show using cv2
# cv2.imshow("result",result_img[...,::-1])
# cv2.waitKey()