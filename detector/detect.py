from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import os 
from utils.align_trans import warp_and_crop_face, get_reference_facial_points
from PIL import Image
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    # unused_pretrained_keys = ckpt_keys - model_keys
    # missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, device):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if device == torch.device("cpu"):
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class FaceDetector():
    def __init__(self, 
                    keep_top_k = 3,
                    conf_th = 0.02, 
                    get_landmarks = False,
                    top_k = 1000,
                    nms_threshold = 0.4,
                    vis_thres = 0.6 ):
        self.device = torch.device("cpu") # if args.cpu else "cuda")
        self.net = self.prepare_model(cfg_mnet).to(self.device)
        self.get_landmarks = get_landmarks
        self.conf_th = conf_th # confidence_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.vis_thres = vis_thres
        self.detections = None
        
    def detect(self, img_raw):
        self.img = img_raw
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)
        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))

        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.conf_th)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)
        self.detections = dets
        return dets
    
    def cut_faces(self, save_face = False):
        imgs_faces = []
        path_faces = []
        if self.detections is None:
            raise Exception('No detections found')

        for id, b in enumerate(self.detections):
            if b[4] < self.vis_thres:
                continue

            face = self.align(self.img, b[5:])
            imgs_faces.append(np.asarray(face))
            if save_face:
                folder = f'data/id{str(id).zfill(2)}/'
                try:  
                    os.mkdir(folder)  
                except:
                    pass  
                name = f"{folder}face{str(id).zfill(2)}.jpg"
                path_faces.append(name)
                cv2.imwrite(name, np.asarray(face))
        self.faces  = np.array(imgs_faces)

    def save_frame(self):
        if self.detections is None:
            raise Exception('No detections found')
        for b in self.detections:
            b = list(map(int, b))
            cv2.rectangle(self.img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cv2.imwrite('data/last_detection.jpg', img_raw)

    def prepare_model(self, cfg):
        torch.set_grad_enabled(False)
        net = RetinaFace(cfg=cfg)
        path_pretrained = './detector/weights/mobilenet0.25_Final.pth'
        net = load_model(net, path_pretrained, self.device)
        net.eval()
        print('Finished loading model!')
        return net

    def align(self, img, landmarks):
        facial5points = [[int(landmarks[2*j]),int(landmarks[2*j+1])] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, crop_size=(112,112))
        return Image.fromarray(warped_face)
    
if __name__ == '__main__':
    detector = FaceDetector(keep_top_k = 3)
    img = cv2.imread('./data/id1_2.jfif', cv2.IMREAD_COLOR)
    if img is None:
        print('read error')
        exit(1)
    detector.detect(img)
    detector.cut_faces(save_face = True)
    for face in detector.faces:
        cv2.imshow('face Capture', face)
        cv2.waitKey(0)
    #print('Done')