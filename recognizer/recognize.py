from recognizer.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch
#from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from PIL import Image
from torchvision import transforms as trans
from recognizer.utils import load_facebank, prepare_facebank
from recognizer.config import get_config
import numpy as np 
from easydict import EasyDict as edict
from pathlib import Path
from torch.nn import CrossEntropyLoss

class FaceRecognizer(object):
    def __init__(self, update_fb = True):
        conf = get_config(training = False, mobile = True)
        self.conf = conf
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        
        self.threshold = conf.threshold

        if update_fb:
            self.targets, self.names = prepare_facebank(conf, self.model)
            print('facebank updated')
        else:
            self.targets, self.names = load_facebank(conf)
            print('facebank loaded')
        
        print(self.names)

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path       
        self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str), map_location=torch.device('cpu')))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
    
    def infer(self, faces, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        conf = self.conf
        embs = []
        
        for img in faces:
            assert img.shape == (112, 112, 3), 'image shape is {}, not 112x112x3'.format(img.shape)
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        diff = source_embs.unsqueeze(-1) - self.targets.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1) 
        idx_identification = self.get_identifications(dist.cpu().numpy(), self.threshold)
        return idx_identification     
    
    def get_identifications(self, dist, th):
        identification = np.full((dist.shape[0],), np.nan)
        dist[dist > th] = np.nan

        def indices_minimo(matriz):
            indice_minimo = np.nanargmin(matriz)
            indice_fila, indice_columna = np.unravel_index(indice_minimo, matriz.shape)
            return indice_fila, indice_columna

        def replace_x_with_nan(mat, idx):
            mat[:, idx[1]] = np.nan
            mat[idx[0], :] = np.nan
            return mat

        while not np.isnan(dist).all():
            min_idx = indices_minimo(dist)
            identification[min_idx[0]] = int(min_idx[1])
            dist = replace_x_with_nan(dist, min_idx)

        identification = np.nan_to_num(identification, nan = -1)
        return identification

    def recognize_faces(self, faces):
        recog_names = []
        if faces.size == 0:
            print('No faces detected')
            return recog_names
        
        results = self.infer(faces)
        for idx, _ in enumerate(results):
            name = self.names[int(results[idx]) + 1]
            recog_names.append(name)
        return recog_names

    def get_config(self, training = True, mobile = False):
        conf = edict()
        conf.data_path = Path('./data')
        conf.work_path = Path('work_space/')
        conf.model_path = conf.work_path/'models'
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
        conf.input_size = [112,112]
        conf.embedding_size = 512
        conf.use_mobilfacenet = mobile
        conf.net_depth = 50
        conf.drop_ratio = 0.6
        conf.net_mode = 'ir_se' # or 'ir'
        conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        conf.test_transform = trans.Compose([
                        trans.ToTensor(),
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
        conf.data_mode = 'emore'
        conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
        conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
        conf.emore_folder = conf.data_path/'faces_emore'
        #conf.batch_size = 100 # irse net depth 50 
        conf.batch_size = 200 # mobilefacenet
    #--------------------Inference Config ------------------------
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 0.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
        return conf

if __name__ == '__main__':
    conf = get_config(training = False, mobile = True)
    recognizer = FaceRecognizer(conf)
    img = Image.open('./data/facebank/img2.jpg')
    img = Image.open('C:/Users/lesli/OneDrive/Documents/GitHub/face_memory/data/facebank/img2.jpg')
    if img is None:
        print('read error')
        exit(1)
    names = recognizer.recognize_faces([img])
    print(names)
