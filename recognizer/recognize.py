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
import os

class FaceRecognizer(object):
    def __init__(self, update_fb = True, buffer_size = 10):
        self.conf = get_config(training = False, mobile = True)
               
        self.model = MobileFaceNet(self.conf.embedding_size).to(self.conf.device)
        print('MobileFaceNet model generated')

        self.threshold = self.conf.threshold # Menor th es más exigente

        if update_fb:
            self.update_facebank()
        else:
            self.targets, self.names = load_facebank(self.conf)
            print('facebank loaded')
            
        print("Know names", self.names)
        self.buffer_faces = []
        self.buffer_size = buffer_size

    def update_facebank(self):
        self.targets, self.names = prepare_facebank(self.conf, self.model)
        print('facebank updated')

    def infer(self, faces, tta=False):
        '''
        faces: list of PIL images
        tta: test time augmentation
        '''
        conf = self.conf
        embs = []
        for img in faces:
            assert img.size == (112, 112), 'image shape is {}, not 112x112'.format(img.size)
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
        idx_identification = self.get_identifications(dist.cpu().numpy())
        return idx_identification     
    
    def get_identifications(self, dist):
        identification = np.full((dist.shape[0],), np.nan)
        dist[dist > self.threshold] = np.nan
        # print(dist)
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

    def recognize_faces(self, faces, by_location=False):
        recog_names = []
        if len(faces) == 0:
            print('No faces detected')
            return recog_names
        
        results = self.infer(faces)
        for idx, _ in enumerate(results):
            name = self.names[int(results[idx]) + 1]
            recog_names.append(name)
        return recog_names

    def filter_by_location(self, unique_names, map_names):
        valid = []
        for id, name in enumerate(unique_names):
            det_img = map_names[:, :, id]
            # Si aparece reconocida en al menos la mitad de los frames
            if np.max(det_img) >= int(0.5 * self.buffer_size): 
                # Calcular posición promedio ponderada
                rows, cols = np.indices(det_img.shape)
                total_sum = np.sum(det_img)
                x_pos = np.sum(rows * det_img) / total_sum if total_sum > 0 else 0
                y_pos = np.sum(cols * det_img) / total_sum if total_sum > 0 else 0
                valid.append({'pos': [x_pos, y_pos], 'name': name})
        return valid

    def generate_map(self, unique_names, img_size):
        map_names = np.zeros((img_size[0], img_size[1], len(unique_names)), np.uint8)
        for id, name in enumerate(unique_names):
            for t in self.buffer_faces:
                for cara in t:
                    if cara['name'] in name:
                        pos = cara['pos']
                        map_names[pos[1]:pos[3], pos[0]:pos[2], id] += 1
        return map_names

    def recognize_faces_by_location(self):
        buff = self.buffer_faces
        unique_names = []

        for t in range(len(buff)):
            faces = [item['face'] for item in buff[t]]
            names = self.recognize_faces(faces)
            unique_names.extend(names)

            for cara in range(len(buff[t])):
                buff[t][cara].update({'name': names[cara]})

        img_size = self.buffer_faces[0][0]['face'].size
        unique_names = list(set(unique_names))
        print(unique_names)
        map_names = self.generate_map(unique_names, img_size)
        valid_identifications = self.filter_by_location(unique_names, map_names)
        self.buffer_faces = []

        return valid_identifications

    def get_config(self, training = True, mobile = True):
        conf = edict()
        conf.data_path = Path('./data')
        conf.work_path = Path('work_space/')
        conf.model_path = conf.work_path/'models'
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
        conf.input_size = [112, 112]
        conf.embedding_size = 1028 #512
        conf.use_mobilfacenet = mobile
        conf.net_depth = 50
        conf.drop_ratio = 0.4
        conf.net_mode = 'ir_se' # or 'ir'
        conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        conf.test_transform = trans.Compose([
                        trans.ToTensor(),
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
        conf.batch_size = 200 # mobilefacenet
    #--------------------Inference Config ------------------------
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 0.5
        conf.face_limit = 10 
        conf.min_face_size = 30 
        return conf

    def save_identities(self, face, name: str):
        path_folder = self.conf.facebank_path+'/'+name
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        if not isinstance(face, np.ndarray):
            face = face.cpu().detach().numpy()
        
        n_files = len([path for path in os.listdir(path_folder) if '.jpg' in path])
        path_img = path_folder + '/' + name +'_'+ str(n_files + 1).zfill(3) + '.jpg'
        
        np.save(path_img, face)
        # update facebank
        self.targets, self.names = prepare_facebank(self.conf, self.model)
        
if __name__ == '__main__':
    conf = get_config(training = False, mobile = True)
    recognizer = FaceRecognizer(conf)
    img = Image.open('./data/facebank/img2.jpg')
    img = Image.open('C:/Users/lesli/OneDrive/Documents/GitHub/face_memory/data/facebank/img2.jpg')
    if img is None:
        print('read error')
        exit(1)
    names = recognizer.recognize_faces([img])
    for name in names:
        print(name)
        if name == 'unknown':
            print('Unknown person')

        else:
            print('Welcome back, {}'.format(name))