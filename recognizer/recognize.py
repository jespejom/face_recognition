from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
#from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
from utils import load_facebank, draw_box_name, prepare_facebank
from config import get_config

class FaceRecognizer(object):
    def __init__(self, conf, update_fb = True):

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

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path       
        self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str), map_location=torch.device('cpu')))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
    
    def infer(self, conf, faces, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - self.targets.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum     

    def recognize_faces(self, faces):
        recog_names = []
        results, score = self.infer(conf, faces, tta=False)
        for idx, res in enumerate(results):
            name = self.names[results[idx] + 1]
            recog_names.append(name)
            print(name)
        return recog_names

if __name__ == '__main__':
    conf = get_config(training = False, mobile = True)
    recognizer = FaceRecognizer(conf)
    img = Image.open('./data/facebank/taylor/face00.jpg')
    if img is None:
        print('read error')
        exit(1)
    names = recognizer.recognize_faces([img])
    
