'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/Update.py
'''
from time import sleep
import torch
import numpy as np
import random
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Dataset
from ..config import for_FL as f
from torchvision import transforms
from ..FL.add_noise import *

import io
import PIL.Image as pilGG
from wand.image import Image


random.seed(f.seed)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        #print('item:',item)
        image, label = self.dataset[self.idxs[item]]

        return image, label


class LocalUpdate_poison(object):

    def __init__(self, dataset = None, idxs = None, user_idx = None, attack_idxs = None):
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = f.local_bs, shuffle = True)
        self.user_idx = user_idx
        # id of attackers
        self.attack_idxs = attack_idxs
        self.attacker_flag = False

    def train(self, net):
        net.train()
        tmp_pos = 0
        tmp_all = 0
        origin_weights = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.SGD(net.parameters(), lr = f.lr, momentum = f.momentum)

        # loss of local epochs
        epoch_loss = []

        for iter in range(f.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                perm = np.random.permutation(len(labels))[0: int(len(labels) * 0.5)]

                for label_idx in range(len(labels)):

                    if (f.attack_mode == 'poison') and (self.user_idx in self.attack_idxs) and label_idx in perm:
                        self.attacker_flag = True
                        labels[label_idx] = f.target_label

                        #### ADD TRIGGER ####
                        TOPIL = transforms.ToPILImage()
                        TOtensor = transforms.ToTensor()

                        im = TOPIL(images[label_idx])
                        
                        #### ADD NOISE ####
                        
                        im.save("tmp.png")

                        # Read image using Image() function
                        with Image(filename="tmp.png") as img:

                            # Generate noise image using spread() function
                            img.noise("gaussian", attenuate = 0.9)

                            # wand to PIL
                            img_buffer = np.asarray(bytearray(img.make_blob(format='png')), dtype='uint8')
                            bytesio = io.BytesIO(img_buffer)
                            pil_img = pilGG.open(bytesio)                       
                        

                            images[label_idx] = TOtensor(pil_img)
                    else:
                        pass

                    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    Normal = transforms.Normalize(*stats,inplace=True)
                    images[label_idx] = Normal(images[label_idx])
                    

                images, labels = images.to(f.device), labels.to(f.device)
                net.zero_grad()
                # probability for each label
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            if f.local_verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                        iter, epoch_loss[iter]))
        
        print('activating~')

        # model after local training
        trained_weights = copy.deepcopy(net.state_dict())

        # large paremeter
        if(f.scale==True):
            scale_up = 20
        else:
            scale_up = 1

        if (f.attack_mode == "poison") and self.attacker_flag:

            attack_weights = copy.deepcopy(origin_weights)

            # parameter of original model
            for key in origin_weights.keys():
                # diff for original and update parameter
                difference =  trained_weights[key] - origin_weights[key]
                # new weights
                attack_weights[key] += scale_up * difference

            # if it is under attack
            return attack_weights, sum(epoch_loss)/len(epoch_loss), self.attacker_flag

        # if it is not under attack
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.attacker_flag
