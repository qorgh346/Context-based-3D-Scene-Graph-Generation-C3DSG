import argparse
import math
import h5py
import numpy as np
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import math
import random
import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from utils.pointcnn_Model import RandPointCNN
from utils.util_layers import Dense
from provider import knn_indices_func_gpu,shuffle_data,jitter_point_cloud,rotate_point_cloud


class Classifier(nn.Module):

    def __init__(self,pointcnn_dim):
        super(Classifier, self).__init__()
        AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)
        self.pointcnn_dim = pointcnn_dim
        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 128, 12, 4, 120),
            AbbPointCNN(128, 512, 12, 6, 120) # arg2 : output feature
        )
        self.fcn = nn.Sequential(
            Dense(512, 384),
            Dense(384, pointcnn_dim, drop_rate=0.5),
        )

    def forward(self, points):

        # Augment batched point clouds by rotation and jittering
        points = points.cpu().numpy()
        rotated_data = rotate_point_cloud(points[:,:,:3])
        jittered_data = jitter_point_cloud(rotated_data)  # P_Sampled

        P_sampled = jittered_data
        P_sampled = torch.from_numpy(P_sampled).float()
        P_sampled = Variable(P_sampled, requires_grad=False).cuda()

        x = self.pcnn1((P_sampled,P_sampled))
        x_ = self.pcnn2(x)
        x = self.pcnn2(x)[1]  # grab features
        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits



if __name__ == '__main__':
    print("------Building model-------")
    model = Classifier(pointcnn_dim=512).cuda()
    print("------Successfully Built model-------")

    pointClouds = torch.randn((24,256,6)).float().cuda() #object num : 24 , #point num = 256

    print(model)
    out = model(pointClouds)

    print(out)