if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import FocalLoss

# 23.02.05 new version

class RelationTypeLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, pred_w,relNum):
        super(RelationTypeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.relNum = relNum
        self.focal_loss_pred = FocalLoss(class_num=relNum, alpha=pred_w, gamma=gamma, size_average=True, use_softmax=True)

    def forward(self,edge_output,gt_rel):
        pred_loss = self.focal_loss_pred(edge_output, gt_rel)
        loss =  self.beta * pred_loss
        return pred_loss,loss

    def prepare_onehot_predgt(self, gt_obj, gt_rel):
        insnum = gt_obj.shape[0]
        onehot_gt = torch.zeros((insnum * insnum - insnum, self.relNum)).to('cuda:0')
        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            if idx_i < idx_j:
                onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(gt_rel[i, 2])] = 1
            elif idx_i > idx_j:
                onehot_gt[int(idx_i * (insnum-1) + idx_j), int(gt_rel[i, 2])] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt



class TotalLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, obj_w, pred_w,objNum,relNum):
        super(TotalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.objNum = objNum
        self.relNum = relNum
        self.focal_loss_pred = FocalLoss(target='rel', class_num=relNum, alpha=pred_w, gamma=gamma, size_average=True, use_softmax=True)
        self.focal_loss_obj = FocalLoss(target ='obj', class_num=objNum, alpha=obj_w, gamma=gamma, size_average=True, use_softmax=False)

    # def forward(self, node_output, edge_output, gt_obj, gt_rel):
        # objgt_onehot = self.prepare_onehot_objgt(gt_obj)
        # predgt_onehot = self.prepare_onehot_predgt(gt_obj, gt_rel)
        # obj_loss = self.focal_loss_obj(node_output, objgt_onehot)
    def forward(self,node_output, edge_output,gt_obj_cls,gt_rel_cls): #obj_pred, rel_pred, gt_obj_cls, gt_rel_cls
        objgt_onehot = self.prepare_onehot_objgt(gt_obj_cls)
        # predgt_onehot = self.prepare_onehot_predgt(gt_obj, gt_rel)
        obj_loss = self.focal_loss_obj(node_output, objgt_onehot)

        pred_loss = self.focal_loss_pred(edge_output, gt_rel_cls)
        loss = self.alpha * obj_loss + self.beta * pred_loss
        return obj_loss,pred_loss,loss

    def prepare_onehot_predgt(self, gt_obj, gt_rel):
        insnum = gt_obj.shape[0]
        onehot_gt = torch.zeros((insnum * insnum - insnum, self.relNum))
        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            if idx_i < idx_j:
                onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(gt_rel[i, 2])] = 1
            elif idx_i > idx_j:
                onehot_gt[int(idx_i * (insnum-1) + idx_j), int(gt_rel[i, 2])] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt

    def prepare_onehot_objgt(self, gt_obj):
        insnum = gt_obj.shape[0]
        onehot = torch.zeros(insnum, self.objNum).float().to(gt_obj.device)
        for i in range(insnum):
            onehot[i, gt_obj[i]] = 1
        return onehot


