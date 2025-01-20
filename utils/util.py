import os,json
import torch.nn as nn
#edit hojun
import torch
import torch.nn.functional as F
from torch.autograd import Variable
class MLP(nn.Module):
    def __init__(self, mlp=[1024, 512, 512], dropout=False, log_sm=False, bn=False, ln=False, bias=True):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(mlp)-1):
            layers.append(nn.Linear(mlp[i], mlp[i+1], bias=bias))
            if bn and i != len(mlp) - 2:
                layers.append(nn.BatchNorm1d(mlp[i+1]))
            if ln and i != len(mlp) - 2:
                layers.append(nn.LayerNorm(mlp[i+1]))
            if i != len(mlp) - 2:
                layers.append(nn.ReLU())
            if dropout and i == 0:
                layers.append(nn.Dropout(0.2))
            if i == len(mlp)-2 and log_sm:
                layers.append(nn.LogSoftmax(dim=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        y = self.layers(x)
        return y
def set_random_seed(seed):
    import random,torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def check_file_exist(path):
    if not os.path.exists(path):
            raise RuntimeError('Cannot open file. (',path,')')

def read_txt_to_list(file):
    output = []
    print('read_txt_to_list - > file name = ',file)
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry)

    f.close()
    return output

def check_file_exist(path):
    if not os.path.exists(path):
            raise RuntimeError('Cannot open file. (',path,')')
            

def read_classes(read_file):
    obj_classes = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            obj_class = line.rstrip().lower() 
            obj_classes.append(obj_class)

    f.close()
    return obj_classes 


def read_relationships(read_file):
    relationships = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            relationship = line.rstrip().lower() 
            relationships.append(relationship)
    f.close()
    return relationships 



def load_semseg(json_file, name_mapping_dict=None, mapping = True):    
    '''
    Create a dict that maps instance id to label name.
    If name_mapping_dict is given, the label name will be mapped to a corresponding name.
    If there is no such a key exist in name_mapping_dict, the label name will be set to '-'

    Parameters
    ----------
    json_file : str
        The path to semseg.json file
    name_mapping_dict : dict, optional
        Map label name to its corresponding name. The default is None.
    mapping : bool, optional
        Use name_mapping_dict as name_mapping or name filtering.
        if false, the query name not in the name_mapping_dict will be set to '-'
    Returns
    -------
    instance2labelName : dict
        Map instance id to label name.

    '''
    instance2labelName = {}
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for segGroups in data['segGroups']:
            # print('id:',segGroups["id"],'label', segGroups["label"])
            # if segGroups["label"] == "remove":continue
            labelName = segGroups["label"]
            if name_mapping_dict is not None:
                if mapping:
                    if not labelName in name_mapping_dict:
                        labelName = 'none'
                    else:
                        labelName = name_mapping_dict[labelName]
                else:
                    if not labelName in name_mapping_dict.values():
                        labelName = 'none'

            instance2labelName[segGroups["id"]] = labelName.lower()#segGroups["label"].lower()
    return instance2labelName

class MLP(nn.Module):
    def __init__(self, mlp=[1024, 512, 512], dropout=False, log_sm=False, bn=False, ln=False, bias=True):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(mlp)-1):
            layers.append(nn.Linear(mlp[i], mlp[i+1], bias=bias))
            if bn and i != len(mlp) - 2:
                layers.append(nn.BatchNorm1d(mlp[i+1]))
            if ln and i != len(mlp) - 2:
                layers.append(nn.LayerNorm(mlp[i+1]))
            if i != len(mlp) - 2:
                layers.append(nn.ReLU())
            if dropout and i == 0:
                layers.append(nn.Dropout(0.2))
            if i == len(mlp)-2 and log_sm:
                layers.append(nn.LogSoftmax(dim=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y


class FocalLoss(nn.Module):
    def __init__(self,target , class_num, alpha=None, gamma=2, size_average=True, use_softmax=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num)) #cuda())
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.target = target
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.use_softmax = use_softmax
        self.reduction = 'mean'
    def forward(self, inputs, class_mask):
        ''' inputs: output from a linear layer
            class_mask: onehot matrix
        '''
        if self.target == 'rel':

            p = torch.where(class_mask >= 0.5, inputs, 1 - inputs)
            logp = - torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
            loss = logp * ((1 - p) ** self.gamma)
            loss = self.class_num * loss.mean()
            return loss

            # # BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
            # BCE_loss = inputs #torch.exp(-inputs)
            # pt = torch.exp(-BCE_loss)
            # class_mask = Variable(class_mask)
            # if inputs.is_cuda and not self.alpha.is_cuda:
            #     self.alpha = self.alpha.to(inputs.device)  # cuda()
            # # alpha = (self.alpha * class_mask).sum(1).view(-1, 1)
            #
            # # probs = (P * class_mask).sum(1).view(-1, 1)
            #
            # log_p = pt.log()
            #
            # batch_loss = -self.alpha * (1 - pt) ** self.gamma * log_p
            # # batch_loss = -self.alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # if self.size_average:
            #     loss = batch_loss.mean()
            # else:
            #     loss = batch_loss.sum()
            # return loss

        else:
            if self.use_softmax:
                P = F.softmax(inputs, dim=1)
            else:
                # P = inputs
                P = torch.exp(inputs.detach()).type(inputs.dtype)
            class_mask = Variable(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.to(inputs.device)  # cuda()
            alpha = (self.alpha * class_mask).sum(1).view(-1, 1)

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
            return loss

    def focal_loss(self, gamma, at, logpt, label):
        label = label.view(-1, 1).contiguous()
        logpt = logpt.gather(1, label)
        at = Variable(at.gather(0, label.data.view(-1))) #cuda()
        pt = Variable(logpt.data.exp()) #cuda().cuda()
        loss = torch.mean(-1 * (1 - pt) ** gamma * logpt)
        return loss
