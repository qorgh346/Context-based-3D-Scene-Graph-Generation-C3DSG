#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:46:24 2020

@author: sc
"""
import sys

import torch
from torch_geometric.nn.conv import MessagePassing
from networks_base import mySequential

def MLP(channels: list, do_bn=False, on_last=False, drop_out=None):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    offset = 0 if on_last else 1
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-offset):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
            
            if drop_out is not None:
                layers.append(torch.nn.Dropout(drop_out))
    return mySequential(*layers)

#[1_dim_in,1_dim_out,2_dim_in,2_dim_out,...]
def build_mlp(dim_list, activation='relu', do_bn=False,
              dropout=0, on_last=False):
   layers = []
   for i in range(len(dim_list) - 1):
     dim_in, dim_out = dim_list[i], dim_list[i + 1]
     layers.append(torch.nn.Linear(dim_in, dim_out))
     final_layer = (i == len(dim_list) - 2)
     if not final_layer or on_last:
       if do_bn:
         layers.append(torch.nn.BatchNorm1d(dim_out))
       if activation == 'relu':
         layers.append(torch.nn.ReLU())
       elif activation == 'leakyrelu':
         layers.append(torch.nn.LeakyReLU())
     if dropout > 0:
       layers.append(torch.nn.Dropout(p=dropout))
   return torch.nn.Sequential(*layers)


class Gen_Index(MessagePassing):
    """ A sequence of scene graph convolution layers  """
    def __init__(self,flow="source_to_target"):
        super().__init__(flow=flow)
        
    def forward(self, x, edges_indices):
        # print('x size -> ',x.size()) #x = (19,512)
        # print('edge_index .size = ',edges_indices.size())
        # print('edge_index', edges_indices)

        # sys.exit()
        # print('???--> edge_indices',edges_indices.size())
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(self.__user_args__,edges_indices,size, {"x":x})
        # print("coll_dict['x_i'].size() --> ",coll_dict['x_i'].size())
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        # print(msg_kwargs)
        x_i, x_j = msg_kwargs['x_i'],msg_kwargs['x_j'] #edit hojun
        #x_i, x_j =  self.message(**msg_kwargs)
        # print('mesaage_passing after x_i sizse = ',x_i.size())
        # print('mesaage_passing after x_i  = ',x_i)
        # print('\n\n')
        # print('mesaage_passing after x_j sizse = ', x_j.size())
        # print('mesaage_passing after x_j  = ', x_j)

        # sys.exit()
        return x_i, x_j
    def message(self, x_i, x_j):
        return x_i,x_j

class Aggre_Index(MessagePassing):
    def __init__(self,aggr='add', node_dim=-2,flow="source_to_target"):
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)
    def forward(self, x,edge_index,dim_size):
        size = self.__check_input__(edge_index, None)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size,{})
        coll_dict['dim_size'] = dim_size
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        x = self.aggregate(x, **aggr_kwargs)
        return x

if __name__ == '__main__':
    flow = 'source_to_target'
    # flow = 'target_to_source'
    g = Gen_Index(flow = flow)
    
    edge_index = torch.LongTensor([[0,1,2],
                                  [2,1,0]])
    x = torch.zeros([3,5])
    x[0,:] = 0
    x[1,:] = 1
    x[2,:] = 2
    x_i,x_j = g(x,edge_index)
    # print('x_i',x_i)
    # print('x_j',x_j)
    
    tmp = torch.zeros_like(x_i)
    tmp = torch.zeros([5,2])
    edge_index = torch.LongTensor([[0,1,2,1,0],
                                  [2,1,1,1,1]])
    for i in range(5):
        tmp[i] = -i
    aggr = Aggre_Index(flow=flow,aggr='max')
    xx = aggr(tmp, edge_index,dim_size=x.shape[0])
    # print(x)
    # print(xx)