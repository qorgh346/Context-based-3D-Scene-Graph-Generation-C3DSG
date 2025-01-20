#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional
import torch
from torch import Tensor
from CSGGN.src.networks.networks_base import BaseNetwork, mySequential
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from typing import Any
import math
'''
EdgeGCN used for SGGpoint (Chaoyi Zhang)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GCNConv


def MLP(channels: list, do_bn=True, on_last=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    offset = 0 if on_last else 1
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - offset):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    return mySequential(*layers)


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

##################################################
#                                                #
#                                                #
#  Core Network: EdgeGCN                         #
#                                                #
#                                                #
##################################################

class TwiningAttentionGCNModel(torch.nn.Module):
    def __init__(self, num_node_in_embeddings, num_edge_in_embeddings, AttnEdgeFlag, AttnNodeFlag):
        super(TwiningAttentionGCNModel, self).__init__()

        self.node_GConv1 = GCNConv(num_node_in_embeddings, num_node_in_embeddings // 2, add_self_loops=True)
        self.node_GConv2 = GCNConv(num_node_in_embeddings // 2, num_node_in_embeddings, add_self_loops=True)

        self.edge_MLP1 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings, num_edge_in_embeddings // 2, 1), nn.ReLU())
        self.edge_MLP2 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings // 2, num_edge_in_embeddings, 1), nn.ReLU())

        self.AttnEdgeFlag = AttnEdgeFlag # boolean (for ablaiton studies)
        self.AttnNodeFlag = AttnNodeFlag # boolean (for ablaiton studies)

        # multi-dimentional (N-Dim) node/edge attn coefficients mappings
        self.edge_attentionND = nn.Linear(num_edge_in_embeddings, num_node_in_embeddings // 2) if self.AttnEdgeFlag else None
        self.node_attentionND = nn.Linear(num_node_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

        self.node_indicator_reduction = nn.Linear(num_edge_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

    def concate_NodeIndicator_for_edges(self, node_indicator, batchwise_edge_index):
        node_indicator = node_indicator.squeeze(0)

        edge_index_list = batchwise_edge_index.t()
        subject_idx_list = edge_index_list[:, 0]
        object_idx_list = edge_index_list[:, 1]

        subject_indicator = node_indicator[subject_idx_list]  # (num_edges, num_mid_channels)
        object_indicator = node_indicator[object_idx_list]    # (num_edges, num_mid_channels)

        edge_concat = torch.cat((subject_indicator, object_indicator), dim=1)
        return edge_concat  # (num_edges, num_mid_channels * 2)

    def forward(self, node_feats, edge_feats,edge_index):
        # prepare node_feats & edge_feats in the following formats
        # node_feats: (1, num_nodes,  num_embeddings)
        # edge_feats: (1, num_edges,  num_embeddings)
        # (num_embeddings = num_node_in_embeddings = num_edge_in_embeddings) = 2 * num_mid_channels

        #node_feats, edge_index = node_feats, edge_index
        edge_feats = edge_feats.unsqueeze(dim=0)
        #### Deriving Edge Attention
        if self.AttnEdgeFlag:
            edge_indicator = self.edge_attentionND(edge_feats.squeeze(0)).unsqueeze(0).permute(0, 2, 1)  # (1, num_mid_channels, num_edges)
            raw_out_row = scatter(edge_indicator, edge_index.t()[:, 0].squeeze(0), dim=2, reduce='mean', dim_size=node_feats.size(0)) # (1, num_mid_channels, num_nodes)
            raw_out_col = scatter(edge_indicator, edge_index.t()[:, 1].squeeze(0), dim=2, reduce='mean', dim_size=node_feats.size(0)) # (1, num_mid_channels, num_nodes)
            agg_edge_indicator_logits = raw_out_row * raw_out_col                                        # (1, num_mid_channels, num_nodes)
            agg_edge_indicator = torch.sigmoid(agg_edge_indicator_logits).permute(0, 2, 1).squeeze(0)    # (num_nodes, num_mid_channels)
        else:
            agg_edge_indicator = 1

        #### Node Evolution Stream (NodeGCN)
        node_feats = F.relu(self.node_GConv1(node_feats, edge_index)) * agg_edge_indicator # applying EdgeAttn on Nodes
        node_feats = F.dropout(node_feats, training=self.training)
        node_feats = F.relu(self.node_GConv2(node_feats, edge_index))
        node_feats = node_feats.unsqueeze(0)  # (1, num_nodes, num_embeddings)

        #### Deriving Node Attention
        if self.AttnNodeFlag:
            node_indicator = F.relu(self.node_attentionND(node_feats.squeeze(0)).unsqueeze(0))                  # (1, num_mid_channels, num_nodes)
            agg_node_indicator = self.concate_NodeIndicator_for_edges(node_indicator, edge_index)               # (num_edges, num_mid_channels * 2)
            agg_node_indicator = self.node_indicator_reduction(agg_node_indicator).unsqueeze(0).permute(0,2,1)  # (1, num_mid_channels, num_edges)
            agg_node_indicator = torch.sigmoid(agg_node_indicator)  # (1, num_mid_channels, num_edges)
        else:
            agg_node_indicator = 1

        #### Edge Evolution Stream (EdgeMLP)
        edge_feats = edge_feats.permute(0, 2, 1)                  # (1, num_embeddings, num_edges)
        edge_feats = self.edge_MLP1(edge_feats)                   # (1, num_mid_channels, num_edges)
        edge_feats = F.dropout(edge_feats, training=self.training) * agg_node_indicator    # applying NodeAttn on Edges
        edge_feats = self.edge_MLP2(edge_feats).permute(0, 2, 1)  # (1, num_edges, num_embeddings)

        return node_feats.squeeze(dim=0), edge_feats.squeeze(dim=0)

if __name__ =='__main__':
    num_node_in_embeddings = 256
    num_edge_in_embeddings = 256
    AttnEdgeFlag = True
    AttnNodeFlag = True
    model = TwiningAttentionGCNModel(num_node_in_embeddings,num_edge_in_embeddings,AttnEdgeFlag,AttnNodeFlag)
    num_node = 6
    num_edge = 16
    node_feats = torch.randn((num_node,256))
    edge_feats = torch.randn((num_edge, 256),dtype=torch.float)
    edge_index = torch.randint(0, num_node, [num_edge, 2])
    print(edge_index)
    edge_index = edge_index.t().contiguous()
    print(edge_index.size())
    print(model)
    # print()
    gcn_node_feat , gcn_edge_feat = model(node_feats, edge_feats, edge_index)
    print(gcn_node_feat.size())
    print(gcn_edge_feat.size())