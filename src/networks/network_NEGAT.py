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
from torch_scatter import scatter, segment_csr, gather_csr
from typing import Any
import math

# 참고 코드 : GAT
# 기존 GAT에서 Edge Feature 고려할 수 있도록 변경
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAT.html#torch_geometric.nn.models.GAT


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


class NE_GAT(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden
        self.heads = 1
        # temp

        # self.nn1 = build_mlp([dim_node * 2 + dim_edge, dim_hidden, dim_hidden * 2 + dim_edge],
        #                      do_bn=use_bn, on_last=True)
        self.nn1 = build_mlp([dim_edge * 2 + dim_edge, dim_hidden, dim_hidden * 2 + dim_edge],
                             do_bn=use_bn, on_last=True)

        self.nn2 = build_mlp([dim_hidden, dim_hidden, dim_node], do_bn=use_bn)
        self.nn3 = build_mlp([dim_hidden, dim_hidden, dim_edge], do_bn=use_bn)
        self.lin_src = Linear(dim_node,  dim_hidden, bias=False, weight_initializer='glorot')

        self.lin_dst = self.lin_src
        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, dim_hidden))
        self.att_dst = Parameter(torch.Tensor(1, dim_hidden))
        # print(self.att_src)
        if dim_edge is not None:
            self.lin_edge = Linear(dim_edge, self.heads * dim_hidden, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, dim_hidden))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        self.glorot(self.att_src)
        self.glorot(self.att_dst)
        self.glorot(self.att_edge)
        # self.zeros(self.bias)


    def forward(self, x, edge_feature, edge_index):
        H, C = self.heads, self.dim_hidden
        if isinstance(x, Tensor):
            assert x.dim() == 2,
            x_src = x_dst = self.lin_src(x).view(-1,C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2,
            x_src = self.lin_src(x_src).view(-1,C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1,C)

        # x = (x_src, x_dst)
        # # a = (x_src * self.att_src)
        #

        alpha_src = (x_src * self.att_src).sum(dim=-1).unsqueeze(dim=1)
        alpha_dst = (x_dst * self.att_dst).sum(-1).unsqueeze(dim=1)
        alpha = (alpha_src, alpha_dst)

        gcn_x, gcn_e = self.propagate(edge_index, x=x,alpha=alpha,edge_feature=edge_feature)
        x = self.nn2(gcn_x)
        gcn_e = self.nn3(gcn_e)
        # print('gcn_e.size() = ',gcn_e.size())
        return x, gcn_e

    def message(self, x_i: Tensor, x_j: Tensor,
                alpha_i: OptTensor,alpha_j:OptTensor,
                edge_feature: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        # (4) attention weight _ a_vivj

        edge_feature = self.lin_edge(edge_feature)
        edge_feature = edge_feature.view(-1, self.dim_hidden)
        alpha_edge = (edge_feature * self.att_edge).sum(dim=-1)

        alpha_ij = alpha_j if alpha_i is None else (alpha_j + alpha_i).view(-1)
        alpha_v_ij = F.leaky_relu(alpha_ij, 0.2)
        alpha_v_ij_edge = alpha_ij + alpha_edge
        alpha_v_ij_edge = F.leaky_relu(alpha_v_ij_edge, 0.2)

        src_v_ij_max = scatter(alpha_ij, index, 0, dim_size=size_i, reduce='max')
        src_v_ij_max = src_v_ij_max.index_select(0, index)
        out_v_ij = (alpha_ij - src_v_ij_max).exp()
        out_sum_v_ij = scatter(out_v_ij, index, 0, dim_size=size_i, reduce='sum')
        out_sum_v_ij = out_v_ij.index_select(0, index)

        src_v_ij_edge_max = scatter(alpha_v_ij_edge, index, 0, dim_size=size_i, reduce='max')
        src_v_ij_edge_max = src_v_ij_edge_max.index_select(0, index)
        out_v_ij_edge = (alpha_v_ij_edge - src_v_ij_edge_max).exp()
        out_sum_v_ij_edge = scatter(out_v_ij_edge, index, 0, dim_size=size_i, reduce='sum')
        out_sum_v_ij_edge = out_sum_v_ij_edge.index_select(0, index)

        pair_NodeAttention = out_v_ij /(out_sum_v_ij  + out_sum_v_ij_edge + 1e-16)
        pair_NodeAttention = F.dropout(pair_NodeAttention,0.1)

        node_EdgeAttention = out_v_ij_edge /(out_sum_v_ij  + out_sum_v_ij_edge + 1e-16)
        node_EdgeAttention = F.dropout(node_EdgeAttention,0.1)

        EdgeAttention = F.dropout(softmax(alpha_v_ij_edge,index,ptr,size_i),p=0.1) #edge_ij_v_i

        # ### torch geometric softmax function ### #
        # Given a value tensor :attr:`src`, this function first groups the values
        # along the first dimension based on the indices specified in :attr:`index`,
        # and then proceeds to compute the softmax individually for each group.

        x = torch.cat([x_i, edge_feature, x_j], dim=1)
        x = self.nn1(x)  # .view(b,-1)
        new_x_i = x[:, :self.dim_hidden]
        new_e = x[:, self.dim_hidden:(self.dim_hidden + self.dim_edge)]
        new_x_j = x[:, (self.dim_hidden + self.dim_edge):]
        x = new_x_i + new_x_j

        N2N_x_i  = new_x_i * pair_NodeAttention.unsqueeze(dim=-1) + new_e * node_EdgeAttention.unsqueeze(-1)
        N2N_x_j = new_x_j * pair_NodeAttention.unsqueeze(dim=-1) + new_e * node_EdgeAttention.unsqueeze(-1)
        N2N_x = N2N_x_i + N2N_x_j

        E2N_x = new_e * EdgeAttention.unsqueeze(dim=-1) +  new_x_i * node_EdgeAttention.unsqueeze(-1) \
                + new_x_j * node_EdgeAttention.unsqueeze(dim=-1)

        # x_j = x_j.mean(dim=1)
        # x_i = x_i.mean(dim=1)
        # print(E2N_x)
        # print(N2N_x)

        return [N2N_x, E2N_x]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x

    def zeros(value: Any):
        constant(value, 0.)

    def glorot(self,value: Any):
        if isinstance(value, Tensor):
            stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
            value.data.uniform_(-stdv, stdv)
        else:
            for v in value.parameters() if hasattr(value, 'parameters') else []:
                glorot(v)
            for v in value.buffers() if hasattr(value, 'buffers') else []:
                glorot(v)

class NEGATModel(BaseNetwork):
    """ proposal models : NE-GAT """

    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(NE_GAT(**kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(node_feature, edge_feature, edges_indices)

            if i < (self.num_layers - 1):
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
        return node_feature, edge_feature


if __name__ == '__main__':
    num_layers = 2
    dim_node = 256
    dim_edge = 256
    dim_hidden = 256
    num_node = 6
    num_edge = 16
    heads = 1
    x = torch.rand(num_node, dim_node)
    edge_feature = torch.rand([num_edge, dim_edge], dtype=torch.float)
    edge_index = torch.randint(0, num_node, [num_edge, 2])
    edge_index = edge_index.t().contiguous()

    net = CSGGN_Model(num_layers, dim_node=dim_node, dim_edge=dim_edge, dim_hidden=dim_hidden)
    y = net(x, edge_feature, edge_index)
    print(y)

    pass