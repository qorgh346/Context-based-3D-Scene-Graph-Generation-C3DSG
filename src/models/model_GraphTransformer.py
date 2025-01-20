"""
    Paper : A Generalization of Transformer Networks to Graphs
    Original source code :
    Domain : 3D Scene Graph Generation
    Graph Transformer Layer with edge features
    Using Tools : pytorch, torch-geometric

    2023.02.20
"""

import torch.nn as nn
import torch
from torch.fx.experimental.unification.match import edge
from typing import Optional
from torch import Tensor
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

class MultiHeadAttentionLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads, use_bias,node_dim=1):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.node_dim = 0
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def forward(self, x, edge_feature, edge_index):
        # print('1. edge =',edge_index.size())
        # print(edge_index)
        # print(x.size())
        x_src = torch.index_select(x,0,edge_index[0])
        x_dst = torch.index_select(x,0,edge_index[1])

        Q_h = self.Q(x_src)
        K_h = self.K(x_dst)
        V_h = self.V(x_dst)
        proj_e = self.proj_e(edge_feature)

        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)
        proj_e = proj_e.view(-1, self.num_heads, self.out_dim)

        score = (Q_h * K_h) / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores
        score = score * proj_e

        # Copy edge features as e_out to be passed to FFN_e
        e_out = score.clone().detach()
        attention_score = torch.exp((score.sum(-1, keepdim=True)).clamp(-5, 5))

        # Send weighted values to target nodes
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print('after. edge =',edge_index.size())
        # print(edge_index)

        h_out, e_out = self.propagate(edge_index, value=V_h, score=attention_score,edge_feature = e_out)

        return h_out,e_out


    def message(self,value,score,edge_feature):

        wv_j = value * score

        return [wv_j,score,edge_feature]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        wv = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr) #wv
        z = scatter(x[1], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr) #z

        h_out = wv / (z + torch.full_like(z, 1e-6))  # adding eps to all values here

        return h_out,x[2] # [ h_out, e_out ]




class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self,h,e,edge_index):
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(h, e, edge_index)
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            # print('h_in1 = {} , h = {}'.format(h_in1.size(),h.size()))
            h = h_in1 + h  # residual connection
            e = e_in1 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h  # for second residual connection
        e_in2 = e  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # residual connection
            e = e_in2 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e

class GraphTransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 256
        n_layers = 2
        num_heads = 4
        dropout = 0.2
        input_feat = 256
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True

        self.embedding_lap_pos_enc = nn.Linear(8, hidden_dim)
        self.embedding_h = nn.Linear(input_feat, hidden_dim)
        self.in_feat_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                           self.layer_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers - 1)])

        self.embedding_e = nn.Linear(input_feat, hidden_dim)

    def forward(self, x, edge_feature, edge_index, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(x)
        h = self.in_feat_dropout(h)
        # h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        # h = h + h_lap_pos_enc

        e = self.embedding_e(edge_feature)

        # convnets
        for conv in self.layers:
            h, e = conv(h, e, edge_index)
        return h,e

if __name__ == '__main__':
    node_num = 10
    edge_index = torch.randint(0,node_num,size=(2,82))
    node_feature = torch.randn(node_num,128)
    edge_feature = torch.randn(82,128)
    lap_pos_enc = gtf_utils.laplacian_positional_encoding(edge_index,node_num,pos_dim=8)

    model = GraphTransformerNet()
    output = model(node_feature,edge_feature,edge_index,lap_pos_enc)
    print(output)