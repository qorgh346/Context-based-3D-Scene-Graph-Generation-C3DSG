#!/usr/bin/env python3
# -*- coding: utf-8 -*-
if __name__ == '__main__' and __package__ is None:
    from os import sys

    sys.path.append('../')
import torch
import torch.optim as optim
import torch.nn.functional as F
from model_base import BaseModel
from network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti
# from network_PointNet2 import PointNet2feat
from network_DGCNN import DGCNNfeat

from network_TripletGCN import TripletGCNModel
from network_CSGGN import NEGATModel
from TwiningAttentionGCN import TwiningAttentionGCNModel
from network_HGT import HGT
from network_GNN import GraphEdgeAttenNetworkLayers

from network_RelationPrediction import RelCls

from config import Config
import op_utils
import optimizer
import math
from model_TransformerNet import Transformer_cfg, PointTransformerfeat
import sys
from utils.util_data import build_word_embedding
from utils.util import MLP
from utils.Loss import TotalLoss,RelationTypeLoss
import torch.nn as nn

import os.path as osp

import torch
import torch.nn.functional as F




# HGT_3DSSG ver. 23.02.02

class HGTModel(BaseModel):
    # obj_w=self.weight_obj, pred_w=self.weight_rel
    def __init__(self, config: Config, name: str, num_class, num_rel, classNames, weight_obj, weight_rel,
                 hidden_channels, out_channels, num_heads, num_layers,meta_data,prior_knowledge):

        super().__init__(name, config)
        models = dict()
        self.config = config
        self.mconfig = mconfig = self.config.MODEL
        self.transformer_config = self.config.TransformerModel
        with_bn = mconfig.WITH_BN
        self.flow = 'source_to_target'  # we want the mess
        self.classNames = classNames
        dim_point = 3
        dim_point_rel = 3
        self.debug = True
        self.embedding_model = build_word_embedding(self.config.WORD_VECTOR_TEXT)
        self.num_rel = num_rel
        self.num_class = num_class
        self.weight_obj = weight_obj
        self.weight_rel = weight_rel
        if mconfig.USE_RGB:
            dim_point += 3
            dim_point_rel += 3
        if mconfig.USE_NORMAL:
            dim_point += 3
            dim_point_rel += 3

        dim_input_rel = 0
        if self.config.Edge_Feature_Selection.USE_Relative_Scale and self.config.Edge_Feature_Selection.USE_Relative_Position:
            dim_input_rel += 11
        elif self.config.Edge_Feature_Selection.USE_Relative_Scale:
            dim_input_rel += 6
        elif self.config.Edge_Feature_Selection.USE_Relative_Position:
            dim_input_rel += 5

        if self.config.Edge_Feature_Selection.USE_Object_Label:
            dim_input_rel += 100
        if self.config.Edge_Feature_Selection.USE_Scene_Label:
            dim_input_rel += 100

        dim_node = self.mconfig.point_feature_size  # 256

        if mconfig.USE_CONTEXT:
            dim_point_rel += 1


        #####################
        # Object Encoder
        if mconfig.USE_PointNet:
            models['obj_encoder'] = PointNetfeat(
                global_feat=True,
                batch_norm=with_bn,
                point_size=dim_point,
                input_transform=False,
                feature_transform=mconfig.feature_transform,
                out_size=mconfig.point_feature_size)
        # elif mconfig.USE_PointNet2:
        #     models['obj_encoder'] = PointNet2feat(
        #         point_size=dim_point,
        #         out_size=mconfig.point_feature_size,
        #         normal_channel=False
        #     )
        elif mconfig.USE_PointTransformer:

            self.transformer_config.num_point = self.config.dataset.num_points
            self.transformer_config.input_dim = dim_point - 3  # PointTransformer --> norm_xyz 제외
            self.transformer_config.transformer_dim = self.mconfig.point_feature_size
            self.transformer_config.relation_point = False
            obj_encoder = PointTransformerfeat(self.transformer_config)
            models['obj_encoder'] = self.transformer_Load(obj_encoder)

        elif mconfig.USE_DGCNN:
            models['obj_encoder'] = DGCNNfeat(
                args=config.DGCNN,
                output_channels=mconfig.point_feature_size
            )
        #####################

        input_dim = (dim_node * 2) + dim_input_rel

        models['non_geoemtric_encoder'] = MLP(mlp=[11,self.mconfig.point_feature_size])

        if mconfig.USE_class_categories:
            models['obj_cls_distrubtion'] = torch.nn.Sequential(
                torch.nn.Linear(self.mconfig.point_feature_size, self.mconfig.point_feature_size // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.mconfig.point_feature_size // 2, self.mconfig.point_feature_size // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(self.mconfig.point_feature_size // 4, num_class),
            )

        if mconfig.GCN_TYPE == "TRIP":
            # SGPN
            models['gcn'] = TripletGCNModel(num_layers=mconfig.N_LAYERS,
                                            dim_node=mconfig.point_feature_size,
                                            dim_edge=mconfig.edge_feature_size,
                                            dim_hidden=mconfig.gcn_hidden_feature_size)

            # TripletGCNModel(num_layers=mconfig.N_LAYERS,
            #                             dim_node=mconfig.point_feature_size,
            #                             dim_edge=mconfig.edge_feature_size,
            #                             dim_hidden=mconfig.gcn_hidden_feature_size)
        elif mconfig.GCN_TYPE == 'EAN':
            # Scene Graph Fusion
            models['gcn'] = GraphEdgeAttenNetworkLayers(self.mconfig.point_feature_size,
                                                        self.mconfig.edge_feature_size,
                                                        self.mconfig.DIM_ATTEN,
                                                        self.mconfig.N_LAYERS,
                                                        self.mconfig.NUM_HEADS,
                                                        self.mconfig.GCN_AGGR,
                                                        flow=self.flow)
        elif mconfig.GCN_TYPE == 'EDGE_GCN':
            # Edge GCN
            models['gcn'] = TwiningAttentionGCNModel(
                num_node_in_embeddings=mconfig.point_feature_size,
                num_edge_in_embeddings=mconfig.edge_feature_size,
                AttnEdgeFlag=True,
                AttnNodeFlag=True)

        elif mconfig.GCN_TYPE == 'NE_GAT':
            models['gcn'] = NEGATModel(num_layers=mconfig.N_LAYERS,
                                       dim_node=mconfig.point_feature_size,
                                       dim_edge=mconfig.edge_feature_size,
                                       dim_hidden=mconfig.gcn_hidden_feature_size)
        elif mconfig.GCN_TYPE == 'HGT':
            models['hgtConv'] = HGT(hidden_channels=mconfig.point_feature_size,
                                    out_channels = mconfig.gcn_hidden_feature_size,
                                    prior_knowledge = prior_knowledge,
                                    num_heads=2,
                                    num_layers=1)

        # node feature classifier
        # models['obj_predictor'] = PointNetCls(num_class, in_size=mconfig.point_feature_size,
        #                                       batch_norm=with_bn, drop_out=True)

        models['rel_predictor'] = RelCls(
            config,prior_knowledge
        )

        # if mconfig.multi_rel_outputs:
        #     models['rel_predictor'] = PointNetRelClsMulti(
        #         num_rel,
        #         in_size=mconfig.edge_feature_size,
        #         batch_norm=with_bn, drop_out=True)
        # else:
        #     models['rel_predictor'] = PointNetRelCls(
        #         num_rel,
        #         in_size=mconfig.edge_feature_size,
        #         batch_norm=with_bn, drop_out=True)

        params = list()
        print('==trainable parameters==')
        for name, model in models.items():
            model = model.to(config.GPU[0])
            self.add_module(name, model)
            params += list(model.parameters())
            print(name, op_utils.pytorch_count_params(model))
        print('')
        self.optimizer = optim.AdamW(
            params=params,
            lr=float(config.LR),
            weight_decay=self.config.W_DECAY,
            amsgrad=self.config.AMSGRAD
        )
        self.optimizer.zero_grad()

        self.scheduler = None
        if self.config.LR_SCHEDULE == 'BatchSize':
            def _scheduler(epoch, batchsize):
                return 1 / math.log(20)

            #                return 1 / math.log(batchsize) if batchsize > 1 else 1
            self.scheduler = optimizer.BatchMultiplicativeLR(self.optimizer, _scheduler)

        print(models)
        if self.weight_rel[0] == 0:
            self.weight_rel[0] += 0.15
        temp_rel_w = torch.ones_like(self.weight_rel)
        temp_rel_w[0] = 0.1

        self.criterion = torch.nn.ModuleDict()

        for edgeType in prior_knowledge['edgeType'].keys():
            self.criterion[edgeType] = RelationTypeLoss(alpha=1,beta=1,gamma=2,pred_w=None,
                                                        relNum = len(prior_knowledge['edgeType'][edgeType]))

    def forward(self, heteroData):
        obj_x_dict = {
            node_type: self.obj_encoder(x)
            for node_type, x in heteroData.x_dict.items()
        }

        for node_type, x in obj_x_dict.items():
            heteroData[node_type].x = x

        for node_type, descriptor in heteroData.descriptor_dict.items():
            encoded_non_geo_feat = self.non_geoemtric_encoder(descriptor[:,:11])
            heteroData[node_type].descriptor = encoded_non_geo_feat


        probs = None
        if self.mconfig.USE_GCN:
            if self.mconfig.GCN_TYPE == 'TRIP':
                # edges = edges.permute(1,0)
                gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'EAN':
                gcn_obj_feature, gcn_rel_feature, probs = self.gcn(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'NE_GAT':
                gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'EDGE_GCN':
                gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, edges)

            elif self.mconfig.GCN_TYPE == 'HGT':
                updated_obj_dict = self.hgtConv(obj_x_dict, heteroData.edge_index_dict)

        rel_pred_output = self.rel_predictor(heteroData,updated_obj_dict)

        result = {'rel_pred_output':rel_pred_output, 'obj_pred_output':None}

        return result


    def process(self, heteroData,edge_storage,mode):
        self.iteration += 1
        result = self(heteroData)
        obj_pred_dict, rel_pred_dict = result['obj_pred_output'], result['rel_pred_output']



        loss = 0
        loss_storage = {'support': 0, 'proximity': 0, 'inclusion': 0, 'comparative': 0}
        for edge_type,v in rel_pred_dict.items():
            pred_output = rel_pred_dict[edge_type]
            gt_output = edge_storage[edge_type]['gt_label']
            loss_rel = self.criterion[edge_type](pred_output, gt_output)[1]
            loss_storage[edge_type] = loss_rel.detach().item()
            loss += loss_rel

        if mode == 'train':
            self.backward(loss)

            if self.scheduler is not None:
                self.scheduler.step()

        logs = [
                ("Loss/support_loss", loss_storage['support']),
            ("Loss/proximity_loss", loss_storage['proximity']),
            ("Loss/inclusion_loss", loss_storage['inclusion']),
            ("Loss/comparative_loss", loss_storage['comparative']),
            ("Loss/loss", loss.detach().item())]

        loss_info = loss.detach().item()

        # probs = result['probs']

        return logs, rel_pred_dict, loss_info,loss_storage

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()




    def transformer_Load(self, model):
        pretrain_path = '/home/baebro/hojun_ws/3D_TripletAttentionSGG[3]/backbone_checkpoints/best_model.pth'
        model_dict = model.state_dict()
        temp_pretrained_dict = dict()
        pretrained_dict = torch.load(pretrain_path)
        for k, v in pretrained_dict['model_state_dict'].items():
            if k in model_dict and 'backbone.fc1' not in k and 'fc2' not in k.split('.')[0]:
                temp_pretrained_dict[k] = v
        model_dict.update(temp_pretrained_dict)
        model.load_state_dict(model_dict, strict=True)

        for name, param in model.named_parameters():
            names = name.split('.')
            if names[1] == 'transformers':
                if names[2] == '2':
                    param.requires_grad = True
            param.requires_grad = False
        # sys.exit()
        return model

if __name__ == '__main__':
    use_dataset = True

    config = Config('../config_hojun.json')
    config.MODEL.USE_RGB = False
    config.MODEL.USE_NORMAL = False
    print(config)
    if not use_dataset:
        num_obj_cls = 40
        num_rel_cls = 26
    else:
        from src.dataset_builder import build_dataset

        # config.dataset.dataset_type = 'rio_graph'
        dataset = build_dataset(config, 'validation_scans', True, multi_rel_outputs=False, use_rgb=False,
                                use_normal=False)
        num_obj_cls = len(dataset.classNames)
        num_rel_cls = len(dataset.relationNames)

    # build model
    mconfig = config.MODEL
    network = SGAPNModel(config, 'SGAPNModel', num_obj_cls, num_rel_cls)

    if not use_dataset:
        max_rels = 80
        n_pts = 10
        n_rels = n_pts * n_pts - n_pts
        n_rels = max_rels if n_rels > max_rels else n_rels
        obj_points = torch.rand([n_pts, 3, 128])
        rel_points = torch.rand([n_rels, 4, 256])
        edges = torch.zeros(n_rels, 2, dtype=torch.long)
        counter = 0
        for i in range(n_pts):
            if counter >= edges.shape[0]: break
            for j in range(n_pts):
                if i == j: continue
                if counter >= edges.shape[0]: break
                edges[counter, 0] = i
                edges[counter, 1] = i
                counter += 1

        obj_gt = torch.randint(0, num_obj_cls - 1, (n_pts,))
        rel_gt = torch.randint(0, num_rel_cls - 1, (n_rels,))

        # rel_gt
        adj_rel_gt = torch.rand([n_pts, n_pts, num_rel_cls])
        rel_gt = torch.zeros(n_rels, num_rel_cls, dtype=torch.float)

        for e in range(edges.shape[0]):
            i, j = edges[e]
            for c in range(num_rel_cls):
                if adj_rel_gt[i, j, c] < 0.5: continue
                rel_gt[e, c] = 1

        network.process(obj_points, rel_points, edges, obj_gt, rel_gt)

    for i in range(100):
        if use_dataset:
            scan_id, instance2mask, obj_points, rel_points, obj_gt, rel_gt, edges = dataset.__getitem__(i)
            obj_points = obj_points.permute(0, 2, 1)
            rel_points = rel_points.permute(0, 2, 1)
        logs, obj_pred, rel_pred = network.process(obj_points, rel_points, edges, obj_gt, rel_gt)
        logs += network.calculate_metrics([obj_pred, rel_pred], [obj_gt, rel_gt])
        # pred_cls = torch.max(obj_pred.detach(),1)[1]
        # acc_obj = (obj_gt == pred_cls).sum().item() / obj_gt.nelement()

        # rel_pred = rel_pred.detach() > 0.5
        # acc_rel = (rel_gt==(rel_pred>0)).sum().item() / rel_pred.nelement()

        # print('{0:>3d} acc_obj: {1:>1.4f} acc_rel: {2:>1.4f} loss: {3:>2.3f}'.format(i,acc_obj,acc_rel,logs[0][1]))
        print('{:>3d} '.format(i), end='')
        for log in logs:
            print('{0:} {1:>2.3f} '.format(log[0], log[1]), end='')
        print('')

