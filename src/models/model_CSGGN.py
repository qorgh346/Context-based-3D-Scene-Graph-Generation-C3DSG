#!/usr/bin/env python3
# -*- coding: utf-8 -*-
if __name__ == '__main__' and __package__ is None:
    from os import sys

    # sys.path.append('./src')
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from model_base import BaseModel
from CSGGN.src.networks.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti
# from network_PointNet2 import PointNet2feat
from CSGGN.src.networks.network_DGCNN import DGCNNfeat
from CSGGN.src.networks.network_pointtransformer_v2 import PointTransformerV2
from CSGGN.src.networks.network_TripletGCN import TripletGCNModel
from CSGGN.src.networks.network_NEGAT import NEGATModel
from CSGGN.src.networks.network_EdgeGCN import TwiningAttentionGCNModel
from model_GraphTransformer import GraphTransformerNet
from CSGGN.src.networks.network_GNN import GraphEdgeAttenNetworkLayers
from CSGGN.src.config import Config
import CSGGN.src.op_utils as op_utils
import CSGGN.src.optimizer as optimizer
import math
from model_TransformerNet import Transformer_cfg,PointTransformerfeat
import sys
from CSGGN.utils.util import MLP
from CSGGN.utils.Loss import TotalLoss
from CSGGN.utils.ssg_eval_tool import Object_Accuracy, Object_Recall, Predicate_Accuracy, Predicate_Recall, Relation_Recall
import torch.nn as nn
import collections

#CSGGN ver.23.01.04
class CSGGNModel(BaseModel):
    # obj_w=self.weight_obj, pred_w=self.weight_rel
    def __init__(self, config: Config, name: str, num_class, num_rel,classNames,weight_obj,weight_rel):

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
        # self.embedding_model = build_word_embedding(self.config.WORD_VECTOR_TEXT)
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
            dim_input_rel +=  11
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
        self.train_pred_acc = Predicate_Accuracy(1170, need_softmax=False)
        self.train_pred_recall = Predicate_Recall(1170, need_softmax=False)

        ####################################################################################
        # Geometric Feature Extraction #
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
            pt_v1_path = self.transformer_config.PTv1_PATH
            models['obj_encoder'] = self.transformerV1_Load(obj_encoder,pt_v1_path)

        elif mconfig.USE_DGCNN:
            models['obj_encoder'] = DGCNNfeat(
                args=config.DGCNN,
                output_channels=mconfig.point_feature_size
            )

        elif mconfig.USE_PointTransformerV2:
            pt_v2 = PointTransformerV2()

            models['obj_encoder'] = self.transformerV2_Load(pt_v2, self.transformer_config.PTv2_PATH)

            models['obj_feature_extractor'] = MLP(mlp=[48+35 , self.mconfig.point_feature_size //2 , self.mconfig.point_feature_size],bn=True)

        ####################################################################################

        # Non-Geometric Feature Extraction
        input_dim =  48 + 11  #+ (dim_node * 2)

        models['rel_encoder'] = MLP(mlp=[input_dim , dim_node, self.mconfig.edge_feature_size])

        ####################################################################################


        if mconfig.USE_class_categories:
            models['obj_cls_distrubtion'] = torch.nn.Sequential(
                torch.nn.Linear(self.mconfig.point_feature_size, self.mconfig.point_feature_size//2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.mconfig.point_feature_size//2, self.mconfig.point_feature_size//4),
                torch.nn.ReLU(),
                torch.nn.Linear(self.mconfig.point_feature_size // 4, num_class),
            )


        ####################################################################################
        # Spatial Context Reasoning

        if mconfig.GCN_TYPE == "TRIP":
            # SGPN (https://3dssg.github.io/)

            models['context_reasoning'] = TripletGCNModel(num_layers=mconfig.N_LAYERS,
                                          dim_node=mconfig.point_feature_size,
                                             dim_edge=mconfig.edge_feature_size,
                                            dim_hidden=mconfig.gcn_hidden_feature_size)

        elif mconfig.GCN_TYPE == 'EAN':
            #Scene Graph Fusion (https://shunchengwu.github.io/SceneGraphFusion )

            models['context_reasoning'] = GraphEdgeAttenNetworkLayers(self.mconfig.point_feature_size,
                                                        self.mconfig.edge_feature_size,
                                                        self.mconfig.DIM_ATTEN,
                                                        self.mconfig.N_LAYERS,
                                                        self.mconfig.NUM_HEADS,
                                                        self.mconfig.GCN_AGGR,
                                                        flow=self.flow)
        elif mconfig.GCN_TYPE == 'EDGE_GCN':
            #Edge GCN (https://github.com/chaoyivision/SGGpoint)
            models['context_reasoning'] = TwiningAttentionGCNModel(
                                            num_node_in_embeddings=mconfig.point_feature_size,
                                            num_edge_in_embeddings=mconfig.edge_feature_size,
                                            AttnEdgeFlag=True,
                                             AttnNodeFlag=True)

        # 신규 제안 모델
        elif mconfig.GCN_TYPE == 'NE_GAT':
            models['context_reasoning'] = NEGATModel(num_layers=mconfig.N_LAYERS,
                                            dim_node=mconfig.point_feature_size,
                                            dim_edge=mconfig.edge_feature_size,
                                            dim_hidden=mconfig.gcn_hidden_feature_size)

        #Graph Transformer Network(https://github.com/graphdeeplearning/graphtransformer)
        elif mconfig.GCN_TYPE == 'GraphTransformer':
            models['context_reasoning'] = GraphTransformerNet()

        ####################################################################################



        ####################################################################################
        # 노드  & 관계간선 분류
        models['obj_predictor'] = PointNetCls(num_class, in_size=mconfig.point_feature_size,
                                              batch_norm=with_bn, drop_out=True)

        if mconfig.multi_rel_outputs:
            models['rel_predictor'] = PointNetRelClsMulti(
                num_rel,
                in_size=mconfig.edge_feature_size,
                batch_norm=with_bn, drop_out=True)
        else:
            models['rel_predictor'] = PointNetRelCls(
                num_rel,
                in_size=mconfig.edge_feature_size,
                batch_norm=with_bn, drop_out=True)
        ####################################################################################
        params = list()
        print('==trainable parameters==')
        for name, model in models.items():
            if len(config.GPU) > 1:
                model = torch.nn.DataParallel(model, config.GPU)
            else:
                model = model.to(config.GPU[0])
            self.add_module(name, model)
            params += list(model.parameters())
            print(name,op_utils.pytorch_count_params(model))
        print('')

        # for name, model in models.items():
        #
        #     model = model.to(config.GPU[0])
        #     self.add_module(name, model)
        #     params += list(model.parameters())
        #     print(name, op_utils.pytorch_count_params(model))
        # print('')


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
                return 1/math.log(20)
#                return 1 / math.log(batchsize) if batchsize > 1 else 1
            self.scheduler = optimizer.BatchMultiplicativeLR(self.optimizer, _scheduler)

        print(models)
        if self.weight_rel[0] == 0:
            self.weight_rel[0]+= 0.15
        temp_rel_w = torch.ones_like(self.weight_rel)
        temp_rel_w[0] = 0.1
        self.criterion = TotalLoss(alpha=0.2, beta=0.8, gamma=0.9, obj_w=self.weight_obj, pred_w=temp_rel_w,
                                   objNum=self.num_class,relNum=self.num_rel)#.to(self.config.DEVICE)

    def transformerV2_Load(self,model,weight_file):
        if os.path.isfile(weight_file):
            checkpoint = torch.load(weight_file)
            state_dict = checkpoint['state_dict']
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=True)
            epochs = checkpoint['epoch']
            for name,param in model.named_parameters():
                print(name[:12])
                module_name = name[:12]
                if module_name in ['enc_stages.0','enc_stages.1','patch_embed.','dec_stages.0','dec_stages.1','dec_stages.2']:
                    param.requires_grad = False
            #         patch_embed.
            # sys.exit()
            for name, param in model.named_parameters():
                print(name, param.requires_grad)
            # sys.exit()
            return model
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(weight_file))




    def forward(self, obj_points, descriptor, edges, data_dict, return_meta_data=False):

        if data_dict:
            #Use PointTransformerV2
            obj_point_feature, scene_point_feature = self.obj_encoder(descriptor.size(0),data_dict)
            obj_feature = torch.concat([obj_point_feature,descriptor],dim=1)
            obj_feature = self.obj_feature_extractor(obj_feature)
        else:
            obj_feature = self.obj_encoder(obj_points)

        # print(obj_feature)
        if self.mconfig.USE_PointNet:
            rel_feature = self.rel_encoder(descriptor)
        else:
            with torch.no_grad():
                relative_descriptor = descriptor[:,:11].detach()
                relative_feature, pair_point_feature = op_utils.Gen_edge_descriptor(flow=self.flow,
                                                                                    feature_selection=self.config.Edge_Feature_Selection)\
                                                                                    (relative_descriptor, edges,scene_point_feature,
                                                                                     data_dict['obj_pair_indices'])


            # 방법 1: 모두 concat
            edge_feature = torch.concat([relative_feature,pair_point_feature],dim=1)
            rel_feature = self.rel_encoder(edge_feature)


        probs = None
        if self.mconfig.USE_GCN:
            if self.mconfig.GCN_TYPE == 'TRIP':
                # edges = edges.permute(1,0)
                gcn_obj_feature, gcn_rel_feature = self.context_reasoning(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'EAN':
                gcn_obj_feature, gcn_rel_feature, probs = self.context_reasoning(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'NE_GAT':
                gcn_obj_feature, gcn_rel_feature = self.context_reasoning(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'EDGE_GCN':
                gcn_obj_feature, gcn_rel_feature = self.context_reasoning(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'GraphTransformer':
                lap_pos_enc = op_utils.laplacian_positional_encoding(edges, obj_feature.size(0), pos_dim=8)
                gcn_obj_feature, gcn_rel_feature = self.context_reasoning(obj_feature, rel_feature, edges, lap_pos_enc)


            if self.mconfig.OBJ_PRED_FROM_GCN:
                obj_cls = self.obj_predictor(gcn_obj_feature)
            else:
                obj_cls = self.obj_predictor(obj_feature)
            rel_cls = self.rel_predictor(gcn_rel_feature)

        else:
            gcn_obj_feature = gcn_rel_feature = None
            obj_cls = self.obj_predictor(obj_feature)
            rel_cls = self.rel_predictor(rel_feature)

        result = {}
        result['obj_pred'] = obj_cls
        result['rel_pred'] = rel_cls
        result['obj_feature'] = obj_feature
        result['rel_feature'] = rel_feature
        result['gcn_obj_feature'] = gcn_obj_feature
        result['gcn_rel_feature'] = gcn_rel_feature
        result['probs'] = probs
        #
        del obj_cls
        del rel_cls
        del obj_feature
        del rel_feature
        del gcn_obj_feature
        del gcn_rel_feature
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if return_meta_data:
            return result  # obj_cls, rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs
        else:
            return result['obj_pred'], result['rel_pred']

    def transformerV1_Load(self,model,pretrain_path):
        model_dict = model.state_dict()
        temp_pretrained_dict = dict()
        pretrained_dict = torch.load(pretrain_path)
        for k, v in pretrained_dict['model_state_dict'].items():
            if k in model_dict and 'backbone.fc1' not in k and 'fc2' not in k.split('.')[0]:
                temp_pretrained_dict[k] = v
        model_dict.update(temp_pretrained_dict)
        model.load_state_dict(model_dict, strict=True)

        for name,param in model.named_parameters():
            names = name.split('.')
            if names[1] =='transformers':
                if names[2] == '2':
                    param.requires_grad = True
            param.requires_grad = False
        # sys.exit()
        return model

    def process(self, obj_points, descriptor, edges, gt_obj_cls, gt_rel_cls,weights_obj=None, weights_rel=None,data_dict=None):
        self.iteration += 1
        result = self(obj_points, descriptor, edges, data_dict, return_meta_data=True)

        obj_pred, rel_pred = result['obj_pred'], result['rel_pred']
        # print('gt_rel= ,',gt_rel_cls.size())
        loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weights_rel)

        loss_obj,loss_rel,loss = self.criterion(obj_pred, loss_rel, gt_obj_cls, gt_rel_cls)

        self.backward(loss)

        if self.scheduler is not None:
            self.scheduler.step()

        logs = [("Loss/cls_loss", loss_obj.detach().item()),
                ("Loss/rel_loss", loss_rel.detach().item()),
                ("Loss/loss", loss.detach().item())]

        loss_info = loss.item()
        probs = result['probs']

        # self.train_pred_acc.calculate_accuracy_binary(obj_points.size()[0], rel_pred, gt_rel_cls)
        # self.train_pred_acc.calculate_recall_binary(obj_points.size()[0], rel_pred, gt_rel_cls)
        # self.train_pred_acc.calculate_accuracy(obj_points.size()[0], rel_pred, gt_rel_cls)

        return logs, obj_pred.detach(), F.sigmoid(rel_pred.detach()), loss_info

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def norm_tensor(self, points, dim):
        # points.shape = [n, 3, npts]
        centroid = torch.mean(points, dim=-1).unsqueeze(-1)  # N, 3
        points -= centroid  # n, 3, npts
        furthest_distance = points.pow(2).sum(1).sqrt().max(1)[0]  # find maximum distance for each n -> [n]
        points /= furthest_distance[0]
        return points

    def calculate_metrics(self, preds, gts):
        assert (len(preds) == 2)
        assert (len(gts) == 2)
        obj_pred = preds[0].detach()
        rel_pred = preds[1].detach()
        obj_gt = gts[0]
        rel_gt = gts[1]

        # print(rel_gt[42])
        pred_cls = torch.max(obj_pred.detach(), 1)[1]
        acc_obj = (obj_gt == pred_cls).sum().item() / obj_gt.nelement()

        if self.mconfig.multi_rel_outputs:
            pred_rel = rel_pred.detach() > 0.5
        else:
            pred_rel = torch.max(rel_pred.detach(), 1)[1]

        acc_rel = (rel_gt == pred_rel).sum().item() / rel_gt.nelement()

        #edit hojun
        # pred_rel
        rel_gt_true = (rel_gt == 1) #TP
        rel_gt_false = (rel_gt == 0 ) #FP ?

        correct_count = torch.eq(pred_rel, rel_gt_true).sum().item()
        total_count = rel_gt.nelement()
        acc_hojun_rel = correct_count/total_count
        logs = [("Accuracy/obj_cls", acc_obj),
                ("Accuracy/rel_cls", acc_rel)
                #("Accuracy/rel_update_cls", acc_hojun_rel)
                ]
        return logs




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
        from CSGGN.src.datasets.dataset_builder import build_dataset

        # config.dataset.dataset_type = 'rio_graph'
        dataset = build_dataset(config, 'validation_scans', True, multi_rel_outputs=False, use_rgb=False,
                                use_normal=False)
        num_obj_cls = len(dataset.classNames)
        num_rel_cls = len(dataset.relationNames)

    # build model
    mconfig = config.MODEL
    network = CSGGNModel(config, 'SGAPNModel', num_obj_cls, num_rel_cls)

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

