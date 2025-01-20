#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from builtins import enumerate
import model_ExternalGT
if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import os
import sys
import torch
import torch.nn as nn
from utils.util import MLP
import torch.optim as optim
import torch.nn.functional as F
from model_base import BaseModel
from network_PointNet import PointNetfeat,PointNetCls,PointNetRelCls,PointNetRelClsMulti
from network_PAconvNet import test_PAConv
from network_TripletGCN import TripletGCNModel
from network_GNN import GraphEdgeAttenNetworkLayers
from config import Config
import op_utils
import optimizer
import math
import  numpy as np
import os
import gc
from utils.Paconv_scene_seg_util.paconv_util import *
from pointnet2_paconv_seg import PointNet2SSGSeg as Model
from pointnet2_paconv_seg import embeddingModule
from utils.Paconv_scene_seg_util import block

from model_TransformerNet import Transformer_cfg,PointTransformerCls

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

class SGFNModel(BaseModel):
    # config,self.model_name,num_obj_class,args,num_rel_class)
    def __init__(self,config:Config,name:str, num_class,args_hobe,num_rel,use_pointTransformer=False,dim_descriptor=11):
        super().__init__(name,config)
        models = dict()
        self.mconfig = mconfig = config.MODEL
        self.args = args_hobe

        # self.device = device #umm....
        with_bn = mconfig.WITH_BN #false
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial # dim_descriptor = 11
        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_class
        self.num_rel=num_rel
        self.use_pointTransformer = use_pointTransformer

        #----------------------
        self.flow = 'target_to_source' # we want the msg
        
        dim_point_feature = self.mconfig.point_feature_size #256
        if self.mconfig.USE_SPATIAL and not self.use_pointTransformer:
            dim_point_feature -= dim_f_spatial-3 # ignore centroid

        # Object Encoder _ PointNet
        if self.use_pointTransformer:
            cfg = Transformer_cfg(point_feature_size = dim_point_feature,input_dim = dim_point)
            models['obj_encoder_pointTransformer'] = PointTransformerCls(cfg)
        else:
            models['obj_encoder_pointNet'] = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)

        models['rel_encoder'] = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        ''' Message passing between segments and segments '''
        if self.mconfig.USE_GCN:
            if mconfig.GCN_TYPE == "TRIP":
                models['gcn'] = TripletGCNModel(num_layers=mconfig.N_LAYERS,
                                                dim_node = mconfig.point_feature_size,
                                                dim_edge = mconfig.edge_feature_size,
                                                dim_hidden = mconfig.gcn_hidden_feature_size)
            elif mconfig.GCN_TYPE == 'EAN':
                models['gcn'] = GraphEdgeAttenNetworkLayers(self.mconfig.point_feature_size,
                                    self.mconfig.edge_feature_size,
                                    self.mconfig.DIM_ATTEN,
                                    self.mconfig.N_LAYERS, 
                                    self.mconfig.NUM_HEADS,
                                    self.mconfig.GCN_AGGR,
                                    flow=self.flow,
                                    attention=self.mconfig.ATTENTION,
                                    use_edge=self.mconfig.USE_GCN_EDGE,
                                    DROP_OUT_ATTEN=self.mconfig.DROP_OUT_ATTEN)
            else:
                raise NotImplementedError('')
        
        ''' node feature classifier '''
        models['obj_predictor'] = PointNetCls(num_class, in_size=mconfig.point_feature_size,
                                 batch_norm=with_bn,drop_out=True)
        
        if mconfig.multi_rel_outputs:
            models['rel_predictor'] = PointNetRelClsMulti(
                num_rel, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        else:
            models['rel_predictor'] = PointNetRelCls(
                num_rel, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
            
            
        params = list()
        print('==trainable parameters==')

        for name, model in models.items():
            if len(config.GPU) > 1:
                print('nONONONONO')
                if name == 'gcn':
                    self.add_module(name, model)
                    params += list(model.parameters())
                    continue
                model = torch.nn.DataParallel(model, config.GPU)
            self.add_module(name, model)
            params += list(model.parameters())
            print(name, op_utils.pytorch_count_params(model))
        print('')

        # for idx,(name, model) in enumerate(models.items()):
        #     # print(name)
        #     if len(config.GPU) > 1:
        #         # print(name)
        #         if name == 'gcn':
        #             self.add_module(name, model)
        #             params += list(model.parameters())
        #             continue
        #         elif name == 'test_hobe_paconv':
        #             pass
        #             # self.add_module(name,model)
        #             # params += list(model.parameters())
        #             # continue
        #         model = torch.nn.DataParallel(model, config.GPU)
        #
        #     self.add_module(name, model)
        #     params += list(model.parameters())
        #     print(name,op_utils.pytorch_count_params(model))
        # print('')
        
        self.optimizer = optim.AdamW(
            params = params,
            lr = float(config.LR),
            weight_decay=self.config.W_DECAY,
            amsgrad=self.config.AMSGRAD
        )
        self.optimizer.zero_grad()
        
        self.scheduler = None
        if self.config.LR_SCHEDULE == 'BatchSize':
            def _scheduler(epoch, batchsize):
                return 1/math.log(batchsize) if batchsize>1 else 1
            self.scheduler = optimizer.BatchMultiplicativeLR(self.optimizer, _scheduler)

        if self.use_pointTransformer:
            for name, param in self.obj_encoder_pointTransformer.named_parameters():
                print(f"Layer: {name} | Size: {param.size()} | Values : {param.requires_grad} \n")
            # model_data = torch.load('/home/baebro/hojun_ws/3DSSG/paconv_model/best_train.pth')
            # print(model_data)

            # for name, param in model_data['state_dict'].items():
            #     print(f"Layer: {name} | Size: {param.size()} | Values : {param.requires_grad} \n")

            # self.obj_encoder_pointTransformer.load_state_dict(model_data['state_dict'],strict=False)
            # self.set_parameter_requires_grad(self.test_hobe_paconv,feature_extracting=True)
            # print(self.test_hobe_paconv.state_dict())
            # sys.exit()
        #---------------------------------

    def forward(self,mode,segments_points, edges, descriptor, imgs = None, covis_graph = None, return_meta_data=False,
                instance2mask = None,className=None,rel_gt=None):
        # print('forward ::',segments_points.size()) #29,9,4096
        # print('edges --> ',edges.size()) #edges --> edge_index
        # print(edges)
        # print('descriptor --> ',descriptor.size())
        # print(self.test_hobe_paconv)
        # for name,v in self.test_hobe_paconv.named_parameters():
        #     print(name,':::',v.requires_grad)
        # sys.exit()
        # print('forward ::', segments_points[:16].size())  # 29,9,4096
        # sys.exit()
        # [num_class,atributes, num_points]
        #num_class 마다 pre-trained 된 point feature 추출기의 input으로 들어간다
        #그 다음 그걸 gcn_input feature 사이즈에 맞게 output을 맞추고 --> dim = 0 으로 concat 하기 ㄱ
        # *참고로 pretrained 된 거 사용할 때는 .permute(0,2,1) 로 해야됌 *

        # sys.exit()
        # obj_feature = self.obj_encoder_PAConv(segments_points,1,2)
        # obj_feature = obj_feature.cuda(non_blocking=True)
        # points.cuda(non_blocking=True)
        # print(obj_feature)
        # print('obj_encoder after = ',obj_feature.size())
        # print('edge feature - before  size = ',edges.size())
        if self.use_pointTransformer:
            segments_points = segments_points.permute(0, 2, 1)
            obj_feature = self.obj_encoder_pointTransformer(segments_points) # ( point_set Number(obj-number) , pointFeature )
            # print('obj_feature = ',obj_feature.size())
            # print('input_points = ',segments_points.size())
        else:
            obj_feature = self.obj_encoder_pointNet(segments_points)
        # print('obj_feature ===> after --> ',obj_feature.size())
        # print('descriptor --> ',descriptor.size())

        if self.mconfig.USE_SPATIAL and not self.use_pointTransformer:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length ---> except --(x,y,z)
            obj_feature = torch.cat([obj_feature, tmp],dim=1)
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor,edges)
            # print('edgeFeature = ',edge_feature.size()) # [ edge_index, relation_feature_size , 1 ]
            # print(edge_feature)
            # print('edge size = ',edges.size())
            rel_feature = self.rel_encoder(edge_feature)
            # sys.exit()
            # print('node_rel  && node_rel --> rel_feature.size = ',rel_feature.size())
            ''' GNN '''
        probs=None
        if self.mconfig.USE_GCN:
            if self.mconfig.GCN_TYPE == 'TRIP':
                gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'EAN':
                # rel_feature = torch.as_tensor(rel_feature, device='cuda:1')
                # edges = torch.as_tensor(edges,device='cuda:1')
                # print(obj_feature.device)
                # print(rel_feature.device)
                # print(edges.device)

                gcn_obj_feature, gcn_rel_feature, probs = self.gcn(obj_feature, rel_feature, edges)
                # print('gcn obj after : {}\n gcn rel after : {}'.format(gcn_obj_feature.size(),gcn_rel_feature.size()))
                # sys.exit()
        else:
            gcn_obj_feature=gcn_rel_feature=probs=None

        ''' Predict '''
        if self.mconfig.USE_GCN:
            if self.mconfig.OBJ_PRED_FROM_GCN:
                obj_cls = self.obj_predictor(gcn_obj_feature)
            else:
                obj_cls = self.obj_predictor(obj_feature)
            rel_cls = self.rel_predictor(gcn_rel_feature)
        else:
            obj_cls = self.obj_predictor(obj_feature)
            rel_cls = self.rel_predictor(rel_feature)

        #edit
        #GT 데이터 속 물체와 물체 사이의 관계의 분포도를 고려하여 더해주는 코드
        # print(rel_cls.size())
        # print(rel_cls)
        # sys.exit()

        # if mode =='train':
        # rel_cls = model_ExternalGT.predict_layer(className=className,instance2Mask=instance2mask,
        #                                              edge_index=edges,rel_gt=rel_gt,output_rel_cls=rel_cls)
        #     # sys.exit()


        result = {}
        result['obj_cls'] = obj_cls
        result['rel_cls'] = rel_cls
        result['obj_feature'] = obj_feature
        result['rel_feature'] = rel_feature
        result['gcn_obj_feature'] = gcn_obj_feature
        result['gcn_rel_feature'] = gcn_rel_feature
        result['probs'] = probs
        # del obj_cls
        # del rel_cls
        # del obj_feature
        # del rel_feature
        # del gcn_obj_feature
        # del gcn_rel_feature

        # gc.collect()
        # torch.cuda.empty_cache()
        if return_meta_data:
            # return obj_cls, rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs
            return result
        else:
            return obj_cls, rel_cls

    def set_parameter_requires_grad(self,model, feature_extracting):
        if feature_extracting:
            # print(model)
            for name,param in model.named_parameters():
                # print(name)
                # if name.split('.')[1] == 'FP_modules':
                    # print('dz')
                param.requires_grad = False

    def process(self, obj_points, edges, descriptor, gt_obj_cls, gt_rel_cls, weights_obj=None, weights_rel=None, ignore_none_rel=False,
                scan_id=None,instance2mask = None,className=None,relationName = None ,imgs = None, covis_graph = None):
        # print('----------------process===============')
        # print('className = ',className)
        # print('relationName = ',relationName)
        print('scene id = ',scan_id)
        # print('ignore : ',ignore_none_rel)
        # print('instance2mask : ',instance2mask)
        # print('gt_rel : ',gt_rel_cls)
        # print('gt_rel : ', gt_rel_cls.size())

        # print('gt_cls : ',gt_obj_cls)
        self.iteration +=1
        # print('gt_obj_cls.size = ', gt_obj_cls.size())
        # print('obj_points size = ',obj_points.size())
        # print('edges = ', edges.size())
        # print(edges)

        mode = 'train'
        result = self(mode,obj_points, edges, descriptor,return_meta_data=True, imgs=imgs, covis_graph=covis_graph,instance2mask = instance2mask,className=className,
                      rel_gt = relationName)

        obj_pred, rel_pred,probs = result['obj_cls'],result['rel_cls'],result['probs']
        if self.mconfig.multi_rel_outputs:
            if self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                batch_mean = torch.sum(gt_rel_cls, dim=(0))
                zeros = (gt_rel_cls ==0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros,batch_mean],dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel:
                    weight[0] = 0
                    weight *= 1e-2 # reduce the weight from ScanNet
                    # print('set weight of none to 0')
                if 'NONE_RATIO' in self.mconfig:
                    weight[0] *= self.mconfig.NONE_RATIO
                    
                weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]
                print(weight)
                # sys.exit()
            else:
                raise NotImplementedError("unknown weight_edge type")

            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weight)
        else:
            if self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                one_hot_gt_rel = torch.nn.functional.one_hot(gt_rel_cls,num_classes = self.num_rel)
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel: 
                    weight[-1] = 0.0 # assume none is the last relationship
                    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            else:
                raise NotImplementedError("unknown weight_edge type")

            if 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
                loss_rel = torch.zeros(1,device=rel_pred.device, requires_grad=False)
            else:
                # print('gt_rel_cls ==> ',gt_rel_cls.size())
                # sys.exit()
                loss_rel = F.nll_loss(rel_pred, gt_rel_cls, weight = weight)

        loss_obj = F.nll_loss(obj_pred, gt_obj_cls, weight = weights_obj)

        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        if 'USE_REL_LOSS' in self.mconfig and not self.mconfig.USE_REL_LOSS:
            loss = loss_obj
        elif 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
            loss = loss_obj
        else:
            loss = lambda_o * loss_obj + lambda_r * loss_rel
            
        if self.scheduler is not None:
            self.scheduler.step(batchsize=edges.shape[1])
        self.backward(loss)
        
        logs = [("Loss/cls_loss",loss_obj.detach().item()),
                ("Loss/rel_loss",loss_rel.detach().item()),
                ("Loss/loss", loss.detach().item())]
        return logs, obj_pred.detach(), rel_pred.detach(), probs
    
    def backward(self, loss):
        loss.backward()        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def calculate_metrics(self, preds, gts):
        assert(len(preds)==2)
        assert(len(gts)==2)
        obj_pred = preds[0].detach()
        rel_pred = preds[1].detach()
        obj_gt   = gts[0]
        rel_gt   = gts[1]
        
        pred_cls = torch.max(obj_pred.detach(),1)[1]
        acc_obj = (obj_gt == pred_cls).sum().item() / obj_gt.nelement()
        
        if self.mconfig.multi_rel_outputs:
            pred_rel= rel_pred.detach() > 0.5
            acc_rel = (rel_gt==pred_rel).sum().item() / rel_gt.nelement()
        else:
            pred_rel = torch.max(rel_pred.detach(),1)[1]
            acc_rel = (rel_gt==pred_rel).sum().item() / rel_gt.nelement()
            
        
        logs = [("Accuracy/obj_cls",acc_obj), 
                ("Accuracy/rel_cls",acc_rel)]
        return logs
    
    def trace(self,path):
        op_utils.create_dir(path)
        params = dict()
        params['USE_GCN']=self.mconfig.USE_GCN
        params['USE_RGB']=self.mconfig.USE_RGB
        params['USE_NORMAL']=self.mconfig.USE_NORMAL
        params['dim_point']=self.dim_point
        params['dim_edge'] =self.dim_edge
        params["DIM_ATTEN"]=self.mconfig.DIM_ATTEN
        params['obj_pred_from_gcn']=self.mconfig.OBJ_PRED_FROM_GCN
        params['dim_o_f']=self.mconfig.point_feature_size
        params['dim_r_f']=self.mconfig.edge_feature_size
        params['dim_hidden_feature']=self.mconfig.gcn_hidden_feature_size
        params['num_classes']=self.num_class
        params['num_relationships']=self.num_rel
        params['multi_rel_outputs']=self.mconfig.multi_rel_outputs
        params['flow'] = self.flow
        
        self.eval()
        params['enc_o'] = self.obj_encoder.trace(path,'obj')
        params['enc_r'] = self.rel_encoder.trace(path,'rel')
        if self.mconfig.USE_GCN:
            params['n_layers']=self.gcn.num_layers
            if self.mconfig.GCN_TYPE == 'EAN':
                for i in range(self.gcn.num_layers):
                    params['gcn_'+str(i)] = self.gcn.gconvs[i].trace(path,'gcn_'+str(i))
            else:
                raise NotImplementedError()
        params['cls_o'] = self.obj_predictor.trace(path,'obj')
        params['cls_r'] = self.rel_predictor.trace(path,'rel')
        return params


def model_fn_decorator(criterion):
    from collections import namedtuple
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # print(inputs.size())
            preds = model(inputs)
            # print(preds.size())
            loss = criterion(preds, labels)
            _, classes = torch.max(preds, 1)
            acc = (classes == labels).float().sum() / labels.numel()
            return ModelReturn(preds, loss, {"acc": acc.item(), 'loss': loss.item()})
    return model_fn





if __name__ == '__main__':
    import torch.optim as optim
    from utils.Paconv_scene_seg_util.util import AverageMeter, intersectionAndUnionGPU, get_logger, get_parser
    from pointnet2_paconv_seg import PointNet2SSGSeg
    args = get_parser()
    B, N, C, K = 1, 4096, 6, 13
    inputs = torch.randn(B, N, 9).cuda()
    labels = torch.randint(0, 3, (B, N)).cuda()
    print('zz?')
    model = PointNet2SSGSeg(c=C, k=K, args=args)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    print("Testing SSGCls with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()
    # sys.exit()
    Model(c=self.args.fea_dim, k=self.args.classes, use_xyz=True, args=self.args)

    use_dataset = False
    
    config = Config('../config_example.json')
    # args = get_parser()
    if not use_dataset:
        num_obj_cls=40
        num_rel_cls=26
    else:
        from src.dataset_builder import build_dataset
        config.dataset.dataset_type = 'SGFN'
        dataset =build_dataset(config, 'train_scans', True, multi_rel_outputs=False, use_rgb=True, use_normal=True)
        num_obj_cls = len(dataset.classNames)
        num_rel_cls = len(dataset.relationNames)


    # build model
    mconfig = config.MODEL
    network = SGFNModel(config,'SceneGraphFusionNetwork',num_obj_cls,num_rel_cls)
    # print(network)
    set =build_dataset(config, 'train_scans', True, multi_rel_outputs=False, use_rgb=True, use_normal=True)
    num_obj_cls = len(dataset.classNames)
    num_rel_cls = len(dataset.relationNames)


    # build model
    mconfig = config.MODEL
    network = SGFNModel(config,'SceneGraphFusionNetwork',num_obj_cls,num_rel_cls)
    # print(network)

    args = get_parser()
    test_model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz, args=args)

    args = get_parser()
    test_model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz, args=args)

    if not use_dataset:
        max_rels = 80    
        n_pts = 10
        n_rels = n_pts*n_pts-n_pts
        n_rels = max_rels if n_rels > max_rels else n_rels
        obj_points = torch.rand([n_pts,9,128]) #use_normal, use_rgb 3 --> 3+3+3 : 9
        rel_points = torch.rand([n_rels, 4, 256])
        edge_indices = torch.zeros(n_rels, 2,dtype=torch.long)
        counter=0
        for i in range(n_pts):
            if counter >= edge_indices.shape[0]: break
            for j in range(n_pts):
                if i==j:continue
                if counter >= edge_indices.shape[0]: break
                edge_indices[counter,0]=i
                edge_indices[counter,1]=i
                counter +=1
    
        temp_des = torch.rand([n_pts,11])

        obj_gt = torch.randint(0, num_obj_cls-1, (n_pts,))
        rel_gt = torch.randint(0, num_rel_cls-1, (n_rels,))
    
        # rel_gt
        adj_rel_gt = torch.rand([n_pts, n_pts, num_rel_cls])
        rel_gt = torch.zeros(n_rels, num_rel_cls, dtype=torch.float)
        
        
        for e in range(edge_indices.shape[0]):
            i,j = edge_indices[e]
            for c in range(num_rel_cls):
                if adj_rel_gt[i,j,c] < 0.5: continue
                rel_gt[e,c] = 1
            
        network.process(obj_points,edge_indices.t().contiguous(),temp_des,obj_gt,rel_gt)

