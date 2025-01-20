import torch.nn as nn
import torch
from torch_geometric.nn.inits import glorot, ones, reset
from torch.nn.functional import softmax

def classification(channels: list,classes_num : int,do_bn=True, on_last=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    offset = 0 if on_last else 1
    for i in range(1, n):
        layers.append(
            nn.Linear(channels[i - 1], channels[i],bias=True))
        if i < (n - offset):
            # if do_bn:
            #     layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    layers.append(
        nn.Linear(channels[-1],classes_num)
    )

    return nn.Sequential(*layers)


class RelCls(nn.Module):
    def __init__(self,config,prior_knowledge):
        super(RelCls,self).__init__()
        self.config = config
        self.prior_knowledge = prior_knowledge

        #
        self.k_lin = nn.ModuleDict()
        self.q_lin = nn.ModuleDict()
        self.v_lin = nn.ModuleDict()
        self.edge_lin = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        hidden_channels = config.MODEL.gcn_hidden_feature_size

        for edgeType in prior_knowledge['edgeType'].keys():
            self.edge_lin[edgeType] = nn.Linear(hidden_channels *2 , hidden_channels)
            self.k_lin[edgeType] = nn.Linear(hidden_channels*2, hidden_channels)
            self.q_lin[edgeType] = nn.Linear(hidden_channels*2, hidden_channels)
            self.v_lin[edgeType] = nn.Linear(hidden_channels*2, hidden_channels)

            classNum = len(prior_knowledge['edgeType'][edgeType])
            self.classifier[edgeType] = classification([hidden_channels,hidden_channels//2,hidden_channels//4],
                                                       classes_num=classNum)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.edge_lin)
        reset(self.classifier)

    def genearte_edge_feature(self,heteroData,updated_obj_dict):
        edge_storage = {
            edge_type[1]: {'geo_edge_feature': torch.tensor([], device=self.config.DEVICE),
                           'non_geo_edge_feature': torch.tensor([], device=self.config.DEVICE)
                           }
            for edge_type in heteroData.edge_types
        }

        for meta_rel,edge_index_list in heteroData.edge_index_dict.items():
            subType,edgeType,objType = meta_rel
            subNodeFeat = updated_obj_dict[subType]
            subEdgeIdx = edge_index_list[0]

            objNodeFeat =  updated_obj_dict[objType]
            objEdgeIdx = edge_index_list[1]

            sub_NonGeoemtricFeature= heteroData[subType].descriptor
            obj_NonGeoemtricFeature= heteroData[objType].descriptor

            if subNodeFeat is None:
                subNodeFeat = heteroData[subType].x
                # continue

            if objNodeFeat is None:
                objNodeFeat = heteroData[objType].x

                # continue

            subInstanceFeature = torch.index_select(subNodeFeat,0,subEdgeIdx)
            objInstanceFeature = torch.index_select(objNodeFeat, 0, objEdgeIdx)

            subInstanceNonGeoemtricFeature = torch.index_select(sub_NonGeoemtricFeature,0,subEdgeIdx)
            objInstanceNonGeometricFeature = torch.index_select(obj_NonGeoemtricFeature, 0, objEdgeIdx)



            geo_edge_feature = torch.concat([subInstanceFeature,objInstanceFeature],dim=1)
            non_geo_edge_feature = torch.concat([subInstanceNonGeoemtricFeature,objInstanceNonGeometricFeature],dim=1)

            edge_storage[edgeType]['geo_edge_feature'] = torch.concat([edge_storage[edgeType]['geo_edge_feature'],
                                                                            geo_edge_feature],dim=0)
            edge_storage[edgeType]['non_geo_edge_feature'] = torch.concat([edge_storage[edgeType]
                                                                                ['non_geo_edge_feature'],non_geo_edge_feature],dim=0)
        return edge_storage


    def attention_compute(self,query,key):
        alpha = query * key #hadamard product
        alpha = softmax(alpha,dim=1)
        return alpha


    def forward(self,heteroData,updated_obj_dict):
        edge_storage = self.genearte_edge_feature(heteroData,updated_obj_dict)

        out_dict = {}

        for edgeType, values in edge_storage.items():
            origin_node_feat = self.edge_lin[edgeType](values['geo_edge_feature'])
            query = self.q_lin[edgeType](values['geo_edge_feature'])
            key = self.k_lin[edgeType](values['non_geo_edge_feature'])
            value = self.v_lin[edgeType](values['non_geo_edge_feature'])
            attnetion_coef = self.attention_compute(query, key)

            fusion_edge_feature = (value * attnetion_coef) + origin_node_feat
            output = self.classifier[edgeType](fusion_edge_feature)
            out_dict[edgeType] = output
        return out_dict
