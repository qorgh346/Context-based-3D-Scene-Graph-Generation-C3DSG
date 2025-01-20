import sys

import numpy as np
import torch, random
from gensim.scripts.glove2word2vec import glove2word2vec, KeyedVectors
import op_utils
from torch_geometric.data import HeteroData
from collections import defaultdict

#add. 23.02.01

def build_edge(subject,predicate,object,flag='source_to_taget'):
    import numpy as np
    #subject,object -> Node Type
    #ex. subject = "item" , object = "furniture"
    #predicate -> Edge Type
    #output : 1. (subject - object).edgeIndex
    #관계 간선 특징에 어떠한 정보를 담을 것인가?
    #1. 두 개체쌍의 geometric feature

    #간선 연결은 일단 fully로 연결
    sub_instanceNum = data[subject].num_nodes
    obj_instanceNum = data[object].num_nodes
    edge_index = []
    for sub in range(sub_instanceNum):
        for obj in range(obj_instanceNum):
            # if sub == obj:
            #     continue
            edge_index.append([sub,obj])
    edge_index = np.array(edge_index).T
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)




    return edge_index


def find_3DSSG_Type(gt_knowledge_data,instanceName, context='nodeType'):
    # t_node = type_node, i_node = instance_node
    for t_node, i_node in gt_knowledge_data[context].items():
        if instanceName in i_node:
            return t_node


def build_word_embedding(file_name):
    model = KeyedVectors.load_word2vec_format(file_name, binary=False)
    return model


def build_edge_from_selection(rel_json, node_ids, neighbor_dict, max_edges_per_node, mode="GT_EDGE"):
    '''
    flow: an edge passes message from i to j is denoted as  [i,j].
    '''
    ''' build trees '''
    edge_indices = list()
    mode = 'GT_EDGE'
    # node_ids = filtered_nodes
    # filtered_nodes = {131, 132, 5, 3, 135, 4, 7, 133, 139, 9, 142, 143, 398, 404, 407,
    # 664, 408, 667, 27, 419, 420, 35, 38, 42, 429, 46,431, 432, 175, 49, 54, 311, 441,
    # 185, 313, 317, 194, 325, 206, 83, 220, 230, 235}
    # print('node_ids : ',node_ids)
    # print('neighbor_dict : ',neighbor_dict)
    # print('max_edges_per_node : ',max_edges_per_node)
    if mode == 'GT_EDGE':
        for rel in rel_json:
            edge_indices.append([rel[0], rel[1]])
    else:
        for s_idx in node_ids:
            if s_idx in neighbor_dict:
                nn = neighbor_dict[s_idx]
            else:
                nn = neighbor_dict[str(s_idx)]
            nn = set(nn).intersection(node_ids)  # only the nodes within node_ids are what we want
            if s_idx in nn: nn.remove(s_idx)  # remove itself
            if max_edges_per_node > 0:
                if len(nn) > max_edges_per_node:  # -1
                    nn = list(np.random.choice(list(nn), max_edges_per_node))

            for t_idx in nn:
                edge_indices.append([s_idx, t_idx])
        # print(len(edge_indices))
        # print(edge_indices)
        # sys.exit()
    return edge_indices


def build_neighbor(nns: dict, instance2labelName: dict, gt_nodes: list, n_times: int, n_seed=1):
    ''' Select node'''
    # print(gt_nodes)
    selected_nodes = list(instance2labelName.keys())
    index = np.random.choice(np.unique(selected_nodes), n_seed).tolist()
    index = list(set(index))  # make them unique
    for n_idx in selected_nodes:
        if str(n_idx) not in nns:
            print('cannot find key', n_idx, 'in', nns.keys())
            assert str(n_idx) in nns.keys()

    ''' loop over n times'''
    filtered_nodes = set()
    n_seletected_nodes = dict()  # this save the neighbor with level n. level n+1 is the neighbors from level n.
    n_seletected_nodes[0] = index  # first layer is the selected node. (408,419,142)
    for n in range(n_times):  # n = 0 or n = 1
        ''' collect neighbors '''
        n_seletected_nodes[n + 1] = list()
        unique_nn_found = set()
        for node_idx in n_seletected_nodes[n]:
            found = set(nns[str(node_idx)])
            found = found.intersection(selected_nodes)  # only choose the node within our selections
            # found = '408': [419, 420, 5, 398, 404, 407, 313, 317]
            if len(found) == 0: continue
            unique_nn_found = unique_nn_found.union(found)
            filtered_nodes = filtered_nodes.union(found)
        n_seletected_nodes[n + 1] = unique_nn_found
    # list(filtered_nodes)
    # a = filtered_nodes.intersection(set(gt_nodes))
    # print(filtered_nodes.intersection(set(gt_nodes)))
    return filtered_nodes


def zero_mean(point):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    ''' without norm to 1  '''
    # furthest_distance = point.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
    # point /= furthest_distance
    return point


def generate_wordTovec(model,word):
    word_list = word.split(' ')
    wordVec = torch.zeros(50)
    for word in word_list:
        wordVec += torch.from_numpy(model[word].astype(np.float32))

    return wordVec


def data_preparation(points, instances, selected_instances, num_points, num_points_union,
                     # use_rgb, use_normal,
                     for_train=True, instance2labelName=None, classNames=None,
                     rel_json=None, relationships=None, multi_rel_outputs=None, use_all=False,
                     padding=0.2, num_max_rel=-1, shuffle_objs=True, nns: dict = None,
                     sample_in_runtime: bool = False, num_nn=1, num_seed=1, glove_model=None, use_predict_class=False,
                     scene_type=None, prior_knowledge = None):
    if sample_in_runtime:
        assert nns is not None

    if for_train:
        assert (instance2labelName is not None)
        assert (rel_json is not None)
        assert (classNames is not None)
        assert (relationships is not None)
        assert (multi_rel_outputs is not None)

    instance_counter = defaultdict(int)
    instances = instances.flatten()
    instances_id = list(np.unique(instances))

    if sample_in_runtime:
        if not use_all:
            filtered_nodes = build_neighbor(nns, instance2labelName, num_nn, num_seed)
        else:
            selected_nodes = list(instance2labelName.keys())
            filtered_nodes = selected_nodes  # use all nodes
        edge_indices = build_edge_from_selection(rel_json,filtered_nodes, nns, max_edges_per_node=-1,mode='NOT_GT_EDGE')

        # if num_max_rel > 0:
        #     choices = np.random.choice(range(len(edge_indices)), num_max_rel).tolist()
        #     edge_indices = [edge_indices[t] for t in choices]
        instances_id = list(filtered_nodes)

    if 0 in instances_id:
        instances_id.remove(0)
    if shuffle_objs:
        random.shuffle(instances_id)

    instance2mask = {}
    instance2mask[0] = 0
    word_Mapping = {'room': 'room', 'bed': 'bedroom', 'toilet': 'bathroom', 'desk': 'office', 'dining': 'dining',
                    'storage': 'storage',
                    'living': 'livingroom', 'kitchen': 'kitchen', 'unknown': 'room'}

    nodeType_list = {t_node: dict() for t_node in prior_knowledge['nodeType'].keys()}
    meta_relation_list = {'{} {} {}'.format(triple[0], triple[1], triple[2]): dict()
                      for triple in prior_knowledge['meta_relations']}


    cat = []
    counter = 0
    ''' Build instance2mask and their gt classes '''
    for instance_id in list(np.unique(instances)):
        # print('instance {} size: {}'.format(instance_id,len(points[np.where(instances == instance_id)])))
        if selected_instances is not None:
            if instance_id not in selected_instances:
                # since we divide the whole graph into sub-graphs if the
                # scene graph is too large to resolve the memory issue, there
                # are instances that are not interested in the current sub-graph
                instance2mask[instance_id] = 0
                continue

        if for_train:
            class_id = -1  # was 0

            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in classNames:  # is it a class we care about?
                class_id = classNames.index(instance_labelName)

            if (class_id >= 0) and (instance_id > 0):  # there is no instance 0?
                cat.append(class_id)
        else:
            class_id = 0

        if class_id != -1:  # was 0
            counter += 1
            instance2mask[instance_id] = counter
        else:
            instance2mask[instance_id] = 0

    '''Map edge indices to mask indices'''
    if sample_in_runtime:
        edge_indices = [[instance2mask[edge[0]] - 1, instance2mask[edge[1]] - 1] for edge in edge_indices]

    num_objects = len(instances_id) if selected_instances is None else len(selected_instances)
    index2label = {v: k for k, v in instance2mask.items()}

    masks = np.array(list(map(lambda l: instance2mask[l], instances)), dtype=np.int32)

    dim_point = points.shape[1]
    obj_points = torch.zeros([num_objects, num_points, dim_point])
    descriptor = torch.zeros([num_objects, 11])

    linguisticVec = torch.zeros([num_objects, 100])

    # create normalized pointsets for each object, sorted like the masks
    bboxes = list()


    for i in range(num_objects):
        # 학습에 사용할 point cloud 임
        # 각 물체들에 대해서 128개의 point를 뽑아서 학습을 하는데 충분할까?
        # 어떤 물체는 2000개가 넘는 point들로 구성하고 어떤거는 900개....이럼
        # 전체 pointset 에서 selected 된 object만 뽑음
        # print('i th objects = ',i)
        # print(np.where(masks == i+1)[0])
        # print('num_obj = ',num_objects)

        obj_pointset = points[np.where(masks == i + 1)[0], :]

        # print('i = ', i)
        # print('masks = ',masks)
        # print(obj_pointset.shape)

        min_box = np.min(obj_pointset[:, :3], 0) - padding
        max_box = np.max(obj_pointset[:, :3], 0) + padding
        bboxes.append([min_box, max_box])
        choice = np.random.choice(len(obj_pointset), num_points, replace=True)
        obj_pointset = obj_pointset[choice, :]
        descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset.astype(np.float32))[:, :3])
        obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))

        obj_pointset[:, :3] = zero_mean(obj_pointset[:, :3])
        obj_points[i] = obj_pointset

        # add 23.02.01
        temp_node_instance_dict = dict()
        scan_instance_id = index2label[i + 1]
        scan_instance_name = instance2labelName[scan_instance_id]
        obj_NodeType = find_3DSSG_Type(prior_knowledge, scan_instance_name, context='nodeType')

        t_instance_cout = len(nodeType_list[obj_NodeType]) + 1

        ######################

        if glove_model and not use_predict_class:
            linguisticVec[i, :50] = generate_wordTovec(glove_model,instance2labelName[index2label[i + 1]])
            #torch.from_numpy(
                # glove_model[instance2labelName[index2label[i + 1]]].astype(np.float32))  # word2vec
            scene_label = word_Mapping[scene_type.split(' ')[0]]
            if 'room' in scene_label:
                linguisticVec[i, 50:] = torch.from_numpy(glove_model[scene_label].astype(np.float32))
            else:
                room = torch.from_numpy(glove_model['room'].astype(np.float32))
                scene = torch.from_numpy(glove_model[scene_label].astype(np.float32))
                linguisticVec[i, 50:] = room + scene
            obj_descriptor = torch.concat([descriptor[i],linguisticVec[i]])
            # linguisticVec[i,50:] = torch.from_numpy(glove_model[scene_type].astype(np.float32))
            # torch.from_numpy(glove_model[instance2labelName[index2label[i+1]]].astype(np.float32))
        instance_counter[scan_instance_name] += 1
        temp_node_instance_dict['feature_idx'] = t_instance_cout
        temp_node_instance_dict['scan_instance_name'] = '{}{}'.format(scan_instance_name,instance_counter[scan_instance_name])
        temp_node_instance_dict['obj_points'] = obj_pointset
        temp_node_instance_dict['obj_non_geometric'] = obj_descriptor.view(1,-1)
        temp_node_instance_dict['instance_id']= scan_instance_id
        nodeType_list[obj_NodeType][scan_instance_id] = temp_node_instance_dict


    if not sample_in_runtime:
        # Build fully connected edges
        edge_indices = list()
        max_edges = -1
        for n in range(len(cat)):
            for m in range(len(cat)):
                if n == m: continue
                edge_indices.append([n, m])
        if max_edges > 0 and len(edge_indices) > max_edges and for_train:
            # for eval, do not drop out any edges.
            indices = list(np.random.choice(len(edge_indices), max_edges))
            edge_indices = edge_indices[indices]

    if for_train:
        ''' Build rel class GT '''
        if multi_rel_outputs:
            # multi classification ( [ num_objects, num_objects, relationships_number ] )
            adj_matrix_onehot = np.zeros([num_objects, num_objects, len(relationships)])
        else:
            adj_matrix = np.zeros([num_objects, num_objects])
            adj_matrix += len(relationships) - 1  # set all to none label.

        for coutIdx,r in enumerate(rel_json):
            temp_meta_instance_dict = {}

            if r[0] not in instance2mask or r[1] not in instance2mask: continue
            index1 = instance2mask[r[0]] - 1
            index2 = instance2mask[r[1]] - 1
            if sample_in_runtime:
                if [index1, index2] not in edge_indices: continue

            if for_train:
                if r[3] not in relationships:
                    continue
                r[2] = relationships.index(r[3])  # remap the index of relationships in case of custom relationNames

            if index1 >= 0 and index2 >= 0:
                if multi_rel_outputs:
                    adj_matrix_onehot[index1, index2, r[2]] = 1
                else:
                    adj_matrix[index1, index2] = r[2]

                # meta-relations add , 23.02.01
                # 관계 데이터엔 라벨이 존재하는데 class 에는 존재하지 않는 경우가 몇몇 존재함
                subIdx = r[0]
                objIdx = r[1]
                subInstanceName = instance2labelName[subIdx]
                objInstanceName = instance2labelName[objIdx]
                subInstanceType = find_3DSSG_Type(prior_knowledge, subInstanceName, context="nodeType")
                objInstanceType = find_3DSSG_Type(prior_knowledge, objInstanceName, context="nodeType")
                edgeType = find_3DSSG_Type(prior_knowledge, r[3], context="edgeType")
                meta_triple = '{} {} {}'.format(subInstanceType, edgeType, objInstanceType)



                if meta_triple in meta_relation_list.keys():
                    subInstance_idx = nodeType_list[subInstanceType][subIdx]['feature_idx']
                    objInstance_idx = nodeType_list[objInstanceType][objIdx]['feature_idx']

                    temp_meta_instance_dict['feature_sub_obj_idx'] = '{}_{}'.format(subInstance_idx,
                                                                                    objInstance_idx)
                    temp_meta_instance_dict['gt_predicate'] = r[3]
                    meta_relation_list[meta_triple][coutIdx] = temp_meta_instance_dict
            # else:
            #     print(r[0],r[1])



        ''' Build rel point cloud '''
        if multi_rel_outputs:
            rel_dtype = np.float32
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=rel_dtype))
        else:
            rel_dtype = np.int64

        if multi_rel_outputs:
            gt_rels = torch.zeros(len(edge_indices), len(relationships), dtype=torch.float)
        else:
            gt_rels = torch.zeros(len(edge_indices), dtype=torch.long)

    rel_points = list()
    for e in range(len(edge_indices)):
        edge = edge_indices[e]
        index1 = edge[0]
        index2 = edge[1]
        if for_train:
            if multi_rel_outputs:
                # print(adj_matrix_onehot[index1,index2,:])
                gt_rels[e, :] = adj_matrix_onehot[index1, index2, :]
                if torch.sum(gt_rels[e, :]) == 0:
                    gt_rels[e, 0] = 1  # relationships.index('none') # 'none'

            else:
                gt_rels[e] = adj_matrix[index1, index2]
        # Bounding box의 겹치는 정도의 mask 된 1차원 배열이 relation point set으로 들어감
        # Respective Masked M_ij --> i = source node : 1 , j = target node : 2 , none : 0
        if not glove_model:
            # 22.08.22 : word-embedding을 false로 해야지만 rel_point를 계산하여 넣도록 코드를 짜놓음.
            # 만약에 rel_point와 word_embedding을 함께 사용될 경우에 코드를 수정해야됌

            mask1 = (masks == index1 + 1).astype(np.int32) * 1
            mask2 = (masks == index2 + 1).astype(np.int32) * 2
            mask_ = np.expand_dims(mask1 + mask2, 1)
            bbox1 = bboxes[index1]
            bbox2 = bboxes[index2]
            min_box = np.minimum(bbox1[0], bbox2[0])
            max_box = np.maximum(bbox1[1], bbox2[1])
            filter_mask = (points[:, 0] > min_box[0]) * (points[:, 0] < max_box[0]) \
                          * (points[:, 1] > min_box[1]) * (points[:, 1] < max_box[1]) \
                          * (points[:, 2] > min_box[2]) * (points[:, 2] < max_box[2])
            points4d = np.concatenate([points, mask_], 1)

            pointset = points4d[np.where(filter_mask > 0)[0], :]
            choice = np.random.choice(len(pointset), num_points_union, replace=True)
            pointset = pointset[choice, :]
            pointset = torch.from_numpy(pointset.astype(np.float32))
            pointset[:, :3] = zero_mean(pointset[:, :3])
            rel_points.append(pointset)

    if not glove_model:
        if for_train:
            try:
                rel_points = torch.stack(rel_points, 0)
            except:
                rel_points = torch.zeros([4, num_points_union])
        else:
            rel_points = torch.stack(rel_points, 0)

    cat = torch.from_numpy(np.array(cat, dtype=np.int64))
    edge_indices = torch.tensor(edge_indices, dtype=torch.long)

    if glove_model and not use_predict_class:
        descriptor = torch.concat([descriptor, linguisticVec], dim=1)


    # Homogeneous Graph --> Heterogeneous Graph
    #nodeType_list, meta_relation_list !

    heteroGraphdata = HeteroData()
    for tNode,iNodeList in nodeType_list.items():
        if len(iNodeList) == 0: continue
        tNode_x = torch.zeros((len(iNodeList),256,6))
        tNode_descriptor = torch.zeros((len(iNodeList)),111)
        instance_names = list()
        for instanceNode,instanceInfo in iNodeList.items():
            tNode_x[instanceInfo['feature_idx']-1][:] = instanceInfo['obj_points']
            tNode_descriptor[instanceInfo['feature_idx']-1][:] = instanceInfo['obj_non_geometric'][0]
            instance_names.append(instanceInfo['scan_instance_name'])
        heteroGraphdata[tNode].x = tNode_x
        heteroGraphdata[tNode].descriptor = tNode_descriptor
        heteroGraphdata[tNode].instanceNames = instance_names


    for meta_type, meta_relations in meta_relation_list.items():
        if len(meta_relations) == 0:
            continue
        m_triple = meta_type.split(' ')
        sType = m_triple[0]
        eType = m_triple[1]
        oType = m_triple[2]
        sNum = heteroGraphdata[sType].num_nodes
        oNum = heteroGraphdata[oType].num_nodes
        edge_index = []

        if use_all:
            #fully connected
            for s in range(sNum):
                for o in range(oNum):
                    edge_index.append([s, o])

        else:
            for meta_instance, meta_info in meta_relations.items():
                sub_obj= meta_info['feature_sub_obj_idx']
                sub_id = sub_obj.split('_')[0]
                obj_id = sub_obj.split('_')[1]
                edge_index.append([int(sub_id)-1, int(obj_id)-1])
                # sub_id : 6, obj_id = 4

                # for edge in edge_indices:
                #     if [sub_id-1,obj_id-1] in edge:
                #         edge_index.append([sub, obj])

        edge_index = np.array(edge_index).T
        heteroGraphdata[sType, eType, oType].edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_type_instance = prior_knowledge['edgeType'][eType]

        meta_relation_gt = torch.zeros((edge_index.shape[1], len(edge_type_instance)), dtype=torch.long)
        # meta_relation_gt[]

        adj_matrix_onehot = torch.zeros((sNum, oNum, len(edge_type_instance)))

        for E_InstanceIdx, E_InstanceInfo in meta_relations.items():
            GT_subobjIdx = E_InstanceInfo['feature_sub_obj_idx'].split('_')
            GT_subIdx = int(GT_subobjIdx[0]) - 1
            GT_objIdx = int(GT_subobjIdx[1]) - 1
            GT_predIdx = edge_type_instance.index(E_InstanceInfo['gt_predicate'])
            adj_matrix_onehot[GT_subIdx, GT_objIdx, GT_predIdx] = 1
        for e in range(edge_index.shape[1]):
            subIdx = edge_index[0][e]
            objIdx = edge_index[1][e]
            meta_relation_gt[e, :] = adj_matrix_onehot[subIdx, objIdx, :]

        heteroGraphdata[sType, eType, oType].gt_label = meta_relation_gt


    if for_train:
        if glove_model:
            return obj_points, descriptor, edge_indices, instance2mask, gt_rels, cat,heteroGraphdata
        else:
            return obj_points, rel_points, edge_indices, instance2mask, gt_rels, cat,heteroGraphdata

    else:
        if glove_model:
            return obj_points, descriptor, edge_indices, instance2mask
        else:
            return obj_points, rel_points, edge_indices, instance2mask

