import sys

import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
gt_data_path = '/home/baebro/hojun_ws/3DSSG/data/NYU40/gen_data_gt'
class_txt = 'classes.txt'
relationship_txt = 'relationships.txt'
gt_json = 'relationships_train.json'
gt_new_json = 'SceneGraphAnnotation.json'

with open(os.path.join(gt_data_path,class_txt),'r') as f:
    class_list = [file.rstrip('\n') for file in f.readlines()]

with open(os.path.join(gt_data_path, relationship_txt), 'r') as f:
    rel_list = [file.rstrip('\n') for file in f.readlines()]

with open(os.path.join(gt_data_path,gt_new_json), 'r') as f:
    json_data = json.dumps(json.load(f))
    gt_datas = json.loads(json_data)

def visual_bar(bar_data,save_path = '/home/baebro/hojun_ws/3DSSG/save_figure/'):
    max_iter = 13

    rel_num = len(bar_data.keys())

    bar_data = dict(sorted(bar_data.items(),key=lambda x : x[1],reverse=True))
    print(bar_data)
    rel_list = list(bar_data.keys())
    rel_num = len(bar_data.keys())
    # bar_data = sort(list(bar_data.values()))

    x = np.arange(rel_num)  # [:10]
    plt.figure(figsize=(50, 50))
    plt.bar(x, list(bar_data.values()), width=0.4)

    column = list()
    for relation_name in rel_list:
        token_idx = relation_name.find(' ')
        if token_idx != -1:
            relation_name = relation_name[:token_idx]+'\n'+relation_name[token_idx:]
            # column.append(relation_name)
        column.append(relation_name)
    plt.xticks(x, column, fontsize=20)
    plt.ylim([0, 25000])
    plt.xlabel('Relation Name')
    plt.ylabel('Frequency')
    # plt.show()
    plt.savefig('{}visual_bar_relation_count_O27R16.png'.format(save_path))

def cal_frequency_rel(scans_data):
    distribution_table = torch.zeros((len(mapping_relationship_idx.keys()),cols))

    bar_data = {rel: 0 for rel in rel_list}

    #### ex ) distribution_table = torch.zeros((len(mapping_relationship_idx.keys()),cols))
    # ['relationships'] 안에 존재하는 물체들만 count를 함 그러나, 여기 안에는 none 이라는 관계가 포함되지 않음
    for scan_id,scan in scans_data.items():
        instance_id = scan['nodes']
        for rel_edge in scan['edges']:
            s_n = rel_edge[0]  # source node
            t_n = rel_edge[1]  # target node
            rel = rel_edge[3]  # relation
            # if rel == 'same part':
            #     continue
            # print(rel)
            # sys.exit()
            bar_data[rel] += 1
            s_n_idx = mapping_class[instance_id[s_n]['rio27_name']]
            t_n_idx = mapping_class[instance_id[t_n]['rio27_name']]

            distribution_row_num = mapping_relationship_idx['{}_{}'.format(s_n_idx, t_n_idx)]
            distribution_col_num = mapping_rel[rel]
            distribution_table[distribution_row_num, distribution_col_num] += 1

    return distribution_table,bar_data

if __name__ == '__main__':
    class_num = len(class_list)
    rows = class_num * (class_num-1) -1
    cols = len(rel_list)

    mapping_relationship_idx = {}
    c = 0
    for i_idx,i in enumerate(class_list):
        for j_idx,j in enumerate(class_list):
            mapping_relationship_idx['{}_{}'.format(i_idx,j_idx)] = c
            c+=1

    mapping_class = {v:k for k,v in enumerate(class_list)} #sofa : 0, door : 1 , box : 2...
    mapping_rel = {v:k for k,v in enumerate(rel_list)} # ' same_part' : 0 , ...

    distribution_rel,bar_data = cal_frequency_rel(gt_datas)
    print(bar_data)
    visual_bar(bar_data)
    for scan_id,data in gt_datas.items():
        print('scan _ id = ',scan_id)
        for rel in data['edges']:
            subject = data['nodes'][rel[0]]['rio27_name']
            object =  data['nodes'][rel[1]]['rio27_name']
            predicate = rel[3]
            print('{} {} {} '.format(str(rel[0])+'_'+subject,predicate,str(rel[1])+'_'+object))
        sys.exit()

