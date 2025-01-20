import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from scipy.special import erf
from sklearn.svm._libsvm import predict

gt_data_path = '/home/baebro/hojun_ws/3DSSG/data/NYU40/gen_data'
class_txt = 'classes.txt'
relationship_txt = 'relationships.txt'
gt_json = 'relationships_train.json'
obj_json = 'objects.json'

with open(os.path.join(gt_data_path,class_txt),'r') as f:
    class_list = [file.rstrip('\n') for file in f.readlines()]

with open(os.path.join(gt_data_path, relationship_txt), 'r') as f:
    rel_list = [file.rstrip('\n') for file in f.readlines()]

with open(os.path.join(gt_data_path,gt_json), 'r') as f:
    json_data = json.dumps(json.load(f))
    gt_data = json.loads(json_data)

class_num = len(class_list)
rows = class_num * (class_num-1) -1

# rel_list.append('none') #add none relationship
cols = len(rel_list)
print(rel_list)

mapping_relationship_idx = {}
c = 0
for i_idx,i in enumerate(class_list):
    for j_idx,j in enumerate(class_list):
        mapping_relationship_idx['{}_{}'.format(i_idx,j_idx)] = c
        c+=1

distribution_table = torch.zeros((len(mapping_relationship_idx.keys()),cols))
mapping_class = {v:k for k,v in enumerate(class_list)} #sofa : 0, door : 1 , box : 2...
mapping_rel = {v:k for k,v in enumerate(rel_list)} # ' same_part' : 0 , ...
# print('mapping_class : ',mapping_class)
# print('mapping_ rel : ',mapping_rel)
cout = 0
edge_index_idx = {}
num_class = len(class_list)
for i in range(num_class):
    for j in range(num_class):
        edge_index_idx['{}_{}'.format(class_list[i],class_list[j])] = cout
        cout +=1
print(edge_index_idx)


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
    plt.savefig('{}visual_bar_relation_count.png'.format(save_path))
    # for i in range(0, rel_num):
    #     print(i)
    #     # years = ['2018', '2019', '2020']
    #     # values = [100, 400, 900]
    #     column = rel_list[i]
    #     values = bar_data[column]
    #     # column = list(bar_data.keys())[i:max_iter + i]
    #     # values = list(bar_data.values())[i:max_iter + i]  # [:10]
    #     plt.figure(figsize=(30, 30))
    #     plt.bar(x, list(bar_data.values()), width=0.8)
    #     plt.xticks(x, column, fontsize=25)
    #
        # plt.show()
        # plt.savefig('{}visual_bar_relation_count{}.png'.format(save_path,i))

def visual_distribution(distribution_table,save_path='/home/baebro/hojun_ws/3DSSG/save_figure/'):

    #Softmax 함수를 통해  source obj --- target obj --> 얼마나 정규분포를 잘 이루었는지??? 시각화 또는 글
    #각 관계별로 시각화
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()

    X_ss = ss.fit_transform(distribution_table.numpy())
    print(X_ss)
    # sys.exit()
    # m = nn.Softmax(dim=1)

    # temp_table =torch.log(distribution_table)  #F.relu(distribution_table)
    # print(temp_table.size())
    print(X_ss[574])
    print(X_ss[444])
    print(distribution_table[574])
    print(distribution_table[444])
    # distribution_table = distribution_table.permute(1,0)
    # print(distribution_table.size())

    # a = F.log_softmax(distribution_table,dim=1)
    # a = m(distribution_table)
    # print(a[0].sum())
    # print(a)
    # sys.exit()
    # b = a.numpy()
    import seaborn as sns
    print('Xss_shape',X_ss.shape)
    print(np.transpose(X_ss).shape)
    b = np.transpose(X_ss)
    print('b.shape: ',b.shape)
    # sys.exit()
    print(b[4][:100])
    plt.rc('font', size=14)
    plt.rc('axes',titlesize=5)
    plt.rc('axes',labelsize=5)
    plt.rc('xtick',labelsize=5)
    plt.rc('ytick',labelsize=5)


    plt.figure(figsize=(10, 10))
    for i in range(len(b)):
        x = [i for i in range(len(b[i]))]
        x = np.linspace(-1, 1, len(x))
        plt.subplot(6,5,i+1)
        plt.title(rel_list[i],fontsize=10)
        plt.plot(x, b[i],alpha=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig('{}visual_distribution_Per_relation.png'.format(save_path))
    np.save('{}save_distribution_GT.npy'.format(save_path), np.transpose(b))

def visual_triple_relationship(data,save_path='/home/baebro/hojun_ws/3DSSG/save_figure/'):

    color_list = ['red','blue','gold','cyan','blue','indigo','violet','purple','ivory','beige','maroon','brown',
             'lightgray','grey','salmon','tan','darkorange','forestgreen','darkcyan','red','blue','gold','cyan','blue','indigo','violet','purple','ivory','beige','maroon','brown',
             'lightgray','grey','salmon','tan','darkorange','forestgreen','darkcyan','red','blue','gold','cyan','blue','indigo','violet','purple','ivory','beige','maroon','brown',
             'lightgray','grey','salmon','tan','darkorange','forestgreen','darkcyan','red','blue','gold','cyan','blue','indigo','violet','purple','ivory','beige','maroon','brown',
             'lightgray','grey','salmon','tan','darkorange','forestgreen','darkcyan','red','blue','gold','cyan','blue','indigo','violet','purple','ivory','beige','maroon','brown',
             'lightgray','grey','salmon','tan','darkorange','forestgreen','darkcyan']

    plt.figure(figsize=(20, 20))
    plt.rc('font', size=10)
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=5)
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=8)
    # plt.rc('title',titlesize=8)
    plt.rc('legend',fontsize=8)

    count = 0
    for c,(src,predicate_list) in enumerate(data.items()):
        plt.title(src)
        X = list()
        Y = list()
        Y_label = list()
        X_label = list()
        if len(predicate_list) == 1:
            continue
        for x_idx, predicate in enumerate(predicate_list):
            datas =predicate.split('/')
            term = datas[0] #attached to ...etc
            target_node = datas[1] #wall,floor,bed ...etc
            value = datas[2] #1652...
            X.append(x_idx)
            Y.append(float(value))
            Y_label.append(term)
            X_label.append(target_node)
        # plt.yticks(Y,Y_label,fontsize=15,rotation=0)

        plt.xticks(X,X_label)
        a = list()
        for idx, i in enumerate(X):
            a.append(plt.bar(i,Y[idx],width=0.4,color=color_list[idx]))

        plt.legend(handles=a, labels=Y_label)

        plt.subplot(6, 7, count + 1)
        count+=1
    plt.tight_layout()
    # plt.show()
    plt.savefig('{}visual_triple_relationship.png'.format(save_path))


def cal_distrubution(distribution_table):
    # source obj --- target obj  --> 릴레이션 개수 ( 총 26개 )에 대해서 얼마나 많이
    # 나왔는지에 대한 표 또는 그림으로 시각화 하기
    print(distribution_table[12])
    print(mapping_rel)
    convert_rel = {v: k for k, v in mapping_relationship_idx.items()}

    temp_count = []
    visual_distribution_data = {}
    for i in range(distribution_table.size(0)):

        knowldege_rel_idx = torch.argmax(distribution_table[i]).item()
        relation = rel_list[knowldege_rel_idx]
        source_node = convert_rel[i].split('_')[0]
        target_node = convert_rel[i].split('_')[1]
        src = class_list[int(source_node)]
        target = class_list[int(target_node)]
        if distribution_table[i][knowldege_rel_idx] >= 300 and relation !='none':
            value = distribution_table[i][knowldege_rel_idx]
            temp_count.append(value)
            try:
                visual_distribution_data[src].append('{}/{}/{}'.format(relation,target,value))
            except:
                visual_distribution_data[src] = list()
                visual_distribution_data[src].append('{}/{}/{}'.format(relation, target, value))
            print('{}  ------- {}  ------- {} ----- value : {} '.format(src, relation, target,value))
    print(visual_distribution_data)
    print(len(temp_count))
    visual_triple_relationship(visual_distribution_data)
    visual_distribution(distribution_table,save_path='/home/baebro/hojun_ws/3DSSG/save_figure/')


    # sys.exit()


def start(gt_data):
    scans_data = gt_data['scans']
    # print('total_scene numbers = ',len(scans_data))
    train_scans_num = len(scans_data)

    bar_data = {rel: 0 for rel in rel_list}


    #### ex ) distribution_table = torch.zeros((len(mapping_relationship_idx.keys()),cols))
    #['relationships'] 안에 존재하는 물체들만 count를 함 그러나, 여기 안에는 none 이라는 관계가 포함되지 않음
    for scan in scans_data:
        instance_id = scan['objects']
        for rel_edge in scan['relationships']:
            s_n = rel_edge[0]  # source node
            t_n = rel_edge[1]  # target node
            rel = rel_edge[3]  # relation
            # if rel == 'same part':
            #     continue
            # print(rel)
            # sys.exit()
            bar_data[rel] += 1
            s_n_idx = mapping_class[instance_id[str(s_n)]]
            t_n_idx = mapping_class[instance_id[str(t_n)]]

            distribution_row_num = mapping_relationship_idx['{}_{}'.format(s_n_idx, t_n_idx)]
            distribution_col_num = mapping_rel[rel]
            distribution_table[distribution_row_num, distribution_col_num] += 1

            # if rel == 'same part':
            #     distribution_table[distribution_row_num, distribution_col_num] += 0.1
            # else:
            #    distribution_table[distribution_row_num, distribution_col_num] += 1

    # 어떤 관계가 몇 개가 있는지 시각화
    # visual_bar(bar_data)
    print(bar_data)
    temp_index = 3
    for scans in gt_data['scans']:
        print('scan _ id = ',scans['scan'])
        for rel in scans['relationships']:
            subject = scans['objects'][str(rel[0])]
            object =  scans['objects'][str(rel[1])]
            predicate = rel[3]
            print('{} {} {} '.format(str(rel[0])+'_'+subject,predicate,str(rel[1])+'_'+object))

        sys.exit()
        temp_index += 1
        if temp_index == 3:
            sys.exit()
    none_relationships_list = list()
    for idx,i in enumerate(distribution_table):
        # print(i)
        a = torch.sum(i)
        # print(a)
        print(torch.max(i))
        if a.item() == 0:
            none_relationships_list.append(idx)
            print(a)
        # sys.exit()
    print(none_relationships_list) #train_scans_num

    for none_relation_idx in none_relationships_list:
        #전체 학습데이터에서 어떠한 관계가 안나온 주어--목적어는 None relationship type에 전체 장면 개수를 넣었음. (빈도수니깐)
        distribution_table[none_relation_idx][-1] = train_scans_num

    # print(distribution_table)

    # for idx,i in enumerate(distribution_table):
    #     print(i)


    # sys.exit()
    cal_distrubution(distribution_table)




# print(a)
# print(a.size())

if __name__ =='__main__':
    start(gt_data)

