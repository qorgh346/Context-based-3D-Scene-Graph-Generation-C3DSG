import sys
import numpy as np
import torch
import os
import json

gt_data_path = '/home/baebro/hojun_ws/3DSSG/data/NYU40/gen_data'
class_txt = 'classes.txt'
relationship_txt = 'relationships.txt'
gt_json = 'relationships_train.json'

def predict_layer(className,instance2Mask,edge_index,rel_gt,output_rel_cls):
    #className =  {1: ['otherfurniture'], 2: ['floor'], 3: ['table'], 4: ['pillow'], 6: ['otherfurniture'], 8: ['pillow'],
    #instance2mask :  {0: tensor([0]), 24: tensor([1]), 454: tensor([2]), 213: tensor([3]), 199: tensor([4]), 300: tensor([5]),
    #edge_index = (2, edge_size)
    #output_rel_cls = (edge_size, num_rel) --> last layer output
    print_debug = True
    # print(instance2Mask)
    # print(className)
    # print(edge_index)
    # print(output_rel_cls.size(0))
    # print(rel_gt)
    rel_gt = {rel[0]:idx for idx,rel in enumerate(rel_gt)}
    # print('rel-gt ',rel_gt)
    rel_gt_reverse = {idx:rel for idx,rel in enumerate(rel_gt)}
    # print('rel-gt ',rel_gt_reverse)

    instance2Mask_convert = {v.item():k for k,v in instance2Mask.items()}
    # print(instance2Mask_convert)
    # sys.exit()

    # GT_external tabel

    mapping_relationship_GT , distribution_GT ,rel_list_GT = process()
    # print(mapping_relationship_GT)
    # print(distribution_GT.size())
    # print(rel_list_GT)
    for i in range(output_rel_cls.size(0)):
        source_num = edge_index[0][i].item()
        target_num = edge_index[1][i].item()
        # print('source_num : ',source_num)
        # print('target_num : ',target_num)
        if source_num == 0 or target_num == 0:
            continue
        source_node_name = className[instance2Mask_convert[source_num]]
        target_node_name = className[instance2Mask_convert[target_num]]

        predicate = '{}_{}'.format(source_node_name[0],target_node_name[0])
        # print('distribution_GT - size : ',distribution_GT.size())
        # print('rel_list_GT length : ',len(rel_list_GT))
        # print(rel_list_GT)

        temp_index_mapping = {}
        with torch.no_grad():
            for idx,knowledge_gt_rel in enumerate(rel_list_GT):

                temp_index_mapping[idx] = rel_gt[knowledge_gt_rel]


            # print('temp_index_mapping : ',torch.tensor(list(temp_index_mapping.values())))
            # print(mapping_relationship_GT)
            # sys.exit()
            GT_Score = distribution_GT[mapping_relationship_GT[predicate]]

            # select_index 를 통해 GT_table의 관계 index 값과 예측된 관계 index값을 일치 시켰음 ( None 은 없음 )
            # print('before - GT Score',GT_Score)
            GT_rel_Score = torch.zeros((len(rel_list_GT)))
            # print(GT_rel_Score.size())

            for temp_i,GT_index in enumerate(torch.tensor(list(temp_index_mapping.values()))):
                GT_rel_Score[GT_index] = GT_Score[temp_i]

            # print('select_after : ',GT_rel_Score,' size :',GT_rel_Score.size())
            # print('GT_argmax index :',torch.argmax(GT_rel_Score))

            # a = GT_rel_Score/torch.sum(GT_rel_Score)
            # print(a)
            # print(torch.sum(GT_rel_Score))
            # print('예측한 관계 클래스 score 점수 :',output_rel_cls[i])
            # print('예측한 관계_argmax index :', torch.argmax(output_rel_cls[i]))
            w = 0.62 # GT : 80% , predict : 20%
            # none_rel_gt = torch.tensor([-0.001])

            # Step1. update weight
            GT_rel_Score = GT_rel_Score * w
            predict_rel_Score = output_rel_cls[i] * (1-w)
            output_rel_cls[i] = GT_rel_Score.cuda() + predict_rel_Score
            # print('GT 가중치 값 적용 후 :',output_rel_cls[i])
            # print('GT 가중치 값 적용 후 argmax _ index :',torch.argmax(output_rel_cls[i]))
            # sys.exit()

            #Step2. distribution
            # multiple_distribution = output_rel_cls[i]*GT_rel_Score.cuda()
            # down = torch.sum(multiple_distribution)
            # output_rel_cls[i] = multiple_distribution / down
            # print('weight all sum = ',torch.sum(output_rel_cls[i]))
            # print('GT after 적용 후 argmax _ index :',torch.argmax(output_rel_cls[i]))
            #
            # print(output_rel_cls[i])
    #
    # sys.exit()

        # continue
    return output_rel_cls
        # print(GT_rel_Score)
        # sys.exit()
        # GT_max_idx = torch.argmax(GT_rel_Score).item()
        # temp_GT_max_idx = (GT_rel_Score >= 0.5 ).nonzero(as_tuple=False)
        # print('-----값이 0.5 이상인 index만 고려----')
        # print('--> ',temp_GT_max_idx)
        #
        # sys.exit()
        # for GT_max_idx in temp_GT_max_idx:
        #     print('-------------Start------------')
        #     GT_max_idx = GT_max_idx.item()
        #
        #     temp_union_data = GT_rel_Score[GT_max_idx]
        #     # print(rel_list_GT[GT_max_idx]) #ex) 16 --> rel_list_GT 의 16번째 관계가 가장 많이 나왔었다는 것
        #     #이 아래 데이터를 json으로 기록하자.
        #     if print_debug:
        #         print('GT : {} {} {}'.format(source_node_name[0],rel_list_GT[GT_max_idx],target_node_name[0]))
        #         print('GT_max_index : ',GT_max_idx)
        #         print('GT :', GT_rel_Score)
        #         #합치기 전 예측한 관계 score 중 가장 높은 것은?
        #
        #         predict_idx = torch.argmax(output_rel_cls[i]).item()
        #         print('Pred :{} {} {}'.format(source_node_name[0], rel_gt_reverse[predict_idx], target_node_name[0]))
        #         print('Pred_max_index : ',predict_idx)
        #         print('Pred :',output_rel_cls[i])
        #
        #     #합치기 후
        #     #이제 GT에서 미리 계산된 통계를 참고해서 두개의 물체가 있을법한 관계에 가중치값 0.2를 더해준다.
        #     # print(temp_union_data)
        #     col_idx = rel_gt[rel_list_GT[GT_max_idx]]
        #     if temp_union_data >= 0:
        #         output_rel_cls[i][col_idx] += temp_union_data
        #     else:
        #         output_rel_cls[i][col_idx] -= temp_union_data
        #
        #
        #     # 합치기 후 예측한 관계 score 중 가장 높은 것은?
        #
        # predict_idx = torch.argmax(output_rel_cls[i]).item()
        # if print_debug:
        #     print('Pred + GT :{} {} {}'.format(source_node_name[0], rel_gt_reverse[predict_idx], target_node_name[0]))
        #     print('Pred + GT :', output_rel_cls[i])
    #
    # sys.exit()
    # return output_rel_cls

def prepare_data():
    with open(os.path.join(gt_data_path, class_txt), 'r') as f:
        class_list = [file.rstrip('\n') for file in f.readlines()]

    with open(os.path.join(gt_data_path, relationship_txt), 'r') as f:
        rel_list = [file.rstrip('\n') for file in f.readlines()]

    return class_list,rel_list

def process():
    class_list,rel_list = prepare_data()
    class_num = len(class_list)
    rel_list.append('none')
    print(rel_list)
    mapping_relationship_idx = {}
    c = 0
    for i_idx, i in enumerate(class_list):
        for j_idx, j in enumerate(class_list):
            mapping_relationship_idx['{}_{}'.format(i, j)] = c
            c += 1
    x = np.load('/home/baebro/hojun_ws/3DSSG/save_figure/save_distribution_GT.npy')
    distribution_GT = torch.from_numpy(x)
    # distribution_GT = distribution_GT.permute(1, 0)

    return mapping_relationship_idx , distribution_GT,rel_list



if __name__ == '__main__':
    process()
