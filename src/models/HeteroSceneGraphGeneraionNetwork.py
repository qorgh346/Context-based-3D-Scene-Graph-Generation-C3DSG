if __name__ == '__main__' and __package__ is None:
    from os import sys

    sys.path.append('../')
import os, torch, time
from tqdm import tqdm
# from model_SGFN import SGFNModel
# from model_SGPN import SGPNModel
# from model_SGPNHojun import SGAPNModel
from model_HeteroSGHojun import HGTModel
from CSGGN.src.datasets.dataset_builder import build_dataset
from torch.utils.tensorboard import SummaryWriter
from config import Config
import op_utils
from utils import plot_confusion_matrix
from torch_geometric.loader import DataLoader
from utils.ssg_eval_tool import  Predicate_Recall_Hetero
from torch.utils.data.distributed import DistributedSampler


def log_string(str):
    print(str)


class HGT_3DSSG():
    def __init__(self, config, args):
        self.config = config
        self.args = args
        try:
            self.model_name = self.config.NAME
        except:
            self.model_name = 'HGT_3DSSG'

        self.mconfig = mconfig = config.MODEL
        ''' set dataset to SGFN if handcrafted edge descriptor is used '''
        self.use_edge_descriptor = self.mconfig.USE_PointNet != True
        # use_edge_descriptor = True
        ''' Build dataset '''
        dataset = None
        if config.MODE == 'train':
            if config.VERBOSE: print('build train dataset')
            self.dataset_train = build_dataset(self.config, split_type='train_scans', shuffle_objs=True,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            self.dataset_train.__getitem__(0)
            self.w_cls_obj = self.w_cls_rel = None
            if self.config.WEIGHTING:
                self.w_cls_obj = self.dataset_train.w_cls_obj
                self.w_cls_rel = self.dataset_train.w_cls_rel
            self.className = self.dataset_train.classNames

            self.rel_w = self.dataset_train.w_cls_rel
            self.obj_w = self.dataset_train.w_cls_obj

        if  config.MODE == 'trace'  or config.MODE == 'train' :
            if config.VERBOSE: print('build valid dataset')
            self.dataset_valid = build_dataset(self.config, split_type='validation_scans', shuffle_objs=False,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            num_obj_class = len(self.dataset_valid.classNames)
            num_rel_class = len(self.dataset_valid.relationNames)
            dataset = self.dataset_valid
            self.className = self.dataset_valid.classNames
            self.rel_w = self.dataset_valid.w_cls_rel
            self.obj_w = self.dataset_valid.w_cls_obj
        if config.MODE == 'eval':
            try:
                if config.VERBOSE: print('build test dataset')
                self.dataset_eval = build_dataset(self.config, split_type='test_scans', shuffle_objs=False,
                                                  multi_rel_outputs=mconfig.multi_rel_outputs,
                                                  use_rgb=mconfig.USE_RGB,
                                                  use_normal=mconfig.USE_NORMAL)
                num_obj_class = len(self.dataset_eval.classNames)
                num_rel_class = len(self.dataset_eval.relationNames)
                dataset = self.dataset_eval
                self.className = self.dataset_eval.classNames
                self.rel_w = self.dataset_eval.w_cls_rel
                self.obj_w = self.dataset_eval.w_cls_obj

            except:
                print('canno build eval dataset.')
                self.dataset_eval = None

        ''' Build Model '''
        if self.use_edge_descriptor:  # True
            print('num_rel_class', num_rel_class)
            meta_data = self.dataset_train.__getitem__(0).metadata()
            prior_knowledge = self.dataset_train.prior_knowledge
            self.model = HGTModel(config, self.model_name, num_obj_class, num_rel_class, self.className, self.obj_w,
                                    self.rel_w,128,256,2,2,meta_data,prior_knowledge).to(config.DEVICE)

            # self.model = SGFNModel(config,self.model_name,num_obj_class,args,num_rel_class,use_pointTransformer=False).to(config.DEVICE)
        else:
            # raise NotImplementedError('not yet cleaned.')
            self.model = HGTModel(config, self.model_name, num_obj_class, num_rel_class, self.className,128,256,2,2,).to(
                config.DEVICE)
            self.dataset_train.meta
            # self.model = SGPNModel(config,self.model_name,num_obj_class, num_rel_class).to(config.DEVICE)

        self.samples_path = os.path.join(config.PATH, self.model_name, 'samples')
        self.results_path = os.path.join(config.PATH, self.model_name, 'results')
        self.trace_path = os.path.join(config.PATH, self.model_name, 'traced')

        if config.MODE == 'train' or config.MODE == 'eval':
            pth_log = os.path.join(config.PATH, "logs", self.model_name)
            self.writter = SummaryWriter(pth_log)

    def load(self, best=False):
        return self.model.load(best)

    def data_processing(self, items, max_edges=-1):
        # print('self.use_edge_descriptor : ',self.use_edge_descriptor)
        if self.use_edge_descriptor:
            with torch.no_grad():
                # items = [item.squeeze(0) for item in items]
                # items = self.cuda(*items)
                items = items.to(self.config.DEVICE)
            return items
        else:
            with torch.no_grad():
                obj_points, rel_points, gt_class, gt_rels, edge_indices = items
                obj_points = obj_points.squeeze(0)
                rel_points = rel_points.squeeze(0)

                edge_indices = edge_indices.squeeze(0)
                gt_class = gt_class.squeeze(0).flatten().long()
                gt_rels = gt_rels.squeeze(0)

                obj_points = obj_points.permute(0, 2, 1)
                rel_points = rel_points.permute(0, 2, 1)
                obj_points, rel_points, edge_indices, gt_class, gt_rels = \
                    self.cuda(obj_points, rel_points, edge_indices, gt_class, gt_rels)
            return obj_points, rel_points, gt_class, gt_rels, edge_indices

    # 반복문
    def train(self):
        ''' create data loader '''
        drop_last = True
        world_size = torch.cuda.device_count()
        rank = 0
        sampler = DistributedSampler(self.dataset_train, num_replicas=world_size, rank=rank, shuffle=True)

        train_loader = DataLoader(self.dataset_train,batch_size=1)
        test_loader = DataLoader(self.dataset_valid,batch_size=1)

        start_epoch = 1
        max_iteration = int(float(self.config.MAX_EPOCHES) * len(self.dataset_train))
        # bar = tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9)
        global_epoch = 0
        best_loss = 9999

        for epoch in range(start_epoch, self.config.MAX_EPOCHES):

            print('\n\nTraining epoch: %d' % epoch)
            train_loss_sum = 0
            loss_support_sum = 0
            loss_proximity_sum = 0
            loss_inclusion_sum = 0
            loss_comparative_sum = 0


            train_support_rel_recall = Predicate_Recall_Hetero(len(train_loader),
                                                         num_classes=6,need_softmax=True)
            train_proximity_rel_recall = Predicate_Recall_Hetero(len(train_loader),
                                                         num_classes=7, need_softmax=True)
            train_inclusion_rel_recall = Predicate_Recall_Hetero(len(train_loader),
                                                         num_classes=7, need_softmax=True)
            train_comparative_rel_recall = Predicate_Recall_Hetero(len(train_loader),
                                                         num_classes=6, need_softmax=True)

            metric_dict = {'support':train_support_rel_recall,'proximity':train_proximity_rel_recall,'inclusion':train_inclusion_rel_recall,
                           'comparative':train_comparative_rel_recall}
            bar = tqdm(enumerate(train_loader), total=len(train_loader))


            for i,items in bar:
                # print('iter = ',i)
                # continue
                # if i == 2: break
                # loss_sum = 0
                self.model.train()
                heteroData = items
                scan_id = heteroData['scan_id'][0]
                edge_storage = {
                    edge_type[1]: {
                        'gt_label': torch.tensor([], device=self.config.DEVICE),
                        'pair_instanceNames': list()
                    }
                    for edge_type in heteroData.edge_types
                }

                ''' get data '''
                tick = time.time()

                if self.use_edge_descriptor:
                    heteroData = self.data_processing(heteroData)

                    for meta_rel, edge_index_list in heteroData.edge_index_dict.items():
                        subType, edgeType, objType = meta_rel

                        meta_gt_label = heteroData[meta_rel]['gt_label']
                        edge_storage[edgeType]['gt_label'] = torch.concat(
                            [edge_storage[edgeType]['gt_label'], meta_gt_label],
                            dim=0)

                        for idx in range(edge_index_list.size(1)):
                            sub_indexs = edge_index_list[0][idx].item()  # 주어 인스턴스노드의 Index 번호
                            obj_indexs = edge_index_list[1][idx].item()  # 목적어 인스턴스 노드의 Index 번호
                            sub_instanceName = heteroData[subType].instanceNames[0][sub_indexs]
                            obj_instanceName = heteroData[objType].instanceNames[0][obj_indexs]
                            # print(sub_instanceName, '--', obj_instanceName)
                            edge_storage[edgeType]['pair_instanceNames'].append(
                                '{}-{}'.format(sub_instanceName, obj_instanceName))


                    logs, rel_pred_dict, loss_info,loss_storage = self.model.process(heteroData,edge_storage,'train')
                    print(loss_info)

                    loss_support_sum += loss_storage['support']
                    loss_proximity_sum += loss_storage['proximity']
                    loss_inclusion_sum += loss_storage['inclusion']
                    loss_comparative_sum += loss_storage['comparative']
                    train_loss_sum += loss_info
                    try:
                        for edge_type, v in rel_pred_dict.items():
                            # print(edge_storage)
                            metric_dict[edge_type].calculate_recall(rel_pred_dict[edgeType],
                                                                  edge_storage[edgeType]['gt_label'])
                    except:
                        print('except ')
                        print(edge_storage)
                        continue

            for edge_type, metric in metric_dict.items():
                metric.final_update()
            # len(train_loader)
            log_string( 'epoch : {} Training mean total loss: {}'.format(epoch,(train_loss_sum / len(train_loader) )))
            log_string( 'Training mean support loss: {}'.format((loss_support_sum / len(train_loader) )))
            log_string( 'Training mean proximity loss: {}'.format((loss_proximity_sum / len(train_loader) )))
            log_string( 'Training mean inclusion loss: {}'.format((loss_inclusion_sum / len(train_loader) )))
            log_string( 'Training mean comparative loss: {}'.format((loss_comparative_sum / len(train_loader) )))

            log_string('Training support ' + metric_dict['support'].print_string())
            log_string('Training inclusion ' + metric_dict['inclusion'].print_string())
            log_string('Training proximity ' + metric_dict['proximity'].print_string())
            log_string('Training comparative ' + metric_dict['comparative'].print_string())

            # test

            # if 'VALID_INTERVAL' in self.config and self.config.VALID_INTERVAL > 0 and epoch % self.config.VALID_INTERVAL == 0:
            #     print('start validation...')
            #
            #     print('')
            #     self.eval()

            test_support_rel_recall = Predicate_Recall_Hetero(len(test_loader),
                                                               num_classes=6, need_softmax=True)
            test_proximity_rel_recall = Predicate_Recall_Hetero(len(test_loader),
                                                                 num_classes=7, need_softmax=True)
            test_inclusion_rel_recall = Predicate_Recall_Hetero(len(test_loader),
                                                                 num_classes=7, need_softmax=True)
            test_comparative_rel_recall = Predicate_Recall_Hetero(len(test_loader),
                                                                   num_classes=6, need_softmax=True)
            test_metric_dict = {'support': test_support_rel_recall, 'proximity': test_proximity_rel_recall,
                           'inclusion': test_inclusion_rel_recall,
                           'comparative': test_comparative_rel_recall}
            with torch.no_grad():
                test_loss_sum = 0
                log_string('---- EPOCH %03d TEST ----' % (global_epoch + 1))
                test_bar = tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9)

                for i, items in test_bar:
                    # print('iter = ',i)
                    # continue
                    # if i == 2: break
                    self.model.eval()
                    heteroData = items
                    scan_id = heteroData['scan_id'][0]
                    edge_storage = {
                        edge_type[1]: {
                            'gt_label': torch.tensor([], device=self.config.DEVICE),
                            'pair_instanceNames': list()
                        }
                        for edge_type in heteroData.edge_types
                    }

                    ''' get data '''
                    tick = time.time()

                    if self.use_edge_descriptor:
                        heteroData = self.data_processing(heteroData)

                        for meta_rel, edge_index_list in heteroData.edge_index_dict.items():
                            subType, edgeType, objType = meta_rel

                            meta_gt_label = heteroData[meta_rel]['gt_label']
                            edge_storage[edgeType]['gt_label'] = torch.concat(
                                [edge_storage[edgeType]['gt_label'], meta_gt_label],
                                dim=0)

                            for idx in range(edge_index_list.size(1)):
                                sub_indexs = edge_index_list[0][idx].item()  # 주어 인스턴스노드의 Index 번호
                                obj_indexs = edge_index_list[1][idx].item()  # 목적어 인스턴스 노드의 Index 번호
                                sub_instanceName = heteroData[subType].instanceNames[0][sub_indexs]
                                obj_instanceName = heteroData[objType].instanceNames[0][obj_indexs]
                                # print(sub_instanceName, '--', obj_instanceName)
                                edge_storage[edgeType]['pair_instanceNames'].append(
                                    '{}-{}'.format(sub_instanceName, obj_instanceName))

                        logs, rel_pred_dict, loss_info,loss_storage = self.model.process(heteroData, edge_storage,'test')
                        print(loss_info)
                        test_loss_sum += loss_info
                        loss_support_sum += loss_storage['support']
                        loss_proximity_sum += loss_storage['proximity']
                        loss_inclusion_sum += loss_storage['inclusion']
                        loss_comparative_sum += loss_storage['comparative']
                        try:
                            for edge_type, v in rel_pred_dict.items():
                                test_metric_dict[edge_type].calculate_recall(rel_pred_dict[edgeType],
                                                                        edge_storage[edgeType]['gt_label'])
                        except:
                            print('test Error')
                            print(edge_storage)
                            continue
                for edge_type, metric in test_metric_dict.items():
                    metric.final_update()
                log_string('epoch : {} Test mean loss: {}'.format(epoch, (test_loss_sum / len(test_loader))))
                log_string('test mean support loss: {}'.format((loss_support_sum / len(test_loader))))
                log_string('test mean proximity loss: {}'.format( (loss_proximity_sum / len(test_loader))))
                log_string('test mean inclusion loss: {}'.format( (loss_inclusion_sum / len(test_loader))))
                log_string(
                    'Training mean comparative loss: {}'.format(epoch, (loss_comparative_sum / len(test_loader))))

                log_string('Test support ' + test_metric_dict['support'].print_string())
                log_string('Test inclusion ' + test_metric_dict['inclusion'].print_string())
                log_string('Test proximity ' + test_metric_dict['proximity'].print_string())
                log_string('Test comparative ' + test_metric_dict['comparative'].print_string())

                curr_loss = train_loss_sum / len(train_loader)
                if best_loss >= curr_loss:
                    best_loss = curr_loss
                    for name, model in self.model._modules.items():
                        # saving_path = os.path.join(self.config.PATH,name)

                        path = os.path.join(self.config.PATH, 'hojun',name+'_best.pth')
                        state = {
                            'epoch': epoch,
                            'current_loss': loss_info,
                            'model_state_dict': model.state_dict()
                        }
                        torch.save(state, path)
                        print('{} saveing model...'.format(name))
                        # if isinstance(model, nn.DataParallel):
                        #     torch.save({
                        #         'model': model.module.state_dict()
                        #     }, path)
                        # else:
                        #     torch.save({
                        #         'model': model.state_dict()
                        #     }, path)



            global_epoch += 1


    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]

    def log(self, logs, iteration):
        # Tensorboard
        if self.writter is not None:
            for i in logs:
                if not i[0].startswith('Misc'):
                    self.writter.add_scalar(i[0], i[1], iteration)

    def save(self):
        self.model.save()

    def validation(self, debug_mode=False):
        val_loader = DataLoader(self.dataset_valid,batch_size=1)

        from utils import util_eva
        eva_tool = util_eva.EvalSceneGraph(self.dataset_valid.classNames, self.dataset_valid.relationNames,
                                           multi_rel_outputs=0.5, k=0,
                                           multi_rel_prediction=self.model.mconfig.multi_rel_outputs)

        total = len(self.dataset_valid)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])

        print('===   start evaluation   ===')
        self.model.eval()
        for i, items in enumerate(val_loader, 0):
            scan_id = items[0][0]
            instance2mask = items[1]
            if items[2].ndim < 4: continue

            ''' get data '''
            tick = time.time()
            if self.use_edge_descriptor:
                obj_points, descriptor, gt_obj_cls, gt_rel_cls, edge_indices = self.data_processing(items[2:])
                if obj_points.size()[0] >= 100 or obj_points.size()[0] <= 3:
                    print('pass')
                    continue
            else:
                obj_points, rel_points, gt_obj_cls, gt_rel_cls, edge_indices = self.data_processing(items[2:])
                if obj_points.size()[0] >= 100 or obj_points.size()[0] <= 3:
                    print('pass')
                    continue
            tock = time.time()

            if edge_indices.ndim == 1:
                # print('no edges found. skip this.')
                continue
            if obj_points.shape[0] < 2:
                # print('need at least two nodes. skip this one. (got ',obj_points.shape,')')
                continue
            if edge_indices.shape[0] == 0:
                # print('no edges! skip')
                continue

            tick = time.time()
            mode = 'test'
            with torch.no_grad():
                if self.use_edge_descriptor:
                    result = self.model(obj_points, descriptor,
                                        edge_indices.t().contiguous(),
                                        return_meta_data=True
                                        )
                    pred_obj_cls, pred_rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs = \
                        result['obj_pred'], result['rel_pred'], result['obj_feature'], result['rel_feature'], result[
                            'gcn_obj_feature'], result['gcn_rel_feature'], result['probs']
                else:
                    result = self.model(obj_points, rel_points, edge_indices.t().contiguous(), return_meta_data=True)
                    pred_obj_cls, pred_rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs = \
                        result['obj_pred'], result['rel_pred'], result['obj_feature'], result['rel_feature'], result[
                            'gcn_obj_feature'], result['gcn_rel_feature'], result['probs']

            ''' calculate metrics '''
            logs = self.model.calculate_metrics([pred_obj_cls, pred_rel_cls], [gt_obj_cls, gt_rel_cls])

            ignore_rel = False
            if 'scene' in scan_id:
                if 'ignore_scannet_rel' in self.config.dataset:
                    ignore_rel = self.config.dataset.ignore_scannet_rel
                else:
                    ignore_rel = True
            if ignore_rel:
                pred_rel_cls = gt_rel_cls = None
            eva_tool.add(scan_id, pred_obj_cls, gt_obj_cls, pred_rel_cls, gt_rel_cls, instance2mask, edge_indices)

            idx2seg = dict()
            for key, item in instance2mask.items():
                idx2seg[item.item() - 1] = key

            logs = [

                   ] + logs
            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])

            if debug_mode:
                if i > 0:
                    break
            # break
        img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_o_cls.c_mat,
                                                                           eva_tool.eva_o_cls.class_names,
                                                                           title='Object Confusion matrix',
                                                                           plot_text=False,
                                                                           plot=False)
        self.writter.add_figure('vali_obj_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)
        img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_r_cls.c_mat,
                                                                           eva_tool.eva_r_cls.class_names,
                                                                           title='Predicate Confusion Matrix',
                                                                           plot_text=False,
                                                                           plot=False)
        self.writter.add_figure('vali_rel_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)

        cls_, rel_ = eva_tool.get_mean_metrics()
        logs = [("IoU/val_obj_cls", cls_[0]),
                ("Precision/val_obj_cls", cls_[1]),
                ("Recall/val_obj_cls", cls_[2]),
                ("IoU/val_rel_cls", rel_[0]),
                ("Precision/val_rel_cls", rel_[1]),
                ("Recall/val_rel_cls", rel_[2])]
        self.log(logs, self.model.iteration)
        return cls_, rel_

    def eval(self, debug_mode=False):
        if self.dataset_eval is None:
            print('no evaludation dataset was built!')
            return
        val_loader = DataLoader(self.dataset_eval,batch_size=1)
        # val_loader = CustomDataLoader(
        #     config=self.config,
        #     dataset=self.dataset_eval,
        #     batch_size=1,
        #     num_workers=0,
        #     drop_last=False,
        #     shuffle=False
        # )
        from utils import util_eva
        eva_tool = util_eva.EvalSceneGraph(self.dataset_eval.classNames, self.dataset_eval.relationNames,
                                           multi_rel_outputs=0.5, k=100,
                                           multi_rel_prediction=self.model.mconfig.multi_rel_outputs)

        total = len(self.dataset_eval)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])

        print('===   start evaluation   ===')
        list_feature_maps = dict()
        list_feature_maps['node_feature'] = list()
        list_feature_maps['edge_feature'] = list()
        list_feature_maps['gcn_node_feature'] = list()
        list_feature_maps['gcn_edge_feature'] = list()
        list_feature_maps['node_names'] = list()
        list_feature_maps['edge_names'] = list()

        self.model.eval()

        for i, items in enumerate(val_loader, 0):
            scan_id = items[0][0]
            print('scan_id = ', scan_id)
            instance2mask = items[1]
            if items[2].ndim < 4: continue

            ''' get data '''
            if self.use_edge_descriptor:
                obj_points, descriptor, gt_obj_cls, gt_rel_cls, edge_indices = self.data_processing(items[2:])
                if obj_points.size()[0] >= 100 or obj_points.size()[0] <= 3:
                    print('pass')
                    continue
            else:
                obj_points, rel_points, gt_obj_cls, gt_rel_cls, edge_indices = self.data_processing(items[2:])
                if obj_points.size()[0] >= 100 or obj_points.size()[0] <= 3:
                    print('pass')
                    continue

            if edge_indices.ndim == 1:
                # print('no edges found. skip this.')
                continue
            if obj_points.shape[0] < 2:
                # print('need at least two nodes. skip this one. (got ',obj_points.shape,')')
                continue
            if edge_indices.shape[0] == 0:
                # print('no edges! skip')
                continue

            with torch.no_grad():
                if self.use_edge_descriptor:
                    print('real_gt_rel_cls : ', gt_rel_cls)
                    result = self.model(obj_points, descriptor, edge_indices.t().contiguous(), return_meta_data=True)
                    pred_obj_cls, pred_rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs = \
                        result['obj_pred'], result['rel_pred'], result['obj_feature'], result['rel_feature'], result[
                            'gcn_obj_feature'], result['gcn_rel_feature'], result['probs']

                    # pred_obj_cls, pred_rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs = \
                    #         self.model(obj_points, edge_indices.t().contiguous(), descriptor, return_meta_data=True)
                else:
                    result = self.model(obj_points, rel_points, edge_indices.t().contiguous(), return_meta_data=True)

                    pred_obj_cls, pred_rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs = \
                        result['obj_pred'], result['rel_pred'], result['obj_feature'], result['rel_feature'], result[
                            'gcn_obj_feature'], result['gcn_rel_feature'], result['probs']

            ''' calculate metrics '''
            logs = self.model.calculate_metrics([pred_obj_cls, pred_rel_cls], [gt_obj_cls, gt_rel_cls])

            ignore_rel = False
            if 'scene' in scan_id:
                if 'ignore_scannet_rel' in self.config.dataset:
                    ignore_rel = self.config.dataset.ignore_scannet_rel
                else:
                    ignore_rel = True
            if ignore_rel:
                pred_rel_cls = gt_rel_cls = None

            eva_tool.add(scan_id, pred_obj_cls, gt_obj_cls, pred_rel_cls, gt_rel_cls, instance2mask, edge_indices)

            idx2seg = dict()
            for key, item in instance2mask.items():
                idx2seg[item.item() - 1] = key

            [list_feature_maps['node_names'].append(self.dataset_eval.classNames[aa]) for aa in gt_obj_cls.tolist()]
            list_feature_maps['node_feature'].append(obj_feature.detach().cpu())
            list_feature_maps['edge_feature'].append(rel_feature.detach().cpu())
            if gcn_obj_feature is not None:
                list_feature_maps['gcn_node_feature'].append(gcn_obj_feature.detach().cpu())

            if not ignore_rel:
                if gcn_rel_feature is not None:
                    list_feature_maps['gcn_edge_feature'].append(gcn_rel_feature.detach().cpu())
                if self.model.mconfig.multi_rel_outputs:
                    for a in range(gt_rel_cls.shape[0]):
                        name = ''
                        for aa in range(gt_rel_cls.shape[1]):
                            if gt_rel_cls[a][aa] > 0:
                                name += self.dataset_eval.relationNames[aa] + '_'
                        if name == '':
                            name = 'none'
                        list_feature_maps['edge_names'].append(name)
                else:
                    for a in range(gt_rel_cls.shape[0]):
                        list_feature_maps['edge_names'].append(self.dataset_eval.relationNames[gt_rel_cls[a]])

            logs = [
                   ] + logs
            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])

            if debug_mode:
                if i > 0:
                    break

        print(eva_tool.gen_text())
        pth_out = os.path.join(self.results_path, str(self.model.iteration))
        op_utils.create_dir(pth_out)

        result_metrics = eva_tool.write(pth_out, self.model_name)
        # result_metrics = {'hparam/'+key: value for key,value in result_metrics.items()}
        tmp_dict = dict()
        for key, item in self.model.mconfig.items():
            if isinstance(item, int) or isinstance(item, float) or isinstance(item, str) or isinstance(item, bool) or \
                    isinstance(item, torch.Tensor):
                tmp_dict[key] = item
        self.writter.add_hparams(tmp_dict, metric_dict=result_metrics)

        img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_o_cls.c_mat,
                                                                           eva_tool.eva_o_cls.class_names,
                                                                           title='Object Confusion matrix',
                                                                           plot_text=False,
                                                                           plot=False)
        self.writter.add_figure('eval_obj_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)
        img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_r_cls.c_mat,
                                                                           eva_tool.eva_r_cls.class_names,
                                                                           title='Predicate Confusion Matrix',
                                                                           plot_text=False,
                                                                           plot=False)
        self.writter.add_figure('eval_rel_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)

        for name, list_tensor in list_feature_maps.items():
            # if name == 'label_names':continue
            if not isinstance(list_tensor, list): continue
            if len(list_tensor) == 0: continue
            if not isinstance(list_tensor[0], torch.Tensor): continue
            if len(list_tensor) < 1: continue
            tmp = torch.cat(list_tensor, dim=0)
            if name.find('node') >= 0:
                names = list_feature_maps['node_names']
            elif name.find('edge') >= 0:
                names = list_feature_maps['edge_names']
            else:
                continue
            print(name)
            self.writter.add_embedding(tmp, metadata=names, tag=self.model_name + '_' + name,
                                       global_step=self.model.iteration)

    def trace(self):
        op_utils.create_dir(self.trace_path)
        args = self.model.trace(self.trace_path)
        with open(os.path.join(self.trace_path, 'classes.txt'), 'w') as f:
            for c in self.dataset_valid.classNames:
                f.write('{}\n'.format(c))

        ''' save relation file'''
        with open(os.path.join(self.trace_path, 'relationships.txt'), 'w') as f:
            for c in self.dataset_valid.relationNames:
                f.write('{}\n'.format(c))

        import json
        with open(os.path.join(self.trace_path, 'args.json'), 'w') as f:
            args['label_type'] = self.dataset_valid.label_type
            json.dump(args, f, indent=2)
        pass


if __name__ == '__main__':
    TEST_CUDA = True
    TEST_EVAL = False
    TEST_TRACE = False

    config = Config('../config_example.json')
    config.dataset.root = "./home/baebro/hojun_ws/3DSSG/data/NYU40/gen_data"
    config.MODEL.GCN_TYPE = ''
    config.MODEL.multi_rel_outputs = False
    config['NAME'] = 'test'

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    if TEST_CUDA and torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda:0")
    else:
        config.DEVICE = torch.device("cpu")

    config.MODE = 'train' if not TEST_EVAL else 'eval'

    pg = SGFN(config)
    if TEST_TRACE:
        pg.trace()
    elif TEST_EVAL:
        pg.train()
    else:
        pg.eval(debug_mode=True)
