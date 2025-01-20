if __name__ == '__main__' and __package__ is None:
    from os import sys, path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os, sys, time, math, torch
import numpy as np
from torch_geometric.nn.conv import MessagePassing
import open3d as o3d
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


def laplacian_positional_encoding(edge_index,node_num,pos_dim=8):
    device = edge_index.device
    edge_index, edge_weight = get_laplacian(edge_index, None,'sym',num_nodes=node_num)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, node_num)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    col = EigVec.shape[1]
    if pos_dim > col:
        lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_dim + 1]).float()
    else:
        lap_pos_enc = torch.from_numpy(EigVec[:, :pos_dim]).float()
    return lap_pos_enc.to(device)



def read_txt_to_list(file):
    output = []
    with open(file, 'r') as f:
        for line in f:
            entry = line.rstrip().lower()
            output.append(entry)
    return output


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def gen_linguistic_feature(object_idx, class_label, embedding_model):
    linguisticVec = torch.zeros([object_idx.size()[0], 50])
    for idx, obj_id in enumerate(object_idx):
        pred_obj_name = class_label[obj_id.item()]
        word2Vector = embedding_model[pred_obj_name]
        linguisticVec[idx] = torch.from_numpy(
            word2Vector.astype(np.float32))  # word2vec
    return linguisticVec


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def gen_descriptor(pts: torch.tensor,o3d_points:np.ndarray):
    # pts : torch.tensor -> ( num_points, xyz )
    '''
    centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths
    [3, 3, 3, 1, 1]
    '''
    assert pts.ndim == 2
    assert pts.shape[-1] == 3
    # centroid [n, 3]
    centroid_pts = pts.mean(0)  # center point
    # # std [n, 3]
    # print(pts)
    std_pts = pts.std(0)  # point distrbution
    # dimensions [n, 3]
    segment_dims = pts.max(dim=0)[0] - pts.min(dim=0)[0]  # size
    # volume [n, 1]
    segment_volume = (segment_dims[0] * segment_dims[1] * segment_dims[2]).unsqueeze(0)  # scale
    # length [n, 1]
    segment_lengths = segment_dims.max().unsqueeze(0)  # size

    # # create oriented bbox
    try:
        o3d_cloud = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(v3d(o3d_points))
        obb_coord = np.asarray(obb.get_box_points()).reshape(-1)
        obb_coord = torch.from_numpy(obb_coord.astype(np.float32))
    except:
        print('obb_coord random generator')
        obb_coord = torch.randn((24))

    return torch.cat([centroid_pts, std_pts, segment_dims, segment_volume, segment_lengths,obb_coord], dim=0)

def gen_obj_pair_indices_within_bbox(edge_indices, total_pcd,instances, mask2instance,nsample=1024):
    pair_instance2point = {}
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(total_pcd[:,:3])
    o3d_cloud.colors = o3d.utility.Vector3dVector(total_pcd[:,3:])

    for edge in edge_indices:
        sub_idx = edge[0].item()
        obj_idx = edge[1].item()
        sub_instance_id = mask2instance[sub_idx+1]
        obj_instance_id = mask2instance[obj_idx+1]
        ind = np.where(((instances == sub_instance_id) | (instances == obj_instance_id)))[0]
        # print('subject = ',sub_instance_id,' object = ',obj_instance_id)

        pair_obj_pcd = o3d_cloud.select_by_index(ind)
        pair_obb = o3d.geometry.OrientedBoundingBox.create_from_points(pair_obj_pcd.points)
        pair_obb.color = (255,0,0)
        pair_obj_indices = pair_obb.get_point_indices_within_bounding_box(o3d_cloud.points)
        pair_obj_indices = np.asarray(pair_obj_indices)

        # sampling
        choice = np.random.choice(len(pair_obj_indices), nsample, replace=True)
        pair_obj_indices = pair_obj_indices[choice]
        pair_instance2point['{} {}'.format(sub_idx, obj_idx)] = pair_obj_indices


        # for visual
        pair_obj_within_pcd = o3d_cloud.select_by_index(np.asarray(pair_obj_indices))
        # o3d.visualization.draw_geometries([pair_obj_within_pcd,pair_obb],width=700,height=700)

    return pair_instance2point
        #
    ''' 
        # # test
        #         sub_ind = np.where(instances==sub_instance_id)[0]
        #         sub_pcd = o3d_cloud.select_by_index(sub_ind)
        #         sub_obb = o3d.geometry.OrientedBoundingBox.create_from_points(sub_pcd.points)
        # 
        #         obj_ind = np.where(instances == obj_instance_id)[0]
        #         obj_pcd = o3d_cloud.select_by_index(obj_ind)
        #         obj_obb = o3d.geometry.OrientedBoundingBox.create_from_points(obj_pcd.points)
        # 
        #         pair_obj_pcd = o3d_cloud.select_by_index(ind)
        #         pair_obb = o3d.geometry.OrientedBoundingBox.create_from_points(pair_obj_pcd.points)
        #         pair_obj_indices = pair_obb.get_point_indices_within_bounding_box(o3d_cloud.points)
        # s_center = sub_obb.center
        # s_extent = sub_obb.extent
        # 
        # o_center = obj_obb.center
        # o_extent = obj_obb.extent
        # 
        # 
        # pair_obb.color = (255, 0, 0)
        # 
        # refine_obb = o3d.geometry.OrientedBoundingBox()
        # refine_obb.center = (o_center + s_center) / 2
        # refine_obb.color = (255,0,0)
        # refine_obb.extent = (o_extent + s_extent) / 2
        # refine_obb.R = pair_obb.R


        # re_pair_obj_indices = refine_obb.get_point_indices_within_bounding_box(o3d_cloud.points)
        # re_pair_obj_within_pcd = o3d_cloud.select_by_index(np.asarray(re_pair_obj_indices))
    '''




class Gen_edge_descriptor(MessagePassing):  # TODO: move to model
    """ A sequence of scene graph convolution layers  """

    def __init__(self, flow="source_to_target", feature_selection=None):
        super().__init__(flow=flow)
        self.feature_selection = feature_selection

    def forward(self, descriptor, edges_indices,scene_point_feature, masks_dict):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(self.__user_args__, edges_indices, size, {"x": descriptor})
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        relative_feature = self.message(**msg_kwargs)

        temp_device = descriptor.device
        pair_point_feature = torch.zeros([len(masks_dict), 48]).to(temp_device)
        # masks_dict = masks.reshape(-1,1)

        for idx, edge in enumerate(edges_indices.T):
            key = '{} {}'.format(edge[0].item(),edge[1].item())
            masks = masks_dict[key].view(-1).to(temp_device)
            subobj_pcd_feat = scene_point_feature[masks]
            subobj_pcd_feat = subobj_pcd_feat.mean(0)
            pair_point_feature[idx] = subobj_pcd_feat


        return relative_feature , pair_point_feature

    def message(self, x_i, x_j):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5: std, 6-8:dims, 9:volume, 10:length
        # to
        # 0-2: offset centroid, 3-5: offset std, 6-8: dim log ratio, 9: volume log ratio, 10: length log ratio

        # temp init

        dim_shape = 11  # bbox
        dim_geometric = 256  # point feature size
        dim_obj_linguistic = 50  # linguistic Obj feature size
        dim_scene_linguistic = 50  # linguistic Scene feature size

        shape_feature = torch.zeros([x_i.size(0), dim_shape]).to(x_i.device)
        if self.feature_selection:
            if self.feature_selection.USE_Relative_Position:
                shape_feature[:, 0:3] = x_i[:, 0:3] - x_j[:, 0:3]
                # std  offset
                shape_feature[:, 3:6] = x_i[:, 3:6] - x_j[:, 3:6]
            if self.feature_selection.USE_Relative_Scale:
                # dim log ratio
                shape_feature[:, 6:9] = torch.log(x_i[:, 6:9] / x_j[:, 6:9])
                # volume log ratio
                shape_feature[:, 9] = torch.log(x_i[:, 9] / x_j[:, 9])
                # length log ratio
                shape_feature[:, 10] = torch.log(x_i[:, 10] / x_j[:, 10])
            if self.feature_selection.USE_Object_Label:
                sub_linguistic_Objfeature = x_i[:, dim_shape:(dim_shape + dim_obj_linguistic)]
                obj_linguistic_Objfeature = x_j[:, dim_shape:(dim_shape + dim_obj_linguistic)]
            if self.feature_selection.USE_Scene_Label:
                strIdx = dim_shape + dim_obj_linguistic  # + dim_scene_linguistic
                sub_linguistic_Scenefeature = x_i[:, strIdx:strIdx + dim_scene_linguistic]
                obj_linguistic_Scenefeature = x_j[:, strIdx:strIdx + dim_scene_linguistic]

        flag_semantic = True

        if self.feature_selection.USE_Object_Label and self.feature_selection.USE_Scene_Label:
            sub_semantic_feature = torch.cat([sub_linguistic_Scenefeature, sub_linguistic_Objfeature], dim=1)
            obj_semantic_feature = torch.cat([obj_linguistic_Scenefeature, obj_linguistic_Objfeature], dim=1)
        elif self.feature_selection.USE_Object_Label:
            sub_semantic_feature = sub_linguistic_Objfeature.detach()
            obj_semantic_feature = obj_linguistic_Objfeature.detach()
        elif self.feature_selection.USE_Scene_Label:
            sub_semantic_feature = sub_linguistic_Scenefeature.detach()
            obj_semantic_feature = obj_linguistic_Scenefeature.detach()
        else:
            sub_semantic_feature = None
            obj_semantic_feature = None
            flag_semantic = False
        dim_linguistic = dim_obj_linguistic + dim_scene_linguistic

        sub_geometric_feature = x_i[:, (dim_shape + dim_linguistic):]
        obj_geometric_feature = x_j[:, (dim_shape + dim_linguistic):]
        geometric_feature = torch.cat([sub_geometric_feature, obj_geometric_feature], dim=1)

        if flag_semantic:
            if self.feature_selection.USE_Relative_Scale and self.feature_selection.USE_Relative_Position:
                semantic_feature = torch.concat([sub_semantic_feature, shape_feature, obj_semantic_feature], dim=1)
            elif self.feature_selection.USE_Relative_Scale:
                semantic_feature = torch.concat([sub_semantic_feature, shape_feature[:, :6], obj_semantic_feature],
                                                dim=1)
            elif self.feature_selection.USE_Relative_Position:
                semantic_feature = torch.concat([sub_semantic_feature, shape_feature[:, 6:], obj_semantic_feature],
                                                dim=1)
            else:
                semantic_feature = torch.concat([sub_semantic_feature, obj_semantic_feature],
                                                dim=1)

            edge_feature = torch.cat([semantic_feature, geometric_feature], dim=1)
        else:
            if self.feature_selection.USE_Relative_Scale and self.feature_selection.USE_Relative_Position:
                edge_feature = torch.cat([shape_feature, geometric_feature], dim=1)
            elif self.feature_selection.USE_Relative_Scale:
                edge_feature = torch.concat([shape_feature[:, :6], geometric_feature], dim=1)
            elif self.feature_selection.USE_Relative_Position:
                edge_feature = torch.concat([shape_feature[:, 6:], geometric_feature], dim=1)
            else:
                edge_feature = geometric_feature.detach()

        # proposal method ( Feature Selection )
        # 50(subject linguistic feature),
        # 11(relative comparision information),
        # 50(object linguistic feature),
        # 512(subject geometric feature + object geometric feature)
        return edge_feature  # geometric_feature#edge_feature



def pytorch_count_params(model, trainable=True):
    "count number trainable parameters in a pytorch model"
    s = 0
    for p in model.parameters():
        if trainable:
            if not p.requires_grad: continue
        try:
            s += p.numel()
        except:
            pass
    return s


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None, silent=False):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if not silent:
                if self._dynamic_display:
                    sys.stdout.write('\b' * prev_total_width)
                    sys.stdout.write('\r')
                else:
                    sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            if not silent:
                sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            if not silent:
                sys.stdout.write(info)
                sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'
                if not silent:
                    sys.stdout.write(info)
                    sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None, silent=False):
        self.update(self._seen_so_far + n, values, silent=silent)


def check(x, y):
    x = x if isinstance(x, list) or isinstance(x, tuple) else [x]
    y = y if isinstance(y, list) or isinstance(y, tuple) else [y]
    [np.testing.assert_allclose(x[i].flatten(), y[i].flatten(), rtol=1e-03, atol=1e-05) for i in range(len(x))]


def export(model: torch.nn.Module, inputs: list, pth: str, input_names: list, output_names: list, dynamic_axes: dict):
    import onnxruntime as ort
    inputs = inputs if isinstance(inputs, list) or isinstance(inputs, tuple) else [inputs]
    torch.onnx.export(model=model, args=tuple(inputs), f=pth,
                      verbose=False, export_params=True,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=12)
    with torch.no_grad():
        model.eval()
        sess = ort.InferenceSession(pth)
        x = model(*inputs)
        ins = {input_names[i]: inputs[i].numpy() for i in range(len(inputs))}
        y = sess.run(None, ins)
        check(x, y)

        inputs = [torch.cat([input, input], dim=0) for input in inputs]
        x = model(*inputs)
        ins = {input_names[i]: inputs[i].numpy() for i in range(len(inputs))}
        y = sess.run(None, ins)
        check(x, y)


def get_tensorboard_logs(pth_log):
    for (dirpath, dirnames, filenames) in os.walk(pth_log):
        break
    l = list()
    for filename in filenames:
        if filename.find('events') >= 0: l.append(filename)
    return l


def create_dir(dir):
    from pathlib import Path
    Path(dir).mkdir(parents=True, exist_ok=True)