import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torch

class PointNet2feat(nn.Module):
    def __init__(self,point_size,out_size,normal_channel=True):
        super(PointNet2feat, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.out_channel = out_size
        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])

        # self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.sa1 = PointNetSetAbstractionMsg(16, [0.1, 0.2, 0.4], [16, 32, 64], in_channel,[[32, 32, 32], [32, 64, 128], [64, 128, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[32, 64, 128], [128, 128, 256], [128, 128, 128]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [128, 256, 512], True)

        self.fc1 = nn.Linear(288*16, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, out_size)
        self.bn2 = nn.BatchNorm1d(out_size)
        self.drop2 = nn.Dropout(0.5)

        # self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l1_points.size())
        # print('B = ', B)
        x = l1_points.view(B,288*16)
        # print('x = ',x.size())
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return x
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        # return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

if __name__ == '__main__':
    model = PointNet2feat(num_class=10,out_channel=256)
    points = torch.randn(23,6,256)
    # print(model(points))
    obj_feature = model(points)
    # print(x.size())
    # print(l3_points.size())
    print(obj_feature.size())