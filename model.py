import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
        def __init__(self, global_feat=True, channel=3, conv_bias = False):
                super(PointNetEncoder, self).__init__()

                self.conv1 = torch.nn.Conv1d(channel, 64, 1, bias = conv_bias)
                self.conv2 = torch.nn.Conv1d(64, 128, 1, bias = conv_bias)
                self.conv3 = torch.nn.Conv1d(128, 1024, 1, bias = conv_bias)
                self.bn1 = nn.BatchNorm1d(64)
                self.bn2 = nn.BatchNorm1d(128)
                self.bn3 = nn.BatchNorm1d(1024)
                self.global_feat = global_feat
 

        def forward(self, x):
                B, D, N = x.size()
                x = F.relu(self.bn1(self.conv1(x)))


                pointfeat = x
                x = F.relu(self.bn2(self.conv2(x)))
                x = self.bn3(self.conv3(x))
                x = torch.max(x, 2, keepdim=True)[0]
                x = x.view(-1, 1024)
                x = x.view(-1, 1024, 1).repeat(1, 1, N)
                return torch.cat([x, pointfeat], 1)



class PointNet(nn.Module):
        def __init__(self, num_class, conv_bias = False):
                super(PointNet, self).__init__()
                self.k = num_class
                self.feat = PointNetEncoder(global_feat=False, channel=3)
                self.conv1 = torch.nn.Conv1d(1088, 512, 1, bias = conv_bias)
                self.conv2 = torch.nn.Conv1d(512, 256, 1, bias = conv_bias)
                self.conv3 = torch.nn.Conv1d(256, 128, 1, bias = conv_bias)
                self.conv4 = torch.nn.Conv1d(128, self.k, 1)
                self.bn1 = nn.BatchNorm1d(512)
                self.bn2 = nn.BatchNorm1d(256)
                self.bn3 = nn.BatchNorm1d(128)

        def forward(self, x):
                # X = [B, D, N]
                batchsize = x.size()[0]
                n_pts = x.size()[2]
                x = self.feat(x)
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                x = self.conv4(x)
                x = x.transpose(1, 2)
                
                return x
