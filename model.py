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

from model_utils import *

class PointNet2SemSegMSG(torch.nn.Module):
    def __init__(self, num_segments = 1):
        super(PointNet2SemSegMSG, self).__init__()
        self.SA_modules = nn.ModuleList()

        c_in = 0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 0, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_segments, kernel_size=1),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz = pointcloud.permute(0, 2, 1).contiguous()
        features = None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        output = self.fc_layer(l_features[0])
        output = output.transpose(1, 2)

        return output



if __name__ == '__main__':
    from dataset import SyntheticDataset
    from torch.utils.data import DataLoader

    model = PointNet(num_class=1)
    
    dataloader = DataLoader(
        SyntheticDataset('train', 128, 1024),
	    batch_size= 32,
	    num_workers= 0,
	    shuffle = True,
	)

    xyz_noise, xyz_gt_labels, xyz_gt = next(iter(dataloader))
    xyz_noise = xyz_noise.transpose(1, 2).float()
    print(xyz_noise.shape)
    output = model(xyz_noise)
    print(output.shape)
