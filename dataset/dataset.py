"""
dataset.py
- This file contains the Dataset class, which is used to load and preprocess
"""

import torch
from torch.utils.data import Dataset, DataLoader

from .utils import create_shape


class SyntheticDataset(Dataset):
    def __init__(self, state, size, num_points):
        self.xyz_noise_full = []
        self.xyz_gt_full = []
        self.xyz_gt_labels_full = []

        num_inliers = int (0.3 * num_points) if state == 'train' else int (0.2 * num_points)
        num_outliers = num_points - num_inliers
        while size > 0:
            batch_size = min(100, size)
            xyz_noise, xyz_gt_labels, xyz_gt = create_shape(num_inliers, num_outliers, batch_size = batch_size)
            self.xyz_noise_full.append(xyz_noise.float())
            self.xyz_gt_full.append(xyz_gt.float())
            self.xyz_gt_labels_full.append(xyz_gt_labels.int())
            size -= batch_size

        self.xyz_noise_full = torch.cat(self.xyz_noise_full, dim = 0)
        self.xyz_gt_full = torch.cat(self.xyz_gt_full, dim = 0)
        self.xyz_gt_labels_full = torch.cat(self.xyz_gt_labels_full, dim = 0)
        self.data = [self.xyz_noise_full, self.xyz_gt_labels_full, self.xyz_gt_full]
        
    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx], self.data[2][idx]


if __name__ == "__main__":
    ds = SyntheticDataset('train', 819, 8192)

    dataloader = DataLoader(ds,
	    batch_size= 32,
	    num_workers= 0,
	    shuffle = True,
	)

    xyz_noise, xyz_gt_labels, xyz_gt = next(iter(dataloader))