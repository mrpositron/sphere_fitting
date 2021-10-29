import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

def create_sphere(num_inliers, num_outliers, batch_size = 32):
    # sphere is define by a coordinates of a sphere senter x0, y0, z0
    # and by the radius r

    xyz0 = torch.rand(batch_size, 3)
    r = 0.5 + torch.rand(batch_size, 1)

    theta = torch.from_numpy(np.random.choice(np.linspace(0, np.pi, 1000), size = (batch_size, num_inliers,), replace = True))
    phi = torch.from_numpy(np.random.choice(np.linspace(0, 2*np.pi, 1000), size = (batch_size, num_inliers,), replace = True))

    x = xyz0[:, 0].unsqueeze(1) + r * np.sin(theta) * np.cos(phi)
    y = xyz0[:, 1].unsqueeze(1) + r * np.sin(theta) * np.sin(phi)
    z = xyz0[:, 2].unsqueeze(1) + r * np.cos(theta)
    x, y, z = x.unsqueeze(2), y.unsqueeze(2), z.unsqueeze(2)

    xyz_gt = torch.cat((x, y, z), dim = 2)

    noise_scaling = 1/32
    x += torch.rand(x.shape) * noise_scaling
    y += torch.rand(y.shape) * noise_scaling
    z += torch.rand(z.shape) * noise_scaling

    xyz_noise = torch.cat((x, y, z), dim = 2)

    if num_outliers > 0:
        xyz_noise = torch.cat((xyz_noise, torch.rand(batch_size, num_outliers, 3)), dim = 1)

    gt_parameters = {
        'sphere_center': xyz0,
        'sphere_radius': r,
    }
    xyz_gt_labels = torch.cat((torch.ones(batch_size, num_inliers), torch.zeros(batch_size, num_outliers)), dim = 1).long()
    

    output = {
        'xyz_noise': xyz_noise,
        'xyz_gt': xyz_gt,
        'xyz_gt_labels': xyz_gt_labels,
        'gt_parameters': gt_parameters
    }

    return output

class SyntheticDataset(Dataset):

    def __init__(self, state, size, num_points):
        self.xyz_noise_full = []
        self.xyz_gt_full = []
        self.xyz_gt_labels_full = []

        if state == 'train':
            num_inliers = int (0.7 * num_points)
            num_outliers = num_points - num_inliers
            while size > 0:
                batch_size = min(100, size)
                output = create_sphere(num_inliers, num_outliers, batch_size = batch_size)
                self.xyz_noise_full.append(output['xyz_noise'])
                self.xyz_gt_full.append(output['xyz_gt'])
                self.xyz_gt_labels_full.append(output['xyz_gt_labels'])
                size -= batch_size

        self.xyz_noise_full = torch.cat(self.xyz_noise_full, dim = 0)
        self.xyz_gt_full = torch.cat(self.xyz_gt_full, dim = 0)
        self.xyz_gt_labels_full = torch.cat(self.xyz_gt_labels_full, dim = 0)
        
    def __len__(self):
        return self.xyz_noise_full.shape[0]

    def __getitem__(self, idx):
        return self.xyz_noise_full[idx], self.xyz_gt_labels_full[idx], self.xyz_gt_full[idx]


if __name__ == "__main__":
    ds = SyntheticDataset('train', 819, 8192)

    dataloader = DataLoader(ds,
	    batch_size= 32,
	    num_workers= 0,
	    shuffle = True,
	)

    xyz_noise, xyz_gt_labels, xyz_gt = next(iter(dataloader))

    print(xyz_noise.shape, xyz_gt_labels.shape, xyz_gt.shape)