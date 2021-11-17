import torch
import numpy as np

def create_sphere(num_points, batch_size = 32):
    """ Create a sphere with num_points points.
    Args:
        num_points: number of points in the sphere
        batch_size: batch size
    Returns:
        xyz_gt: Tensor of size (batch_size, num_points, 3)
    """

    xyz0 = torch.rand(batch_size, 3)
    r = 0.5 + torch.rand(batch_size, 1)

    theta = torch.from_numpy(np.random.choice(np.linspace(0, np.pi, 1000), size = (batch_size, num_points,), replace = True))
    phi = torch.from_numpy(np.random.choice(np.linspace(0, 2*np.pi, 1000), size = (batch_size, num_points,), replace = True))

    x = xyz0[:, 0].unsqueeze(1) + r * np.sin(theta) * np.cos(phi)
    y = xyz0[:, 1].unsqueeze(1) + r * np.sin(theta) * np.sin(phi)
    z = xyz0[:, 2].unsqueeze(1) + r * np.cos(theta)
    x, y, z = x.unsqueeze(2), y.unsqueeze(2), z.unsqueeze(2)

    xyz_gt = torch.cat((x, y, z), dim = 2)

    return xyz_gt

def create_plane(num_points, batch_size = 32):
    """ Create a plane with num_points points.
    Args:
        num_points: number of points in the plane
        batch_size: batch size
    Returns:
        xyz_gt: Tensor of size (batch_size, num_points, 3)
    """
    x = torch.rand(batch_size, num_points, 1) - 2 * torch.rand(batch_size, num_points, 1)
    y = torch.rand(batch_size, num_points, 1) - 2 * torch.rand(batch_size, num_points, 1)

    a = torch.rand(batch_size, 1, 1)
    b = torch.rand(batch_size, 1, 1)
    c = torch.rand(batch_size, 1, 1)

    z = a * x + b * y + c

    xyz_gt = torch.cat((x, y, z), dim = 2)

    return xyz_gt

def create_shape(num_inliers, num_outliers, batch_size = 32):
    """ Create a shape with num_inliers inliers and num_outliers outliers.
    Args:
        num_inliers: number of inliers
        num_outliers: number of outliers
        batch_size: batch size
    Returns:
        xyz_noise: Tensor of size (batch_size, num_inliers + num_outliers, 3)
    """
    xyz_gt_sphere = create_sphere(num_inliers, batch_size= batch_size)
    xyz_gt_plane = create_plane(num_outliers, batch_size= batch_size)

    xyz_noise = torch.cat((xyz_gt_sphere, xyz_gt_plane), dim = 1)
    xyz_gt_labels = torch.cat((torch.ones(batch_size, num_inliers, 1), torch.zeros(batch_size, num_outliers, 1)), dim = 1)
    
    return xyz_noise, xyz_gt_labels, xyz_gt_sphere

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    batch_size = 32

    xyz_noise, xyz_gt_labels, xyz_gt_sphere = create_shape(num_inliers = 1000, num_outliers = 1000, batch_size = batch_size)
    for i in range(batch_size):
        ax = plt.axes(projection='3d')
        ax.scatter3D(xyz_noise[i, :, 0], xyz_noise[i, :, 1], xyz_noise[i, :, 2], c = xyz_gt_labels[i] )
        plt.show()
        plt.close()


