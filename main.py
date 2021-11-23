from tqdm import tqdm
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from dataset.dataset import SyntheticDataset


from model.model import PointNet2SemSegMSG, PointNet
from pytorch_pipeline.ptpl import PyTorchPipeline

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_num', default=-1, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    args = parser.parse_args()

    hparams = {
        'batch_size' : 32,
        'num_workers': 8,
        'gpu_num' : args.gpu_num,
        'num_epochs' : args.num_epochs,
        'K': {
            'train': 32,
            'val': 1,
            'test': 1,
        },
        'hyp_count': {
            'train': 64,
            'val': 512,
        },
        'threshold': {
            'train': 1/512,
            'val': 1/16, 
            },
        'lr': 1e-3,
        'path2save': './checkpoint.pt',
    }

    num_points = 2048
    train_size = 800
    test_size = val_size = 100
    # define training dataloader
    train_dataloader = DataLoader(
        SyntheticDataset('train', train_size, num_points),
        batch_size = hparams['batch_size'],
        num_workers = hparams['num_workers'],
        shuffle = True,
    )
    # define validation dataloader
    val_dataloader = DataLoader(
        SyntheticDataset('val', val_size, num_points),
        batch_size = hparams['batch_size'],
        num_workers = hparams['num_workers'],
        shuffle = False,
    )
    # define test dataloader
    test_dataloader = DataLoader(
        SyntheticDataset('test', test_size, num_points),
        batch_size = hparams['batch_size'],
        num_workers = hparams['num_workers'],
        shuffle = False,
    )

    device = torch.device('cuda:' + str(args.gpu_num) if args.gpu_num > -1 else 'cpu')
    model = PointNet(1)
    model = model.to(device)




    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams['lr'],
        # betas=(0.9, 0.999),
        # eps=1e-08,
        # weight_decay = 1e-4,
    )

    ptpl = PyTorchPipeline(
        project_name = "sphere_fitting",
        configs = {
        'device': device,
        'criterion': None,
        'optimizer': optimizer,
        'train_dataloader': train_dataloader, 
        'val_dataloader': val_dataloader,
        'test_dataloader': test_dataloader,
        'print_logs': True,
        'model': model,
        'wb': False,
        },
        hparams = hparams,
    )

    # ptpl.train(num_epochs= hparams['num_epochs'], path2save =  hparams['path2save'])
    
    batch = next(iter(train_dataloader))
    # l, _ = ptpl.run_pn_ngransac(batch, "train")
    # xyz_noise = torch.from_numpy(np.load('xyz_noise.npy'))
    # xyz_gt_label = torch.from_numpy(np.load('xyz_gt_label.npy'))
    # xyz_gt = torch.from_numpy(np.load('xyz_gt.npy'))

    # batch = [ xyz_noise, xyz_gt_label, xyz_gt ]

    for i in range(1000):
        l, _ = ptpl.run_pn_ngransac(batch, "train")
        print(l.mean().item()) 
        
    # l = ptpl.run_ransac("train")

