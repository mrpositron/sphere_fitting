import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np
from tqdm import tqdm

from utils import *

class PyTorchPipeline:
    def __init__(self, project_name, configs, hparams):
        print(f"PyTorch pipeline for {project_name} is set up")
        ## assertion checks
        self.configs = configs
        self.model = self.configs['model'] if 'model' in self.configs else None
        self.device = self.configs['device'] if 'device' in self.configs else torch.device('cpu')


        # set training details
        self.criterion = self.configs['criterion'] if 'criterion' in self.configs else None
        self.optimizer = self.configs['optimizer'] if 'optimizer' in self.configs else None

        # set dataloaders
        self.train_dataloader = self.configs['train_dataloader'] if 'train_dataloader' in self.configs else None
        self.val_dataloader = self.configs['val_dataloader'] if 'val_dataloader' in self.configs else None
        self.test_dataloader = self.configs['test_dataloader'] if 'test_dataloader' in self.configs else None

        # et cetera
        self.print_logs = self.configs["print_logs"]
        self.K = hparams['K']
        self.hyp_count = hparams['hyp_count']
        self.threshold = hparams['threshold']

    def predict(self, batch, state):
        xyz_noise, xyz_gt_labels, xyz_gt = batch
        xyz_noise = xyz_noise.to(self.device)
        xyz_noise = xyz_noise.transpose(1, 2)
        output_data = self.model(xyz_noise)
        return output_data


    def run_pn_ngransac(self, batch, state):
        xyz_noise, xyz_gt_labels, xyz_gt = batch
        xyz_noise = xyz_noise.to(self.device)
        xyz_gt = xyz_gt.to(self.device)
        xyz_gt_labels = xyz_gt_labels.to(self.device)

        if state == 'train':
            self.optimizer.zero_grad()
        output_data = self.predict(batch, state)
        prob = F.softmax(output_data, dim = 1).squeeze(2)
        log_prob = torch.log(prob)

        
        grads = []
        distLosses = []

        for k in range(self.K[state]):
            gradients = torch.zeros(xyz_noise.shape[0], xyz_noise.shape[1]).to(self.device) if state == "train" else None
            _, parameters, indices = fit_sphere(xyz_noise, prob, self.threshold[state], self.hyp_count[state], gradients)

            distLoss = calculate_error(xyz_gt, parameters)
            distLosses.append(distLoss)

            if state == "train":
                grads.append(gradients)
        
        distLosses = torch.stack(distLosses, dim = 1)	

        
        loss_temp = distLosses.mean(dim = 1)

        losses = distLosses
        if state == "train":
            losses = (losses - losses.mean(dim = 1).unsqueeze(1)).unsqueeze(2)

            grads = torch.stack(grads, dim = 2).transpose(1, 2)
            grads = (grads * losses).mean(dim = 1)

            torch.autograd.backward((log_prob), (grads))
            self.optimizer.step() 

        return loss_temp, output_data

    def run_ransac(self, state):
        cum_loss = .0
        total_cnt = 0

        if state == "train":
            dataloader = self.train_dataloader
        elif state == "test":
            dataloader = self.test_dataloader
        else:
            dataloader = self.val_dataloader

        
        for _, batch in enumerate(tqdm(dataloader)):
            xyz_noise, xyz_gt_labels, xyz_gt = batch
            probabilities = torch.ones(xyz_noise.shape[0], xyz_noise.shape[1]).to(self.device)

            total_cnt += xyz_noise.shape[0]

            xyz_noise = xyz_noise.to(self.device)
            xyz_gt = xyz_gt.to(self.device)        
            pred_num_inliers, parameters, indices = fit_sphere(xyz_noise, probabilities, self.threshold[state], self.hyp_count[state])

            loss = calculate_error(xyz_gt, parameters)

            cum_loss += loss.sum().item()
        
        print(f"Distance loss: {cum_loss/total_cnt}")
        return cum_loss

    def train(self, num_epochs = None, path2save = None):
        
        min_loss = 1000

        for epoch in range(num_epochs):
            self.model.train()
            total_cnt = 0
            cum_dist_loss = .0
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                batch_size = batch[0].shape[0]
                total_cnt += batch_size

                distance_loss, output_data = self.run_pn_ngransac(batch, state = "train")
                cum_dist_loss += distance_loss.sum().item()

            if self.print_logs:
                print()
                print("=" * 100)
                print(f"Epoch: {epoch + 1}/{num_epochs}")
                print("=" * 100)
                print()
                print(f"TRAIN || Distance Loss: {round(cum_dist_loss/total_cnt, 5)}")
                
            # val_dist_loss = self.check("val")
            # if val_dist_loss < min_loss:
            #     min_loss = val_dist_loss
            #     print("Best model is saved!")
            #     self.save(path2save)
        

        
    @torch.no_grad()
    def check(self, state):
        self.model.eval()

        total_cnt = 0
        cum_dist_loss = .0

        dataloader = self.test_dataloader if state == 'test' else self.val_dataloader
        for _, batch in enumerate(dataloader):
            batch_size = batch[0].shape[0]
            total_cnt += batch_size

            distance_loss, output_data = self.run_pn_ngransac(batch, state)
            cum_dist_loss += distance_loss.sum().item()

        if self.print_logs:
            print(f"{state.upper()} || Distance Loss: {round(cum_dist_loss/total_cnt, 5)}")

        return cum_dist_loss/total_cnt

    def save(self, path2save):
        print(f"The model is saved under the path {path2save}")
        torch.save(self.model.state_dict(), path2save)
    
    def load(self, path2load):
        self.model.load_state_dict(torch.load(path2load))
    


if __name__ == "__main__":
    pass