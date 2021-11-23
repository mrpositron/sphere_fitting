"""
utils.py 
- This file contains functions required for NG-RANSAC
"""
import torch
import numpy as np
from typing import List

def batched_index_select(input: torch.Tensor, dim: int, index: torch.Tensor):
    """ Index-wise selection per batch.
    Args:
        input: B x * x ... x *
        dim: 0 < scalar
        index: B x M
    Returns:
        B x M x * x ... x *
    Source:
        https://discuss.pytorch.org/t/batched-index-select/9115/8

    """
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

def interleave(tensors_arr: List[torch.Tensor], hyp_count: int):
    """ Interleave a list of tensors.
    Args:
        tensors_arr: list of tensors
        hyp_count: number of hypotheses
    Returns:
        tensors_arr: list of interleaved tensors
    """
    for i in range(len(tensors_arr)):
        tensors_arr[i] = tensors_arr[i].repeat_interleave(hyp_count, dim = 0)
    return tensors_arr

def sample_from_probs(input: torch.Tensor, probs: torch.Tensor, min_set: int):
    """ Sample from a tensor with a probability distribution.
    Args:
        input: tensor of shape [B, N, ...]
        probs: tensor of shape [B, N]
        min_set: integer, minimum number of elements to sample
    """

    # input: [B * hyp_count, N, D]
    # probs: [B * hyp_count, N]	
    sampledIndices = probs.multinomial(min_set, replacement = False)
    sampledValues =  batched_index_select(input, 1, sampledIndices)

    # sampledValues: [B * hyp_count, min_set, D]
    # sampledIndices: [B * hyp_count, min_set]
    return [sampledIndices, sampledValues]

def sphere_fitting(point_cloud: torch.Tensor, weights: torch.Tensor):
    eps = 1e-10
    weights = weights.float()
    diag_w = torch.diag_embed(weights)

    point_cloud = point_cloud.float()
    # point_cloud: [B, N, D]
    # weights: [B, N]
    
    b = (point_cloud ** 2).sum(dim = 2)
    b -= ((weights * b).sum(dim = 1)/(weights.sum(dim = 1) + eps)).unsqueeze(1)
    b = torch.bmm(diag_w, b.unsqueeze(2)).squeeze()
    

    A =  2 * (point_cloud - ((point_cloud * weights.unsqueeze(2)).sum(dim = 1) / weights.sum(dim = 1).unsqueeze(1)  ).unsqueeze(1))
    A = torch.bmm(diag_w, A)
    
    AtA = torch.bmm(A.transpose(1, 2), A)
    Atb = torch.bmm(A.transpose(1, 2), b.unsqueeze(2))

    c = torch.linalg.solve(AtA, Atb).squeeze(2).unsqueeze(1)
    r = torch.sqrt( (((point_cloud - c)**2).sum(dim = 2) * weights).sum(dim = 1) / (weights.sum(dim = 1)) )
    
    c = c.squeeze(1)
    return c, r


def sample_from_probabilities(correspondences, probabilities, min_set):
    # correspondences: [B * hyp_count, N, D]
    # probabilities: [B * hyp_count, N]	
    sampledIndices = probabilities.multinomial(min_set, replacement = False)
    sampledValues =  batched_index_select(correspondences, 1, sampledIndices)

    # sampledValues: [B * hyp_count, min_set, D]
    # sampledIndices: [B * hyp_count, min_set]
    return [sampledIndices, sampledValues]

@torch.no_grad()
def fit_sphere(correspondences, probabilities, thresh, hyp_count, gradients = None):
    B, N, D = correspondences.shape
    # minimal set
    min_set = 4
    # correspondences: [B, N, D]
    # probabilities: [B, N]
    device = correspondences.device
    probabilities, correspondences = interleave([probabilities, correspondences], hyp_count)
    # probabilities: [B * hyp_count, N, D]
    # correspondences: [B * hyp_count, N, D]
    sampledIndices, sampledValues = sample_from_probabilities(correspondences, probabilities, min_set)
    sampledIndices = sampledIndices.reshape(B, hyp_count *  min_set)
    if gradients != None:
        sampledIndices = sampledIndices.to(device)
        added = torch.tensor([1.0]).expand(sampledIndices.shape).to(device)
        gradients.scatter_add_ (1, sampledIndices, added)
    
    del probabilities
    # 1. fit a sphere, i.e. caculate hypotheses for each minimal set
    temp = (sampledValues ** 2).sum(dim = 2).unsqueeze(2)
    temp = torch.cat((temp, sampledValues, torch.ones(sampledValues.shape[0], sampledValues.shape[1], 1).to(device) ), dim = 2)
    sub_deter = []



    for i in range(5):
        matrix = torch.cat((temp[:, :, :i], temp[:, :, i+1:]), dim = 2)
        sub_deter.append(torch.linalg.det(matrix))
    del temp
    a = sub_deter[1]/(2 * sub_deter[0] + 1e-6)
    b = - sub_deter[2]/(2 * sub_deter[0] + 1e-6)
    c = sub_deter[3] / (2 * sub_deter[0] + 1e-6)
    r = torch.sqrt(a**2 + b**2 + c**2 - (sub_deter[4]/sub_deter[0]))

    # 2. Calculate inliers
    # 2.1. Calculate the distance from a sphere

    # Distance is calculated as following
    dist = abs(torch.sqrt((a.unsqueeze(1) - correspondences[:, :, 0]) ** 2 + (b.unsqueeze(1) - correspondences[:, :, 1]) ** 2 + (c.unsqueeze(1) - correspondences[:, :, 2])**2) - r.unsqueeze(1))
    # 2.2. Calculate inliers within a threshold

    indices = torch.where(dist < thresh, 1, 0)

    ### free some memory ###
    del dist, a, b, c, r
    ### ---------------- ###

    inlierCnt = torch.sum( indices, dim = 1).reshape(B, hyp_count)
    correspondences = correspondences.reshape(B, hyp_count, N, D)
    indices = indices.reshape(B, hyp_count, N)
    maxInlierCnt, idx = inlierCnt.max(dim = 1)

    # 3. Refine the hypothesis given inliers
    indices = batched_index_select(indices, 1, idx).squeeze()
    correspondences = batched_index_select(correspondences, 1, idx).squeeze()

    
    parameters = {}
    c, r = sphere_fitting(correspondences, indices)
    parameters['sphere_center'] = c
    parameters['sphere_radius'] = r

    return maxInlierCnt.float(), parameters, indices


def calculate_error(gt_point_cloud, parameters):
    # parameters: {'sphere_center': [B, D], 'sphere_radius': [B]}
    # gt_point_cloud: [N, D]
    a = parameters['sphere_center'][:, 0]
    b = parameters['sphere_center'][:, 1]
    c = parameters['sphere_center'][:, 2]
    r = parameters['sphere_radius']

    # print shapes for tensors a, b, c, r
    # print(a.shape, b.shape, c.shape, r.shape)

    # distance is computed as following:
    # dist_i = (((a - x_i)^2 + (b - y_i)^2 + (c - z_i)^2)^(1/2) - r)^2
    dist = ( torch.sqrt(
            (a.unsqueeze(1) - gt_point_cloud[:, :, 0]) ** 2 + 
            (b.unsqueeze(1) - gt_point_cloud[:, :, 1]) ** 2 + 
            (c.unsqueeze(1) - gt_point_cloud[:, :, 2])**2) - r.unsqueeze(1)) ** 2
    # then we are taking mean of all losses
    # dist = ((dist_0 + ... + dist_N)/ N)^(1/2)
    return torch.sqrt(dist.mean(dim = 1))