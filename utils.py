import torch
import numpy as np

# source: https://discuss.pytorch.org/t/batched-index-select/9115/8?u=mrpositron
# input: B x * x ... x *
# dim: 0 < scalar
# index: B x M
def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

def interleave(tensors_arr, hyp_count):
    for i in range(len(tensors_arr)):
        tensors_arr[i] = tensors_arr[i].repeat_interleave(hyp_count, dim = 0)
    return tensors_arr

def sample_from_probs(input, probs, min_set):
    # input: [B * hyp_count, N, D]
    # probs: [B * hyp_count, N]	
    sampledIndices = probs.multinomial(min_set, replacement = False)
    sampledValues =  batched_index_select(input, 1, sampledIndices)

    # sampledValues: [B * hyp_count, min_set, D]
    # sampledIndices: [B * hyp_count, min_set]
    return [sampledIndices, sampledValues]

def sphere_fitting(point_cloud, weights):
	# point_cloud: [B, N, D]
	# weights: [B, N]

	A =  2 * (-point_cloud + ((point_cloud * weights.unsqueeze(2)).sum(dim = 1) / weights.sum(dim = 1).unsqueeze(1)  ).unsqueeze(1))
	b =  (-(point_cloud ** 2).sum(dim = 2) + ( ((((point_cloud ** 2).sum(dim = 2) * weights)).sum(dim = 1)) /(weights.sum(dim = 1))).unsqueeze(1))

	AtA = torch.bmm(A.transpose(1, 2), A)
	Atb = torch.bmm(A.transpose(1, 2), b.unsqueeze(2))

	c = torch.linalg.solve(AtA, Atb).squeeze(2).unsqueeze(1)
	r = torch.sqrt( (((point_cloud - c)**2).sum(dim = 2) * weights).sum(dim = 1) / (weights.sum(dim = 1)) )
	
	c = c.squeeze(1)
	r = r.unsqueeze(1)
	return c, r

@torch.no_grad()
def fit_sphere(point_cloud, probs, thresh, hyp_count, gradients = None, epsilon = 1e-6):
    B, N, D = point_cloud.shape
    # minimal set
    min_set = 4
    # point_cloud: [B, N, D]
    # probs: [B, N]
    device = point_cloud.device
    probs, point_cloud = interleave([probs, point_cloud], hyp_count)
    # probs: [B * hyp_count, N, D]
    # point_cloud: [B * hyp_count, N, D]
    sampledIndices, sampledValues = sample_from_probs(point_cloud, probs, min_set)
    sampledIndices = sampledIndices.reshape(B, hyp_count *  min_set)
    if gradients != None:
        sampledIndices = sampledIndices.to(device)
        added = torch.tensor([1.0]).expand(sampledIndices.shape).to(device)
        gradients.scatter_add_ (1, sampledIndices, added)
    
    del probs
    # 1. fit a sphere, i.e. caculate hypotheses for each minimal set
    temp = (sampledValues ** 2).sum(dim = 2).unsqueeze(2)
    temp = torch.cat((temp, sampledValues, torch.ones(sampledValues.shape[0], sampledValues.shape[1], 1).to(device) ), dim = 2)
    sub_deter = []



    for i in range(5):
        matrix = torch.cat((temp[:, :, :i], temp[:, :, i+1:]), dim = 2)
        sub_deter.append(torch.linalg.det(matrix))
    del temp
    a = sub_deter[1]/(2 * sub_deter[0] + epsilon)
    b = - sub_deter[2]/(2 * sub_deter[0] + epsilon)
    c = sub_deter[3] / (2 * sub_deter[0] + epsilon)
    r = torch.sqrt( (a**2 + b**2 + c**2 - (sub_deter[4]/ (sub_deter[0] + epsilon) )))

    # 2. Calculate inliers
    # 2.1. Calculate the distance from a sphere

    # Distance is calculated as following
    dist = abs(torch.sqrt((a.unsqueeze(1) - point_cloud[:, :, 0]) ** 2 + (b.unsqueeze(1) - point_cloud[:, :, 1]) ** 2 + (c.unsqueeze(1) - point_cloud[:, :, 2])**2) - r.unsqueeze(1))
    # 2.2. Calculate inliers within a threshold

    indices = torch.where(dist < thresh, 1, 0)

    ### free some memory ###
    del dist, a, b, c, r
    ### ---------------- ###

    inlierCnt = torch.sum( indices, dim = 1).reshape(B, hyp_count)
    point_cloud = point_cloud.reshape(B, hyp_count, N, D)
    indices = indices.reshape(B, hyp_count, N)
    maxInlierCnt, idx = inlierCnt.max(dim = 1)

    # 3. Refine the hypothesis given inliers
    indices = batched_index_select(indices, 1, idx).squeeze()
    point_cloud = batched_index_select(point_cloud, 1, idx).squeeze()

    
    #hypotheses = sphere_fitting(point_cloud, indices)

    parameters = {}
    sphere_center, sphere_radius = sphere_fitting(point_cloud, indices)
    parameters['sphere_center'] = sphere_center
    parameters['sphere_radius'] = sphere_radius

    return maxInlierCnt.float(), parameters, indices