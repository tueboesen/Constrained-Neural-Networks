import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None,dim=8):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // dim

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, dim)
    x = x.view(batch_size, num_points, 1, num_dims, dim).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        x: point features of shape [Batch, N_feat, Dimensions, N_samples, Number of neighbours]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]
            # u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            # u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)

            # compute the cross product of the two output vectors
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0




class neural_network_equivariant_simple(torch.nn.Module):
    """
    """
    def __init__(self,nlayers,gamma,dim,con_fnc,con_type,discretization,n_vec_low=4,n_vec_high=40,n_features_in=1,n_features_low=2,n_features_high=20) -> None:
        super().__init__()

        self.nlayers = nlayers
        self.gamma = gamma
        self.dim = dim

        self.n_vec_low = n_vec_low
        self.n_vec_high = n_vec_high
        self.n_features_in = n_features_in
        self.n_features_low = n_features_low
        self.n_features_high = n_features_high

        self.con_fnc = con_fnc
        self.con_type = con_type
        self.discretization = discretization
        assert self.discretization in ['leapfrog', 'euler','rk4']

        self.gamma = 0
        self.nlayers = nlayers
        self.dim = dim
        self.h = torch.nn.Parameter(torch.ones(nlayers)*1e-2)
        self.lin = torch.nn.Linear(self.n_vec_low, self.n_vec_high)
        self.ortho = torch.nn.utils.parametrizations.orthogonal(self.lin)

        self.conv1s = nn.ModuleList()
        self.conv2s = nn.ModuleList()
        self.vn_lins = nn.ModuleList()
        for i in range(nlayers):
            conv1 = VNLinearLeakyReLU(self.n_features_low, self.n_features_high)
            conv2 = VNLinearLeakyReLU(self.n_features_high, self.n_features_low)
            vn_lin = nn.Linear(self.n_features_low, self.n_features_in , bias=False)
            self.conv1s.append(conv1)
            self.conv2s.append(conv2)
            self.vn_lins.append(vn_lin)

        #TODO add a second set of lin for the velocity components
        self.params = nn.ModuleDict({
            "base": nn.ModuleList([self.conv1s, self.conv2s, self.vn_lins]),
            "h": nn.ParameterList([self.h]),
            "close": nn.ModuleList([self.lin])
        })

        return

    def project(self, y):
        x = y @ self.lin.weight
        return x

    def uplift(self, x):
        y = x @ self.lin.weight.T
        return y

    def forward_propagation(self,y,i):
        y = y.unsqueeze(1).transpose(2,3)
        y = get_graph_feature(y, k=5,dim=self.n_vec_high)
        y = self.conv1s[i](y)
        y = self.conv2s[i](y)
        y1 = y.mean(dim=-1)
        y = self.vn_lins[i](y1.transpose(1, -1)).transpose(1, -1)
        y = y.transpose(2,3).squeeze()
        return y

    def forward(self, x, batch, node_attr, edge_src, edge_dst,wstatic=None,weight=1) -> torch.Tensor:
        x_vec = x.view(batch.max()+1,-1,self.dim*2)
        y = self.uplift(x_vec)

        ndimx = x.shape[-1]
        ndimy = y.shape[-1]
        # y_old = y
        reg = torch.tensor(0.0)
        reg2 =  torch.tensor(0.0)

        for i in range(self.nlayers):
            y_new = self.forward_propagation(y,i)
            if self.gamma > 0:
                if self.discretization == 'rk4':
                    q1 = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy), self.project, self.uplift, weight)
                    q2 = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy) + q1 / 2 * self.h[i] ** 2, self.project, self.uplift, weight)
                    q3 = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy) + q2 / 2 * self.h[i] ** 2, self.project, self.uplift, weight)
                    q4 = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy) + q3 * self.h[i] ** 2, self.project, self.uplift, weight)
                    dy = (q1 + 2 * q2 + 2 * q3 + q4) / 6
                else:
                    dy = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1,-1,ndimy),self.project,self.uplift,weight)
                dy = dy.view(-1,ndimy)
            else:
                dy = 0
            if self.discretization == 'leapfrog':
                tmp = y.clone()
                y = 2*y - y_old - self.h[i]**2 * (y_new + self.gamma*dy)
                y_old = tmp
            elif self.discretization == 'euler':
                y = y + self.h[i]**2 * (y_new - self.gamma*dy)
            elif self.discretization == 'rk4':
                k1 = self.forward_propagation(y.clone(),i)
                k2 = self.forward_propagation(y.clone() + k1 * self.h[i] ** 2 / 2,i)
                k3 = self.forward_propagation(y.clone() + k2 * self.h[i] ** 2 / 2,i)
                k4 = self.forward_propagation(y.clone() + k3 * self.h[i] ** 2,i)

                y_new = (k1 + 2*k2 + 2*k3 + k4)/6
                y = y + self.h[i]**2 * (y_new - self.gamma*dy)
            else:
                raise NotImplementedError(f"Discretization method {self.discretization} not implemented.")

            if self.con_fnc is not None and self.con_type == 'high':
                if y.isnan().any():
                    raise ValueError("NaN detected")
                y, regi, regi2 = self.con_fnc(y.view(batch.max() + 1,-1,ndimy),self.project,self.uplift,weight)
                y = y.view(-1,ndimy)
                reg = reg + regi
                reg2 = reg2 + regi2
                if y.isnan().any():
                    raise ValueError("NaN detected")

            x = self.project(y)

        if self.con_fnc is not None and self.con_type == 'low':
            x, reg, reg2 = self.con_fnc(x.view(batch.max() + 1, -1, ndimx), weight=weight)
            x = x.view(-1, ndimx)
        if x.isnan().any():
            raise ValueError("NaN detected")

        if self.con_fnc is not None:
            _, cv_mean, cv_max = self.con_fnc.compute_constraint_violation(x.view(batch.max() + 1, -1, ndimx))
        else:
            cv_mean, cv_max = torch.tensor(-1.0), torch.tensor(-1.0)

        x = x.view(-1,ndimx)

        return x, cv_mean, cv_max, reg, reg2

