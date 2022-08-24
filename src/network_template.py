import math

import e3nn.o3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from src.utils import smooth_cutoff, atomic_masses


class network_template(nn.Module):
    """

    """
    def __init__(self, node_dim_in, node_dim_latent, nlayers, nmax_atom_types=20,atom_type_embed_dim=8,max_radius=50,con_fnc=None,con_type=None,dim=3,embed_node_attr=True,discretization='leapfrog',gamma=0,regularization=0):
        super().__init__()
        """

        """
        self.nlayers = nlayers
        self.gamma = gamma
        self.dim = dim
        self.low_dim = node_dim_in
        self.high_dim = node_dim_latent
        self.lin = torch.nn.Linear(self.low_dim,self.high_dim)
        self.ortho = torch.nn.utils.parametrizations.orthogonal(self.lin)
        self.node_attr_embedder = torch.nn.Embedding(nmax_atom_types,atom_type_embed_dim)
        self.max_radius = max_radius
        self.h = torch.nn.Parameter(torch.ones(nlayers)*1e-2)
        self.con_fnc = con_fnc
        self.con_type = con_type
        self.embed_node_attr = embed_node_attr
        self.discretization = discretization
        assert self.discretization in ['leapfrog', 'euler','rk4']

        self.PropagationBlocks = nn.ModuleList()
        for i in range(nlayers):
            block = PropagationBlock(xn_dim=node_dim_latent, xn_attr_dim=atom_type_embed_dim)
            self.PropagationBlocks.append(block)
        self.params = nn.ModuleDict({
            "base": self.PropagationBlocks,
            "h": nn.ParameterList([self.h]),
            "close": nn.ModuleList([self.lin])
        })
        return

    # def inverse(self,x):
    #     y = x @ self.K.T @ torch.inverse(self.K @ self.K.T) #This is for the right side
        # y = x @ torch.inverse(self.K.T @ self.K) @ self.K.T
        # return y

    def uplift(self,x):
        y = x @ self.lin.weight.T
        # y = x @ self.K.T
        return y

    def project(self,y):
        x = y @ self.lin.weight
        # x = y @ self.K
        return x

    def propgation(self,y,batch,weight,i,edge_src,edge_dst):
        ndimy = y.shape[-1]

        if self.gamma > 0:
            if self.discretization == 'rk4':
                q1 = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy), self.project, self.uplift, weight)
                q2 = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy) + q1 / 2 * self.h[i] ** 2, self.project, self.uplift, weight)
                q3 = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy) + q2 / 2 * self.h[i] ** 2, self.project, self.uplift, weight)
                q4 = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy) + q3 * self.h[i] ** 2, self.project, self.uplift, weight)
                dy = (q1 + 2 * q2 + 2 * q3 + q4) / 6
            else:
                dy = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy), self.project, self.uplift, weight)
            dy = dy.view(-1, ndimy)
        else:
            dy = 0
        if self.discretization == 'leapfrog':
            tmp = y.clone()
            y = 2 * y - y_old - self.h[i] ** 2 * (y_new + self.gamma * dy)
            y_old = tmp
        elif self.discretization == 'euler':
            y = y + self.h[i] ** 2 * (y_new - self.gamma * dy)
        elif self.discretization == 'rk4':
            k1 = self.PropagationBlocks[i](y.clone(), edge_attr, edge_src, edge_dst)
            k2 = self.PropagationBlocks[i](y.clone() + k1 * self.h[i] ** 2 / 2, edge_attr, edge_src, edge_dst)
            k3 = self.PropagationBlocks[i](y.clone() + k2 * self.h[i] ** 2 / 2, edge_attr, edge_src, edge_dst)
            k4 = self.PropagationBlocks[i](y.clone() + k3 * self.h[i] ** 2, edge_attr, edge_src, edge_dst)
            y_new = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            y = y + self.h[i] ** 2 * (y_new - self.gamma * dy)
        else:
            raise NotImplementedError(f"Discretization method {self.discretization} not implemented.")
        if self.con_fnc is not None and self.con_type == 'high' and ignore_con is False:
            if y.isnan().any():
                raise ValueError("NaN detected")
            y, regi, regi2 = self.con_fnc(y.view(batch.max() + 1,-1,ndimy),self.project,self.uplift,weight)
            y = y.view(-1,ndimy)
            reg = reg + regi
            reg2 = reg2 + regi2
            if y.isnan().any():
                raise ValueError("NaN detected")

        x = self.project(y)




    def forward(self, x, batch, node_attr, edge_src, edge_dst,wstatic=None,ignore_con=False,weight=1):
        if x.isnan().any():
            raise ValueError("NaN detected")

        if self.embed_node_attr:
            node_attr_embedded = self.node_attr_embedder(node_attr.min(dim=-1)[0].to(dtype=torch.int64)).squeeze()
        else:
            node_attr_embedded = node_attr
        y = self.uplift(x)
        ndimx = x.shape[-1]
        ndimy = y.shape[-1]
        y_old = y
        reg = torch.tensor(0.0)
        reg2 =  torch.tensor(0.0)

        for i in range(self.nlayers):
            if x.isnan().any():
                raise ValueError("NaN detected")
            if wstatic is None:
                edge_vec = x[:,0:self.dim][edge_src] - x[:,0:self.dim][edge_dst]
                edge_len = edge_vec.norm(dim=1)
                w = smooth_cutoff(edge_len / self.max_radius) / edge_len
            else:
                w = wstatic
            edge_attr = torch.cat([node_attr_embedded[edge_src], node_attr_embedded[edge_dst], w[:,None]],dim=-1)

            y_new = self.PropagationBlocks[i](y.clone(), edge_attr, edge_src, edge_dst)



        if self.con_fnc is not None and self.con_type == 'low' and ignore_con is False:
            x, reg, reg2 = self.con_fnc(x.view(batch.max() + 1,-1,ndimx),weight=weight)
            x = x.view(-1,ndimx)
        if x.isnan().any():
            raise ValueError("NaN detected")

        if self.con_fnc is not None:
            _, cv_mean,cv_max = self.con_fnc.compute_constraint_violation(x.view(batch.max() + 1,-1,ndimx))
        else:
            cv_mean,cv_max = torch.tensor(-1.0),  torch.tensor(-1.0)

        return x, cv_mean, cv_max, reg, reg2
