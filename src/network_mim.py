import math
import time

import e3nn.o3
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from src.utils import smooth_cutoff, atomic_masses


class Mixing(nn.Module):
    """
    A mixing function for non-equivariant networks, it has the options to use the bilinear operation, and for the bilinear operation it has the option to use the e3nn version or the standard pytorch version.
    When this program was made the standard pytorch version had some serious runtime issues, and the e3nn bilinear operation was much faster.
    """
    def __init__(self, dim_in1,dim_in2,dim_out,use_bilinear=True,use_e3nn=True):
        super(Mixing, self).__init__()
        self.use_bilinear = use_bilinear
        if use_bilinear:
            if use_e3nn:
                irreps1 = e3nn.o3.Irreps("{:}x0e".format(dim_in1))
                irreps2 = e3nn.o3.Irreps("{:}x0e".format(dim_in2))
                self.bilinear = e3nn.o3.FullyConnectedTensorProduct(irreps1, irreps2, irreps1)
            else:
                self.bilinear = nn.Bilinear(dim_in1, dim_in2, dim_out, bias=False)
        self.lin = nn.Linear(2*dim_in1+dim_in2,dim_out)

    def forward(self, x1,x2):
        x = torch.cat([x1,x2],dim=-1)
        if self.use_bilinear:
            x_bi = self.bilinear(x1,x2)
            x = torch.cat([x,x_bi],dim=-1)
        x = self.lin(x)
        return x


class NodesToEdges(nn.Module):
    """
    A mimetic node to edges operation utilizing average and gradient operations
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self,xn,xe_src,xe_dst, W):
        xe_grad = W * (xn[xe_src] - xn[xe_dst])
        xe_ave = W * (xn[xe_src] + xn[xe_dst]) / 2
        return xe_grad,xe_ave


class EdgesToNodes(nn.Module):
    """
    A mimetic edges to node operation utilizing divergence and averaging operations.
    """
    def __init__(self, dim_in, dim_out, num_neighbours=20,use_e3nn=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.norm = 1 / math.sqrt(num_neighbours)
        self.mix = Mixing(dim_in, dim_in, dim_out)

    def forward(self, xe, xe_src, xe_dst, W):
        xn_1 = scatter(W * xe,xe_dst,dim=0) * self.norm
        xn_2 = scatter(W * xe,xe_src,dim=0) * self.norm
        xn_div = xn_1 - xn_2
        xn_ave = xn_1 + xn_2
        xn = self.mix(xn_div, xn_ave)
        return xn

class FullyConnectedNet(nn.Module):
    """
    A simple fully connected network, with an activation function at the end
    """
    def __init__(self,dimensions,activation_fnc):
        super(FullyConnectedNet, self).__init__()
        self.layers = nn.ModuleList()
        self.nlayers = len(dimensions)-1
        for i in range(self.nlayers):
            ll = nn.Linear(dimensions[i],dimensions[i+1])
            self.layers.append(ll)
        self.activation = activation_fnc
        return

    def forward(self,x):
        for i in range(self.nlayers):
            x = self.layers[i](x)
        x = self.activation(x)
        return x


def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X



class DoubleLayer(nn.Module):
    """
    A double layer building block used to generate our mimetic neural network
    """
    def __init__(self,dim_x):
        super().__init__()
        self.dim_x = dim_x
        self.lin1 = nn.Linear(dim_x,dim_x,bias=False)
        self.lin2 = nn.Linear(dim_x,dim_x,bias=False)


        return

    def forward(self,x):
        x = torch.tanh(x)
        x = self.lin1(x)
        x = tv_norm(x)
        x = torch.tanh(x)
        x = self.lin1(x)
        x = torch.tanh(x)
        return x


class PropagationBlock(nn.Module):
    """
    A propagation block (layer) used in our mimetic neural network.
    """
    def __init__(self, xn_dim, xn_attr_dim):
        super().__init__()

        self.fc1 = FullyConnectedNet([xn_attr_dim*2+1, xn_dim],activation_fnc=torch.nn.functional.silu)
        self.doublelayer = DoubleLayer(xn_dim*5)

        return

    def forward(self, xn, xe_attr, xe_src, xe_dst):
        W = self.fc1(xe_attr)
        gradX = W * (xn[xe_src] - xn[xe_dst])
        aveX = W * (xn[xe_src] + xn[xe_dst]) / 2

        gradXaveX = gradX * aveX
        gradXsq = gradX * gradX
        aveXsq = aveX * aveX

        dxe = torch.cat([gradX, aveX, gradXaveX, gradXsq, aveXsq], dim=1)
        dxe = self.doublelayer(dxe)

        xn_1 = scatter(W.repeat(1,5) * dxe,xe_dst,dim=0)
        xn_2 = scatter(W.repeat(1,5) * dxe,xe_src,dim=0)
        xn_div = xn_1 - xn_2
        xn_ave = (xn_1 + xn_2) / 2.0

        d = xn.shape[-1]
        xn = xn_div[:,:d] + xn_ave[:,d:2*d] + xn_ave[:,2*d:3*d] + xn_ave[:,3*d:4*d] + xn_ave[:,4*d:]
        if xn.isnan().any():
            raise ValueError("NaN detected")
        return xn


class neural_network_mimetic(nn.Module):
    """
    This network is designed to predict the 2D/3D coordinates of a set of particles.
    """
    def __init__(self, dim_in, dim_latent, layers, nmax_atom_types=20, atom_type_embed_dim=8, max_radius=50, con_fnc=None, con_type=None, position_indices=(0, 1), embed_node_attr=True, discretization_method='leapfrog', penalty_strength=0, regularization_strength=0, orthogonal_projection=True):
        super().__init__()
        """
        node_dimn_latent:   The dimension of the latent space
        nlayers:            The number of propagationBlocks to include in the network 
        nmax_atom_types:    Max number of particle types to handle, for proteins this should be the types of amino acids
        atom_type_embed_dim: The out dimension of the embedding done to the atom_types
        max_radius:         The maximum radius used when building the edges in the graph between the particles
        con_fnc:            A handler to the constrain function wrapped in a nn.sequential.
        con_type:           The type of constraints being used (high, low, reg)
        dim:                The dimension that the data lives in (default 3)
        """
        self.nlayers = layers
        self.penalty_strength = penalty_strength
        self.regularization_strength = regularization_strength
        self.position_indices = position_indices
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.orthogonal_K = orthogonal_projection
        self.lin = torch.nn.Linear(self.dim_in, self.dim_latent)
        self.node_attr_embedder = torch.nn.Embedding(nmax_atom_types,atom_type_embed_dim)
        self.max_radius = max_radius
        self.h = torch.nn.Parameter(torch.ones(layers) * 1e-2)
        if self.orthogonal_K:
            # self.h = torch.nn.Parameter(torch.ones(nlayers) * 1e-2)
            self.ortho = torch.nn.utils.parametrizations.orthogonal(self.lin)
        else:
            # self.h = torch.nn.Parameter(torch.ones(nlayers) * 1e-3)
            pass

        self.con_fnc = con_fnc
        self.con_type = con_type
        self.embed_node_attr = embed_node_attr
        self.discretization_method = discretization_method
        assert self.discretization_method in ['leapfrog', 'euler', 'rk4']

        self.PropagationBlocks = nn.ModuleList()
        for i in range(layers):
            block = PropagationBlock(xn_dim=dim_latent, xn_attr_dim=atom_type_embed_dim)
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

    @property
    def K(self):
        return self.lin.weight.T

    def project(self,y):
        x = y @ self.lin.weight
        # x = y @ self.K
        return x

    def forward(self, x, batch, node_attr, edge_src, edge_dst,wstatic=None,ignore_con=False,weight=1):
        if x.isnan().any():
            raise ValueError("NaN detected")
        if self.embed_node_attr:
            node_attr_embedded = self.node_attr_embedder(node_attr.min(dim=-1)[0].to(dtype=torch.int64)).squeeze()
        else:
            node_attr_embedded = node_attr
        y = self.uplift(x)
        # y2 = x @ self.K().T
        ndimx = x.shape[-1]
        ndimy = y.shape[-1]
        y_old = y
        reg = torch.tensor(0.0)

        for i in range(self.nlayers):
            t1 = time.time()
            dt = max(min(self.h[i] ** 2, 0.1),1e-4)
            if x.isnan().any():
                raise ValueError("NaN detected")
            if wstatic is None:
                edge_vec = x[:,self.position_indices][edge_src] - x[:, self.position_indices][edge_dst]
                edge_len = edge_vec.norm(dim=1)
                w = smooth_cutoff(edge_len / self.max_radius) / edge_len
            else:
                w = wstatic
            t2 = time.time()
            edge_attr = torch.cat([node_attr_embedded[edge_src], node_attr_embedded[edge_dst], w[:,None]],dim=-1)

            y_new = self.PropagationBlocks[i](y.clone(), edge_attr, edge_src, edge_dst)

            if self.penalty_strength > 0:
                if self.discretization_method == 'rk4':
                    q1 = self.con_fnc.constraint_penalty(y.view(batch.max() + 1, -1, ndimy), self.project, self.uplift, weight)
                    q2 = self.con_fnc.constraint_penalty(y.view(batch.max() + 1, -1, ndimy) + q1 / 2 * dt, self.project, self.uplift, weight)
                    q3 = self.con_fnc.constraint_penalty(y.view(batch.max() + 1, -1, ndimy) + q2 / 2 * dt, self.project, self.uplift, weight)
                    q4 = self.con_fnc.constraint_penalty(y.view(batch.max() + 1, -1, ndimy) + q3 * dt, self.project, self.uplift, weight)
                    dy = (q1 + 2 * q2 + 2 * q3 + q4) / 6
                else:
                    dy = self.con_fnc.constraint_penalty(y.view(batch.max() + 1,-1,ndimy),self.project,self.uplift,weight)
                dy = dy.view(-1,ndimy)
            else:
                dy = 0
            if self.discretization_method == 'leapfrog':
                tmp = y.clone()
                y = 2*y - y_old - dt * (y_new + self.penalty_strength * dy)
                y_old = tmp
            elif self.discretization_method == 'euler':
                y = y + dt * (y_new - self.penalty_strength * dy)
            elif self.discretization_method == 'rk4':
                k1 = self.PropagationBlocks[i](y.clone(), edge_attr, edge_src, edge_dst)
                k2 = self.PropagationBlocks[i](y.clone() + k1 * dt / 2, edge_attr, edge_src, edge_dst)
                k3 = self.PropagationBlocks[i](y.clone() + k2 * dt / 2, edge_attr, edge_src, edge_dst)
                k4 = self.PropagationBlocks[i](y.clone() + k3 * dt, edge_attr, edge_src, edge_dst)
                y_new = (k1 + 2*k2 + 2*k3 + k4)/6
                y = y + dt * (y_new - self.penalty_strength * dy)
            else:
                raise NotImplementedError(f"Discretization method {self.discretization_method} not implemented.")

            t3 = time.time()
            if self.con_fnc is not None and self.con_type == 'high' and ignore_con is False:
                if y.isnan().any():
                    raise ValueError("NaN detected")
                # y, regi, regi2 = self.con_fnc(y.view(batch.max() + 1,-1,ndimy),self.K(),self.K().T,weight)
                y, _, regi = self.con_fnc(y.view(batch.max() + 1,-1,ndimy),self.project,self.uplift,weight,K=self.K)
                y = y.view(-1,ndimy)
                reg = reg + regi
                if y.isnan().any():
                    raise ValueError("NaN detected")

            x = self.project(y)
            t4 = time.time()
            print(f"Inner {t2-t1:2.2f},{t3-t2:2.2f},{t4-t3:2.2f}")

        if self.con_fnc is not None and self.con_type == 'low' and ignore_con is False:
            x, _, reg = self.con_fnc(x.view(batch.max() + 1,-1,ndimx),weight=weight)
            x = x.view(-1,ndimx)
        if x.isnan().any():
            raise ValueError("NaN detected")

        t1 = time.time()
        if self.con_fnc is not None:
            cv, cv_mean,cv_max = self.con_fnc.constraint_violation(x.view(batch.max() + 1, -1, ndimx))
            if reg == 0:
                reg = (cv*cv).mean()
        else:
            cv_mean,cv_max = torch.tensor(-1.0),  torch.tensor(-1.0)
        t2 = time.time()
        print(f"Con time = {t2-t1:2.2f}")
        return x, cv_mean, cv_max, reg * self.regularization_strength
