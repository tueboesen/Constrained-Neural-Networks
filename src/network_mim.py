import math

import e3nn.o3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from src.utils import smooth_cutoff


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
    This network is designed to predict the 3D coordinates of a set of particles.
    """
    def __init__(self, node_dim_latent, nlayers, PU, nmax_atom_types=20,atom_type_embed_dim=8,max_radius=50,con_fnc=None,con_type=None,dim=3,embed_node_attr=True,discretization='leapfrog'):
        super().__init__()
        """
        node_dimn_latent:   The dimension of the latent space
        nlayers:            The number of propagationBlocks to include in the network 
        PU:                 A handler to a projection uplifting class instance
        nmax_atom_types:    Max number of particle types to handle, for proteins this should be the types of amino acids
        atom_type_embed_dim: The out dimension of the embedding done to the atom_types
        max_radius:         The maximum radius used when building the edges in the graph between the particles
        con_fnc:            A handler to the constrain function wrapped in a nn.sequential.
        con_type:           The type of constraints being used (high, low, reg)
        dim:                The dimension that the data lives in (default 3)
        """
        self.nlayers = nlayers
        self.PU = PU
        self.dim = dim
        # self.PU.make_matrix_semi_unitary()

        self.node_attr_embedder = torch.nn.Embedding(nmax_atom_types,atom_type_embed_dim)
        self.max_radius = max_radius
        self.h = torch.nn.Parameter(torch.ones(nlayers)*1e-2)
        self.con_fnc = con_fnc
        self.con_type = con_type
        self.embed_node_attr = embed_node_attr
        self.discritization = discretization
        assert self.discritization in ['leapfrog','euler']

        self.PropagationBlocks = nn.ModuleList()
        for i in range(nlayers):
            block = PropagationBlock(xn_dim=node_dim_latent, xn_attr_dim=atom_type_embed_dim)
            self.PropagationBlocks.append(block)
        return

    def forward(self, x, batch, node_attr, edge_src, edge_dst,wstatic=None,ignore_con=False):
        gamma = 1000
        if x.isnan().any():
            raise ValueError("NaN detected")

        if self.embed_node_attr:
            node_attr_embedded = self.node_attr_embedder(node_attr.to(dtype=torch.int64)).squeeze()
        else:
            node_attr_embedded = node_attr
        y = self.PU.uplift(x)

        y_old = y
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
            tmp = y.clone()

            if self.con_type == 'stabhigh':
                data = self.con_fnc({'y': y, 'batch': batch, 'z': node_attr})  # TODO should this use y_new or y?
                lam_y = data['lam_y']
            else:
                lam_y = 0
            if self.discritization == 'leapfrog':
                y = 2*y - y_old - self.h[i]**2 * (y_new - gamma*lam_y)
                y_old = tmp
            elif self.discritization == 'euler':
                y = y + self.h[i]**2 * (y_new - gamma*lam_y)
            else:
                raise NotImplementedError(f"Discretization method {self.discritization} not implemented.")

            if self.con_fnc is not None and self.con_type == 'high' and ignore_con is False:
                if y.isnan().any():
                    raise ValueError("NaN detected")
                data = self.con_fnc({'y': y, 'batch': batch,'z':node_attr})
                y = data['y']
                if y.isnan().any():
                    raise ValueError("NaN detected")

            x = self.PU.project(y)

        if self.con_fnc is not None and self.con_type == 'low' and ignore_con is False:
            data = self.con_fnc({'x': x, 'batch': batch,'z':node_attr})
            x = data['x']
        if self.con_fnc is not None and self.con_type == 'reg' and ignore_con is False:
            data = self.con_fnc({'x': x, 'batch': batch, 'z': node_attr})
            reg = data['c']
        else:
            reg = 0
        if x.isnan().any():
            raise ValueError("NaN detected")

        if self.con_fnc is not None:
            cv = self.con_fnc[0].compute_constraint_violation({'x': x, 'batch': batch, 'z': node_attr})
        else:
            cv = -1

        return x, reg, cv
