"""model with self-interactions and gates
Exact equivariance to :math:`E(3)`
version of february 2021
"""
import math
from typing import Dict, Union
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr, Activation, Extract
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from torch.autograd import grad
import torch.nn.functional as F

from src.EQ_operations import SelfInteraction, Convolution
from src.utils import smooth_cutoff


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class constrained_network(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        irreps_out,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        max_radius,
        number_of_basis,
        radial_neurons,
        num_neighbors,
        num_nodes,
        reduce_output=True,
        PES_predictor=None,
        masses=None
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        if PES_predictor is not None:
            self.EMC = EnergyMomentumConstraints(PES_predictor,masses)
        else:
            self.EMC = None



        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        num_vec_out = ExtractIr(irreps_hidden,'1o').irreps_out.num_irreps
        w = torch.empty((self.irreps_in.num_irreps, num_vec_out))
        torch.nn.init.xavier_normal_(w,gain=1/math.sqrt(self.irreps_in.num_irreps)) # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
        self.projection_matrix = torch.nn.Parameter(w)

        self.ext_z = ExtractIr(self.irreps_node_attr, '0e')
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }
        nmax_atoms = 20
        embed_dim = 8
        self.node_embedder = torch.nn.Embedding(nmax_atoms,embed_dim)
        # irreps = o3.Irreps("{:}x0e".format(embed_dim))
        self.irreps_node_attr = o3.Irreps("{:}x0e".format(embed_dim))

        self.self_interaction = torch.nn.ModuleList()
        irreps = self.irreps_hidden
        self.self_interaction.append(SelfInteraction(irreps,irreps))
        for _ in range(1,layers):
            self.self_interaction.append(SelfInteraction(self.irreps_hidden,self.irreps_hidden))


        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                self.irreps_hidden,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                radial_neurons,
                num_neighbors
            )
            # self.norms.append(TvNorm(gate.irreps_in))
            irreps = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)
        return

    def make_matrix_semi_unitary(self, M,debug=False):
        I = torch.eye(M.shape[-2])
        if debug:
            M_org = M.clone()
        for i in range(10):
            M = M - 0.5 * (M @ M.t() - I) @ M

        if debug:
            pre_error = torch.norm(I - M_org @ M_org.t())
            post_error = torch.norm(I - M @ M.t())
            print(f"Deviation from unitary before: {pre_error:2.2e}, deviation from unitary after: {post_error:2.2e}")
        return M

    def uplift(self,x):
        """
        :param x:
        :return:
        """
        irreps_out = self.irreps_hidden
        nd_out = irreps_out.dim
        x2 = x.reshape(x.shape[0],-1,3).transpose(1,2)
        M = self.projection_matrix
        M = self.make_matrix_semi_unitary(M)
        y2 = x2 @ M
        y_vec = y2.transpose(1,2).reshape(x.shape[0],-1)

        y = torch.zeros((x.shape[0],nd_out),dtype=x.dtype,device=x.device)
        idx0=0
        for mul, ir in irreps_out:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                y[:,idx0:idx1] = y_vec
                break
            idx0 = idx1
        # print("embedding is still needed!")
        return y


    def project(self,y):
        irreps_in = self.irreps_hidden
        irreps_out = self.irreps_out
        ir_vec = ExtractIr(irreps_in,'1o')
        idx0 = 0
        for mul, ir in irreps_in:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                y_vec = y[:,idx0:idx1]
                break
            idx0 = idx1
        M = self.projection_matrix
        M = self.make_matrix_semi_unitary(M)
        y2 = y_vec.reshape(y.shape[0],-1,3).transpose(1,2)
        x2 = y2 @ M.t()
        x = x2.transpose(1,2).reshape(y.shape[0],-1)
        # print("embedding is still needed!")
        return x

    def apply_constraints(self, y, batch, z, n=10):
        for j in range(n):
            x = self.project(y)
            r = x[:,-3:]
            v = x[:,:-3]
            c,lam_x = self.EMC(r,v,batch,z)
            lam_y = self.uplift(lam_x)
            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 / lam_y.norm()
                lsiter = 0
                while True:
                    ytry = y - alpha * lam_y
                    x = self.project(ytry)
                    r = x[:, -3:]
                    v = x[:, :-3]
                    ctry = self.EMC.constraint(r,v,batch,z,return_gradient=False)
                    if torch.norm(ctry) < torch.norm(c):
                        break
                    alpha = alpha / 2
                    lsiter = lsiter + 1
                    if lsiter > 10:
                        break
                if lsiter == 0:
                    alpha = alpha * 1.5
            y = y - alpha * lam_y
        return y

    def save_reference_energy(self,x,batch,z):
        r = x[:, -3:]
        v = x[:, :-3]
        self.EMC.Energy(r, v, batch, z, save_E0=True)
        return

    def forward(self, x, batch, node_attr, edge_src, edge_dst) -> torch.Tensor:
        """evaluate the network
        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``x`` the input coordinates, perhaps for multiple states
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        h = 0.1
        node_attr_embedded = self.node_embedder(node_attr.to(dtype=torch.int64)).squeeze()
        y = self.uplift(x)
        # x2 = self.project(y)
        y_old = y
        if self.EMC is not None:
            self.save_reference_energy(x,batch,node_attr)

        for i,(conv,gate) in enumerate(zip(self.convolutions,self.gates)):

            edge_vec = x[:,-3:][edge_src] - x[:,-3:][edge_dst]
            edge_sh = o3.spherical_harmonics(o3.Irreps("1x1o"), edge_vec, True, normalization='component')
            edge_length = edge_vec.norm(dim=1)
            edge_features = soft_one_hot_linspace(
                x=edge_length,
                start=0.0,
                end=self.max_radius,
                number=self.number_of_basis,
                basis='bessel',
                cutoff=False
            ).mul(self.number_of_basis**0.5)
            edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

            y_new = conv(y.clone(), node_attr_embedded, edge_src, edge_dst, edge_attr, edge_features)
            y_new = gate(y_new)
            y_new = self.self_interaction[i](y_new)
            tmp = y.clone()
            y = 2*y - y_old + h*y_new
            y_old = tmp
            if self.EMC is not None:
                y = self.apply_constraints(y,batch,node_attr,n=100)
            x = self.project(y)

        return x


class EnergyMomentumConstraints(nn.Module):
    def __init__(self,potential_energy_predictor,m):
        super(EnergyMomentumConstraints, self).__init__()
        self.F = potential_energy_predictor
        self.E0 = None
        self.m = m
        return



    def Energy(self,r,v, batch, z, save_E0=False,return_gradient=False):
        m = self.m
        F = self.F
        E_kin = 0.5*scatter((m*torch.sum(v**2,dim=-1)), batch, dim=0)
        E_pot = F(r,batch,z)
        E = E_pot + E_kin
        if save_E0:
            self.E0 = E
        if return_gradient:
            E_grad = grad(E, r, create_graph=False)[0]
            return E, E_grad
        else:
            return E

    def constraint(self,r,v,batch,z,return_gradient):
        m = self.m
        E1, E_grad = self.Energy(r,v,batch, z,return_gradient=return_gradient)
        E = E1 - self.E0
        p = m[None,:] @ v
        # c = torch.cat([E,p],dim=-1)
        return E, p, E_grad

    def dConstraintT(self,E,p,E_grad):
        m = self.m
        J1 = E_grad * E
        J2 = p * E + m * p
        J = torch.cat([J1,J2],dim=-1)
        return J

    def forward(self,r,v,batch, z):
        E,p, E_grad = self.constraint(r,v,batch, z,return_gradient=True)
        c = torch.cat([E, p], dim=-1)
        J = self.dConstraintT(E,p,E_grad)
        return c, J
