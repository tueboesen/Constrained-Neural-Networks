"""model with self-interactions and gates
Exact equivariance to :math:`E(3)`
version of february 2021
"""
import math
from typing import Dict, Union
import numpy as np
import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr, Activation
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from torch.autograd import grad
import torch.nn.functional as F


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
from torch_scatter import scatter


@compile_mode('script')
class Convolution(torch.nn.Module):
    r"""equivariant convolution
    Parameters
    ----------
    irreps_node_input : `Irreps`
        representation of the input node features
    irreps_node_attr : `Irreps`
        representation of the node attributes
    irreps_edge_attr : `Irreps`
        representation of the edge attributes
    irreps_node_output : `Irreps` or None
        representation of the output node features
    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    num_neighbors : float
        typical number of nodes convolved over
    """
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel],
            torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out


class SelfExpandingGate(torch.nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps_in = irreps
        self.irreps_out = irreps

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0 ])
        ir = "0e"
        irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])
        if o3.Irreps(irreps_gated).dim == 0:
            self.si = Identity()
            activation_fnc = []
            for mul,ir in o3.Irreps(irreps_scalars):
                if ir.p == 1:
                    activation_fnc.append(torch.nn.functional.silu)
                else:
                    activation_fnc.append(torch.tanh)
            self.gate = Activation(irreps_scalars, activation_fnc)
        else:
            self.gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated)  # gated tensors
            self.si = SelfInteraction(irreps, self.gate.irreps_in)
        return

    def forward(self,x):
        x = self.si(x)
        x = self.gate(x)
        return x


#
# class Compose(torch.nn.Module):
#     def __init__(self, first, second):
#         super().__init__()
#         self.first = first
#         self.second = second
#         self.irreps_in = self.first.irreps_in
#         self.irreps_out = self.second.irreps_out
#
#     def forward(self, *input):
#         x = self.first(*input)
#         return self.second(x)

class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            return
        def forward(self, input):
            return input

class Filter(torch.nn.Module):
    def __init__(self, number_of_basis, radial_layers, radial_neurons, irrep):
        super().__init__()
        self.irrep = irrep
        nd = irrep.dim
        nr = irrep.num_irreps
        self.net = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [nr], torch.nn.functional.silu)
        S = torch.empty(nr,dtype=torch.int64)
        idx = 0
        for mul,ir in irrep:
            li = 2*ir.l+1
            for i in range(mul):
                S[idx+i] = li
            idx += i+1
        self.register_buffer("degen", S)
        return

    def forward(self, x):
        x = self.net(x)
        y = x.repeat_interleave(self.degen,dim=1)
        return y

class SelfInteraction(torch.nn.Module):
    def __init__(self, irreps_in,irreps_out):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.tp = o3.FullyConnectedTensorProduct(irreps_in,irreps_in,irreps_out)

        nd = irreps_out.dim
        nr = irreps_out.num_irreps
        degen = torch.empty(nr, dtype=torch.int64)
        m_degen = torch.empty(nr, dtype=torch.bool)
        idx = 0
        for mul, ir in irreps_out:
            li = 2 * ir.l + 1
            for i in range(mul):
                degen[idx + i] = li
                m_degen[idx + i] = ir.l == 0
            idx += i + 1
        M = m_degen.repeat_interleave(degen)
        self.register_buffer("m_scalar", M)
        return

    def forward(self, x, normalize_variance=True, eps=1e-9, debug=False):
        y = self.tp(x,x)
        if normalize_variance:
            nb, _ = y.shape
            ms = self.m_scalar
            std = torch.std(y[:, ms], dim=1)
            y[:, ms] /= std[:,None] + eps

            mv = ~ms
            if torch.sum(mv) > 0:
                tmp = y[:,mv].clone()
                yy = tmp.view(nb,-1,3).clone()
                norm1 = torch.sqrt(torch.sum(yy**2,dim=2)+eps)
                std = norm1.std(dim=1)
                yy2 = yy / (std[:,None,None]+eps)
                y[:,mv] = yy2.view(nb,-1)
        if debug:
            if normalize_variance and torch.sum(mv) > 0:
                with torch.no_grad():
                    scalars_in = torch.cat([x[:, ms].view(-1),norm1.view(-1)])
                    tmp1 = y[:, mv]
                    tmp2 = tmp1.view(nb, -1, 3)
                    norm2 = tmp2.norm(dim=2)
                    scalars_out = torch.cat([y[:, ms].view(-1),norm2.view(-1)])
                print(f"input var: {scalars_in.var():2.2f}, output var: {scalars_out.var():2.2f}")
            else:
                print(f"Normalize variance={normalize_variance}, input var: {x.var():2.2f}, output var: {y.var():2.2f}")
        return y



class TvNorm(torch.nn.Module):
        def __init__(self,irreps):
            super().__init__()
            self.irreps_in = irreps
            self.irreps_out = irreps

            nd = irreps.dim
            nr = irreps.num_irreps
            degen = torch.empty(nr, dtype=torch.int64)
            m_degen = torch.empty(nr,dtype=torch.bool)
            idx = 0
            for mul, ir in irreps:
                li = 2 * ir.l + 1
                for i in range(mul):
                    degen[idx + i] = li
                    m_degen[idx+i] = ir.l == 0
                idx += i + 1
            M = m_degen.repeat_interleave(degen)
            self.register_buffer("m_scalar", M)
            return

        def forward(self, x, eps=1e-6):
            nb,_ = x.shape
            ms = self.m_scalar
            # x[:,ms] = x[:,ms] - torch.mean(x[:,ms], dim=1, keepdim=True)
            x[:,ms] /= torch.sqrt(torch.sum(x[:,ms] ** 2, dim=1, keepdim=True) + eps)

            mv = ~ms
            if torch.sum(mv) > 0:
                #We need to decide how to handle the vectors, eps is the tricky part
                tmp = x[:,mv].clone()
                xx = tmp.view(nb,-1,3).clone()
                tmp2 = torch.sum(xx**2,dim=1)
                # if (tmp2 == 0).any():
                    # print("??")
                norm1 = torch.sqrt(tmp2+eps)
                # if (norm1 < 1e-6).any():
                #     print("Stop here")
                norm_mean = torch.mean(norm1,dim=1)
                xx2 = xx / (norm_mean[:,None,None]+eps)
                x[:,mv] = xx2.view(nb,-1)
            return x



class Norm(torch.nn.Module):
        def __init__(self,irreps):
            super().__init__()
            self.irreps_in = irreps
            self.irreps_out = irreps

            nd = irreps.dim
            nr = irreps.num_irreps
            degen = torch.empty(nr, dtype=torch.int64)
            m_degen = torch.empty(nr,dtype=torch.bool)
            idx = 0
            for mul, ir in irreps:
                li = 2 * ir.l + 1
                for i in range(mul):
                    degen[idx + i] = li
                    m_degen[idx+i] = ir.l == 0
                idx += i + 1
            M = m_degen.repeat_interleave(degen)
            self.register_buffer("m_scalar", M)
            return

        def forward(self, x, eps=1e-6):
            nb,_ = x.shape
            ms = self.m_scalar
            # x[:,ms] = x[:,ms] - torch.mean(x[:,ms], dim=1, keepdim=True)
            x[:,ms] /= torch.sqrt(torch.sum(x[:,ms] ** 2, dim=1, keepdim=True) + eps)

            mv = ~ms
            if torch.sum(mv) > 0:
                #We need to decide how to handle the vectors, eps is the tricky part
                xx = x[:,mv].view(nb,-1,3)
                norm = xx.norm(dim=-1)
                xx /= norm[:,:,None]
                x[:,mv] = xx.view(nb,-1)
            return x


class Activation(torch.nn.Module):
    def __init__(self, irreps, activation_fnc):
        super().__init__()
        self.irreps_in = irreps
        self.irreps_out = irreps
        self.activation_fnc = activation_fnc

        nd = irreps.dim
        nr = irreps.num_irreps
        degen = torch.empty(nr, dtype=torch.int64)
        m_degen = torch.empty(nr, dtype=torch.bool)
        idx = 0
        for mul, ir in irreps:
            li = 2 * ir.l + 1
            for i in range(mul):
                degen[idx + i] = li
                m_degen[idx + i] = ir.l == 0
            idx += i + 1
        M = m_degen.repeat_interleave(degen)
        self.register_buffer("m_scalar", M)
        return

    def forward(self, x, eps=1e-6):
        nb, _ = x.shape
        ms = self.m_scalar
        # x[:,ms] = x[:,ms] - torch.mean(x[:,ms], dim=1, keepdim=True)
        x[:, ms] = self.activation_fnc(x[:, ms])

        mv = ~ ms
        # if torch.sum(mv) > 0:
        #     tmp = x[:, mv].clone()
        #     xx = tmp.view(nb, -1, 3).clone()
        #     norm1 = xx.norm(dim=-1)
        #     if (norm1 < 1e-6).any():
        #         print("stop here")
        #     xx_normalized = xx / (norm1[:,:,None])
        #     norm_activated = self.activation_fnc(norm1)
        #     xx_activated = norm_activated[:,:,None] * xx_normalized
        #     x[:, mv] = xx_activated.view(nb, -1)
        return x



class DoubleLayer(torch.nn.Module):
    def __init__(self, irreps_in,irreps_hidden,irreps_out):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out_intended = irreps_out
        # self.non_linear_act1 = e3nn.nn.NormActivation(self.irreps_in,torch.sigmoid,normalize=False,bias=True)
        self.non_linear_act1 = Activation(self.irreps_in,torch.tanh)
        # self.g1 = SelfExpandingGate(irreps_in)


        irreps = o3.Irreps([(mul, ir) for mul, ir in irreps_hidden if
                   tp_path_exists(irreps_in, irreps_in, ir)])
        self.si1 = SelfInteraction(self.irreps_in,irreps)
        # self.normalize_and_non_linear_act2 = e3nn.nn.NormActivation(irreps,torch.sigmoid,normalize=True,bias=False)

        self.tv = TvNorm(irreps)
        # self.g2 = SelfExpandingGate(irreps)
        irreps2 = o3.Irreps([(mul, ir) for mul, ir in irreps_out if
                   tp_path_exists(irreps, irreps, ir)])
        self.si2 = SelfInteraction(irreps,irreps2)
        # self.g3 = SelfExpandingGate(irreps2)
        self.non_linear_act2 = Activation(irreps2,torch.sigmoid)
        # self.non_linear_act3 = e3nn.nn.NormActivation(irreps2, torch.sigmoid, normalize=False, bias=True)

        self.irreps_out = irreps2
        return

    def forward(self, x):
        assert ~x.isnan().any()
        x1 = self.non_linear_act1(x)
        assert ~x1.isnan().any()
        x2 = self.si1(x1)
        assert ~x2.isnan().any()
        x3 = self.tv(x2)
        assert ~x3.isnan().any()
        x4 = self.si2(x3)
        assert ~x4.isnan().any()
        x5 = self.non_linear_act2(x4)
        assert ~x5.isnan().any()
        return x5

def zero_small_numbers(x,eps=1e-6):
    M = torch.abs(x) < eps
    x[M] = 0
    return x


class Concatenate(torch.nn.Module):
    def __init__(self,irreps_in):
        super().__init__()
        self.irreps_in = irreps_in
        irreps_sorted, J, I = irreps_in.sort()
        I = torch.tensor(I)
        J = torch.tensor(J)
        self.irreps_out = irreps_sorted.simplify()
        idx_conversion = torch.empty(irreps_in.dim,dtype=torch.int64)
        S = torch.empty(len(I),dtype=torch.int64)
        for i, (mul,ir) in enumerate(irreps_in):
            li = 2*ir.l+1
            S[i] = li*mul
        idx_cum = torch.zeros((len(I)+1),dtype=torch.int64)
        idx_cum[1:] = torch.cumsum(S,dim=0)
        ii0 = 0
        for i,Ii in enumerate(I):
            idx_conversion[ii0:ii0+S[Ii]] = torch.arange(idx_cum[Ii],idx_cum[Ii]+S[Ii])
            ii0 += S[Ii]
        idx_conversion_rev = torch.argsort(idx_conversion)
        self.register_buffer("idx_conversion", idx_conversion)
        self.register_buffer("idx_conversion_rev", idx_conversion_rev)
        return

    def forward(self, x,dim):
        x = torch.cat(x,dim)
        x = torch.index_select(x, dim, self.idx_conversion)
        return x

    def reverse_idx(self,x,dim):
        x = torch.index_select(x, dim, self.idx_conversion_rev)
        return x





if __name__ == '__main__':
    #
    # irreps = o3.Irreps("2x0e+1x1e")
    # catfnc = Concatenate(3*irreps)
    #
    # x = irreps.randn(5, -1, normalization='norm')
    # y = catfnc([x,x,x],dim=1)
    # print('done')
    a = torch.tensor([0,2,4,1,3])
    b = torch.argsort(a)
    # b = torch.arange(5)
    # c = b[a]
    # d = a[b]
    print('done')