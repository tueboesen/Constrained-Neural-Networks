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
    """
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

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

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars,num_neighbors) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(num_neighbors**0.5)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out


class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            return
        def forward(self, input):
            return input

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

class ProjectUpliftEQ(torch.nn.Module):
    '''
    Note that this will only work if the low dimension is purely a vector space for now
    '''
    def __init__(self,irreps_low,irreps_high):
        super(ProjectUpliftEQ, self).__init__()
        self.irreps_low = irreps_low
        self.irreps_high = irreps_high
        self.n_vec_low = ExtractIr(irreps_low, '1o').irreps_out.num_irreps
        self.n_vec_high = ExtractIr(irreps_high, '1o').irreps_out.num_irreps

        w = torch.empty((self.n_vec_low, self.n_vec_high))
        torch.nn.init.xavier_normal_(w,gain=1/math.sqrt(self.n_vec_low)) # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
        # self.M = torch.nn.Parameter(w)
        self.register_buffer("M", w)

        return

    def make_matrix_semi_unitary(self, debug=False):
        M = self.M.clone()
        I = torch.eye(M.shape[-2],device=M.device)
        for i in range(100):
            M = M - 0.5 * (M @ M.t() - I) @ M

        if debug:
            pre_error = torch.norm(I - self.M @ self.M.t())
            post_error = torch.norm(I - M @ M.t())
            print(f"Deviation from unitary before: {pre_error:2.2e}, deviation from unitary after: {post_error:2.2e}")
        self.Mu = M
        return

    def uplift(self,x):
        irreps_in = self.irreps_low
        irreps_out = self.irreps_high
        nd_out = irreps_out.dim
        idx0 = 0
        for mul, ir in irreps_in:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                x_vec = x[:,idx0:idx1]
                break
            idx0 = idx1
        x2 = x_vec.reshape(x_vec.shape[0], -1, 3).transpose(1, 2)
        y2 = x2 @ self.Mu
        y_vec = y2.transpose(1,2).reshape(x.shape[0],-1)
        y = torch.zeros((x_vec.shape[0],nd_out),dtype=x.dtype,device=x.device)
        idx0=0
        for mul, ir in irreps_out:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                y[:,idx0:idx1] = y_vec
                break
            idx0 = idx1
        return y

    def project(self,y):
        irreps_in = self.irreps_high
        irreps_out = self.irreps_low
        ir_vec = ExtractIr(irreps_in,'1o')
        idx0 = 0
        for mul, ir in irreps_in:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                y_vec = y[:,idx0:idx1]
                break
            idx0 = idx1
        y2 = y_vec.reshape(y.shape[0],-1,3).transpose(1,2)
        x2 = y2 @ self.Mu.t()
        x = x2.transpose(1,2).reshape(y.shape[0],-1)
        return x


class ProjectUplift(torch.nn.Module):
    '''
    Note that this will only work if the low dimension is purely a vector space for now
    '''
    def __init__(self,low_dim,high_dim):
        super(ProjectUplift, self).__init__()
        self.low_dim = low_dim
        self.high_dim = high_dim
        w = torch.empty((self.low_dim, self.high_dim))
        torch.nn.init.xavier_normal_(w,gain=1/math.sqrt(self.low_dim)) # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
        # self.M = torch.nn.Parameter(w)
        self.register_buffer("M", w)

        return

    def make_matrix_semi_unitary(self, debug=False):
        M = self.M.clone()
        I = torch.eye(M.shape[-2],device=M.device)
        for i in range(100):
            M = M - 0.5 * (M @ M.t() - I) @ M

        if debug:
            pre_error = torch.norm(I - self.M @ self.M.t())
            post_error = torch.norm(I - M @ M.t())
            print(f"Deviation from unitary before: {pre_error:2.2e}, deviation from unitary after: {post_error:2.2e}")
        self.Mu = M
        return

    def project(self, y):
        x = y @ self.Mu.t()
        return x

    def uplift(self, x):
        y = x @ self.Mu
        return y


class ProjectUplift_conv(torch.nn.Module):
    '''
    Note that this will only work if the low dimension is purely a vector space for now
    '''
    def __init__(self,irreps_low,irreps_high):
        super(ProjectUplift_conv, self).__init__()
        self.irreps_low = irreps_low
        self.irreps_high = irreps_high
        self.n_vec_low = ExtractIr(irreps_low, '1o').irreps_out.num_irreps
        self.n_vec_high = ExtractIr(irreps_high, '1o').irreps_out.num_irreps

        w = torch.empty((self.n_vec_low, self.n_vec_high,1))
        torch.nn.init.xavier_normal_(w,gain=1/math.sqrt(self.n_vec_low)) # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
        # self.w = torch.nn.Parameter(torch.eye(self.n_vec_low, self.n_vec_high).unsqueeze(-1))
        self.w = torch.nn.Parameter(w)
        return

    def make_matrix_semi_unitary(self, debug=False):
        pass
        return

    def uplift(self,x):
        irreps_in = self.irreps_low
        irreps_out = self.irreps_high
        nd_out = irreps_out.dim
        idx0 = 0
        for mul, ir in irreps_in:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                x_vec = x[:,idx0:idx1]
                break
            idx0 = idx1
        x2 = x_vec.reshape(x_vec.shape[0], -1, 3)
        y2 = F.conv_transpose1d(x2,self.w)
        y_vec = y2.reshape(x.shape[0],-1)
        y = torch.zeros((x_vec.shape[0],nd_out),dtype=x.dtype,device=x.device)
        idx0=0
        for mul, ir in irreps_out:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                y[:,idx0:idx1] = y_vec
                break
            idx0 = idx1
        return y

    def project(self,y):
        irreps_in = self.irreps_high
        irreps_out = self.irreps_low
        ir_vec = ExtractIr(irreps_in,'1o')
        idx0 = 0
        for mul, ir in irreps_in:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                y_vec = y[:,idx0:idx1]
                break
            idx0 = idx1
        y2 = y_vec.reshape(y.shape[0],-1,3)
        x2 = F.conv1d(y2,self.w)
        x = x2.reshape(y.shape[0],-1)
        return x





if __name__ == '__main__':
    import e3nn.o3
    from e3nn.util.test import equivariance_error
    from e3nn.util.test import assert_equivariant
    # torch.set_default_dtype(torch.float64)
    torch.set_default_dtype(torch.float32)

    irreps_in = o3.Irreps("2x1o")
    irreps_out = o3.Irreps("20x0o+20x0e+10x1o+5x1e")

    PU = ProjectUplift(irreps_in,irreps_out)
    PU.make_matrix_semi_unitary()
    assert_equivariant(
        PU.uplift,
        irreps_in=[irreps_in],
        irreps_out=[irreps_out]
    )
    assert_equivariant(
        PU.project,
        irreps_in=[irreps_out],
        irreps_out=[irreps_in]
    )

    SI = SelfInteraction(irreps_in,irreps_out)
    assert_equivariant(
        SI,
        irreps_in=[irreps_in],
        irreps_out=[irreps_out]
    )
    n = 100
    irreps_node = irreps_out
    irreps_node_attr = o3.Irreps("8x0e")
    irreps_edge_attr = o3.Irreps("1x1o")
    irreps_conv_out = o3.Irreps("20x0o+35x0e+10x1o+5x1e")
    radial_neurons = [8, 16]
    num_neighbors = n
    Conv = Convolution(
        irreps_node,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_conv_out,
        radial_neurons,
        num_neighbors
    )

    # n = 4
    node_input = irreps_node.randn(n,-1)
    node_attr = irreps_node_attr.randn(n,-1)
    edge_src = torch.arange(n).repeat_interleave(n)
    edge_dst = torch.arange(n).repeat(n)
    edge_attr = irreps_edge_attr.randn(edge_dst.shape[0],-1)
    edge_features = irreps_node_attr.randn(edge_dst.shape[0],-1)
    #
    out = Conv(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_features)
    rot = o3.rand_matrix()
    Dnode_input = irreps_node.D_from_matrix(rot)
    Dnode_attr = irreps_node_attr.D_from_matrix(rot)
    Dedge_attr = irreps_edge_attr.D_from_matrix(rot)
    Dedge_features = irreps_node_attr.D_from_matrix(rot)
    Dout = irreps_conv_out.D_from_matrix(rot)
    #
    node_input_rot = node_input@Dnode_input
    node_attr_rot = node_attr@Dnode_attr
    edge_attr_rot = edge_attr@Dedge_attr
    edge_features_rot = edge_features@Dedge_features
    #
    out_pre_rot = Conv(node_input_rot, node_attr_rot, edge_src, edge_dst, edge_attr_rot, edge_features_rot)
    out_rot = out @ Dout
    assert torch.allclose(out_pre_rot, out_rot, rtol=1e-4, atol=1e-4)

    irreps_scalars = o3.Irreps("20x0o+20x0e")
    irreps_gates = o3.Irreps("10x0e+5x0e")
    irreps_gated = o3.Irreps("10x1o+5x1e")
    act = [torch.tanh, torch.nn.functional.silu]
    act_gates = [torch.sigmoid, torch.sigmoid]
    gate = Gate(
        irreps_scalars, act,  # scalar
        irreps_gates, act_gates,  # gates (scalars)
        irreps_gated  # gated tensors
    )
    # Now lets test a small network that does both uplift conv and projection
    SI2 = SelfInteraction(irreps_out, irreps_out)

    masses = torch.ones(n)
    con = MomentumConstraints(masses, project=PU.project, uplift=PU.uplift)

    def f(x,edge_src,edge_dst,con,edge_feature_ref,edge_attr_ref):
        max_radius = 2
        number_of_basis = 8
        edge_vec = x[:,-3:][edge_src] - x[:,-3:][edge_dst]
        edge_sh = o3.spherical_harmonics(o3.Irreps("1x1o"), edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_features = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=max_radius,
            number=number_of_basis,
            basis='bessel',
            cutoff=False
        ).mul(number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / max_radius)[:, None] * edge_sh

        y = PU.uplift(x)
        for i in range(5):
            y_new = Conv(y.clone(), node_attr, edge_src, edge_dst, edge_attr, edge_features)
            y_new = gate(y_new)
            y = y + 0.1 * y_new
            y = con(y,torch.zeros(n,dtype=torch.int64))
        xout = PU.project(y)
        return xout

    x = irreps_in.randn(n,-1)
    xmean = x.pow(2).mean()
    # x = x / torch.sqrt(xmean)

    Dx = irreps_in.D_from_matrix(rot)
    Dedge_attr = irreps_edge_attr.D_from_matrix(rot)
    max_radius = 2
    batch = torch.zeros(n,dtype=torch.int64)
    edge_index = radius_graph(x[:,-3:], max_radius, batch)
    edge_src = edge_index[0]
    edge_dst = edge_index[1]

    max_radius = 2
    number_of_basis = 8
    edge_vec = x[:, -3:][edge_src] - x[:, -3:][edge_dst]
    edge_sh = o3.spherical_harmonics(o3.Irreps("1x1o"), edge_vec, True, normalization='component')
    edge_length = edge_vec.norm(dim=1)
    edge_features = soft_one_hot_linspace(
        x=edge_length,
        start=0.0,
        end=max_radius,
        number=number_of_basis,
        basis='bessel',
        cutoff=False
    ).mul(number_of_basis ** 0.5)
    edge_attr = smooth_cutoff(edge_length / max_radius)[:, None] * edge_sh

    xout = f(x,edge_src,edge_dst,con,edge_features,edge_attr)

    ne = edge_dst.shape[0]
    nes = torch.randperm(ne)

    edge_src_per = edge_src[nes]
    edge_dst_per = edge_dst[nes]
    edge_attr_per = edge_attr[nes]
    edge_features_per = edge_features[nes]

    xout_per = f(x, edge_src_per, edge_dst_per, con, edge_features_per,edge_attr_per)

    assert torch.allclose(xout, xout_per, rtol=1e-4, atol=1e-4)

    xout_rot = xout @ Dx

    x_rot = x @ Dx
    edge_index_rot = radius_graph(x_rot[:,-3:], max_radius, batch)
    edge_src_rot = edge_index_rot[0]
    edge_dst_rot = edge_index_rot[1]

    a = torch.cat([edge_dst[:,None],edge_src[:,None]],dim=1)
    a2 = torch.cat([edge_dst_rot[:, None], edge_src_rot[:, None]], dim=1)

    an = a.numpy()
    a2n = a2.numpy()
    ind = np.lexsort((an[:, 1], an[:, 0]))
    an_sorted = an[ind]
    ind = np.lexsort((a2n[:, 1], a2n[:, 0]))
    a2n_sorted = a2n[ind]

    edge_attr_rot = edge_attr @ Dedge_attr
    edge_features_rot = edge_features

    xout_rot_pre = f(x_rot,edge_src_per,edge_dst_per,con,edge_features_rot,edge_attr_rot)
    print("torch.get_default_dtype()={:} norm_eq_error={:}".format(torch.get_default_dtype(),(xout_rot_pre-xout_rot).norm()))
    assert torch.allclose(xout_rot_pre, xout_rot, rtol=1e-4, atol=1e-4)

    print("done")
    # equivariance_error(
    #     Conv,
    #     args_in=
    #     irreps_in=[irreps_node, irreps_node_attr, irreps_edge_attr],
    #     irreps_out=[irreps_out]
    # )