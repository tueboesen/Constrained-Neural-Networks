"""model with self-interactions and gates
Exact equivariance to :math:`E(3)`
version of february 2021
"""
import torch
import torch.nn
import torch.nn as nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr, Activation, Extract
import torch.nn.functional as F
from torch_scatter import scatter

from src.EQ_operations import SelfInteraction, Convolution
from src.constraints import MomentumConstraints
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
        irreps_inout,
        irreps_hidden,
        layers,
        max_radius,
        number_of_basis,
        radial_neurons,
        num_neighbors,
        num_nodes,
        embed_dim,
        max_atom_types,
        constraints=None,
        constrain_all_layers=True,
        PU=None,
        particles_pr_node = 1,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.irreps_in = o3.Irreps(irreps_inout)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_inout)
        self.irreps_edge_attr = o3.Irreps(irreps_inout)
        self.constraints = constraints
        self.constrain_all_layers = constrain_all_layers
        self.PU = PU
        self.particles_pr_nodes = particles_pr_node
        if self.num_neighbors < 0:
            self.automatic_neighbors = True
        else:
            self.automatic_neighbors = False
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }
        self.node_embedder = torch.nn.Embedding(max_atom_types,embed_dim)
        self.irreps_node_attr = o3.Irreps("{:}x0e".format(embed_dim))

        self.self_interaction = torch.nn.ModuleList()
        irreps = self.irreps_hidden
        self.self_interaction.append(SelfInteraction(irreps,irreps))
        for _ in range(1,layers):
            self.self_interaction.append(SelfInteraction(self.irreps_hidden,self.irreps_hidden))
        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.h = torch.nn.Parameter(torch.ones(layers)*1e-2)
        # self.h = torch.ones(layers)*1e-2
        self.mix = torch.nn.Parameter(torch.ones(layers)*0.5)
        radial_neurons_prepend = [2*particles_pr_node*self.number_of_basis] + radial_neurons
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
                radial_neurons_prepend
            )
            # self.norms.append(TvNorm(gate.irreps_in))
            irreps = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)
        return

    def get_edge_info(self,x,edge_src,edge_dst):
        nvec = x.shape[-1] // 3
        edge_features =[]
        edge_attrs = []
        for i in range(nvec):
            edge_vec = x[:,i*3:i*3+3][edge_src] - x[:,i*3:i*3+3][edge_dst]
            edge_sh = o3.spherical_harmonics(o3.Irreps("1x1o"), edge_vec, True, normalization='component')
            edge_length = edge_vec.norm(dim=1)
            edge_feature = soft_one_hot_linspace(
                x=edge_length,
                start=0.0,
                end=self.max_radius,
                number=self.number_of_basis,
                basis='bessel',
                cutoff=False
            ).mul(self.number_of_basis ** 0.5)
            edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh
            edge_features.append(edge_feature)
            edge_attrs.append(edge_attr)
        edge_attrs = torch.cat(edge_attrs, dim=-1)
        edge_features = torch.cat(edge_features, dim=-1)
        return edge_features, edge_attrs

    def forward(self, x, batch, node_attr, edge_src, edge_dst,tmp=None,tmp2=None) -> torch.Tensor:
        if self.automatic_neighbors:
            self.num_neighbors = edge_dst.shape[0]/x.shape[0]

        node_attr_embedded = self.node_embedder(node_attr.to(dtype=torch.int64)).squeeze()
        self.PU.make_matrix_semi_unitary()
        y = self.PU.uplift(x)
        y_old = y

        for i,(conv,gate) in enumerate(zip(self.convolutions,self.gates)):
            edge_features,edge_attr = self.get_edge_info(x,edge_src,edge_dst)

            y_new = conv(y.clone(), node_attr_embedded, edge_src, edge_dst, edge_attr, edge_features,self.num_neighbors)
            y_new = gate(y_new)
            y_new2 = self.self_interaction[i](y.clone())
            tmp = y.clone()
            y = 2*y - y_old + self.h[i]**2 *(self.mix[i]*y_new + (self.mix[i]-1) * y_new2)
            y_old = tmp
            if self.constraints is not None and self.constrain_all_layers is True:
                data = self.constraints({'y':y,'batch':batch})
                y = data['y']
            x = self.PU.project(y)
        if self.constraints is not None and self.constrain_all_layers is False:
            data = self.constraints({'x':x,'batch':batch})
            x = data['x']
        return x
