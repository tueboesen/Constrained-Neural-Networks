"""models with self-interactions and gates
Exact equivariance to :math:`E(3)`
version of february 2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate

from src.EQ_operations import SelfInteraction, Convolution, tp_path_exists
from src.project_uplift import ProjectUpliftEQ
from src.utils import smooth_cutoff, atomic_masses
from src.vizualization import plot_water


class neural_network_equivariant(torch.nn.Module):
    """
    An equivariant neural network built in e3nn inspired by their MD simulating neural network paper (the network has not been published at this time)
    """
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
        con_fnc=None,
        con_type=None,
        particles_pr_node=1,
        discretization_method='leapfrog',
        penalty_strength=0,
        regularization_strength=1
    ) -> None:
        super().__init__()
        irreps_inout = o3.Irreps(irreps_inout)
        irreps_hidden = o3.Irreps(irreps_hidden)

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.irreps_in = o3.Irreps(irreps_inout)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_inout)
        self.irreps_edge_attr = o3.Irreps(irreps_inout)
        self.con_fnc = con_fnc
        self.con_type = con_type
        self.PU = ProjectUpliftEQ(irreps_inout,irreps_hidden)
        self.project = self.PU.project
        self.uplift = self.PU.uplift
        self.discretization_method = discretization_method
        self.penalty_strength = penalty_strength
        self.regularization_strength = regularization_strength
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
        self.mix = torch.nn.Parameter(torch.ones(layers)*0.75)
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
        self.params = nn.ModuleDict({
            "base": nn.ModuleList([self.convolutions, self.gates, self.self_interaction]),
            "h": nn.ParameterList([self.h,self.mix]),
            "close": nn.ModuleList([self.PU.lin])
        })
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

    def forward_propagation(self,y,node_attr_embedded,edge_src,edge_dst,edge_attr,edge_features,num_neighbors,i):
        y_new = self.convolutions[i](y.clone(), node_attr_embedded, edge_src, edge_dst, edge_attr, edge_features, num_neighbors)
        y_new = self.gates[i](y_new)
        y_new2 = self.self_interaction[i](y.clone())
        y_out = min(self.mix[i]**2,1) * y_new + (1 - min(self.mix[i]**2,1)) * y_new2
        return y_out

    def forward(self, x, batch, node_attr, edge_src, edge_dst,wstatic=None,weight=1) -> torch.Tensor:

        if self.automatic_neighbors:
            self.num_neighbors = edge_dst.shape[0]/x.shape[0]

        node_attr_embedded = self.node_embedder(torch.min(node_attr,dim=-1)[0].to(dtype=torch.int64)).squeeze()
        y = self.uplift(x)

        ndimx = x.shape[-1]
        ndimy = y.shape[-1]
        y_old = y
        reg = torch.tensor(0.0)

        for i,(conv,gate) in enumerate(zip(self.convolutions,self.gates)):
            dt = max(min(self.h[i]**2,0.1),1e-4)
            edge_features,edge_attr = self.get_edge_info(x,edge_src,edge_dst)

            if self.penalty_strength > 0:
                if self.discretization_method == 'rk4':
                    q1 = self.con_fnc.constraint_penalty(y.view(batch.max() + 1, -1, ndimy), self.project, self.uplift, weight)
                    q2 = self.con_fnc.constraint_penalty(y.view(batch.max() + 1, -1, ndimy) + q1 / 2 * dt, self.project, self.uplift, weight)
                    q3 = self.con_fnc.constraint_penalty(y.view(batch.max() + 1, -1, ndimy) + q2 / 2 * dt, self.project, self.uplift, weight)
                    q4 = self.con_fnc.constraint_penalty(y.view(batch.max() + 1, -1, ndimy) + q3 * dt, self.project, self.uplift, weight)
                    dy = (q1 + 2 * q2 + 2 * q3 + q4) / 6
                else:
                    dy = self.con_fnc.constrain_stabilization(y.view(batch.max() + 1, -1, ndimy), self.project, self.uplift, weight)
                dy = dy.view(-1, ndimy)
            else:
                dy = 0
            if self.discretization_method == 'leapfrog':
                y_new = self.forward_propagation(y.clone(), node_attr_embedded, edge_src, edge_dst, edge_attr, edge_features, self.num_neighbors, i)
                tmp = y.clone()
                y = 2*y - y_old + dt *(y_new + self.penalty_strength*dy)
                y_old = tmp
            elif self.discretization_method == 'euler':
                y_new = self.forward_propagation(y.clone(), node_attr_embedded, edge_src, edge_dst, edge_attr, edge_features, self.num_neighbors, i)
                y = y + dt * (y_new - self.penalty_strength * dy)
            elif self.discretization_method == 'rk4':
                k1 = self.forward_propagation(y.clone(), node_attr_embedded, edge_src, edge_dst, edge_attr, edge_features, self.num_neighbors, i)
                k2 = self.forward_propagation(y.clone() + k1 * dt / 2, node_attr_embedded, edge_src, edge_dst, edge_attr, edge_features, self.num_neighbors, i)
                k3 = self.forward_propagation(y.clone() + k2 * dt / 2, node_attr_embedded, edge_src, edge_dst, edge_attr, edge_features, self.num_neighbors, i)
                k4 = self.forward_propagation(y.clone() + k3 * dt, node_attr_embedded, edge_src, edge_dst, edge_attr, edge_features, self.num_neighbors, i)
                y_new = (k1 + 2*k2 + 2*k3 + k4)/6
                y = y + dt * (y_new - self.penalty_strength*dy)
            else:
                raise NotImplementedError(f"Discretization method {self.discretization_method} not implemented.")


            if self.con_fnc is not None and self.con_type == 'high':
                y, _, regi = self.con_fnc(y.view(batch.max() + 1,-1,ndimy),self.project,self.uplift,weight)
                y = y.view(-1,ndimy)
                reg = reg + regi


            x = self.project(y)
        if self.con_fnc is not None and self.con_type == 'low':
            x, _, reg = self.con_fnc(x.view(batch.max() + 1,-1,ndimx),weight=weight)
            x = x.view(-1,ndimx)

        if self.con_fnc is not None:
            _, cv_mean,cv_max = self.con_fnc.constraint_violation(x.view(batch.max() + 1, -1, ndimx))
        else:
            cv_mean,cv_max = torch.tensor(-1.0),  torch.tensor(-1.0)

        return x, cv_mean, cv_max, reg * self.regularization_strength

    def get_water_viz(self, y_new, y_old, batch):
        x_new = self.PU.project(y_new)
        ndim = x_new.shape[-1] // 2
        nb = batch.max() + 1

        r_new = x_new[:, 0:ndim].view(nb, -1, ndim).detach().cpu().numpy()
        v_new = x_new[:, ndim:].view(nb, -1, ndim).detach().cpu().numpy()

        x_old = self.PU.project(y_old)
        r_old = x_old[:, 0:ndim].view(nb, -1, ndim).detach().cpu().numpy()
        v_old = x_old[:, ndim:].view(nb, -1, ndim).detach().cpu().numpy()
        plot_water(r_new,v_new,r_old,v_old)

    def get_water_viz_low(self, x_new, x_old, x_org, batch):
        ndim = x_new.shape[-1] // 2
        nb = batch.max() + 1

        r_new = x_new[:, 0:ndim].view(nb, -1, ndim).detach().cpu().numpy()
        v_new = x_new[:, ndim:].view(nb, -1, ndim).detach().cpu().numpy()

        r_old = x_old[:, 0:ndim].view(nb, -1, ndim).detach().cpu().numpy()
        v_old = x_old[:, ndim:].view(nb, -1, ndim).detach().cpu().numpy()

        r_org = x_org[:, 0:ndim].view(nb, -1, ndim).detach().cpu().numpy()
        v_org = x_org[:, ndim:].view(nb, -1, ndim).detach().cpu().numpy()

        plot_water(r_new,v_new,r_old,v_old,r_org,v_org)

