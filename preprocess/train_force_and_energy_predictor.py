"""model with self-interactions and gates
Exact equivariance to :math:`E(3)`
version of february 2021
"""
import math
import time
from typing import Dict, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr, Activation
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from torch.autograd import grad
import torch.nn.functional as F

from src.EQ_operations import SelfInteraction, Convolution, TvNorm


def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class NequIP(torch.nn.Module):
    r"""equivariant neural network
    Parameters
    ----------
    irreps_in : `Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features
    irreps_hidden : `Irreps`
        representation of the hidden features
    irreps_out : `Irreps`
        representation of the output features
    irreps_node_attr : `Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes
    irreps_edge_attr : `Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials
    layers : int
        number of gates (non linearities)
    max_radius : float
        maximum radius for the convolution
    number_of_basis : int
        number of basis on which the edge length are projected
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes at a distance ``max_radius``
    num_nodes : float
        typical number of nodes in a graph
    """
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
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

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
        irreps = self.irreps_in if self.input_has_node_in else o3.Irreps("1x0e")
        self.self_interaction.append(SelfInteraction(irreps,irreps))
        for _ in range(1,layers):
            self.self_interaction.append(SelfInteraction(self.irreps_hidden,self.irreps_hidden))
        # n_0e = o3.Irreps(self.irreps_hidden).count('0e')
        second_to_last_irrep = o3.Irreps("16x0e")
        last_irrep = o3.Irreps("1x0e")
        self.self_interaction.append(SelfInteraction(self.irreps_hidden,second_to_last_irrep))
        self.self_interaction.append(SelfInteraction(second_to_last_irrep,last_irrep))
        self.activation = Activation("16x0e", [torch.nn.functional.silu])
        # n_1e = o3.Irreps(self.irreps_hidden).count('0e')
        # n_1o = o3.Irreps(self.irreps_hidden).count('1o')


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
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                radial_neurons,
                num_neighbors
            )
            self.norms.append(TvNorm(gate.irreps_in))
            irreps = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)
        return

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """evaluate the network
        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        h = 0.1
        edge_index = radius_graph(data['pos'], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='bessel',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and 'x' in data:
            assert self.irreps_in is not None
            x = data['x']
        else:
            assert self.irreps_in is None
            x = data['pos'].new_ones((data['pos'].shape[0], 1))

        if self.input_has_node_attr and 'z' in data:
            z = data['z']
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data['pos'].new_ones((data['pos'].shape[0], 1))

        # scalar_z = self.ext_z(z)
        edge_features = edge_length_embedded

        # print(f'mean={x.pow(2).mean():2.2f}')
        z = self.node_embedder(z.to(dtype=torch.int64)).squeeze()
        # print(f'mean={x.pow(2).mean():2.2f}')
        # x = self.self_interaction[0](x)
        # print(f'mean={x.pow(2).mean():2.2f}')

        for i,(conv,norm,gate) in enumerate(zip(self.convolutions,self.norms,self.gates)):
            y = conv(x, z, edge_src, edge_dst, edge_attr, edge_features)
            # y = norm(y)
            # print(f'mean={x.pow(2).mean():2.2f}')
            y = gate(y)
            # print(f'mean={x.pow(2).mean():2.2f}')
            if y.shape == x.shape:
                y = self.self_interaction[i](y)
                x = x + h*y
            else:
                x = y
            # print(f'mean(abs(x))={torch.abs(x).mean():2.2f},norm={x.norm():2.2f}')
        x = self.self_interaction[-2](x,normalize_variance=False)
        x = self.activation(x)
        x = self.self_interaction[-1](x,normalize_variance=False)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x

class Dataset_ForceEnergy(torch.utils.data.Dataset):
    def __init__(self, R, F, E, z):
        self.R = R
        self.F = F
        self.E = E
        self.z = z
        return

    def __getitem__(self, index):
        R = self.R[index]
        F = self.F[index]
        E = self.E[index]
        z = self.z[:,None]
        return R, F, E, z

    def __len__(self):
        return len(self.R)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'

def use_model_eq(model,dataloader,train,max_samples,optimizer,batch_size=1):
    aloss = 0.0
    aloss_E = 0.0
    aloss_F = 0.0
    Fps = 0.0
    Fts = 0.0
    MAE = 0.0
    t_dataload = 0.0
    t_prepare = 0.0
    t_model = 0.0
    t_backprop = 0.0
    if train:
        model.train()
    else:
        model.eval()
    t3 = time.time()
    for i, (Ri, Fi, Ei, zi) in enumerate(dataloader):
        nb = Ri.shape[0]
        t0 = time.time()
        Ri.requires_grad_(True)
        Ri_vec = Ri.reshape(-1,Ri.shape[-1])
        zi_vec = zi.reshape(-1,zi.shape[-1])
        batch = torch.arange(Ri.shape[0]).repeat_interleave(Ri.shape[1]).to(device=Ri.device)

        data = {
                'batch': batch,
                'pos': Ri_vec,
                'z': zi_vec
                }

        optimizer.zero_grad()
        t1 = time.time()
        E_pred = model(data)
        E_pred_tot = torch.sum(E_pred)
        t2 = time.time()

        if train:
            F_pred = -grad(E_pred_tot, Ri, create_graph=True)[0].requires_grad_(True)
        else:
            F_pred = -grad(E_pred_tot, Ri, create_graph=False)[0]
        loss_F = F.mse_loss(F_pred, Fi) / nb
        loss_E = F.mse_loss(E_pred, Ei) / nb
        loss = loss_E + loss_F
        Fps += torch.mean(torch.sqrt(torch.sum(F_pred.detach() ** 2, dim=1)))
        Fts += torch.mean(torch.sqrt(torch.sum(Fi ** 2, dim=1)))
        MAEi = torch.mean(torch.abs(F_pred - Fi)).detach()
        MAE += MAEi
        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        aloss_F += loss_F.detach()
        aloss_E += loss_E.detach()
        t_dataload += t0 - t3
        t3 = time.time()
        t_prepare += t1 - t0
        t_model += t2 - t1
        t_backprop += t3 - t2
        if (i+1)*batch_size >= max_samples:
            break
    aloss /= (i+1)
    aloss_E /= (i+1)
    aloss_F /= (i+1)
    MAE /= (i+1)
    Fps /= (i+1)
    Fts /= (i+1)
    t_dataload /= (i+1)
    t_prepare /= (i+1)
    t_model /= (i+1)
    t_backprop /= (i+1)
    return aloss,aloss_F, aloss_E



if __name__ == '__main__':
    n_train = 1000
    n_val = 1000
    batch_size = 50
    model_name = './../results/force_energy_model.pt'


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    print_distograms = False
    print_3d_structures = False
    use_mean_map = False
    # load training data
    # data = np.load('../../../data/MD/MD17/aspirin_dft.npz')
    data = np.load('../../../data/MD/water_jones/water.npz')
    E = data['PE']
    Force = data['F']
    R = data['R']
    epochs_for_lr_adjustment = 50
    z = torch.from_numpy(data['z']).to(dtype=torch.float32, device=device)
    ndata = E.shape[0]
    natoms = z.shape[0]
    R_mean = None

    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))


    ndata_rand = 0 + np.arange(ndata)
    np.random.shuffle(ndata_rand)


    E_train = torch.from_numpy(E[ndata_rand[:n_train]]).to(dtype=torch.float32, device=device)
    F_train = torch.from_numpy(Force[ndata_rand[:n_train]]).to(dtype=torch.float32, device=device)
    R_train = torch.from_numpy(R[ndata_rand[:n_train]]).to(dtype=torch.float32, device=device)

    E_val = torch.from_numpy(E[ndata_rand[n_train:n_train+n_val]]).to(dtype=torch.float32, device=device)
    F_val = torch.from_numpy(Force[ndata_rand[n_train:n_train+n_val]]).to(dtype=torch.float32, device=device)
    R_val = torch.from_numpy(R[ndata_rand[n_train:n_train+n_val]]).to(dtype=torch.float32, device=device)

    E_test = torch.from_numpy(E[ndata_rand[n_train+n_val:]]).to(dtype=torch.float32, device=device)
    F_test = torch.from_numpy(Force[ndata_rand[n_train+n_val:]]).to(dtype=torch.float32, device=device)
    R_test = torch.from_numpy(R[ndata_rand[n_train+n_val:]]).to(dtype=torch.float32, device=device)


    dataset_train = Dataset_ForceEnergy(R_train, F_train, E_train, z)
    dataset_val = Dataset_ForceEnergy(R_val, F_val, E_val, z)
    dataset_test = Dataset_ForceEnergy(R_test, F_test, E_test, z)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=False)

    # Setup the network and its parameters

    irreps_in = None    #o3.Irreps("0x0e")
    irreps_hidden = o3.Irreps("10x0e+10x0o+5x1e+5x1o")
    irreps_out = o3.Irreps("1x0e")
    irreps_node_attr = o3.Irreps("1x0e")
    irreps_edge_attr = o3.Irreps("1x0e+1x1o")
    layers = 6
    max_radius = 5
    number_of_basis = 8
    radial_neurons = [8,16]
    num_neighbors = 15
    num_nodes = natoms
    model = NequIP(irreps_in=irreps_in, irreps_hidden=irreps_hidden, irreps_out=irreps_out, irreps_node_attr=irreps_node_attr, irreps_edge_attr=irreps_edge_attr, layers=layers, max_radius=max_radius,
                    number_of_basis=number_of_basis, radial_neurons=radial_neurons, num_neighbors=num_neighbors, num_nodes=num_nodes)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ', total_params)


    #### Start Training ####
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    alossBest = 1e6
    epochs = 1000

    bestModel = model
    hist = torch.zeros(epochs)
    eps = 1e-10
    nprnt = 1
    nprnt2 = min(nprnt, n_train)
    t0 = time.time()
    aloss_best = 1e6
    epochs_since_best = 0
    for epoch in range(epochs):
        t1 = time.time()
        aloss_t,aloss_F_t,aloss_E_t = use_model_eq(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=batch_size)
        t2 = time.time()
        aloss_v,aloss_F_v,aloss_E_v  = use_model_eq(model, dataloader_val, train=False, max_samples=10, optimizer=optimizer, batch_size=batch_size)
        t3 = time.time()

        if aloss_v < aloss_best:
            aloss_best = aloss_v
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= epochs_for_lr_adjustment:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.8
                    lr = g['lr']
                epochs_since_best = 0

        # print(f' t_dataloader(train): {t_dataload_t:.3f}s  t_dataloader(val): {t_dataload_v:.3f}s  t_prepare(train): {t_prepare_t:.3f}s  t_prepare(val): {t_prepare_v:.3f}s  t_model(train): {t_model_t:.3f}s  t_model(val): {t_model_v:.3f}s  t_backprop(train): {t_backprop_t:.3f}s  t_backprop(val): {t_backprop_v:.3f}s')
        print(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss(val): {aloss_v:.2e}  Loss_F(train): {aloss_F_t:.2e}  Loss_F(val): {aloss_F_v:.2e}  Loss_E(train): {aloss_E_t:.2e}  LossE_(val): {aloss_E_v:.2e}  loss(best): {aloss_best:.2e}  Time(train): {t2-t1:.1f}s  Time(val): {t3-t2:.1f}s  Lr: {lr:2.2e} ')

    # Specify a path
    torch.save(model.state_dict(), f"{model_name}")
