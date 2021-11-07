from e3nn import o3
import torch
import numpy as np
from src.constraints import load_constraints
from src.dataloader import load_data
from src.network_mim import network_simple
from src.network_e3 import constrained_network
from src.project_uplift import ProjectUpliftEQ, ProjectUplift
from src.utils import atomic_masses, convert_snapshots_to_future_state_dataset

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    model_file = './../pretrained_networks/force_energy_model.pt'
    data = './../../../data/MD/argon/argon.npz'
    network_type = 'EQ'
    nskip = 9999
    ntrain = 10000
    nval = 0
    use_val = False
    use_test = False
    batch_size = 1
    n_ensemples = 10
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if network_type.lower() == 'eq':
        cn = {
            'irreps_inout': o3.Irreps("2x1o"),
            'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
            # 'irreps_node_attr': o3.Irreps("1x0e"),
            # 'irreps_edge_attr': o3.Irreps("{:}x1o".format(args.n_input_samples)),
            'irreps_edge_attr': o3.Irreps("2x1o"),
            'layers': 4,
            'max_radius': 15,
            'number_of_basis': 8,
            'embed_dim': 8,
            'max_atom_types': 20,
            'radial_neurons': [16, 16],
            'num_neighbors': -1,
            'constraints': '',
        }
    elif network_type.lower() == 'mim':
        cn = {
            'node_dim_in': 6,
            'node_attr_dim_in': 1,
            'node_dim_latent': 60,
            'nlayers': 6,
            'max_radius': 15,
            'constraints': 'EP',
        }

    data = np.load(data)

    R = torch.from_numpy(data['R']).to(device=device,dtype=torch.get_default_dtype())
    z = torch.from_numpy(data['z']).to(device=device)
    if 'V' in data.files:
        V = torch.from_numpy(data['V']).to(device=device,dtype=torch.get_default_dtype())
    masses = atomic_masses(z)

    #We rescale the data
    Rscale = torch.sqrt(R.pow(2).mean())
    Vscale = torch.sqrt(V.pow(2).mean())
    R /= Rscale
    V /= Vscale

    # Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    # Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)

    if network_type == 'EQ':
        PU = ProjectUpliftEQ(cn['irreps_inout'], cn['irreps_hidden'])
    elif network_type == 'mim':
        PU = ProjectUplift(cn['node_dim_in'], cn['node_dim_latent'])

    constraints = None

    if network_type == 'EQ':
        model = constrained_network(irreps_inout=cn['irreps_inout'], irreps_hidden=cn['irreps_hidden'], layers=cn['layers'],
                                    max_radius=cn['max_radius'],
                                    number_of_basis=cn['number_of_basis'], radial_neurons=cn['radial_neurons'], num_neighbors=cn['num_neighbors'],
                                    num_nodes=ds.Rin.shape[1], embed_dim=cn['embed_dim'], max_atom_types=cn['max_atom_types'], constraints=constraints, PU=PU)
    elif network_type == 'mim':
        model = network_simple(cn['node_dim_in'], cn['node_attr_dim_in'], cn['node_dim_latent'], cn['nlayers'], PU=PU, constraints=constraints)

    n,natoms,_ = R.shape
    nsteps = np.floor((n-n_ensemples-1)/(nskip+1))

    Rhist = torch.zeros((n_ensemples,nsteps,natoms,3))
    Vhist = torch.zeros((n_ensemples,nsteps,natoms,3))
    for i in range(n_ensemples):
        Rin = R[i]
        Vin = V[i]
        for j in range(nsteps):
            Rout,Vout = model(Rin,Vin)
            Rhist[i,j] = Rout
            Vhist[i,j] = Vout

            Rin = Rout
            Vin = Vout

    Rpred_mean = Rhist.mean(dim=0)
    Rpred_std = Rhist.std(dim=0)
    Vpred_mean = Vhist.mean(dim=0)
    Vpred_std = Vhist.std(dim=0)

