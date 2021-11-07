import argparse
import os

import torch
from e3nn import o3

from src.batch_jobs import job_planner, job_runner

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 10
    args.n_val = 10
    args.batch_size = 1
    args.nskip = 9999
    args.epochs_for_lr_adjustment = 10
    args.lr_adjustment = 0.8
    args.use_val = False
    args.use_test = False
    args.perform_endstep_MD_propagation = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = [1234,1235,1236,1237,1238]
    args.use_same_data = True
    args.epochs = 300
    args.network_type = 'mim'
    args.loss = 'mim'
    # args.train_idx = None
    args.data = './../../../data/casp11/casp11_sel.npz'
    args.data_type = 'protein'
    args.con = ['chain','triangle','chaintriangle']
    # args.con = ['triangle','chaintriangle']
    # args.con_type = ['high', 'low','reg']
    args.con_type = ['reg']
    args.con_data = './../../../data/casp11/casp11_sel_cons.pt'
    if args.network_type.lower() == 'eq':
        args.network = {
            'irreps_inout': o3.Irreps("6x1o"),
            'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
            'layers': 8,
            'max_radius': 15,
            'number_of_basis': 8,
            'embed_dim': 2,
            'max_atom_types': 20,
            'radial_neurons': [48],
            'num_neighbors': -1,
        }
    elif args.network_type.lower() == 'mim':
        args.network = {
           'node_dim_in': 9,
            'node_dim_latent': 120,
            'nlayers': 3,
            'max_radius': 15,
        }
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)

    cs, legends, results = job_planner(c)

    job_runner(cs,legends,results)
