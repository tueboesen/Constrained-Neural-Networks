import argparse
import os

import torch

from src.batch_jobs import job_planner, job_runner

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 10000
    args.n_val = 1000
    args.batch_size = 10
    args.nskip = 100
    args.epochs_for_lr_adjustment = 1000
    args.lr_adjustment = 0.8
    args.use_val = True
    args.use_test = False
    args.perform_endstep_MD_propagation = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = [1234,1235,1236,1237,1238]
    args.use_same_data = True
    args.epochs = 500
    args.network_type = ['mim']  #Note if you use multiple network types equivariant networks always needs to go first or you will have memory trouble, this is likely due to the JIT compiler, though I'm not 100% sure.
    args.loss = 'eq'
    args.data = ''
    # args.data = './../Data/water.npz'
    args.data_type = 'pendulum'
    args.data_dim = 2
    # args.con = ['triangle']
    args.con = ['','pendulum']
    # args.con_type = ['reg']
    args.con_type = ['high','low','reg']
    args.regularizationparameter = [1e-2]
    args.con_data = ""
    args.use_double = True
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)
    c['network'] = {
        'node_dim_in': 2,
        'node_dim_latent': 120,
        'nlayers': 8,
        'max_radius': 15,
    }

    cs, legends, results = job_planner(c)

    job_runner(cs,legends,results)
