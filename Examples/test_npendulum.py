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
    args.nskip = 20
    args.epochs_for_lr_adjustment = 1000
    args.lr_adjustment = 0.8
    args.use_val = True
    args.use_test = False
    args.perform_endstep_MD_propagation = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = [1234,1235,1236]
    args.use_same_data = True
    args.epochs = 50
    args.network_type = ['mim']  #Note if you use multiple network types equivariant networks always needs to go first or you will have memory trouble, this is likely due to the JIT compiler, though I'm not 100% sure.
    args.network_discretization = 'euler'
    args.loss = 'eq'
    args.data = ''
    args.data_val = ''
    # args.data = './../Data/water.npz'
    args.data_type = 'n-pendulum'
    args.data_dim = 2
    args.con = ['','n-pendulum']
    # args.con = ['n-pendulum']
    # args.con = ['angles']
    args.ignore_cons = True
    args.con_type = ['stabhigh']
    # args.con_type = ['high','stabhigh']
    # args.con_type = ['high','low','reg']
    args.model_specific = {'n': 5,
                           'dt': 0.01,
                           'L': [1,1,1,1,1],
                           'M': [1,1,1,1,1],
                           'angles': False
                           }
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
