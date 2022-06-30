import argparse
import os

import torch

from src.batch_jobs import job_planner, job_runner
if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 100
    args.n_val = 500
    args.batch_size = 5
    args.nskip = 99
    args.epochs_for_lr_adjustment = 1000
    args.lr_adjustment = 0.8
    args.use_training = True
    args.use_val = True
    args.use_test = False
    args.perform_endstep_MD_propagation = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = [1234,1235,1236]
    args.use_same_data = True
    args.epochs = 100
    args.load_previous_model_file = ''
    args.network_type = ['mim']  #Note if you use multiple network types equivariant networks always needs to go first or you will have memory trouble, this is likely due to the JIT compiler, though I'm not 100% sure.
    args.network_discretization = 'leapfrog'
    args.loss = ''
    # args.data = ''
    args.data_val = ''
    args.data = './../Data/water.npz'
    args.data_type = 'water'
    args.data_dim = 3
    args.con = ['water']
    # args.con = ['n-pendulum','n-pendulum-seq','n-pendulum-seq-start']
    args.ignore_cons = False
    # args.con_type = ['','high','stabhigh','low']
    # args.con_type = ['','high','stabhigh','low']
    args.con_type = ['stabhigh']
    args.model_specific = {'n': 5,
                           'dt': 0.01,
                           'L': [1,1,1,1,1],
                           'M': [1,1,1,1,1],
                           'angles': False
                           }
    args.regularizationparameter = [1e-2]
    args.con_data = ""
    args.gamma = [0,100,500,1000,5000,10000]
    # args.gamma = [100]
    args.use_double = True
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)
    # c['network'] = {
    #     'node_dim_in': 9,
    #     'node_dim_latent': 120,
    #     'nlayers': 8,
    #     'max_radius': 15,
    # }

    cs, legends, results = job_planner(c)

    job_runner(cs,legends,results)
