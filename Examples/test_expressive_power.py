import argparse
import os

import torch

from src.batch_jobs import job_planner, job_runner

"""
This is an example designed to test whether inherently constrained neural networks have a higher expressive power than non-constrained networks.
This is tested by using a very small network on a very large dataset, and seeing how well the network can fit the entire dataset.
In this particular case we test 2 kinds of constraints on the problem of protein folding
  
"""


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = -1
    args.n_val = 0
    args.batch_size = 1
    args.nskip = 999
    args.epochs_for_lr_adjustment = 1000
    args.lr_adjustment = 0.8
    args.use_val = False
    args.use_test = False
    args.perform_endstep_MD_propagation = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = [1234,1235,1236,1237,1238]
    args.use_same_data = True
    args.epochs = 50
    args.network_type = ['mim']  #Note if you use multiple network types equivariant networks always needs to go first or you will have memory trouble, this is likely due to the JIT compiler, though I'm not 100% sure.
    args.loss = ''
    args.data = './../../../data/casp11/casp11_sel.npz'
    args.data_type = 'protein'
    args.con = ['','triangle','chaintriangle']
    args.con_type = ['high']
    args.con_data = ''
    args.basefolder = os.path.basename(__file__).split(".")[0]
    args.network = {
        'node_dim_in': 9,
        'node_attr_dim_in': 1,
        'node_dim_latent': 60,
        'nlayers': 3,
        'max_radius': 15,
    }
    c = vars(args)

    cs, legends, results = job_planner(c)

    job_runner(cs,legends,results)