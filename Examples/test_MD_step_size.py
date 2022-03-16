import argparse
import os

import torch

from src.batch_jobs import job_planner, job_runner

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 1000
    args.n_val = 1000
    args.batch_size = 5
    args.nskip = 999
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
    args.epochs = 100
    args.network_type = ['eq','mim']  #Note if you use multiple network types equivariant networks always needs to go first or you will have memory trouble, this is likely due to the JIT compiler, though I'm not 100% sure.
    args.loss = ''
    args.data = './../../../data/MD/water_jones/water.npz'
    # args.data = './../Data/water.npz'
    args.data_type = 'water'
    # args.con = ['triangle']
    args.con = ['','triangle']
    args.con_type = ['high']
    args.con_data = ''
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)

    cs, legends, results = job_planner(c)

    job_runner(cs,legends,results)
