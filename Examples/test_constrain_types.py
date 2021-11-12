import argparse
import os

import torch

from src.batch_jobs import job_planner, job_runner

"""
This example tests the three different types of constraints: 
    "high" - which are constraints in high dimensional space, done in every layer of a neural network
    "low" - which are constraints in low dimensional space (the output space), done at the end of the neural network
    "reg" - which are constraints through regularization, which is done at the end of the neural network and then applied as a regularization term to the loss function
    
    The constraint types are tested on the problem of protein folding. 
"""

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 10
    args.n_val = 0
    args.batch_size = 1
    args.nskip = 9999
    args.epochs_for_lr_adjustment = 20
    args.lr_adjustment = 0.8
    args.use_val = False
    args.use_test = False
    args.perform_endstep_MD_propagation = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = [11,12,13,14,15,16,17,18,19,110,111,112,113,114,115]
    args.use_same_data = True
    args.epochs = 300
    args.network_type = ['mim']
    args.loss = ''
    args.data = './../../../data/casp11/casp11_sel.npz'
    args.data_type = 'protein'
    args.con = ['', 'chain', 'triangle', 'chaintriangle']
    args.con_type = ['high', 'low', 'reg']
    args.con_data = './../../../data/casp11/casp11_sel_cons.pt'
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)

    cs, legends, results = job_planner(c)

    job_runner(cs,legends,results)
