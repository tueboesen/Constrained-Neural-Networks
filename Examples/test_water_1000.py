import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner3, standard_network_sizes

if __name__ == '__main__':

    mutable_parameters = {
        # 'network_discretization': ['rk4','rk4','rk4','rk4','rk4','rk4','rk4','euler'],
        'con_type': [''],
        'regularization': [10],
    }

    # mutable_parameters = {
    #     # 'network_discretization': ['rk4','rk4','rk4','rk4','rk4','rk4','rk4','euler'],
    #     'con_type': ['','','low','high'],
    #     'penalty': [0,10,10,10],
    #     'regularization': [0, 0, 10, 10],
    #     'lr': [1e-2,1e-2,1e-3,1e-3],
    #     'epochs': [75,75,20,20]
    # }

    c = load_base_parameters_npendulum()
    c['data'] = './../Data/water.npz'
    c['data_type'] = 'water'
    c['con'] = 'water'
    c['epochs'] = 50
    # c['lr_adjustment'] = 0.99
    # args.epochs_for_lr_adjustment = 1
    # c['epochs_for_lr_adjustment'] = 1000
    c['lr'] = 1e-2
    c['nskip'] = 49
    c['n_val'] = 100
    c['n_train'] = 1000

    # c['epochs'] = 2
    c['use_test'] = True
    c['basefolder'] = os.path.basename(__file__).split(".")[0]
    c['network_type'] = 'eq'  #Note if you use multiple network types equivariant networks always needs to go first or you will have memory trouble, this is likely due to the JIT compiler, though I'm not 100% sure.
    c = standard_network_sizes(c, c['network_type'])

    cs, legends, results,results_test = job_planner3(c,mutable_parameters)

    job_runner(cs,legends,results,results_test)
