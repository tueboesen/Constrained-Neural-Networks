import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner, standard_network_sizes
"""
This is an example of the water molecule simulation. In order to run this simulation you need to have the appropriate dataset.
The dataset water.npz is too large for github, but send an email to me on tue.boesen@protonmail.com and I will find a way to send it to you, if enough people requests it I will find a smarter solution. 
"""

if __name__ == '__main__':
    # mutable_parameters = {
    #     'con_type': ['','','','low','high'],
    #     'penalty': [0,0,1,1,1],
    #     'regularization': [0, 1, 0, 100, 100],
    # }

    mutable_parameters = {
        'con_type': ['low','high'],
        'penalty': [10,1],
        'regularization': [10, 100],
    }


    c = load_base_parameters_npendulum()
    c['data'] = './../data/water/water.npz'
    c['data_type'] = 'water'
    c['con'] = 'water'
    c['epochs'] = 150
    c['lr_adjustment'] = 0.99
    c['lr'] = 1e-2
    c['nskip'] = 200
    c['n_val'] = 100
    c['n_train'] = 100
    c['use_test'] = True
    c['basefolder'] = os.path.basename(__file__).split(".")[0]
    c['network_type'] = 'eq'
    c = standard_network_sizes(c, c['network_type'])

    cs, legends, results,results_test= job_planner(c, mutable_parameters)

    job_runner(cs,legends,results, results_test)
