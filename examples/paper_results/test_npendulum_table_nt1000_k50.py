import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner

if __name__ == '__main__':

    c = load_base_parameters_npendulum()
    mutable_parameters = {
        # 'network_discretization': ['rk4','rk4','rk4','rk4','rk4','rk4','rk4','euler'],
        'con_type': ['low','high'],
        'penalty': [1,1],
        'regularization': [1, 1],
        'nskip': [50]*2
    }
    c['basefolder'] = os.path.basename(__file__).split(".")[0]
    c['epochs'] = 100
    c['n_val'] = 100
    c['n_train'] = 1000


    cs, legends, results,results_test = job_planner(c, mutable_parameters)

    job_runner(cs,legends,results,results_test)
