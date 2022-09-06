import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner3

if __name__ == '__main__':

    mutable_parameters = {
        # 'network_discretization': ['rk4','rk4','rk4','rk4','rk4','rk4','rk4','euler'],
        'con_type': ['','','low','high'],
        'penalty': [0,50,50,50],
        'regularization': [0, 0, 5, 5],
    }
    c = load_base_parameters_npendulum()
    c['basefolder'] = os.path.basename(__file__).split(".")[0]
    c['n_train'] = 1000


    cs, legends, results = job_planner3(c,mutable_parameters)

    job_runner(cs,legends,results)