import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner3

if __name__ == '__main__':

    c = load_base_parameters_npendulum()
    mutable_parameters = {
        # 'network_discretization': ['rk4','rk4','rk4','rk4','rk4','rk4','rk4','euler'],
        'con_type': ['','','','low','high']*2,
        'penalty': [0,0,10,10,10]*2,
        'regularization': [0, 10, 0, 10, 10]*2,
        'nskip': [20,20,20,20,20,50,50,50,50,50]
    }
    c['basefolder'] = os.path.basename(__file__).split(".")[0]
    c['epochs'] = 2

    cs, legends, results,results_test = job_planner3(c,mutable_parameters)

    job_runner(cs,legends,results,results_test)
