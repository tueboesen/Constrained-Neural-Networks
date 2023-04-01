import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner

"""
This is a simple script that will simulate a 5 body pendulum, and then train a neural network to simulate the results.
"""

if __name__ == '__main__':

    c = load_base_parameters_npendulum()
    mutable_parameters = {
        'con_type': ['','','','low','high'],
        'penalty': [0,0,10,10,10],
        'regularization': [0, 10, 0, 10, 10],
        'nskip': [200]*5
    }
    c['basefolder'] = os.path.basename(__file__).split(".")[0]
    c['epochs'] = 150
    c['con'] = 'n-pendulum'
    c['data'] = './../data/multibodypendulum/multibodypendulum.npz'

    cs, legends, results,results_test = job_planner(c, mutable_parameters)
    job_runner(cs,legends,results,results_test)
