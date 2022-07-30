import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner3

if __name__ == '__main__':

    # mutable_parameters = {
    #     'con_type': ['high']*6,
    #     'penalty': [0,1,10,50,100,200],
    # }
    #
    # mutable_parameters = {
    #     'con_type': ['high']*4,
    #     'regularization': [0,0.1,1,10],
    # }
    #
    # mutable_parameters = {
    #     'con_type': ['high']*5,
    #     'regularization': [1]*5,
    #     'penalty': [1,10,50,100,200],
    # }

    mutable_parameters = {
        'con_type': ['high']*14,
        'regularization': [0,0,0,0,0,0,0.1,1,10,1,1,1,1,1],
        'penalty': [0,1,10,50,100,200,0,0,0,1,10,50,100,200],
    }

    c = load_base_parameters_npendulum()
    c['basefolder'] = os.path.basename(__file__).split(".")[0]


    cs, legends, results = job_planner3(c,mutable_parameters)

    job_runner(cs,legends,results)
