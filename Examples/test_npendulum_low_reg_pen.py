import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner3

if __name__ == '__main__':

    # mutable_parameters = {
    #     'con_type': ['low']*6,
    #     'penalty': [0,10,50,100,200,500],
    # }
    #
    # mutable_parameters = {
    #     'con_type': ['low']*4,
    #     'regularization': [0,1,10,20],
    # }
    #
    # mutable_parameters = {
    #     'con_type': ['low']*5,
    #     'regularization': [1]*5,
    #     'penalty': [10,50,100,200,500],
    # }

    mutable_parameters = {
        'con_type': ['low']*19,
        'regularization': [0,0,0,0,0,0,1,10,20,1,1,1,1,1,10,10,10,10,10],
        'penalty': [0,10,50,100,200,500,0,0,0,10,50,100,200,500,10,50,100,200,500],
    }

    c = load_base_parameters_npendulum()
    c['basefolder'] = os.path.basename(__file__).split(".")[0]
    c['n_val'] = 100


    cs, legends, results = job_planner3(c,mutable_parameters)

    job_runner(cs,legends,results)
