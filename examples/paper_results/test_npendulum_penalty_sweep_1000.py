import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner

if __name__ == '__main__':
    #Penalty sweep low
    mutable_parameters = {
        'con_type': ['']*6,
        'penalty': [0,1,10,50,100,200],
    }
    c = load_base_parameters_npendulum()
    c['basefolder'] = os.path.basename(__file__).split(".")[0]
    c['n_train'] = 1000

    cs, legends, results = job_planner(c, mutable_parameters)

    job_runner(cs,legends,results)
