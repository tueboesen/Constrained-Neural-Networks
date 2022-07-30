import argparse
import os

import torch

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_planner, job_runner, job_planner3

if __name__ == '__main__':
    #Penalty sweep low
    mutable_parameters = {
        # 'con_type': ['low','low','low','low','low','low'],
        'penalty': [0,1,10,50,100,200],
    }
    c = load_base_parameters_npendulum()


    cs, legends, results = job_planner3(c,mutable_parameters)

    job_runner(cs,legends,results)
