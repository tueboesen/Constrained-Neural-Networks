import argparse
import copy
from datetime import datetime
import os
import numpy as np

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_runner, job_planner2, job_planner

base= '/home/tue/PycharmProjects/results/test_npendulum_final_test_1000/2022-08-03_12_16_42/'
mutable_parameters = {
    # 'network_discretization': ['rk4'],
    'load_previous_model_file' : [base+'con_type=high_penalty=50_regularization=5_1/model_best.pt',base+'con_type=low_penalty=50_regularization=5_0/model_best.pt',base+'con_type=_penalty=0_regularization=0_0/model_best.pt',base+'con_type=_penalty=50_regularization=0_0/model_best.pt'],
    'con_type': ['high','low','',''],
    'penalty': [10,50,0,50],
    'regularization': [1,5,0,0],
}
c = load_base_parameters_npendulum()
c['basefolder'] = os.path.basename(__file__).split(".")[0]
c['epochs'] = 1
c['use_training'] = False
c['use_val'] = True
c['use_test'] = True
c['n_test'] = 1000
c['seed'] = [1230]
c['nviz'] = 20
c['batchsize'] = 10
cs, legends, results = job_planner(c, mutable_parameters=mutable_parameters)
job_runner(cs, legends, results)