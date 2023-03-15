import argparse
import copy
from datetime import datetime
import os
import numpy as np

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_runner, job_planner2, job_planner

base= '/home/tue/PycharmProjects/results/test_npendulum_table_nt1000/2022-09-06_11_34_15/'
mutable_parameters = {
    # 'network_discretization': ['rk4'],
    'load_previous_model_file' : [base+'con_type=high_penalty=10_regularization=10_nskip=20_0/model_best.pt',base+'con_type=low_penalty=10_regularization=10_nskip=20_0/model_best.pt',base+'con_type=_penalty=10_regularization=0_nskip=20_0/model_best.pt',base+'con_type=_penalty=0_regularization=10_nskip=20_0/model_best.pt',base+'con_type=_penalty=0_regularization=0_nskip=20_0/model_best.pt'],
    'con_type': ['high','low','','',''],
    'penalty': [10,10,10,0,0],
    'regularization': [10,10,0,10,0],
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
cs, legends, results,results_test = job_planner(c, mutable_parameters=mutable_parameters)
job_runner(cs, legends, results,results_test)