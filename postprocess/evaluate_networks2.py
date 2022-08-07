import argparse
import copy
from datetime import datetime
import os
import numpy as np

from src.base_parameters import load_base_parameters_npendulum
from src.batch_jobs import job_runner, job_planner2, job_planner3

mutable_parameters = {
    # 'network_discretization': ['rk4'],
    'load_previous_model_file' : ['/home/tue/PycharmProjects/results/test_npendulum_final_test/2022-08-02_12_15_57/con_type=high_penalty=10_regularization=1_0/model_best.pt','/home/tue/PycharmProjects/results/test_npendulum_final_test/2022-08-02_12_15_57/con_type=low_penalty=50_regularization=5_0/model_best.pt','/home/tue/PycharmProjects/results/test_npendulum_final_test/2022-08-02_12_15_57/con_type=_penalty=0_regularization=0_0/model_best.pt','/home/tue/PycharmProjects/results/test_npendulum_final_test/2022-08-02_12_15_57/con_type=_penalty=50_regularization=0_0/model_best.pt','/home/tue/PycharmProjects/results/test_npendulum_final_test/2022-08-02_12_15_57/con_type=_penalty=0_regularization=0_0/model_best.pt'],
    'con_type': ['high','low','','','low'],
    'penalty': [10,50,0,50,0],
    'regularization': [1,5,0,0,0],
}
c = load_base_parameters_npendulum()
c['basefolder'] = os.path.basename(__file__).split(".")[0]
c['epochs'] = 1
c['use_training'] = False
c['use_val'] = True
c['use_test'] = True
c['seed'] = [1234]
c['nviz'] = 10
c['batchsize'] = 10
cs, legends, results = job_planner3(c,mutable_parameters=mutable_parameters)
job_runner(cs, legends, results)