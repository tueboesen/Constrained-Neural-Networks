import argparse
import copy
from datetime import datetime
import os
import numpy as np

from src.batch_jobs import job_runner, job_planner2

parser = argparse.ArgumentParser(description='Constrained MD')
args = parser.parse_args()
args.n_train = 1
args.n_val = 1000
args.batch_size = 10
args.nskip = 20
args.epochs_for_lr_adjustment = 1000
args.lr_adjustment = 0.8
args.use_training = False
args.use_val = True
args.use_test = False
args.perform_endstep_MD_propagation = False
args.debug = False
args.viz = False
args.nviz = 8
args.lr = 1e-3
args.seed = [1234]
args.use_same_data = True
args.epochs = 1
args.load_previous_model_file = ['/home/tue/PycharmProjects/results/test_npendulum/2022-07-05_18_51_18/mim_n-pendulum__1_1_0/model_best.pt','/home/tue/PycharmProjects/results/test_npendulum/2022-07-05_18_51_18/mim_n-pendulum_high_1_1_0/model_best.pt']
args.network_type = ['mim','mim']  # Note if you use multiple network types equivariant networks always needs to go first or you will have memory trouble, this is likely due to the JIT compiler, though I'm not 100% sure.
args.network_discretization = 'rk4'
args.loss = 'eq'
args.data = ''
args.data_val = ''
# args.data = './../Data/water.npz'
args.data_type = 'n-pendulum'
args.data_dim = 2
args.con = 'n-pendulum'
# args.con = ['n-pendulum','n-pendulum-seq','n-pendulum-seq-start']
args.ignore_cons = False
args.con_type = ['','high']
args.model_specific = {'n': 5,
                       'dt': 0.01,
                       'L': [1, 1, 1, 1, 1],
                       'M': [1, 1, 1, 1, 1],
                       'angles': False
                       }
args.regularizationparameter = [1e-2]
args.con_data = ""
args.gamma = 500
args.use_double = True
args.basefolder = os.path.basename(__file__).split(".")[0]
c = vars(args)
c['network'] = {
    'node_dim_in': 2,
    'node_dim_latent': 120,
    'nlayers': 8,
    'max_radius': 15,
}
#
# c['result_dir_base'] = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
#     root='results',
#     runner_name=c['basefolder'],
#     date=datetime.now(),
# )
# os.makedirs(c['result_dir_base'])
#
# # c_base = copy.deepcopy(c)
# cs = []
# for file_network,legend,con_type in zip(file_networks,legends,con_types):
#     c_new = copy.deepcopy(c)
#     c_new['load_previous_model_file'] = file_network
#     c_new['con_type'] = con_type
#     cs.append(c_new)
# results = np.zeros((len(legends),1,10,c['epochs']))

cs, legends, results = job_planner2(c)
job_runner(cs, legends, results)