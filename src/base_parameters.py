import argparse
import os

import torch

from src.batch_jobs import job_planner, job_runner, job_planner
"""
Standard parameters used for most simulations. Each script overwrites the parameters not used by that particular script.
"""
def load_base_parameters_npendulum():
    """
    Standard simulation parameters for npendulum
    """
    parser = argparse.ArgumentParser(description='Constrained Neural Network')
    args = parser.parse_args()
    args.n_train = 100
    args.n_val = 1000
    args.n_test = 1000
    args.batch_size = 10
    args.nskip = 20
    args.epochs_for_lr_adjustment = 1
    args.lr_adjustment = 0.98
    args.use_training = True
    args.nviz = 0
    args.use_val = True
    args.use_test = True
    args.perform_endstep_MD_propagation = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = [1234,1235,1236]
    args.use_same_data = True
    args.epochs = 150
    args.load_previous_model_file = ''
    args.network_type = 'mim'  #Note if you use multiple network types equivariant networks always needs to go first or you will have memory trouble, this is likely due to the JIT compiler, though I'm not 100% sure.
    args.network_discretization = 'rk4'
    args.loss = 'eq'
    args.data = ''
    args.data_val = ''
    args.data_type = 'n-pendulum'
    args.data_dim = 2
    args.con = 'n-pendulum'
    args.ignore_cons = False
    args.con_type = ''
    args.model_specific = {'n': 5,
                           'dt': 0.01, #TODO if we want energy conservation constraints, we should set this to 0.001, set nskip to 200 and extra_simulation_steps to 10000 otherwise the energy will change slightly over a simulation
                           'L': [1,1,1,1,1],
                           'M': [1,1,1,1,1],
                           'angles': False,
                           'extra_simulation_steps': 1000
                           }
    args.regularization = 0
    args.penalty = 0
    args.con_data = ""
    args.use_double = True
    c = vars(args)
    c['network'] = {
        'node_dim_in': 2,
        'node_dim_latent': 120,
        'nlayers': 8,
        'max_radius': 15,
    }
    return c
