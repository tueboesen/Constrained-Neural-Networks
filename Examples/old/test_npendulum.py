import argparse
import os

import torch

from src.batch_jobs import job_planner, job_runner, job_planner3

if __name__ == '__main__':

    # mutable_parameters = {
    #     # 'network_discretization': ['rk4','rk4','rk4','rk4','rk4','rk4','rk4','euler'],
    #     'con_type': ['','','low','low','high','high','','low','high'],
    #     'penalty': [0,100,0,100,0,100,0,0,0],
    #     'regularization': [0,0,0,0,0,0,1,1,1]
    # }

    # #Penalty sweep high
    # mutable_parameters = {
    #     'con_type': ['high','high','high','high','high','high'],
    #     'penalty': [0,1,10,50,100,200],
    # }

    #Reg sweep high
    # mutable_parameters = {
    #     'con_type': ['high','high','high','high'],
    #     'regularization': [0,0.1,1,10],
    # }


    # #Reg sweep high
    # mutable_parameters = {
    #     'con_type': ['high','high'],
    #     'regularization': [1,0],
    # }


    #Penalty sweep low
    mutable_parameters = {
        # 'con_type': ['low','low','low','low','low','low'],
        'penalty': [0,1,10,50,100,200],
    }

    # mutable_parameters = {
    #     # 'network_discretization': ['rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','euler','euler','euler','euler','euler','euler','euler','euler','euler','euler','euler','euler'],
    #     'regularization': [1,1,1],
    #     'penalty': [10,50,100],
    # }

    # mutable_parameters = {
    #     'network_discretization': ['rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','rk4','euler','euler','euler','euler','euler','euler','euler','euler','euler','euler','euler','euler'],
    #     'regularization': [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1],
    #     'penalty': [0,1,10,50,100,200,0,1,10,50,100,200,0,1,10,50,100,200,0,1,10,50,100,200],
    # }


    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 100
    args.n_val = 1000
    args.batch_size = 10
    args.nskip = 20
    args.epochs_for_lr_adjustment = 1000
    args.lr_adjustment = 0.8
    args.use_training = True
    args.nviz = 0
    args.use_val = True
    args.use_test = False
    args.perform_endstep_MD_propagation = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    # args.seed = [1234,1235]
    args.seed = [1234,1235,1236,1237,1238]
    args.use_same_data = True
    args.epochs = 150
    args.load_previous_model_file = ''
    args.network_type = 'mim'  #Note if you use multiple network types equivariant networks always needs to go first or you will have memory trouble, this is likely due to the JIT compiler, though I'm not 100% sure.
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
    args.con_type = ''
    # args.con_type = ['stabhigh','high','low','']
    # args.con_type = ['high','low','reg']
    args.model_specific = {'n': 5,
                           'dt': 0.01,
                           'L': [1,1,1,1,1],
                           'M': [1,1,1,1,1],
                           'angles': False
                           }
    args.regularization = 0
    args.penalty = 0
    args.con_data = ""
    # args.gamma = [100]
    # args.gamma = [500]
    args.use_double = True
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)
    c['network'] = {
        'node_dim_in': 2,
        'node_dim_latent': 120,
        'nlayers': 8,
        'max_radius': 15,
    }

    cs, legends, results = job_planner3(c,mutable_parameters)

    job_runner(cs,legends,results)
