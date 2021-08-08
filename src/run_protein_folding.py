import argparse
from datetime import datetime
import os, sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import torch.autograd.profiler as profiler

from torch.utils.data import DataLoader
from e3nn import o3

from preprocess.train_force_and_energy_predictor import generate_FE_network
from src import log
from src.log import log_all_parameters
from src.main import main
from src.main_protein import main_protein
from src.network_e3 import constrained_network
from src.utils import fix_seed, convert_snapshots_to_future_state_dataset, run_network, run_network_eq, run_network_e3, atomic_masses

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 1000
    args.n_val = 100
    args.batch_size = 1
    args.n_input_samples = 1
    args.nskip = 9999
    args.epochs_for_lr_adjustment = 50
    args.use_val = True
    args.use_test = True
    args.debug = False
    args.lr = 1e-3
    args.seed = 123545
    args.epochs = 100
    args.network_type = 'mim'
    args.loss = 'distogram'
    args.train_idx = None
    args.data = './../../../data/casp11/'

    if args.network_type.lower() == 'eq':
        args.network = {
            'irreps_inout': o3.Irreps("6x1o"),
            'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
            # 'irreps_node_attr': o3.Irreps("1x0e"),
            # 'irreps_edge_attr': o3.Irreps("{:}x1o".format(args.n_input_samples)),
            'layers': 8,
            'max_radius': 15,
            'number_of_basis': 8,
            'embed_dim': 2,
            'max_atom_types': 20,
            'radial_neurons': [48],
            'num_neighbors': -1,
            'constraints': '',
        }
    elif args.network_type.lower() == 'mim':
        args.network = {
            'node_dim_in': 9,
            'node_attr_dim_in': 1,
            'node_dim_latent': 120,
            'nlayers': 9,
            'max_radius': 15,
            # 'constraints': 'chaintriangle',
        }


    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)

    result_dir_base = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )

    constrain_all_layers = [True,False]
    constraints = ['chain','triangle','chaintriangle']
    nskips = [9999]
    job = 0
    job += 1
    c['network']['constrain_all_layers'] = True
    c['result_dir'] = "{:}/{:}".format(result_dir_base, job)
    c['network']['constraints'] = ''
    results = main_protein(c)
    for nskip in nskips:
        c['nskip'] = nskip
        for constrain_all_layersi in constrain_all_layers:
            c['network']['constrain_all_layers'] = constrain_all_layersi
            for constraint in constraints:
                c['network']['constraints'] = constraint
                job += 1
                c['result_dir'] = "{:}/{:}".format(result_dir_base,job)
                results = main_protein(c)

