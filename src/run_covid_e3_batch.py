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
from src.main_covid import main_covid
from src.network_e3 import constrained_network
from src.network_eq import network_eq_simple
from src.utils import fix_seed, convert_snapshots_to_future_state_dataset, DatasetFutureState, run_network, run_network_eq, run_network_e3, atomic_masses

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.mode ='standard'
    args.n_train = 1000
    args.n_val = 1000
    args.batch_size = 1
    args.n_input_samples = 1
    args.nskip = 0
    args.epochs_for_lr_adjustment = 1000
    args.use_validation = True
    args.use_test = True
    args.lr = 1e-4
    args.seed = 123545
    args.epochs = 10000
    args.PE_predictor = ''
    args.data = './../../../data/covid/covid_backbone.npz'
    # args.data = './../../../data/MD/water_jones/water.npz'
    # args.data = './../../../data/MD/MD17/ethanol_dft.npz'
    args.network = {
        'irreps_inout': o3.Irreps("6x1o"),
        'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
        # 'irreps_node_attr': o3.Irreps("1x0e"),
        # 'irreps_edge_attr': o3.Irreps("{:}x1o".format(args.n_input_samples)),
        'irreps_edge_attr': o3.Irreps("1x1o"),
        'layers': 4,
        'max_radius': 10,
        'number_of_basis': 8,
        'embed_dim': 8,
        'max_atom_types': 25,
        'radial_neurons': [8, 16],
        'num_neighbors': 15,
        'constraints': 'bindingall'

    }
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)

    result_dir_base = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )


    constraints_hist = ['','binding','bindingall']
    nskips = [0,9,99,999]
    job = 0
    for nskip in nskips:
        c['nskip'] = nskip
        for constraint in constraints_hist:
            c['network']['constraints'] = constraint
            job += 1
            c['result_dir'] = "{:}/{:}".format(result_dir_base,job)
            results = main_covid(c)

