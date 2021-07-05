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
from src.network_e3 import constrained_network
from src.network_eq import network_eq_simple
from src.utils import fix_seed, convert_snapshots_to_future_state_dataset, DatasetFutureState, run_network, run_network_eq, run_network_e3, atomic_masses

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(False)
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 10000
    args.n_val = 1000
    args.batch_size = 8
    args.n_input_samples = 1
    args.nskip = 9999
    args.train_idx = None
    args.epochs_for_lr_adjustment = 10
    args.use_validation = True
    args.use_test = True
    args.debug = True
    args.lr = 5e-3
    args.seed = 123545
    args.loss = 'EQ'
    args.network_type = 'EQ' #EQ or mim
    args.epochs = 10000
    args.PE_predictor = './../pretrained_networks/force_energy_model.pt'
    args.data = './../../../data/MD/argon/argon.npz'
    # args.data = './../../../data/MD/water_jones/water.npz'
    # args.data = './../../../data/MD/MD17/ethanol_dft.npz'
    args.network = {
        'irreps_inout': o3.Irreps("2x1o"),
        'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
        # 'irreps_node_attr': o3.Irreps("1x0e"),
        # 'irreps_edge_attr': o3.Irreps("{:}x1o".format(args.n_input_samples)),
        'irreps_edge_attr': o3.Irreps("2x1o"),
        'layers': 4,
        'max_radius': 15,
        'number_of_basis': 8,
        'embed_dim': 8,
        'max_atom_types': 20,
        'radial_neurons': [16, 16],
        'num_neighbors': -1,
        'constraints': ''

    }
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)
    results = main(c)
