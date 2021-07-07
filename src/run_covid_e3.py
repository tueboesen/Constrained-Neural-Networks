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
    args.n_train = 1
    args.n_val = 4000
    args.batch_size = 1
    args.n_input_samples = 1
    args.nskip = 0
    args.epochs_for_lr_adjustment = 1000
    args.use_validation = False
    args.use_test = False
    args.lr = 1e-2
    args.seed = 123545
    args.epochs = 10000
    args.PE_predictor = ''
    args.data = './../../../data/covid/covid_backbone.npz'
    # args.data = './../../../data/MD/water_jones/water.npz'
    # args.data = './../../../data/MD/MD17/ethanol_dft.npz'
    args.network = {
        'irreps_inout': o3.Irreps("6x1o"),
        'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
        'layers': 4,
        'max_radius': 10,
        'number_of_basis': 8,
        'embed_dim': 8,
        'max_atom_types': 25,
        'radial_neurons': [16],
        'num_neighbors': 15,
        'constraints': 'chaintriangle'             #chain, triangle, chaintriangle

    }
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)
    main_covid(c)
