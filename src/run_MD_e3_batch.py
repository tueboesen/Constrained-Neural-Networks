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
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.mode ='standard'
    args.n_train = 1000
    args.n_val = 1000
    args.batch_size = 100
    args.n_input_samples = 1
    args.nskip = 0
    args.epochs_for_lr_adjustment = 50
    args.use_validation = True
    args.use_test = True
    args.lr = 1e-3
    args.seed = 123545
    args.epochs = 1000
    args.train_idx = None
    args.PE_predictor = './../pretrained_networks/force_energy_model.pt'
    args.data = './../../../data/MD/ethanol/ethanol_heating.npz'
    # args.data = './../../../data/MD/water_jones/water.npz'
    # args.data = './../../../data/MD/MD17/ethanol_dft.npz'
    args.network = {
        'irreps_inout': o3.Irreps("2x1o"),
        'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
        # 'irreps_node_attr': o3.Irreps("1x0e"),
        # 'irreps_edge_attr': o3.Irreps("{:}x1o".format(args.n_input_samples)),
        'irreps_edge_attr': o3.Irreps("1x1o"),
        'layers': 4,
        'max_radius': 15,
        'number_of_basis': 8,
        'embed_dim': 8,
        'max_atom_types': 20,
        'radial_neurons': [8, 16],
        'num_neighbors': 15,
        'constraints': ''
    }
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)

    result_dir_base = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )

    # data = np.load(c['data'])
    # R = torch.from_numpy(data['R'])
    # Rin, Rout = convert_snapshots_to_future_state_dataset(c['nskip'], R)
    # ndata_rand = 0 + np.arange(Rin.shape[0])
    # np.random.shuffle(ndata_rand)
    # train_idx = ndata_rand[:c['n_train']]
    # val_idx = ndata_rand[c['n_train']:c['n_train'] + c['n_val']]
    # test_idx = ndata_rand[c['n_train'] + c['n_val']:]
    #
    # c['train_idx'] = train_idx
    # c['val_idx'] = val_idx
    # c['test_idx'] = test_idx
    # splitting_indices = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
    # os.makedirs(result_dir_base)
    # np.savez("{:}/splitting_indices".format(result_dir_base), **splitting_indices)


    constraints_hist = ['','P']
    nskips = [0,9,99,999,9999]
    job = 0
    for nskip in nskips:
        c['nskip'] = nskip
        for constraint in constraints_hist:
            c['network']['constraints'] = constraint
            job += 1
            c['result_dir'] = "{:}/{:}".format(result_dir_base,job)
            results = main(c)
