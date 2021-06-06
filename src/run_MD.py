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

from src import log
from src.log import log_all_parameters
from src.network import network_simple
from src.network_eq import network_eq_simple
from src.utils import fix_seed, convert_snapshots_to_future_state_dataset, DatasetFutureState, run_network

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.mode ='standard'
    args.n_train = 100
    args.n_val = 1000
    args.batch_size = 50
    args.n_input_samples = 2
    args.n_skips = 0
    args.epochs_for_lr_adjustment = 50
    args.use_validation = False
    args.lr = 1e-2
    args.seed = 123545
    args.epochs = 100000
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)

    fix_seed(c['seed']) #Set a seed, so we make reproducible results.

    c['result_dir'] = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )

    os.makedirs(c['result_dir'])
    logfile_loc = "{}/{}.log".format(c['result_dir'], 'output')
    LOG = log.setup_custom_logger('runner',logfile_loc,c['mode'])
    log_all_parameters(LOG, c)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    # load training data
    data = np.load('./../../../data/MD/MD17/aspirin_dft.npz')
    R = torch.from_numpy(data['R']).to( device=device)
    Rin, Rout = convert_snapshots_to_future_state_dataset(c['n_input_samples'], c['n_skips'], R)

    z = torch.from_numpy(data['z']).to( device=device)

    ndata = Rout.shape[0]
    natoms = z.shape[0]

    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

    ndata_rand = 0 + np.arange(ndata)
    np.random.shuffle(ndata_rand)


    Rin_train = Rin[ndata_rand[:c['n_train']]]
    Rout_train = Rout[ndata_rand[:c['n_train']]]

    Rin_val = Rin[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    Rout_val = Rout[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]

    dataset_train = DatasetFutureState(Rin_train, Rout_train, z)
    dataset_val = DatasetFutureState(Rin_val, Rout_val, z)

    dataloader_train = DataLoader(dataset_train, batch_size=c['batch_size'], shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=c['batch_size'], shuffle=True, drop_last=False)

    node_dim_in = 6
    node_attr_dim_in = 1
    node_dim_latent = 60
    nlayers = 6
    model  = network_simple(node_dim_in, node_attr_dim_in, node_dim_latent, nlayers)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    LOG.info('Number of parameters {:}'.format(total_params))

    #### Start Training ####
    optimizer = torch.optim.Adam(model.parameters(), lr=c['lr'])

    alossBest = 1e6
    lr = c['lr']
    t0 = time.time()
    epochs_since_best = 0
    for epoch in range(c['epochs']):
        t1 = time.time()
        aloss_t, aloss_ref_t, MAE_t, t_dataload_t, t_prepare_t, t_model_t, t_backprop_t = run_network(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=c['batch_size'])
        t2 = time.time()
        if c['use_validation']:
            aloss_v, aloss_ref_v, MAE_v, t_dataload_v, t_prepare_v, t_model_v, t_backprop_v = run_network(model, dataloader_val, train=False, max_samples=50, optimizer=optimizer, batch_size=c['batch_size'])
        else:
            aloss_v, aloss_ref_v, MAE_v, t_dataload_v, t_prepare_v, t_model_v, t_backprop_v = 0,0,0,0,0,0,0
        t3 = time.time()

        if aloss_v < alossBest:
            alossBest = aloss_v
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= c['epochs_for_lr_adjustment']:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.8
                    lr = g['lr']
                epochs_since_best = 0

        # print(f' t_dataloader(train): {t_dataload_t:.3f}s  t_dataloader(val): {t_dataload_v:.3f}s  t_prepare(train): {t_prepare_t:.3f}s  t_prepare(val): {t_prepare_v:.3f}s  t_model(train): {t_model_t:.3f}s  t_model(val): {t_model_v:.3f}s  t_backprop(train): {t_backprop_t:.3f}s  t_backprop(val): {t_backprop_v:.3f}s')
        LOG.info(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss_ref(train): {aloss_ref_t:.2e} MAE(train) {MAE_t:.2e}  Loss(val): {aloss_v:.2e}   Loss_ref(val): {aloss_ref_v:.2e} MAE(val) {MAE_v:.2e}  Time(train): {t2 - t1:.1f}s  Time(val): {t3 - t2:.1f}s  Lr: {lr:2.2e} ')


