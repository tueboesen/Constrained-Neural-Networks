import argparse
import pandas as pd
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
from src.dataloader import load_data, load_data_protein
from src.log import log_all_parameters, close_logger
from src.network import network_simple
from src.network_e3 import constrained_network
from src.constraints import MomentumConstraints, PointChain, PointToPoint, EnergyMomentumConstraints, load_constraints, load_constraints_protein
from src.project_uplift import ProjectUpliftEQ, ProjectUplift
from src.utils import fix_seed, convert_snapshots_to_future_state_dataset, run_network_eq, run_network_e3, atomic_masses, Distogram, LJ_potential, use_proteinmodel


def main_protein(c,dataloader_train=None,dataloader_val=None,dataloader_test=None):
    cn = c['network']
    fix_seed(c['seed'])  # Set a seed, so we make reproducible results.

    if 'result_dir' not in c:
        c['result_dir'] = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
            root='results',
            runner_name=c['basefolder'],
            date=datetime.now(),
        )
    model_name = "{}/{}.pt".format(c['result_dir'], 'model')
    model_name_best = "{}/{}.pt".format(c['result_dir'], 'model_best')
    model_name_last = "{}/{}.pt".format(c['result_dir'], 'model_last')
    optimizer_name_last = "{}/{}.pt".format(c['result_dir'], 'optimizer_last')
    os.makedirs(c['result_dir'])
    logfile_loc = "{}/{}.log".format(c['result_dir'], 'output')
    LOG = log.setup_custom_logger('runner', logfile_loc, c['debug'])
    log_all_parameters(LOG, c)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    # load training data
    if dataloader_train is None:
        dataloader_train, dataloader_val, dataloader_test = load_data_protein(c['data'], device, c['n_train'], c['n_val'], c['use_val'], c['use_test'], c['batch_size'])

    if c['network_type'] == 'EQ':
        PU = ProjectUpliftEQ(cn['irreps_inout'], cn['irreps_hidden'])
    elif c['network_type'] == 'mim':
        PU = ProjectUplift(cn['node_dim_in'], cn['node_dim_latent'])

    reg = cn['constrain_method'] ==  'reg'
    ds = dataloader_train.dataset
    constraints = load_constraints_protein(cn['constraints'], PU,device=device,reg=reg)

    if c['network_type'] == 'EQ':
        model = constrained_network(irreps_inout=cn['irreps_inout'], irreps_hidden=cn['irreps_hidden'], layers=cn['layers'],
                                    max_radius=cn['max_radius'],
                                    number_of_basis=cn['number_of_basis'], radial_neurons=cn['radial_neurons'], num_neighbors=cn['num_neighbors'],
                                    num_nodes=20, embed_dim=cn['embed_dim'], max_atom_types=cn['max_atom_types'], constraints=constraints, constrain_method=cn['constrain_method'], PU=PU, particles_pr_node=3)
    elif c['network_type'] == 'mim':
        model = network_simple(cn['node_dim_in'], cn['node_attr_dim_in'], cn['node_dim_latent'], cn['nlayers'], PU=PU, constraints=constraints,constrain_method=cn['constrain_method'])
    else:
        raise NotImplementedError("Network type is not implemented")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    LOG.info('Number of parameters {:}'.format(total_params))

    #### Start Training ####
    optimizer = torch.optim.Adam(model.parameters(), lr=c['lr'])

    alossBest = 1e6
    lr = c['lr']
    t0 = time.time()
    epochs_since_best = 0
    results=None
    epoch = 0

    # model_name_prev = 'C:/Users/Tue/PycharmProjects/results/run_MD_e3_batch/2021-07-07_13_28_06/5/model_best.pt'
    # # optimizer_name_prev = 'C:/Users/Tue/PycharmProjects/results/run_MD_e3_batch/2021-07-07_11_34_05/1/optimizer_last.pt'
    # LOG.info(f'Loading model from file')
    # model.load_state_dict(torch.load(model_name_prev))
    # # optimizer.load_state_dict(torch.load(optimizer_name_prev))
    # epoch = 9999
    # if c['viz']:
    #     vizdir = "{:}/figures/".format(c['result_dir'])
    #     os.makedirs(vizdir)
    csv_file = f"{c['result_dir']}/training.csv"

    while epoch < c['epochs']:
        # if c['viz']:
        #     viz = "{:}{:}".format(vizdir, epoch)
        # else:
        #     viz = ''

        t1 = time.time()
        aloss_t, aloss_rel_t = use_proteinmodel(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=c['batch_size'], reg=reg)
        t2 = time.time()
        if c['use_val']:
            aloss_v, aloss_rel_v = use_proteinmodel(model, dataloader_val, train=False, max_samples=1000, optimizer=optimizer, batch_size=c['batch_size'],reg=reg)
        else:
            aloss_v, alossr_v, alossv_v, alossD_v, alossDr_v, alossDv_v, MAEr_v,MAEv_v, P_mean_v, E_rel_diff_v = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            aloss_v = torch.tensor(0.0)
            aloss_rel_v = torch.tensor(0.0)
        t3 = time.time()
        LOG.info(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss_rel(train): {aloss_rel_t:.2e}  Loss(val): {aloss_v:.2e}  Loss_rel(val): {aloss_rel_v:.2e}  Time(train): {t2 - t1:.1f}s  Time(val): {t3 - t2:.1f}s  Lr: {lr:2.2e} ')
        if aloss_v < alossBest:
            alossBest = aloss_v
            # epochs_since_best = 0
            torch.save(model.state_dict(), f"{model_name_best}")
        else:
            epochs_since_best += 1
            if epochs_since_best >= c['epochs_for_lr_adjustment']:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.8
                    lr = g['lr']
                epochs_since_best = 0

        result = pd.DataFrame({
        'epoch': [epoch],
        'loss_t': [aloss_rel_t.cpu().numpy()],
        'loss_v': [aloss_rel_v.cpu().numpy()],
        },dtype=np.float32)
        result = result.astype({'epoch': np.int64})
        if epoch == 0:
            aresults = result
            aresults.iloc[-1:].to_csv(csv_file, header=True, sep='\t')
        else:
            aresults = aresults.append(result, ignore_index=True)
            aresults.iloc[-1:].to_csv(csv_file, mode='a', header=False, sep='\t')
        epoch += 1

    torch.save(model.state_dict(), f"{model_name}")
    if c['use_test']:
        model.load_state_dict(torch.load(model_name_best)) #TODO UNCLEAR THIS
        aloss, aloss_rel = use_proteinmodel(model, dataloader_test, train=False, max_samples=999999,optimizer=optimizer, batch_size=c['batch_size'],reg=reg)
        LOG.info(f'Loss: {aloss:.2e}  Loss_rel: {aloss_rel:.2e}')
    close_logger(LOG)
    fig = plt.figure(num=2)
    plt.clf()
    plt.semilogy(aresults['epoch'],aresults['loss_t'],label='training')
    plt.semilogy(aresults['epoch'],aresults['loss_v'],label='validation')
    plt.ylim(1e-4,1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    # plt.show()
    plt.savefig(f"{c['result_dir']}/loss.png")


    return aresults, dataloader_train, dataloader_val, dataloader_test
