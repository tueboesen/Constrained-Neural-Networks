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
from src.constraints import BindingConstraintsNN, BindingConstraintsAB
from src.log import log_all_parameters, close_logger
from src.network_e3 import constrained_network
from src.network_eq import network_eq_simple
from src.utils import fix_seed, convert_snapshots_to_future_state_dataset, DatasetFutureState, run_network, run_network_eq, run_network_e3, atomic_masses, run_network_covid_e3


def main_covid(c):
    cn = c['network']
    fix_seed(c['seed'])  # Set a seed, so we make reproducible results.

    if 'result_dir' not in c:
        c['result_dir'] = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
            root='results',
            runner_name=c['basefolder'],
            date=datetime.now(),
        )
    model_name = "{}/{}.pt".format(c['result_dir'], 'model')
    os.makedirs(c['result_dir'])
    logfile_loc = "{}/{}.log".format(c['result_dir'], 'output')
    LOG = log.setup_custom_logger('runner', logfile_loc, c['mode'])
    log_all_parameters(LOG, c)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    # load training data
    data = np.load(c['data'], allow_pickle=True)
    Ra = torch.from_numpy(data['RCA']).to(dtype=torch.get_default_dtype())
    Rb = torch.from_numpy(data['RCB']).to(dtype=torch.get_default_dtype())
    Rn = torch.from_numpy(data['RN']).to(dtype=torch.get_default_dtype())
    z = torch.from_numpy(data['aa_num']).to(dtype=torch.int64)
    fragids = torch.from_numpy(data['fragid']).to(dtype=torch.get_default_dtype())
    R_org = torch.cat([Ra,Rb,Rn],dim=2)
    R = R_org[1:]
    V = R_org[1:] - R_org[:-1]
    nz = len(z.unique())

    fragid_unique = torch.unique(fragids)
    d_nn = 3.8*torch.ones(nz,nz)
    count_nn = torch.ones((nz,nz),dtype=torch.int64)
    d_ab = torch.zeros(nz)
    count_ab = torch.zeros(nz,dtype=torch.int64)
    d_an = torch.zeros(nz)
    count_an = torch.zeros(nz,dtype=torch.int64)

    for fragid_i in fragid_unique:
        idx = fragid_i == fragids
        Ri = R[:, idx, :]
        dRi = Ri[:, 1:, :] - Ri[:, :-1, :]
        dRa = dRi[:,:,:3]
        for i in range(dRa.shape[1]):
            d_nn[z[i],z[i+1]] += torch.sum(torch.sqrt(torch.sum(dRa[:,i,:]**2,dim=-1)))
            count_nn[z[i],z[i+1]] += dRa.shape[0]
        dRab = Ri[:,:,:3] - Ri[:,:,3:6]
        dRan = Ri[:,:,:3] - Ri[:,:,6:9]
        for i in range(Ri.shape[1]):
            d_ab[z[i]] += torch.sum(torch.sqrt(torch.sum(dRab[:,i,:]**2,dim=-1)))
            count_ab[z[i]] += dRab.shape[0]
            d_an[z[i]] += torch.sum(torch.sqrt(torch.sum(dRan[:,i,:]**2,dim=-1)))
            count_an[z[i]] += dRan.shape[0]

        distRa = torch.norm(dRa,dim=2)
        distRab = torch.norm(dRab,dim=2)
        distRan = torch.norm(dRan,dim=2)
        print(f"max={distRa.max():3.2f}, min={distRa.min():3.2f}, mean={distRa.mean():3.2f}. AlphaBeta_Dist(mean) = {distRab.mean():3.2f}, AlphaBeta_Dist(min) = {distRab.min():3.2f}, AlphaBeta_Dist(max) = {distRab.max():3.2f}, AlphaN_Dist(mean) = {distRan.mean():3.2f},AlphaN_Dist(min) = {distRan.min():3.2f} AlphaN_Dist(max) = {distRan.max():3.2f}")

    dist_nn = d_nn / count_nn
    dist_ab = d_ab / count_ab
    dist_an = d_an / count_an

    # dist_nnz = dist_nn[z[:-1],z[1:]]
    dist_abz = dist_ab[z]
    dist_anz = dist_an[z]

    # dR = R[:,1:,:] - R[:,:-1,:]
    # dRa = dR[:,:,:3]
    # dRb = dR[:,:,3:6]
    # dRn = dR[:,:,6:9]
    # distRa = torch.norm(dRa,dim=2)
    # distRb = torch.norm(dRb,dim=2)
    # distRn = torch.norm(dRn,dim=2)


    Rin, Rout = convert_snapshots_to_future_state_dataset(c['nskip'], R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(c['nskip'], V)

    # p = torch.sum(V * masses[None,:,None],dim=1)

    ndata = np.min([Rout.shape[0]])
    natoms = z.shape[0]

    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

    ndata_rand = 0 + np.arange(ndata)
    np.random.shuffle(ndata_rand)

    Rin_train = Rin[ndata_rand[:c['n_train']]]
    Rout_train = Rout[ndata_rand[:c['n_train']]]
    Vin_train = Vin[ndata_rand[:c['n_train']]]
    Vout_train = Vout[ndata_rand[:c['n_train']]]
    dataset_train = DatasetFutureState(Rin_train, Rout_train, z, Vin_train, Vout_train,device=device)
    dataloader_train = DataLoader(dataset_train, batch_size=c['batch_size'], shuffle=True, drop_last=True)

    if c['use_validation']:
        Rin_val = Rin[ndata_rand[c['n_train']:c['n_train'] + c['n_val']]]
        Rout_val = Rout[ndata_rand[c['n_train']:c['n_train'] + c['n_val']]]
        Vin_val = Vin[ndata_rand[c['n_train']:c['n_train'] + c['n_val']]]
        Vout_val = Vout[ndata_rand[c['n_train']:c['n_train'] + c['n_val']]]
        dataset_val = DatasetFutureState(Rin_val, Rout_val, z, Vin_val, Vout_val,device=device)
        dataloader_val = DataLoader(dataset_val, batch_size=c['batch_size'], shuffle=True, drop_last=False)

    if c['use_test']:
        Rin_test = Rin[ndata_rand[c['n_train'] + c['n_val']:]]
        Rout_test = Rout[ndata_rand[c['n_train'] + c['n_val']:]]
        Vin_test = Vin[ndata_rand[c['n_train'] + c['n_val']:]]
        Vout_test = Vout[ndata_rand[c['n_train'] + c['n_val']:]]
        dataset_test = DatasetFutureState(Rin_test, Rout_test, z, Vin_test, Vout_test,device=device)
        dataloader_test = DataLoader(dataset_test, batch_size=c['batch_size'], shuffle=False, drop_last=False)


    if cn['constraints'] == 'binding':
        constraints = BindingConstraintsNN(3.8, fragmentid=fragids)
        constraints2 = None
    elif cn['constraints'] == 'bindingall':
        constraints = BindingConstraintsNN(3.8, fragmentid=fragids)
        constraints2 = BindingConstraintsAB(d_ab=dist_abz.to(device),d_an=dist_anz.to(device),fragmentid=fragids)
    else:
        constraints = None
        constraints2 = None

    model = constrained_network(irreps_inout=cn['irreps_inout'], irreps_hidden=cn['irreps_hidden'], irreps_edge_attr=cn['irreps_edge_attr'], layers=cn['layers'],
                                max_radius=cn['max_radius'],
                                number_of_basis=cn['number_of_basis'], radial_neurons=cn['radial_neurons'], num_neighbors=cn['num_neighbors'],
                                num_nodes=natoms, embed_dim=cn['embed_dim'], max_atom_types=cn['max_atom_types'], constraints=constraints, constraints2=constraints2)
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
        aloss_t, alossr_t, alossv_t, alossD_t, alossDr_t, alossDv_t, aloss_ref_t, ap_t, _, _ = run_network_covid_e3(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=c['batch_size'], max_radius=cn['max_radius'])
        t2 = time.time()
        if c['use_validation']:
            aloss_v, alossr_v, alossv_v, alossD_v, alossDr_v, alossDv_v, aloss_ref_v, ap_v, _,_ = run_network_covid_e3(model, dataloader_val, train=False, max_samples=100, optimizer=optimizer, batch_size=c['batch_size'], max_radius=cn['max_radius'])
        else:
            aloss_v, alossr_v, alossv_v, alossD_v, alossDr_v, alossDv_v, aloss_ref_v, ap_v, _,_ = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
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

        LOG.info(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss(val): {aloss_v:.2e}  Loss_ref(train): {aloss_ref_t:.2e}  LossD(train): {alossD_t:.2e}  LossD(val): {alossD_v:.2e} Loss_r(train): {alossr_t:.2e}  Loss_v(train): {alossv_t:.2e}   Loss_r(val): {alossr_v:.2e}  Loss_v(val): {alossv_v:.2e}   P(train): {ap_t:.2e}  P(val): {ap_v:.2e}  Loss_best(val): {alossBest:.2e}  Time(train): {t2 - t1:.1f}s  Time(val): {t3 - t2:.1f}s  Lr: {lr:2.2e} ')
    torch.save(model.state_dict(), f"{model_name}")
    if c['use_test']:
        aloss, alossr, alossv, alossD, alossDr, alossDv, aloss_ref, ap, MAEr, MAEv = run_network_e3(model, dataloader_test, train=False, max_samples=100, optimizer=optimizer, batch_size=c['batch_size'], max_radius=cn['max_radius'])
        LOG.info(f'Loss: {aloss:.2e}  Loss_rel: {aloss/aloss_ref:.2e}  Loss_ref: {aloss_ref:.2e}  LossD: {alossD:.2e}  Loss_r: {alossr:.2e}  Loss_v: {alossv:.2e}  P: {ap:.2e}  MAEr:{MAEr:.2e} MAEv:{MAEv:.2e}')
        results = {'loss': aloss,
            'loss_rel': aloss/aloss_ref,
            'loss_ref': aloss_ref,
            'loss_D': alossD,
            'loss_Dr': alossDr,
            'loss_Dv': alossDv,
            'loss_r': alossr,
            'loss_v': alossv,
            'momentum': ap,
            'mean_absolute_error_r': MAEr,
            'mean_absolute_error_r': MAEv,
                   }
        output = {"config":c,
               'results': results,
               }
        np.save("{:}/test_results".format(c['result_dir']),output)
    else:
        results = None

    close_logger(LOG)
    return results
