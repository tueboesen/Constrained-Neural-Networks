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
from src.EQ_operations import ProjectUplift, ProjectUplift_conv
from src.log import log_all_parameters, close_logger
from src.network import network_simple
from src.network_e3 import constrained_network
from src.constraints import MomentumConstraints, PointChain, PointToPoint
from src.network_eq import network_eq_simple
from src.utils import fix_seed, convert_snapshots_to_future_state_dataset, DatasetFutureState, run_network, \
    run_network_eq, run_network_e3, atomic_masses, Distogram


def main(c):
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
    data = np.load(c['data'])
    R = torch.from_numpy(data['R']).to(device=device,dtype=torch.get_default_dtype())
    V = torch.from_numpy(data['V']).to(device=device,dtype=torch.get_default_dtype())
    F = torch.from_numpy(data['F']).to(device=device,dtype=torch.get_default_dtype())
    KE = torch.from_numpy(data['KE']).to(device=device,dtype=torch.get_default_dtype())
    PE = torch.from_numpy(data['PE']).to(device=device,dtype=torch.get_default_dtype())

    #We rescale the data
    R /= torch.sqrt(R.pow(2).mean())
    V /= torch.sqrt(V.pow(2).mean())

    DR0 = Distogram(R[0]).fill_diagonal_(99.0)
    nn_dist_start = torch.min(DR0,dim=0)[0].max()
    DRend = Distogram(R[-1]).fill_diagonal_(99.0)
    nn_dist_end = torch.min(DRend,dim=0)[0].max()

    Rin, Rout = convert_snapshots_to_future_state_dataset(c['nskip'], R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(c['nskip'], V)
    Fin, Fout = convert_snapshots_to_future_state_dataset(c['nskip'], F)
    KEin, KEout = convert_snapshots_to_future_state_dataset(c['nskip'], KE)
    PEin, PEout = convert_snapshots_to_future_state_dataset(c['nskip'], PE)
    z = torch.from_numpy(data['z']).to(device=device)
    masses = atomic_masses(z)

    R = None
    V = None
    F = None

    # p = torch.sum(V * masses[None,:,None],dim=1)

    ndata = Rout.shape[0]
    natoms = z.shape[0]

    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

    ndata_rand = 0 + np.arange(ndata)
    np.random.shuffle(ndata_rand)
    if c['train_idx'] is None:
        train_idx = ndata_rand[:c['n_train']]
        val_idx = ndata_rand[c['n_train']:c['n_train'] + c['n_val']]
        test_idx = ndata_rand[c['n_train'] + c['n_val']:]
    else:
        train_idx = c['train_idx']
        val_idx = c['val_idx']
        test_idx = c['test_idx']


    Rin_train = Rin[train_idx]
    Rout_train = Rout[train_idx]
    Vin_train = Vin[train_idx]
    Vout_train = Vout[train_idx]
    Fin_train = Fin[train_idx]
    Fout_train = Fout[train_idx]
    KEin_train = KEin[train_idx]
    KEout_train = KEout[train_idx]
    PEin_train = PEin[train_idx]
    PEout_train = PEout[train_idx]



    if c['use_validation']:
        Rin_val = Rin[val_idx]
        Rout_val = Rout[val_idx]
        Vin_val = Vin[val_idx]
        Vout_val = Vout[val_idx]
        Fin_val = Fin[val_idx]
        Fout_val = Fout[val_idx]
        KEin_val = KEin[val_idx]
        KEout_val = KEout[val_idx]
        PEin_val = PEin[val_idx]
        PEout_val = PEout[val_idx]

    if c['use_test']:
        Rin_test = Rin[test_idx]
        Rout_test = Rout[test_idx]
        Vin_test = Vin[test_idx]
        Vout_test = Vout[test_idx]
        Fin_test = Fin[test_idx]
        Fout_test = Fout[test_idx]
        KEin_test = KEin[test_idx]
        KEout_test = KEout[test_idx]
        PEin_test = PEin[test_idx]
        PEout_test = PEout[test_idx]


    Fin = None
    Fout = None
    KEin = None
    KEout = None
    PEin = None
    PEout = None
    Fin_train = None
    Fout_train = None
    KEin_train = None
    KEout_train = None
    PEin_train = None
    PEout_train = None

    dataset_train = DatasetFutureState(Rin_train, Rout_train, z, Vin_train, Vout_train, Fin_train, Fout_train, KEin_train, KEout_train, PEin_train, PEout_train, masses,device=device)
    if c['use_validation']:
        dataset_val = DatasetFutureState(Rin_val, Rout_val, z, Vin_val, Vout_val, Fin_val, Fout_val, KEin_val, KEout_val, PEin_val, PEout_val, masses,device=device)
        dataloader_val = DataLoader(dataset_val, batch_size=c['batch_size'], shuffle=True, drop_last=False)


    if c['use_test']:
        dataset_test = DatasetFutureState(Rin_test, Rout_test, z, Vin_test, Vout_test, Fin_test, Fout_test, KEin_test, KEout_test, PEin_test, PEout_test, masses,device=device)
        dataloader_test = DataLoader(dataset_test, batch_size=c['batch_size'], shuffle=False, drop_last=False)

    dataloader_train = DataLoader(dataset_train, batch_size=c['batch_size'], shuffle=True, drop_last=True)

    PU = ProjectUplift(cn['irreps_inout'], cn['irreps_hidden'])
    if cn['constraints'] == 'chain':
        constraints = torch.nn.Sequential(PointChain(PU.project,PU.uplift,3.8, fragmentid=fragids))
    elif cn['constraints'] == 'triangle':
        constraints = torch.nn.Sequential(PointToPoint(PU.project,PU.uplift,r=dist_abz.to(device)),PointToSphereSphereIntersection(PU.project,PU.uplift,r1=dist_anz.to(device),r2=dist_bnz.to(device)))
    elif cn['constraints'] == 'chaintriangle':
        constraints = torch.nn.Sequential(PointChain(PU.project,PU.uplift,3.8, fragmentid=fragids),PointToPoint(PU.project,PU.uplift,r=dist_abz.to(device)),PointToSphereSphereIntersection(PU.project,PU.uplift,r1=dist_anz.to(device),r2=dist_bnz.to(device)))
        # constraints2 = BindingConstraintsAB(d_ab=dist_abz.to(device), d_an=dist_anz.to(device), fragmentid=fragids)
    else:
        constraints = None

    if c['network_type'] == 'EQ':
        model = constrained_network(irreps_inout=cn['irreps_inout'], irreps_hidden=cn['irreps_hidden'], layers=cn['layers'],
                                    max_radius=cn['max_radius'],
                                    number_of_basis=cn['number_of_basis'], radial_neurons=cn['radial_neurons'], num_neighbors=cn['num_neighbors'],
                                    num_nodes=natoms, embed_dim=cn['embed_dim'], max_atom_types=cn['max_atom_types'], constraints=constraints, PU=PU)
    elif c['network_type'] == 'mim':
        model = network_simple(cn['node_dim_in'], cn['node_attr_dim_in'], cn['node_dim_latent'], cn['nlayers'], constraints=cn['constraints'], masses=masses)
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

    # model_name_prev = 'C:/Users/Tue/PycharmProjects/results/run_MD_e3_batch/2021-07-07_11_34_05/1/model_last.pt'
    # optimizer_name_prev = 'C:/Users/Tue/PycharmProjects/results/run_MD_e3_batch/2021-07-07_11_34_05/1/optimizer_last.pt'
    # LOG.info(f'Loading model from file')
    # model.load_state_dict(torch.load(model_name_prev))
    # optimizer.load_state_dict(torch.load(optimizer_name_prev))

    while epoch < c['epochs']:
        t1 = time.time()
        aloss_t, alossr_t, alossv_t, alossD_t, alossDr_t, alossDv_t, ap_t, MAEr_t, MAEv_t = run_network_e3(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, loss_fnc=c['loss'], batch_size=c['batch_size'], max_radius=cn['max_radius'], debug=c['debug'], log=LOG)
        t2 = time.time()
        if c['use_validation']:
            aloss_v, alossr_v, alossv_v, alossD_v, alossDr_v, alossDv_v, ap_v, MAEr_v,MAEv_v = run_network_e3(model, dataloader_val, train=False, max_samples=1000, optimizer=optimizer, loss_fnc=c['loss'], batch_size=c['batch_size']*10, max_radius=cn['max_radius'], log=LOG)
        else:
            aloss_v, alossr_v, alossv_v, alossD_v, alossDr_v, alossDv_v, ap_v, MAEr_v,MAEv_v = 0, 0, 0, 0, 0, 0, 0, 0, 0
        t3 = time.time()

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
        # if (epoch+1) % c['epochs_for_lr_adjustment'] == 0:
        #     for g in optimizer.param_groups:
        #         g['lr'] *= 0.8
        #         lr = g['lr']
        #

        LOG.info(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss_r(train): {alossr_t:.2e}  Loss_v(train): {alossv_t:.2e}   Loss(val): {aloss_v:.2e}  LossD(train): {alossD_t:.2e}  LossD(val): {alossD_v:.2e} MAE_r(val): {MAEr_v:.2e}  MAE_v(val): {MAEv_v:.2e}   MAE_r(train): {MAEr_t:.2e}  MAE_v(train): {MAEv_t:.2e} P(train): {ap_t:.2e}  P(val): {ap_v:.2e}  Loss_best(val): {alossBest:.2e}  Time(train): {t2 - t1:.1f}s  Time(val): {t3 - t2:.1f}s  Lr: {lr:2.2e} ')
        if torch.isnan(aloss_t):
            LOG.info(f'nan detected, reloading model, resetting epoch and lowering lr')
            model.load_state_dict(torch.load(model_name_last))
            epoch -= 1
            for g in optimizer.param_groups:
                g['lr'] *= 0.8
                lr = g['lr']
            epochs_since_best = 0
            optimizer.load_state_dict(torch.load(optimizer_name_last))
        else:
            torch.save(model.state_dict(), f"{model_name_last}")
            torch.save(optimizer.state_dict(),f"{optimizer_name_last}")
        epoch += 1

    torch.save(model.state_dict(), f"{model_name}")
    if c['use_test']:
        model.load_state_dict(torch.load(model_name_best))
        aloss, alossr, alossv, alossD, alossDr, alossDv, ap, MAEr, MAEv = run_network_e3(model, dataloader_test, train=False, max_samples=999999, optimizer=optimizer, loss_fnc=c['loss'], batch_size=c['batch_size'], max_radius=cn['max_radius'], log=LOG, debug=c['debug'])
        LOG.info(f'Loss: {aloss:.2e}  LossD: {alossD:.2e}  Loss_r: {alossr:.2e}  Loss_v: {alossv:.2e}  P: {ap:.2e}  MAEr:{MAEr:.2e} MAEv:{MAEv:.2e}')
        results = {'loss': aloss,
            'loss_rel': aloss,
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

    close_logger(LOG)
    return results
