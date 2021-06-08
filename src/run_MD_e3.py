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
from src.network_e3 import constrained_network
from src.network_eq import network_eq_simple
from src.utils import fix_seed, convert_snapshots_to_future_state_dataset, DatasetFutureState, run_network, run_network_eq, run_network_e3, atomic_masses

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.mode ='standard'
    args.n_train = 1
    args.n_val = 1000
    args.batch_size = 1
    args.n_input_samples = 1
    args.n_skips = 0
    args.epochs_for_lr_adjustment = 50
    args.use_validation = False
    args.lr = 1e-2
    args.seed = 123545
    args.epochs = 1000
    args.PE_predictor = './../pretrained_networks/force_energy_model.pt'
    args.data = './../../../data/MD/ethanol/ethanol.npz'
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
        'constraints': 'P'

    }
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)
    cn = c['network']

    fix_seed(c['seed']) #Set a seed, so we make reproducible results.

    c['result_dir'] = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )
    model_name = "{}/{}.pt".format(c['result_dir'], 'model')
    os.makedirs(c['result_dir'])
    logfile_loc = "{}/{}.log".format(c['result_dir'], 'output')
    LOG = log.setup_custom_logger('runner',logfile_loc,c['mode'])
    log_all_parameters(LOG, c)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    # load training data
    data = np.load(c['data'])
    R = torch.from_numpy(data['R']).to( device=device)
    V = torch.from_numpy(data['V']).to( device=device)
    F = torch.from_numpy(data['F']).to( device=device)
    KE = torch.from_numpy(data['KE']).to(device=device)
    PE = torch.from_numpy(data['PE']).to( device=device)

    Rin, Rout = convert_snapshots_to_future_state_dataset(c['n_skips'], R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(c['n_skips'], V)
    Fin, Fout = convert_snapshots_to_future_state_dataset(c['n_skips'], F)
    KEin, KEout = convert_snapshots_to_future_state_dataset(c['n_skips'], KE)
    PEin, PEout = convert_snapshots_to_future_state_dataset(c['n_skips'], PE)
    z = torch.from_numpy(data['z']).to(device=device)
    masses = atomic_masses(z)

    # p = torch.sum(V * masses[None,:,None],dim=1)

    ndata = Rout.shape[0]
    natoms = z.shape[0]

    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

    ndata_rand = 0 + np.arange(ndata)
    np.random.shuffle(ndata_rand)


    Rin_train = Rin[ndata_rand[:c['n_train']]]
    Rout_train = Rout[ndata_rand[:c['n_train']]]
    Vin_train = Vin[ndata_rand[:c['n_train']]]
    Vout_train = Vout[ndata_rand[:c['n_train']]]
    Fin_train = Fin[ndata_rand[:c['n_train']]]
    Fout_train = Fout[ndata_rand[:c['n_train']]]
    KEin_train = KEin[ndata_rand[:c['n_train']]]
    KEout_train = KEout[ndata_rand[:c['n_train']]]
    PEin_train = PEin[ndata_rand[:c['n_train']]]
    PEout_train = PEout[ndata_rand[:c['n_train']]]

    Rin_val = Rin[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    Rout_val = Rout[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    Vin_val = Vin[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    Vout_val = Vout[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    Fin_val = Fin[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    Fout_val = Fout[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    KEin_val = KEin[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    KEout_val = KEout[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    PEin_val = PEin[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    PEout_val = PEout[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]

    dataset_train = DatasetFutureState(Rin_train, Rout_train, z,Vin_train,Vout_train,Fin_train,Fout_train,KEin_train,KEout_train,PEin_train,PEout_train,masses)
    dataset_val = DatasetFutureState(Rin_val, Rout_val, z,Vin_val,Vout_val,Fin_val,Fout_val,KEin_val,KEout_val,PEin_val,PEout_val,masses)

    dataloader_train = DataLoader(dataset_train, batch_size=c['batch_size'], shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=c['batch_size'], shuffle=True, drop_last=False)

    if cn['constraints'] == 'EP':
        PE_predictor = generate_FE_network(natoms=natoms)
        PE_predictor.load_state_dict(torch.load(c['PE_predictor'], map_location=torch.device('cpu')))
        PE_predictor.eval()
    else:
        PE_predictor = None


    model = constrained_network(irreps_inout=cn['irreps_inout'], irreps_hidden=cn['irreps_hidden'], irreps_edge_attr=cn['irreps_edge_attr'], layers=cn['layers'],
                   max_radius=cn['max_radius'],
                   number_of_basis=cn['number_of_basis'], radial_neurons=cn['radial_neurons'], num_neighbors=cn['num_neighbors'],
                   num_nodes=natoms, embed_dim=cn['embed_dim'], max_atom_types=cn['max_atom_types'], masses=masses, constraints=cn['constraints'])
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
        aloss_t, alossr_t, alossv_t, aloss_ref_t, ap_t, MAE_t = run_network_e3(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=c['batch_size'], max_radius=cn['max_radius'])
        t2 = time.time()
        if c['use_validation']:
            aloss_v, alossr_v, alossv_v, aloss_ref_v, ap_v, MAE_v= run_network_e3(model, dataloader_val, train=False, max_samples=100, optimizer=optimizer, batch_size=c['batch_size'],max_radius=cn['max_radius'])
        else:
            aloss_v, alossr_v, alossv_v, aloss_ref_v, ap_v, MAE_v = 0,0,0,0,0,0
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
        LOG.info(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss(val): {aloss_v:.2e}  Loss_ref(train): {aloss_ref_t:.2e} Loss_ref(val): {aloss_ref_v:.2e} Loss_r(train): {alossr_t:.2e}  Loss_v(train): {alossv_t:.2e}   Loss_r(val): {alossr_v:.2e}  Loss_v(val): {alossv_v:.2e}   P(train): {ap_t:.2e}  P(val): {ap_v:.2e}  Loss_best(val): {alossBest:.2e}  Time(train): {t2 - t1:.1f}s  Time(val): {t3 - t2:.1f}s  Lr: {lr:2.2e} ')
    torch.save(model.state_dict(), f"{model_name}")

