import argparse
import pickle
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
from src.utils import fix_seed, convert_snapshots_to_future_state_dataset, run_network, run_network_eq, run_network_e3, atomic_masses

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 10000
    args.n_val = 1000
    args.batch_size = 15
    args.n_input_samples = 1
    args.nskip = 999
    args.epochs_for_lr_adjustment = 10000
    args.use_val = True
    args.use_test = True
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = 123545
    args.epochs = 100
    args.network_type = 'mim'
    args.loss = 'distogram'
    args.train_idx = None
    args.PE_predictor = './../pretrained_networks/force_energy_model.pt'
    # args.data = './../../../data/MD/argon/argon.npz'
    args.data = './../../../data/MD/water_jones/water.npz'
    # args.data = './../../../data/MD/MD17/ethanol_dft.npz'



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


    dataloader_train=None
    dataloader_val=None
    dataloader_test=None
    # network_types = ['EQ']
    network_types = ['EQ','mim']
    constraints = ['','triangle']
    # nskips = [9999]
    job = 0
    res_his = []
    result_dir_base = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )
    # seeds = [1234,1235]
    seeds = [1234,1235,1236,1237,1238]
    res_his = []
    for ii,seed in enumerate(seeds):
        c['seed'] = seed
        dataloader_train = None
        dataloader_val = None
        dataloader_test = None
        job = 0
        # c['network']['constrain_method'] = 'all_layers'
        # c['network']['constraints'] = ''
        for jj,network_type in enumerate(network_types):
            c['network_type'] = network_type
            if c['network_type'].lower() == 'eq':
                c['loss'] = 'EQ'
                c['network'] = {
                    'irreps_inout': o3.Irreps("6x1o"),
                    'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
                    # 'irreps_node_attr': o3.Irreps("1x0e"),
                    # 'irreps_edge_attr': o3.Irreps("{:}x1o".format(args.n_input_samples)),
                    'layers': 6,
                    'max_radius': 100,
                    'number_of_basis': 8,
                    'embed_dim': 8,
                    'max_atom_types': 20,
                    'radial_neurons': [48],
                    'num_neighbors': -1,
                    'constraints': 'triangle',
                    'constrain_method': 'all_layers',
                }
                c['lr'] = 1e-2
                c['batch_size'] = 25
            elif c['network_type'].lower() == 'mim':
                c['loss'] = 'distogram'
                c['network'] = {
                    'node_dim_in': 18,
                    'node_attr_dim_in': 1,
                    'node_dim_latent': 90,
                    'nlayers': 6,
                    'max_radius': 100,
                    'constraints': 'triangle',
                    'constrain_method': 'all_layers',
                }
                c['lr'] = 1e-3
                c['batch_size'] = 25
            else:
                raise NotImplementedError("network type not recognized")
            for kk,constraint in enumerate(constraints):
                c['result_dir'] = "{:}/{:}_{:}_{:}".format(result_dir_base, network_type,constraint,ii)
                c['network']['constraints'] = constraint
                results,dataloader_train,dataloader_val,dataloader_test = main(c,dataloader_train,dataloader_val,dataloader_test)
                if ii==0:
                    res_his.append([])
                res_his[job].append(results)
                job += 1

    outputfile = "{:}/results_history.pickle".format(result_dir_base)
    with open(outputfile, "wb") as fp:  # Pickling
        pickle.dump(res_his, fp)

    nl = len(res_his) #Number of different tries
    nr = len(res_his[0]) #Number of repeats
    ne = c['epochs']
    loss_types = ['loss_t','loss_v','lossD_t','lossD_v']
    nlosses = len(loss_types)
    res_numpy = np.zeros((nlosses,nl,nr,ne))
    for kk,loss_type  in enumerate(loss_types):
        for ii,a  in enumerate(res_his):
            for jj,b in enumerate(a):
                res_numpy[kk,ii,jj,:] = b[loss_type]
    res_std = res_numpy.std(axis=2)
    res_mean = res_numpy.mean(axis=2)
    print("next")
    x = np.arange(ne)

    legends=['mim','mim_constraints']
    for kk,loss_type  in enumerate(loss_types):
        for ii in range(nl):
            pngfile = "{:}/training_semi_{:}_{:}.png".format(result_dir_base,loss_type,ii)
            fig, ax = plt.subplots()
            ax.semilogy(x, res_mean[kk,ii], '-',label=legends[kk])
            # ax.plot(x, res_mean[ii], '-')
            ax.fill_between(x, res_mean[kk,ii] - res_std[kk,ii], res_mean[kk,ii] + res_std[kk,ii], alpha=0.2)
            plt.ylim(1e-4,2)
            ax.legend()
            plt.savefig(pngfile)
            plt.clf()


    for kk,loss_type  in enumerate(loss_types):
        fig, ax = plt.subplots()
        pngfile = "{:}/training_all_{:}.png".format(result_dir_base,loss_type)
        # legends = ['no constraint','constraints']
        for ii,legend in enumerate(legends):
            ax.semilogy(x, res_mean[kk,ii], '-',label=legend)
            # ax.plot(x, res_mean[ii], '-')
            ax.fill_between(x, res_mean[kk,ii] - res_std[kk,ii], res_mean[kk,ii] + res_std[kk,ii], alpha=0.2)
        ax.legend()
        plt.ylim(1e-4,2)

        plt.savefig(pngfile)
        plt.clf()


    # for ii in range(nl):
    #     pngfile = "{:}/training_{:}.png".format(result_dir_base,ii)
    #     fig, ax = plt.subplots()
    #     ax.plot(x, res_mean[ii], '-')
    #     ax.fill_between(x, res_mean[ii] - res_std[ii], res_mean[ii] + res_std[ii], alpha=0.2)
    #     plt.savefig(pngfile)
    #     plt.clf()


    # with open(outputfile, "rb") as fp:   # Unpickling
    #     b = pickle.load(fp)
