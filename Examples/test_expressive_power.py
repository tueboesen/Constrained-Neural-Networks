import pickle
import argparse
from datetime import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from e3nn import o3

from src.main import main
from src.main_protein import main_protein

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 39589
    args.n_val = 0
    args.batch_size = 1
    args.n_input_samples = 1
    args.nskip = 9999
    args.epochs_for_lr_adjustment = 2
    args.use_val = False
    args.use_test = False
    args.use_endstep = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = 133548
    args.epochs = 20
    args.network_type = 'mim'
    args.loss = 'distogram'
    args.train_idx = None
    args.data = './../../../data/casp11/casp11_sel.npz'
    args.data_type = 'protein'

    if args.network_type.lower() == 'eq':
        args.network = {
            'irreps_inout': o3.Irreps("6x1o"),
            'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
            # 'irreps_node_attr': o3.Irreps("1x0e"),
            # 'irreps_edge_attr': o3.Irreps("{:}x1o".format(args.n_input_samples)),
            'layers': 8,
            'max_radius': 15,
            'number_of_basis': 8,
            'embed_dim': 2,
            'max_atom_types': 20,
            'radial_neurons': [48],
            'num_neighbors': -1,
        }
    elif args.network_type.lower() == 'mim':
        args.network = {
            'node_dim_in': 9,
            'node_attr_dim_in': 1,
            'node_dim_latent': 60,
            'nlayers': 3,
            'max_radius': 15,
        }


    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)

    result_dir_base = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )
    seeds = [1234,1235,1236,1237,1238]
    res_his = []
    for ii,seed in enumerate(seeds):
        c['seed'] = seed
        dataloader_train = None
        dataloader_val = None
        dataloader_test = None
        dataloader_endstep = None
        constrain_method = ['all_layers'] #['all_layers','reg','end']
        constraints = ['chain','triangle','chaintriangle']
        nskips = [9999]
        job = 0
        c['network']['constraints'] = 'chain'
        c['network']['constraint_type'] = 'high'  # high, low, reg
        c['network']['constraint_data'] = './../../../data/casp11/casp11_sel_constraints.pt'
        # c['network']['constrain_method'] = 'all_layers'
        # c['network']['constraints'] = ''
        c['result_dir'] = "{:}/{:}_{:}".format(result_dir_base, job,ii)
        results,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep = main(c,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep)
        if ii==0:
            res_his.append([])
        res_his[job].append(results)

        for nskip in nskips:
            c['nskip'] = nskip
            for constrain_methodi in constrain_method:
                c['network']['constrain_method'] = constrain_methodi
                for constraint in constraints:
                    c['network']['constraints'] = constraint
                    job += 1
                    c['result_dir'] = "{:}/{:}_{:}".format(result_dir_base, job, ii)
                    results,dataloader_train,dataloader_val,dataloader_test = main_protein(c,dataloader_train,dataloader_val,dataloader_test)
                    if ii==0:
                        res_his.append([])
                    res_his[job].append(results)


    outputfile = "{:}/results_history.pickle".format(result_dir_base)
    with open(outputfile, "wb") as fp:  # Pickling
        pickle.dump(res_his, fp)

    nl = len(res_his) #Number of different tries
    nr = len(res_his[0]) #Number of repeats
    ne = c['epochs']
    res_numpy = np.zeros((nl,nr,ne))
    for ii,a  in enumerate(res_his):
        for jj,b in enumerate(a):
            res_numpy[ii,jj,:] = b
    res_std = res_numpy.std(axis=1)
    res_mean = res_numpy.mean(axis=1)
    print("next")
    x = np.arange(ne)
    for ii in range(nl):
        pngfile = "{:}/training_{:}.png".format(result_dir_base,ii)
        fig, ax = plt.subplots()
        ax.plot(x, res_mean[ii], '-')
        ax.fill_between(x, res_mean[ii] - res_std[ii], res_mean[ii] + res_std[ii], alpha=0.2)
        plt.savefig(pngfile)
        plt.clf()