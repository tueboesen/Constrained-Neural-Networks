import pickle
import argparse
from datetime import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from e3nn import o3

from src.batch_jobs import job_planner, job_runner
from src.main import main
from src.main_protein import main_protein
from src.vizualization import plot_training_and_validation_accumulated

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Constrained MD')
    args = parser.parse_args()
    args.n_train = 10
    args.n_val = 0
    args.batch_size = 1
    args.n_input_samples = 1
    args.nskip = 9999
    args.epochs_for_lr_adjustment = 300
    args.lr_adjustment = 0.8
    args.use_val = False
    args.use_test = False
    args.perform_endstep_MD_propagation = False
    args.debug = False
    args.viz = False
    args.lr = 1e-3
    args.seed = [1,2,3,4,5]
    args.use_same_data = True
    args.epochs = 300
    args.network_type = 'mim'
    args.loss = 'mim'
    args.train_idx = None
    args.data = './../../../data/casp11/casp11_sel.npz'
    args.data_type = 'protein'
    args.con = ['','chain','triangle','chaintriangle']
    args.con_type = ['high', 'low','reg']
    args.con_data = './../../../data/casp11/casp11_sel_cons.pt'
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

    cs, legends, results = job_planner(c)

    job_runner(cs,legends,results)

    print("jobs")

    # seeds = [1234,1235,1236,1237,1238]
    # res_his = []
    # legend_his = []
    # for ii,seed in enumerate(seeds):
    #     c['seed'] = seed
    #     dataloader_train = None
    #     dataloader_val = None
    #     dataloader_test = None
    #     dataloader_endstep = None
    #     nskips = [9999]
    #     job = 0
    #     c['network']['con'] = ''
    #     c['network']['con_type'] = 'high'  # high, low, reg
    #     c['result_dir'] = "{:}/{:}_{:}".format(result_dir_base, job,ii)
    #     results,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep = main(c,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep)
    #     if ii==0:
    #         res_his.append([])
    #         legend_his.append([])
    #     res_his[job].append(results)
    #
    #     for nskip in nskips:
    #         c['nskip'] = nskip
    #         for con_type in con_types:
    #             c['network']['con_type'] = con_type
    #             for con in cons:
    #                 c['network']['con'] = con
    #                 job += 1
    #                 c['result_dir'] = "{:}/{:}_{:}".format(result_dir_base, job, ii)
    #                 results,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep = main(c,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep)
    #                 if ii==0:
    #                     res_his.append([])
    #                     legend_his.append([])
    #                 res_his[job].append(results)
    #                 legend_his[job] = f"con:{c['network']['con']:}, type:{c['network']['con_type']:}"
    #
    #
    # outputfile = "{:}/results_history.pickle".format(result_dir_base)
    # with open(outputfile, "wb") as fp:  # Pickling
    #     pickle.dump(res_his, fp)
    #     pickle.dump(legend_his, fp)
    #
    # plot_training_and_validation_accumulated(res_his, legend_his, result_dir_base)