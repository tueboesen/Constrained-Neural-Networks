import os
import time
from datetime import datetime

import torch

from src import log
from src.constraints import load_constraints, load_constraint_parameters
from src.dataloader import load_data
from src.log import log_all_parameters, close_logger
from src.loss import find_relevant_loss
from src.network_e3 import neural_network_equivariant
from src.network_eq_simple import neural_network_equivariant_simple
from src.network_mim import neural_network_mimetic
from src.optimization import run_model
from src.project_uplift import ProjectUpliftEQ
from src.utils import fix_seed, update_results_and_save_to_csv, run_model_MD_propagation_simulation, save_test_results_to_csv
from src.vizualization import plot_training_and_validation


def main(c,dataloader_train=None,dataloader_val=None,dataloader_test=None,dataloader_endstep=None):
    """
    The main function which should be called with a fitting configuration dictionary.
    c is the input configuration dictionary, which dictates the run.
    """
    cn = c['network']
    if c['use_double']:
        torch.set_default_dtype(torch.float64)
    fix_seed(c['seed'])  # Set a seed, so we make reproducible results.
    if 'result_dir' not in c:
        c['result_dir'] = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
            root='results',
            runner_name=c['basefolder'],
            date=datetime.now(),
        )
    model_name = "{}/{}.pt".format(c['result_dir'], 'model')
    model_name_best = "{}/{}.pt".format(c['result_dir'], 'model_best')
    os.makedirs(c['result_dir'])
    logfile_loc = "{}/{}.log".format(c['result_dir'], 'output')
    debug_folder = f"{c['result_dir']}/debug"
    LOG = log.setup_custom_logger('runner', logfile_loc, c['debug'])
    log_all_parameters(LOG, c)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # load training data
    if dataloader_train is None:
        dataloader_train, dataloader_val, dataloader_test, dataloader_endstep = load_data(c['data'], c['data_type'], device, c['nskip'], c['n_train'], c['n_val'], c['n_test'], c['use_val'], c['use_test'], c['batch_size'], use_endstep=c['perform_endstep_MD_propagation'],file_val=c['data_val'],model_specific=c['model_specific'])
    ds = dataloader_train.dataset

    if c['network_type'].lower() == 'eq':
        PU = ProjectUpliftEQ(cn['irreps_inout'], cn['irreps_hidden'])

    cv = load_constraint_parameters(c['con'], c['con_type'], c['data_type'], con_data=c['con_data'],model_specific=c['model_specific'])

    con_fnc = load_constraints(c['con'], c['con_type'], con_variables=cv,rscale=ds.rscale,vscale=ds.vscale,device=device,debug_folder=debug_folder)

    if c['network_type'].lower() == 'eq':
        model = neural_network_equivariant(irreps_inout=cn['irreps_inout'], irreps_hidden=cn['irreps_hidden'], layers=cn['layers'],
                                    max_radius=cn['max_radius'],
                                    number_of_basis=cn['number_of_basis'], radial_neurons=cn['radial_neurons'], num_neighbors=cn['num_neighbors'],
                                    num_nodes=ds.Rin.shape[1], embed_dim=cn['embed_dim'], max_atom_types=cn['max_atom_types'], con_fnc=con_fnc, con_type=c['con_type'], PU=PU, particles_pr_node=ds.particles_pr_node,discretization=c['network_discretization'],gamma=c['penalty'])
    elif c['network_type'].lower() == 'eq_simple':
        model = neural_network_equivariant_simple(cn['nlayers'],gamma=c['penalty'],dim=c['data_dim'],con_fnc=con_fnc,con_type=c['con_type'],discretization=c['network_discretization'])
    elif c['network_type'] == 'mim':
        node_dim_in = cn['node_dim_in'] if ds.pos_only else cn['node_dim_in'] * 2
        model = neural_network_mimetic(node_dim_in,cn['node_dim_latent'], cn['nlayers'], con_fnc=con_fnc, con_type=c['con_type'],dim=c["data_dim"],discretization=c['network_discretization'],gamma=c['penalty'],regularization=c['regularization'])
    else:
        raise NotImplementedError("Network type is not implemented")

    if c['load_previous_model_file'] != '':
        model.load_state_dict(torch.load(c['load_previous_model_file'], map_location=device))

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    LOG.info('Number of parameters {:}'.format(total_params))

    lr = c['lr']
    optimizer = torch.optim.Adam([{"params": model.params.base.parameters()},
                                  {"params": model.params.h.parameters()},
                                  {'params': model.params.close.parameters(), 'lr': lr*0.1}], lr=lr)
    lossBest = 1e20
    epochs_since_best = 0
    results=None
    results_test=None
    epoch = 0
    csv_file = f"{c['result_dir']}/training.csv"

    t0 = time.time()
    while epoch < c['epochs']:
        t1 = time.time()
        if c['use_training']:
            loss_r_t, loss_v_t,drmsd_t,cv_t, cv_max_t,cv_energy_t,cv_energy_max_t, MAE_r_t, MAE_v_t,reg_t, reg2_t = run_model(c['data_type'], model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, loss_fnc=c['loss'], max_radius=cn['max_radius'], debug=c['debug'],epoch=epoch, output_folder=c['result_dir'],nviz=c['nviz'],regularization=c['regularization'])
        else:
            loss_r_t, loss_v_t, drmsd_t,cv_t, cv_max_t,cv_energy_t,cv_energy_max_t,MAE_r_t, MAE_v_t, reg_t, reg2_t = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        t2 = time.time()
        if c['use_val']:
            loss_r_v, loss_v_v, drmsd_v,cv_v, cv_max_v,cv_energy_v,cv_energy_max_v, MAE_r_v, MAE_v_v, reg_v, reg2_v = run_model(c['data_type'], model, dataloader_val, train=False, max_samples=1000, optimizer=optimizer, loss_fnc=c['loss'], max_radius=cn['max_radius'], debug=c['debug'],epoch=epoch, output_folder=c['result_dir'],nviz=c['nviz'],regularization=c['regularization'])
        else:
            loss_r_v, loss_v_v, drmsd_v,cv_v, cv_max_v,cv_energy_v,cv_energy_max_v, MAE_r_v, MAE_v_v, reg_v, reg2_v = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        t3 = time.time()
        if c['ignore_cons']:
            _,_,_,_,_ = run_model(c['data_type'], model, dataloader_train, train=False, max_samples=9, optimizer=optimizer, loss_fnc=c['loss'], max_radius=cn['max_radius'], debug=c['debug'],epoch=epoch, output_folder=c['result_dir'],ignore_cons=True)


        # Next save results
        # loss_t = (loss_r_t + loss_v_t) / 2
        # loss_v = (loss_r_v + loss_v_v) / 2

        results = update_results_and_save_to_csv(results, epoch, loss_r_t,loss_v_t,cv_max_t,loss_r_v,loss_v_v, cv_max_v, MAE_r_t, MAE_r_v, MAE_v_t, MAE_v_v, csv_file,cv_t,cv_v,cv_energy_t,cv_energy_max_t,cv_energy_v,cv_energy_max_v)
        if c['use_val']:
            loss = loss_r_v
        else:
            loss = loss_r_t
        if loss < lossBest: # Check if model was better than previous
            lossBest = loss
            epochs_since_best = 0
            torch.save(model.state_dict(), f"{model_name_best}")
        else:
            epochs_since_best += 1

        LOG.info(f'{epoch:2d}  cv_energy={cv_energy_t:.2e} ({cv_energy_v:.2e})  cv={cv_t:.2e} ({cv_v:.2e})  cvm={cv_max_t:.2e} ({cv_max_v:.2e}) reg={reg_t:.2e} ({reg_v:.2e})  reg2={reg2_t:.2e} ({reg2_v:.2e}) MAEv={MAE_v_t:.2e} ({MAE_v_v:.2e})   MAEr={MAE_r_t:.2e} ({MAE_r_v:.2e})  Loss_r={loss_r_t:.2e}({loss_r_v:.2e}) Lr: {lr:2.2e}  Time={t2 - t1:.1f}s ({t3 - t2:.1f}s)  '
                 f'Time(total) {(time.time() - t0)/3600:.1f}h')
        epoch += 1
        if (epoch % c['epochs_for_lr_adjustment']) == 0:
            for g in optimizer.param_groups:
                g['lr'] *= c['lr_adjustment']
                lr = g['lr']

    model.load_state_dict(torch.load(model_name_best))  # We load the best performing model
    if c['use_test']:
        t4 = time.time()
        loss_r_test, loss_v_test, drmsd_test, cv_test, cv_max_test,cv_energy_test,cv_energy_max_test, MAE_r_test, MAE_v_test, reg_test, reg2_test = run_model(c['data_type'], model, dataloader_test, train=False, max_samples=1000, optimizer=optimizer, loss_fnc=c['loss'],
                                                                                        max_radius=cn['max_radius'], debug=c['debug'], epoch=epoch,
                                                                                        output_folder=c['result_dir'], nviz=c['nviz'], regularization=c['regularization'],viz_paper=True)
        LOG.info(f'Test cv_energy={cv_energy_test:.2e}  cv={cv_test:.2e} cvm={cv_max_test:.2e} reg={reg_test:.2e} reg2={reg2_test:.2e}  MAEr={MAE_r_test:.2e}  MAEv={MAE_v_test:.2e}  Loss_r={loss_r_test:.2e} Time={t4 - time.time():.1f}s ')

        csv_file_test = f"{c['result_dir']}/test.csv"
        results_test = save_test_results_to_csv(loss_r_test, loss_v_test, cv_max_test, MAE_r_test, MAE_v_test, cv_test, cv_energy_test, cv_energy_max_test, csv_file_test)

    if c['viz']:
        plot_training_and_validation(results,c['result_dir'])

    if c['perform_endstep_MD_propagation']:
        assert c['data_type'] == 'water'
        run_model_MD_propagation_simulation(model, dataloader_endstep, log=LOG,viz=c['result_dir'])

    close_logger(LOG)
    # print(f"{torch.cuda.memory_allocated() / 1024 / 1024:2.2f}MB")
    # print(f"{torch.cuda.max_memory_allocated() / 1024 / 1024:2.2f}MB")
    # print(torch.cuda.memory_summary())
    return results, results_test, dataloader_train, dataloader_val, dataloader_test, dataloader_endstep
